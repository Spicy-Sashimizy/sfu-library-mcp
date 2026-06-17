"""ThinClientRetriever — drop-in replacement for OpenSearchRetriever.

Same public surface (search / dense_search / is_available, same result dict
shape) backed by tantivy + BMP + usearch over a built index root. The
FederatedSearchRouter and reranker need zero changes.

Cross-section score merging:
  - SPLADE (BMP) and dense scores are corpus-independent dot products — exact
    merge across sections/shards.
  - BM25F scores use per-section IDF, so cross-section merge is approximate
    (same caveat as multi-shard OpenSearch; measured 40/80 score-multiset drift
    for 2-shard in docs/archive/COMPRESSION_EVAL_RESULTS.md). RRF fusion downstream is
    rank-based per leg, which absorbs most of the drift.

Filters: tantivy enforces year/type/is_oa natively (fast fields); BMP/usearch
legs over-fetch and post-filter against meta.sqlite (the sidecar-mask pattern
from docs/THIN_CLIENT_STACK_RESEARCH.md).

Abstract policy: hot sections answer locally from the zstd-dict sidecar; ids
missing locally are batch-fetched from the OpenAlex API (inverted-abstract
parser in lib.openalex) when remote_abstracts=True — BEFORE rerank, never
display-only.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any

from lib.thinclient.sections import ERA_BOUNDARY_YEAR, ERAS, classify_query

logger = logging.getLogger("sfu_library_mcp")

# Per-leg in-memory metrics (reset on restart), mirrors tools._metrics.
_leg_metrics: dict[str, dict] = {}
_metrics_lock = threading.Lock()


def _mem_available_gb() -> float | None:
    """MemAvailable in GiB (kernel's reclaim-aware free estimate), or None if
    /proc/meminfo is unreadable (non-Linux)."""
    try:
        with open("/proc/meminfo") as fh:
            for line in fh:
                if line.startswith("MemAvailable"):
                    return int(line.split()[1]) / 1024 / 1024
    except OSError:
        return None
    return None


def _guard_load_mem(next_shard_bytes: int, floor_gb: float, ctx: str) -> None:
    """Abort the load CLEANLY if pulling the next BMP shard would drive
    MemAvailable below `floor_gb`. BMP loads resident (~3.07x on disk), so a big
    shard can swing several GB; we project the shard's resident cost and refuse
    *before* the allocation rather than letting the OOM-killer SIGKILL us mid-
    construction (which leaves no actionable error). See BMP_RESIDENT_RATIO."""
    avail = _mem_available_gb()
    if avail is None:
        return
    projected = next_shard_bytes / 1073741824 * BMP_RESIDENT_RATIO
    if avail - projected < floor_gb:
        raise RuntimeError(
            f"thinclient load aborted at {ctx}: MemAvailable {avail:.1f} GB, "
            f"next BMP shard needs ~{projected:.1f} GB resident "
            f"(~{BMP_RESIDENT_RATIO:.2f}x on disk), which would breach the "
            f"{floor_gb:.1f} GB floor. BMP has no mmap mode; the full 150M "
            f"SPLADE set is ~211 GB resident and cannot serve on this host. "
            f"Raise host RAM, serve fewer sections, or set "
            f"SFU_LOAD_MEM_FLOOR_GB lower to override (will risk OOM-kill).")


def _record_leg(leg: str, ms: float, error: bool = False) -> None:
    with _metrics_lock:
        m = _leg_metrics.setdefault(leg, {"count": 0, "errors": 0, "total_ms": 0.0})
        m["count"] += 1
        m["total_ms"] += ms
        if error:
            m["errors"] += 1

# BMP is a load-into-memory engine (no mmap mode in 0.2.6): `bmp.Searcher`
# deserializes each *.bmp shard into ANONYMOUS RAM at a measured ~3.07x its
# on-disk size (probe 2026-06-17, 569 MB shard -> 1747 MB steady-state RSS;
# ratio holds from 12 MB to 569 MB shards). The full 150M SPLADE set is 68.6 GB
# on disk -> ~211 GB resident. _load() builds ALL shards eagerly, so on a host
# that can't hold the set the process is OOM-killed *inside* construction —
# before any per-query mem-floor guard can run. The guard below samples
# MemAvailable before each shard and aborts CLEANLY (RuntimeError) instead, so
# the failure is diagnosable rather than a SIGKILL. Gate: SFU_LOAD_MEM_FLOOR_GB.
BMP_RESIDENT_RATIO = 3.07       # measured anon-RAM expansion of bmp.Searcher
LOAD_MEM_FLOOR_GB = 1.5         # abort load if MemAvailable would drop below this

RRF_K = 60
QUANT_SCALE = 70                # must match builder.QUANT_SCALE (saturation-free)
SPLADE_QUERY_TERMS = 64         # top query terms, same cap as the OpenSearch leg
OVERFETCH = 4                   # sparse/dense over-fetch multiplier (no filters)
OVERFETCH_FILTERED = 10         # ... when post-filtering
# BMP block-max approximation: measured on the 1M build (10 SPLADE queries,
# 64 terms): alpha=0.8 keeps top-50 overlap 1.000 at -29% latency; beta<1.0
# prunes query terms and visibly costs quality (0.5 -> overlap 0.80).
BMP_ALPHA = 0.8
BMP_BETA = 1.0


class ThinClientRetriever:
    """Retriever over a thin-client index root (tantivy + BMP + usearch)."""

    def __init__(
        self,
        index_root: str = "",
        splade_model_path: str = "",
        dense_model_path: str = "",
        remote_abstracts: bool = True,
        openalex_mailto: str = "",
        query_log_path: str = "",
    ):
        repo_root = Path(__file__).resolve().parents[3]
        self.root = Path(index_root or repo_root / "data" / "thinclient_index")
        self.splade_model_path = splade_model_path or str(repo_root / "models" / "splade_onnx")
        self.dense_model_path = dense_model_path or str(repo_root / "models" / "sfu-academic-embed-v5")
        self.remote_abstracts = remote_abstracts
        self.openalex_mailto = openalex_mailto
        self.query_log_path = query_log_path
        self._qlog_lock = threading.Lock()

        self._lock = threading.Lock()
        self._sections: dict[str, dict] = {}   # name -> {tantivy, searcher, bmp:[...]}
        self._abstract_stores: dict[str, Any] = {}
        self._meta: sqlite3.Connection | None = None
        self._meta_numeric = False
        self._meta_lock = threading.Lock()
        self._dense: dict | None = None
        self._dense_cache = None   # query-driven warm cache (dense_cache.py)
        self._loaded = False

    # ── lazy loading ────────────────────────────────────────────────────────

    def _manifest(self) -> dict:
        p = self.root / "manifest.json"
        return json.loads(p.read_text()) if p.exists() else {}

    def _load(self) -> None:
        if self._loaded:
            return
        with self._lock:
            if self._loaded:
                return
            import os

            import bmp
            import tantivy

            from lib.thinclient.abstracts import AbstractStore

            load_floor = float(os.environ.get("SFU_LOAD_MEM_FLOOR_GB",
                                              LOAD_MEM_FLOOR_GB))
            sections_dir = self.root / "sections"
            if sections_dir.is_dir():
                for sdir in sorted(sections_dir.iterdir()):
                    if not (sdir / "tantivy" / "meta.json").exists():
                        continue
                    idx = tantivy.Index.open(str(sdir / "tantivy"))
                    idx.reload()
                    shards = []
                    for p in sorted(sdir.glob("splade_*.bmp")):
                        _guard_load_mem(p.stat().st_size, load_floor,
                                        f"{sdir.name}/{p.name}")
                        vocab_path = p.with_suffix(".vocab.zst")
                        vocab = None
                        if vocab_path.exists():
                            import zstandard
                            try:
                                vocab = set(zstandard.ZstdDecompressor().decompress(
                                    vocab_path.read_bytes()).decode().split("\n"))
                            except Exception as e:
                                # A corrupt sidecar must not take down the whole
                                # index; vocab=None falls back to querying the
                                # shard without the zero-overlap guard.
                                logger.warning("thinclient: bad vocab sidecar %s "
                                               "(%s) — loading shard without it",
                                               vocab_path, e)
                                vocab = None
                        shards.append({"searcher": bmp.Searcher(str(p)),
                                       "vocab": vocab})
                    self._sections[sdir.name] = {
                        "index": idx, "searcher": idx.searcher(),
                        "schema": idx.schema, "bmp": shards,
                    }
                    abs_path = sdir / "abstracts.sqlite"
                    if abs_path.exists():
                        self._abstract_stores[sdir.name] = AbstractStore(abs_path)
            meta_path = self.root / "meta.sqlite"
            if meta_path.exists():
                self._meta = sqlite3.connect(str(meta_path), check_same_thread=False)
                # v2 schema stores W-ids as INTEGER PRIMARY KEY (+ docs_other
                # TEXT overflow); v1 indexes (e.g. thinclient_1m) are TEXT.
                cols = {r[1]: (r[2] or "").upper() for r in
                        self._meta.execute("PRAGMA table_info(docs)")}
                self._meta_numeric = cols.get("id", "").startswith("INT")
            dense_dir = self.root / "dense"
            if (dense_dir / "b1.usearch").exists():
                import numpy as np
                from usearch.index import Index
                view = Index.restore(str(dense_dir / "b1.usearch"), view=True)
                self._dense = {
                    "index": view,
                    "rescore": np.load(dense_dir / "rescore_int8.npy", mmap_mode="r"),
                    "ids": json.loads((dense_dir / "ids.json").read_text()),
                }
            if os.environ.get("SFU_DENSE_WARMCACHE", "1") != "0":
                try:
                    from lib.thinclient.dense_cache import DenseWarmCache
                    self._dense_cache = DenseWarmCache(
                        dense_dir, seed_ids=(self._dense or {}).get("ids", []),
                        model_path=self.dense_model_path)
                except Exception as exc:
                    logger.warning("dense warm cache unavailable: %s", exc)
                    self._dense_cache = None
            self._loaded = True
            logger.info("thinclient: loaded %d live sections %s, dense=%s",
                        len(self._sections), sorted(self._sections),
                        bool(self._dense))

    # ── public surface (OpenSearchRetriever-compatible) ─────────────────────

    def is_available(self) -> bool:
        try:
            self._load()
        except Exception as exc:
            logger.warning("thinclient index unavailable: %s", exc)
            return False
        return bool(self._sections)

    def live_sections(self) -> list[str]:
        self._load()
        return sorted(self._sections)

    def cold_section_hint(self, query: str,
                          filters: dict | None = None) -> str | None:
        """Sub-section this query classifies to when it is NOT live — i.e.
        results may be missing because the home (sub-)section is packed/absent.
        Era-aware: a year filter that excludes an era suppresses hints for it.
        Used for routing telemetry and client-visible unpack hints."""
        self._load()
        base = classify_query(query)
        if base in self._sections:          # legacy era-less layout
            return None
        f = _normalize_filters(filters)
        for era in ERAS:                    # recent first: the era axis leads
            sub = f"{base}__{era}"
            if sub in self._sections or _era_skip(sub, f):
                continue
            return sub
        return None

    def metrics(self) -> dict:
        """Per-leg latency/error counters + index coverage (for /health)."""
        try:
            self._load()
        except Exception:
            pass
        with _metrics_lock:
            legs = {k: dict(v) for k, v in _leg_metrics.items()}
        for m in legs.values():
            m["avg_ms"] = round(m["total_ms"] / max(m["count"], 1), 1)
            m["total_ms"] = round(m["total_ms"], 1)
        return {
            "legs": legs,
            "live_sections": sorted(self._sections),
            "bmp_shards": {n: len(s["bmp"]) for n, s in self._sections.items()},
            "dense_vectors": len(self._dense["ids"]) if self._dense else 0,
            "dense_warm_cache": (self._dense_cache.info()
                                 if self._dense_cache else None),
            "meta_schema": "v2-numeric" if self._meta_numeric else "v1-text",
            "manifest": self._manifest(),
        }

    def reload(self) -> dict:
        """Atomically re-load the index root (zero-downtime swap after a
        rebuild/unpack). In-flight queries finish on the old objects; the
        swap is a single attribute rebind under the load lock."""
        with self._lock:
            old = (self._sections, self._abstract_stores, self._meta,
                   self._dense, self._dense_cache)
            self._sections, self._abstract_stores = {}, {}
            self._meta, self._dense, self._dense_cache = None, None, None
            self._loaded = False
        try:
            self._load()
        except Exception:
            # Failed swap: restore the previous live objects.
            with self._lock:
                (self._sections, self._abstract_stores, self._meta,
                 self._dense, self._dense_cache) = old
                self._loaded = True
            raise
        if old[2] is not None:
            try:
                old[2].close()
            except sqlite3.Error:
                pass
        if old[4] is not None:
            old[4].close()
        return {"live_sections": sorted(self._sections),
                "dense_vectors": len(self._dense["ids"]) if self._dense else 0}

    def _log_query(self, leg: str, query: str, n: int, ms: float,
                   f: dict | None) -> None:
        if not self.query_log_path:
            return
        try:
            entry = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "leg": leg,
                     "query": query, "results": n, "latency_ms": round(ms, 1),
                     "filtered": bool(f), "cold_hint": self.cold_section_hint(query)}
            with self._qlog_lock, open(self.query_log_path, "a") as fh:
                fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except OSError:
            pass  # a bad log path must never break a query

    def search(self, query: str, top_k: int = 50, mode: str | None = None,
               filters: dict | None = None) -> list[dict]:
        """mode: 'bm25f' (default) or 'splade' — same contract as the
        OpenSearch retriever so FederatedSearchRouter can RRF-fuse both."""
        self._load()
        if not self._sections:
            return []
        use_splade = mode == "splade"
        leg = "splade" if use_splade else "bm25f"
        f = _normalize_filters(filters)
        t0 = time.perf_counter()
        try:
            if use_splade:
                ranked = self._splade_leg(query, top_k, f)
            else:
                ranked = self._bm25f_leg(query, top_k, f)
        except Exception:
            _record_leg(leg, (time.perf_counter() - t0) * 1000, error=True)
            raise
        ms = (time.perf_counter() - t0) * 1000
        _record_leg(leg, ms)
        out = self._hydrate(ranked, source="thinclient" if not use_splade
                            else "thinclient_splade")
        self._log_query(leg, query, len(out), ms, f)
        return out

    def dense_search(self, query: str, top_k: int = 50,
                     dense_index: str = "", dense_model_path: str = "",
                     filters: dict | None = None) -> list[dict]:
        """usearch b1 + int8 rescore. Coverage = the dense POC subset until the
        full corpus is encoded (same partial-coverage caveat as the OpenSearch
        dense leg)."""
        self._load()
        cache = self._dense_cache
        if not self._dense and not (cache and cache.info()["docs"]):
            return []
        import numpy as np

        from lib.opensearch_retriever import encode_dense
        t0 = time.perf_counter()
        q = np.asarray(encode_dense(query, dense_model_path or self.dense_model_path),
                       dtype=np.float32)
        f = _normalize_filters(filters)
        fetch = top_k * (OVERFETCH_FILTERED if f else OVERFETCH)
        qbits = np.packbits((q > 0).astype(np.uint8))
        merged: dict[str, float] = {}
        if self._dense:
            hits = self._dense["index"].search(qbits, fetch)
            keys = np.asarray(hits.keys, dtype=np.int64).ravel()
            if keys.size:
                rescore = self._dense["rescore"]
                scores = (rescore[keys].astype(np.float32) / 127.0) @ q
                ids_all = self._dense["ids"]
                for i in range(keys.size):
                    merged[ids_all[int(keys[i])]] = float(scores[i])
        if cache:
            # Warm-cache delta: exact same int8·fp32 scoring — merges by score.
            for did, score in cache.search(qbits, q, fetch):
                if score > merged.get(did, float("-inf")):
                    merged[did] = score
        if not merged:
            return []
        ranked = sorted(merged.items(), key=lambda t: -t[1])
        if f:
            ranked = self._post_filter(ranked, f)
        ms = (time.perf_counter() - t0) * 1000
        _record_leg("dense", ms)
        out = self._hydrate(ranked[:top_k], source="thinclient_dense")
        self._log_query("dense", query, len(out), ms, f)
        return out

    def search_rrf(self, query: str, top_k: int = 50,
                   filters: dict | None = None, include_dense: bool = False) -> list[dict]:
        """Convenience 2/3-leg RRF (k=60) — same fusion the router applies."""
        legs = [self.search(query, top_k, mode="bm25f", filters=filters),
                self.search(query, top_k, mode="splade", filters=filters)]
        if include_dense and self._dense:
            legs.append(self.dense_search(query, top_k, filters=filters))
        scores: dict[str, float] = {}
        by_id: dict[str, dict] = {}
        for leg in legs:
            for rank, doc in enumerate(leg, start=1):
                did = doc["openalex_id"]
                scores[did] = scores.get(did, 0.0) + 1.0 / (RRF_K + rank)
                by_id.setdefault(did, doc)
        out = []
        for did in sorted(scores, key=lambda d: scores[d], reverse=True)[:top_k]:
            doc = dict(by_id[did])
            doc["score"] = scores[did]
            doc["source"] = "thinclient_rrf"
            out.append(doc)
        return out

    # ── legs ─────────────────────────────────────────────────────────────────

    def _bm25f_leg(self, query: str, top_k: int,
                   f: dict | None) -> list[tuple[str, float]]:
        import tantivy
        text = "".join(c if c.isalnum() or c.isspace() else " " for c in query)
        if not text.strip():
            return []
        merged: list[tuple[str, float]] = []
        for name, sec in self._sections.items():
            if _era_skip(name, f):
                continue
            idx, searcher = sec["index"], sec["searcher"]
            qt = idx.parse_query(text, ["title"])
            qa = idx.parse_query(text, ["abstract"])
            clauses = [
                (tantivy.Occur.Should, tantivy.Query.boost_query(qt, 3.0)),
                (tantivy.Occur.Should, qa),
            ]
            q = tantivy.Query.boolean_query(clauses)
            must = [(tantivy.Occur.Must, q)]
            if f and f.get("year_range"):
                lo, hi = f["year_range"]
                must.append((tantivy.Occur.Must, tantivy.Query.range_query(
                    sec["schema"], "year", tantivy.FieldType.Integer, lo, hi)))
            if f and f.get("type"):
                must.append((tantivy.Occur.Must, tantivy.Query.term_query(
                    sec["schema"], "doctype", f["type"])))
            if f and f.get("is_oa"):
                must.append((tantivy.Occur.Must, tantivy.Query.term_query(
                    sec["schema"], "is_oa", True)))
            final = tantivy.Query.boolean_query(must) if len(must) > 1 else q
            for score, addr in searcher.search(final, top_k).hits:
                merged.append((searcher.doc(addr)["id"][0], float(score)))
        merged.sort(key=lambda t: -t[1])
        return merged[:top_k]

    def _splade_leg(self, query: str, top_k: int,
                    f: dict | None) -> list[tuple[str, float]]:
        from lib.opensearch_retriever import encode_splade
        try:
            sparse = encode_splade(query, self.splade_model_path)
        except Exception as exc:
            # Missing/corrupt ONNX model must degrade the leg, not the query:
            # RRF still fuses BM25F (+dense) while this leg mirrors BM25F.
            logger.warning("SPLADE encoder failed (%s); falling back to BM25F leg", exc)
            sparse = None
        if not sparse:
            logger.warning("SPLADE encoding empty; falling back to BM25F leg")
            return self._bm25f_leg(query, top_k, f)
        top_terms = dict(sorted(sparse.items(), key=lambda x: -x[1])[:SPLADE_QUERY_TERMS])
        qvec = {t: max(1, int(round(w * QUANT_SCALE))) for t, w in top_terms.items()}
        fetch = top_k * (OVERFETCH_FILTERED if f else 1)
        merged: list[tuple[str, float]] = []
        for name, sec in self._sections.items():
            if _era_skip(name, f):
                continue
            for shard in sec["bmp"]:
                # BMP panics (Rust unwrap) when NO query term exists in the
                # shard — skip via the vocab sidecar; guard for old builds.
                q_here = qvec
                if shard["vocab"] is not None:
                    q_here = {t: w for t, w in qvec.items() if t in shard["vocab"]}
                    if not q_here:
                        continue
                try:
                    ids, scores = shard["searcher"].search(
                        q_here, k=fetch, alpha=BMP_ALPHA, beta=BMP_BETA)
                except BaseException as exc:  # pyo3 PanicException subclasses BaseException
                    if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                        raise
                    logger.warning("BMP shard search failed in %s: %s", name, exc)
                    continue
                merged.extend(zip(ids, map(float, scores)))
        merged.sort(key=lambda t: -t[1])
        if f:
            merged = self._post_filter(merged, f)
        return merged[:top_k]

    # ── helpers ───────────────────────────────────────────────────────────────

    def _meta_rows(self, ids: list[str]) -> dict[str, tuple]:
        if not ids:
            return {}
        out: dict[str, tuple] = {}
        if self._meta is None:
            if self._dense_cache is not None:
                out.update(self._dense_cache.metadata_rows(ids))
            return out
        # (warm-cache rows for ids missing from meta are filled in below)

        def _query(table: str, keys: list, back: dict | None) -> None:
            for start in range(0, len(keys), 500):
                chunk = keys[start:start + 500]
                marks = ",".join("?" * len(chunk))
                for row in self._meta.execute(
                        f"SELECT id, title, doi, year, type, is_oa, section "
                        f"FROM {table} WHERE id IN ({marks})", chunk):
                    did = back[row[0]] if back is not None else row[0]
                    out[did] = (did,) + tuple(row[1:])

        with self._meta_lock:
            if not self._meta_numeric:
                _query("docs", ids, None)
            else:
                from lib.thinclient.builder import encode_meta_id
                enc: dict[int, str] = {}
                other: list[str] = []
                for s in ids:
                    n = encode_meta_id(s)
                    (other.append(s) if n is None else enc.__setitem__(n, s))
                if enc:
                    _query("docs", list(enc), enc)
                if other:
                    _query("docs_other", other, None)
        if self._dense_cache is not None:
            missing = [i for i in ids if i not in out]
            if missing:
                out.update(self._dense_cache.metadata_rows(missing))
        return out

    def _post_filter(self, ranked: list[tuple[str, float]],
                     f: dict) -> list[tuple[str, float]]:
        rows = self._meta_rows([d for d, _ in ranked])
        keep = []
        for did, score in ranked:
            row = rows.get(did)
            if row is None:
                continue
            _, _, _, year, dtype, is_oa, _ = row
            if f.get("year_range"):
                lo, hi = f["year_range"]
                if not (year and lo <= year <= hi):
                    continue
            if f.get("type") and dtype != f["type"]:
                continue
            if f.get("is_oa") and not is_oa:
                continue
            keep.append((did, score))
        return keep

    def _hydrate(self, ranked: list[tuple[str, float]], source: str) -> list[dict]:
        ids = [d for d, _ in ranked]
        rows = self._meta_rows(ids)
        abstracts = self.fetch_abstracts(ids)
        results = []
        for did, score in ranked:
            row = rows.get(did)
            title, doi, year, dtype, is_oa = (
                (row[1], row[2], row[3], row[4], bool(row[5])) if row
                else ("", "", None, "", False))
            results.append({
                "doi": doi,
                "openalex_id": did,
                "title": title,
                "abstract": abstracts.get(did, ""),
                "publication_year": year,
                "date": str(year) if year is not None else "",
                "year": year,
                "score": score,
                "source": source,
                "type": dtype,
                "is_oa": is_oa,
            })
        if self._dense_cache is not None and results:
            # Query-driven dense growth: every surfaced doc is a candidate for
            # the warm-cache delta (fire-and-forget; dedup inside).
            self._dense_cache.enqueue(results)
        return results

    def fetch_abstracts(self, ids: list[str]) -> dict[str, str]:
        """Hot-section sidecars first; missing ids via OpenAlex API (cold-section
        policy: fetch BEFORE rerank, ~100 docs ≈ one mget)."""
        out: dict[str, str] = {}
        for store in self._abstract_stores.values():
            missing = [i for i in ids if i not in out]
            if not missing:
                break
            out.update(store.fetch(missing))
        missing = [i for i in ids if i not in out]
        if missing and self.remote_abstracts:
            out.update(self._fetch_remote_abstracts(missing))
        return out

    def _fetch_remote_abstracts(self, ids: list[str]) -> dict[str, str]:
        import requests
        out: dict[str, str] = {}
        try:
            for start in range(0, len(ids), 50):  # OpenAlex max 50 ids per filter
                chunk = ids[start:start + 50]
                params = {
                    "filter": "openalex_id:" + "|".join(chunk),
                    "per-page": str(len(chunk)),
                    "select": "id,abstract_inverted_index",
                }
                if self.openalex_mailto:
                    params["mailto"] = self.openalex_mailto
                r = requests.get("https://api.openalex.org/works", params=params,
                                 timeout=15)
                r.raise_for_status()
                for w in r.json().get("results", []):
                    wid = (w.get("id") or "").rsplit("/", 1)[-1]
                    inv = w.get("abstract_inverted_index")
                    if wid and inv:
                        out[wid] = _invert_abstract(inv)
        except requests.RequestException as exc:
            logger.warning("remote abstract fetch failed (%d ids): %s", len(ids), exc)
        return out


def _invert_abstract(inv: dict[str, list[int]]) -> str:
    """OpenAlex inverted-abstract-index -> plain text."""
    positions: list[tuple[int, str]] = []
    for word, idxs in inv.items():
        positions.extend((i, word) for i in idxs)
    positions.sort()
    return " ".join(w for _, w in positions)


def _era_skip(section_name: str, f: dict | None) -> bool:
    """Era pruning — the fast axis: a year-bounded query never needs the other
    era's sub-sections, so they are skipped outright (cross-discipline pruning
    is impossible; cross-era pruning is free)."""
    if not f or not f.get("year_range"):
        return False
    lo, hi = f["year_range"]
    if section_name.endswith("__recent"):
        return hi < ERA_BOUNDARY_YEAR
    if section_name.endswith("__archive"):
        return lo >= ERA_BOUNDARY_YEAR
    return False


def _normalize_filters(filters: dict | None) -> dict | None:
    """OpenAlex-style filter dict -> {year_range, type, is_oa} (mirrors
    OpenSearchRetriever._build_filter_clauses semantics)."""
    if not filters:
        return None

    def _to_year(value) -> int | None:
        try:
            return int(str(value)[:4])
        except (ValueError, TypeError):
            return None

    out: dict = {}
    lo, hi = None, None
    py = filters.get("publication_year")
    if py:
        text = str(py)
        if "-" in text:
            a, _, b = text.partition("-")
            lo, hi = _to_year(a), _to_year(b)
        else:
            lo = hi = _to_year(text)
    fd = filters.get("from_publication_date")
    if fd and lo is None:
        lo = _to_year(fd)
    td = filters.get("to_publication_date")
    if td and hi is None:
        hi = _to_year(td)
    if lo is not None or hi is not None:
        out["year_range"] = (lo if lo is not None else 0,
                             hi if hi is not None else 3000)
    if filters.get("type"):
        out["type"] = filters["type"]
    oa = filters.get("open_access.is_oa", filters.get("is_oa"))
    if oa not in (None, "", False, "false"):
        out["is_oa"] = True
    return out or None
