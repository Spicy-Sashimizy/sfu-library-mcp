#!/usr/bin/env python3
"""Migrate the OpenSearch corpus into the thin-client stack
(tantivy BM25F + BMP SPLADE + usearch dense + sidecars), sectioned hot/cold.

Phases (all checkpointed in <index_root>/build_status.json):
  1. EXPORT  sliced scroll from the source index (_source includes the
     production sparse_field weights — no re-encoding needed) -> per-section
     zstd spool files. A slice that dies restarts cleanly (its files are
     deleted); completed slices are never re-exported.
  2. BUILD   per section: one streaming pass over its spools -> tantivy index
     + BMP shards + meta.sqlite rows + abstracts.sqlite (hot sections only).
  3. PACK    cold sections (everything not in the persona's hot list) ->
     packed/<name>.tar.zst (zstd-19 LDM), live dir removed.
  4. DENSE   usearch b1 + int8 rescore from the existing dense-POC vectors.

Usage
─────
    # 1M-doc validation build (single scroll, everything hot):
    .venv/bin/python3 scripts/build_thinclient_index.py \
        --limit 1000000 --persona all_hot --index-root data/thinclient_1m

    # Full 150M migration (defaults: data/thinclient_index, 8 slices/4 workers):
    SFU_MIGRATION_SOURCE=http://host.docker.internal:9200 \
    .venv/bin/python3 scripts/build_thinclient_index.py --persona political_science

Spool files are deleted after each section builds (disk headroom at 150M);
pass --keep-spool to retain them.
"""

import argparse
import json
import logging
import os
import shutil
import sys
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from pathlib import Path

import requests
import zstandard

try:  # ~6x faster JSON for the 150M-doc stream
    import orjson
    _loads = orjson.loads
    def _dumps(o) -> str:
        return orjson.dumps(o).decode()
except ImportError:
    _loads = json.loads
    _dumps = json.dumps

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("build_thinclient")

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from lib.thinclient.builder import (BMP_BLOCK_SIZE, META_SCHEMA, QUANT_SCALE,  # noqa: E402
                                    SectionBuilder, build_dense_leg, open_meta_db)
from lib.thinclient.packer import pack_section  # noqa: E402
from lib.thinclient.sections import (ERAS, PERSONAS, SECTION_NAMES,  # noqa: E402
                                     SUBSECTION_NAMES, classify_doc, era_of)

SOURCE_URL = os.environ.get("SFU_MIGRATION_SOURCE",
                            "http://host.docker.internal:9200").rstrip("/")
SOURCE_INDEX = os.environ.get("SFU_MIGRATION_INDEX", "openalex_works")
EXPORT_FIELDS = ["title", "abstract", "publication_year", "type", "is_oa",
                 "doi", "openalex_id", "sparse_field"]
SCROLL_BATCH = 4000
DENSE_VECS = REPO_ROOT / "data/dense_compression/vectors_600k.npy"
DENSE_IDS = REPO_ROOT / "data/dense_compression/ids_600k.json"


# ── status / checkpointing ────────────────────────────────────────────────────

def load_status(root: Path) -> dict:
    p = root / "build_status.json"
    if p.exists():
        return json.loads(p.read_text())
    return {"phase": "export", "slices_done": [], "sections_built": [],
            "sections_packed": [], "dense_done": False, "docs_exported": 0,
            "started": time.strftime("%Y-%m-%dT%H:%M:%S")}


def save_status(root: Path, status: dict) -> None:
    status["updated"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    tmp = root / "build_status.json.tmp"
    tmp.write_text(json.dumps(status, indent=2))
    os.replace(tmp, root / "build_status.json")


# ── phase 1: export ───────────────────────────────────────────────────────────

class SectionSpool:
    """Per-slice writer set: spool/<section>/slice_NNN.jsonl.zst"""

    def __init__(self, spool_dir: Path, slice_id: int):
        self.spool_dir = spool_dir
        self.slice_id = slice_id
        self._writers: dict[str, tuple] = {}

    def _writer(self, section: str):
        if section not in self._writers:
            d = self.spool_dir / section
            d.mkdir(parents=True, exist_ok=True)
            fh = open(d / f"slice_{self.slice_id:03d}.jsonl.zst", "wb")
            zw = zstandard.ZstdCompressor(level=3).stream_writer(fh)
            self._writers[section] = (fh, zw)
        return self._writers[section][1]

    def write(self, section: str, doc: dict) -> None:
        self._writer(section).write((_dumps(doc) + "\n").encode())

    def close(self) -> None:
        for fh, zw in self._writers.values():
            zw.close()
            fh.close()

    def delete_partial(self) -> None:
        self.close()
        for section in SECTION_NAMES:
            p = self.spool_dir / section / f"slice_{self.slice_id:03d}.jsonl.zst"
            p.unlink(missing_ok=True)


def export_slice(spool_dir: Path, slice_id: int, max_slices: int,
                 limit: int | None) -> int:
    """Scroll one slice of the source index into per-section spool files."""
    spool = SectionSpool(spool_dir, slice_id)
    n = 0
    try:
        body: dict = {"size": SCROLL_BATCH, "query": {"match_all": {}},
                      "_source": EXPORT_FIELDS}
        if max_slices > 1:
            body["slice"] = {"id": slice_id, "max": max_slices}
        r = requests.post(f"{SOURCE_URL}/{SOURCE_INDEX}/_search?scroll=10m",
                          json=body, timeout=120)
        r.raise_for_status()
        data = _loads(r.content)
        scroll_id = data["_scroll_id"]
        t0 = time.perf_counter()
        try:
            while True:
                hits = data["hits"]["hits"]
                if not hits:
                    break
                for h in hits:
                    doc = h["_source"]
                    doc["id"] = h["_id"]
                    spool.write(classify_doc(doc.get("title"), doc.get("abstract")), doc)
                    n += 1
                if limit and n >= limit:
                    break
                if n % 200_000 < SCROLL_BATCH:
                    rate = n / max(time.perf_counter() - t0, 1)
                    logger.info("slice %d: %s docs (%.0f docs/s)", slice_id, f"{n:,}", rate)
                data = _loads(requests.post(
                    f"{SOURCE_URL}/_search/scroll",
                    json={"scroll": "10m", "scroll_id": scroll_id},
                    timeout=120).content)
                scroll_id = data.get("_scroll_id", scroll_id)
        finally:
            requests.delete(f"{SOURCE_URL}/_search/scroll",
                            json={"scroll_id": scroll_id}, timeout=30)
        spool.close()
        logger.info("slice %d complete: %s docs", slice_id, f"{n:,}")
        return n
    except Exception:
        logger.exception("slice %d failed — removing partial spool files", slice_id)
        spool.delete_partial()
        raise


def phase_export(root: Path, status: dict, slices: int, workers: int,
                 limit: int | None) -> None:
    spool_dir = root / "spool"
    todo = [s for s in range(slices) if s not in status["slices_done"]]
    if not todo:
        return
    logger.info("EXPORT: %d/%d slices to go (source %s/%s)",
                len(todo), slices, SOURCE_URL, SOURCE_INDEX)
    # On resume, budget only what's left of the limit, or the rerun slices
    # re-export a full share each and the total overshoots.
    remaining = (max(limit - status["docs_exported"], 0) if limit else None)
    per_slice_limit = (remaining // max(len(todo), 1)) if remaining else None
    # Processes, not threads: orjson parse + classify + spool write are
    # GIL-bound (measured: 4 threads aggregate ~1.6k docs/s, same as 1).
    pool_cls = ThreadPoolExecutor if (limit and slices == 1) else ProcessPoolExecutor
    with pool_cls(max_workers=workers) as pool:
        futs = {pool.submit(export_slice, spool_dir, s, slices, per_slice_limit): s
                for s in todo}
        for fut in as_completed(futs):
            s = futs[fut]
            n = fut.result()  # raises on slice failure -> rerun resumes cleanly
            status["slices_done"].append(s)
            status["docs_exported"] += n
            save_status(root, status)
    status["phase"] = "build"
    save_status(root, status)


# ── phase 2: build sections ──────────────────────────────────────────────────

def iter_spool(spool_dir: Path, section: str):
    dctx = zstandard.ZstdDecompressor()
    for path in sorted((spool_dir / section).glob("slice_*.jsonl.zst")):
        with open(path, "rb") as fh, dctx.stream_reader(fh) as reader:
            buf = b""
            while True:
                chunk = reader.read(8 << 20)
                if not chunk:
                    break
                buf += chunk
                *lines, buf = buf.split(b"\n")
                for ln in lines:
                    if ln:
                        yield _loads(ln)
            if buf.strip():
                yield _loads(buf)


def build_section_worker(root_str: str, section: str, hot_subs: list[str],
                         keep_spool: bool, bmp_shard_docs: int) -> dict:
    """Process-pool worker: builds one BASE section's spool into its two era
    sub-sections (sections/<base>__<era>/), with its OWN meta db
    (meta_<section>.sqlite, merged into meta.sqlite afterwards)."""
    import sqlite3

    root = Path(root_str)
    spool_dir = root / "spool"
    for era in ERAS:
        live = root / "sections" / f"{section}__{era}"
        if live.exists():
            shutil.rmtree(live)

    meta_path = root / f"meta_{section}.sqlite"
    meta_path.unlink(missing_ok=True)
    meta_db = sqlite3.connect(str(meta_path))
    meta_db.executescript(META_SCHEMA)
    t0 = time.perf_counter()
    builders: dict[str, SectionBuilder] = {}
    for doc in iter_spool(spool_dir, section):
        era = era_of(int(doc.get("publication_year") or 0))
        b = builders.get(era)
        if b is None:
            sub = f"{section}__{era}"
            b = builders[era] = SectionBuilder(
                root, sub, hot=sub in hot_subs, meta_db=meta_db,
                bmp_shard_docs=bmp_shard_docs)
        b.add(doc)
    infos: dict[str, dict] = {}
    for era, b in builders.items():
        infos[f"{section}__{era}"] = b.finish()
    meta_db.close()
    secs = round(time.perf_counter() - t0, 1)
    for info in infos.values():
        info["build_secs"] = secs
    if not keep_spool:
        shutil.rmtree(spool_dir / section, ignore_errors=True)
    return infos


def merge_section_meta(root: Path, sections: list[str]) -> None:
    main = open_meta_db(root)
    for section in sections:
        part = root / f"meta_{section}.sqlite"
        if not part.exists():
            continue
        main.execute("ATTACH DATABASE ? AS part", (str(part),))
        main.execute("INSERT OR REPLACE INTO docs SELECT * FROM part.docs")
        main.execute("INSERT OR REPLACE INTO docs_other "
                     "SELECT * FROM part.docs_other")
        main.commit()
        main.execute("DETACH DATABASE part")
        part.unlink()
        logger.info("meta: merged %s", section)
    main.close()


def phase_build(root: Path, status: dict, hot_sections: list[str],
                keep_spool: bool, bmp_shard_docs: int,
                build_workers: int) -> None:
    spool_dir = root / "spool"
    todo = []
    for section in SECTION_NAMES:
        if section in status["sections_built"]:
            continue
        if not (spool_dir / section).is_dir():
            logger.info("BUILD: section %s has no spool (0 docs) — skipping", section)
            status["sections_built"].append(section)
            save_status(root, status)
            continue
        todo.append(section)
    if todo:
        # Largest sections first so the long pole starts immediately.
        todo.sort(key=lambda s: -sum(f.stat().st_size
                                     for f in (spool_dir / s).glob("*")))
        logger.info("BUILD: %s with %d workers", todo, build_workers)
        with ProcessPoolExecutor(max_workers=build_workers) as pool:
            futs = {pool.submit(build_section_worker, str(root), s,
                                hot_sections, keep_spool, bmp_shard_docs): s
                    for s in todo}
            for fut in as_completed(futs):
                section = futs[fut]
                infos = fut.result()   # {sub_section: info} (per era)
                status.setdefault("section_info", {}).update(infos)
                status["sections_built"].append(section)
                save_status(root, status)
    # Over ALL sections, not just this run's: a crash between a section build
    # and the merge leaves its meta_<section>.sqlite orphaned on resume
    # (merge skips files that no longer exist, so this stays idempotent).
    merge_section_meta(root, list(SECTION_NAMES))
    status["phase"] = "pack"
    save_status(root, status)


# ── phase 3: pack cold sections ──────────────────────────────────────────────

def phase_pack(root: Path, status: dict, hot_sections: list[str]) -> None:
    for section in SUBSECTION_NAMES:
        if section in hot_sections or section in status["sections_packed"]:
            continue
        if not (root / "sections" / section).is_dir():
            continue
        logger.info("PACK: %s (cold)", section)
        info = pack_section(root, section)
        status.setdefault("pack_info", {})[section] = info
        status["sections_packed"].append(section)
        save_status(root, status)
    status["phase"] = "dense"
    save_status(root, status)


# ── phase 4: dense leg ───────────────────────────────────────────────────────

def phase_dense(root: Path, status: dict) -> None:
    if status["dense_done"]:
        return
    if not DENSE_VECS.exists():
        logger.warning("DENSE: %s missing — skipping dense leg", DENSE_VECS)
    else:
        status["dense_info"] = build_dense_leg(root, DENSE_VECS, DENSE_IDS)
    status["dense_done"] = True
    status["phase"] = "done"
    save_status(root, status)


def write_manifest(root: Path, persona: str, hot_sections: list[str],
                   status: dict) -> None:
    from lib.thinclient.packer import _load_manifest, _save_manifest
    manifest = _load_manifest(root)
    manifest.update({
        "built": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "source": f"{SOURCE_URL}/{SOURCE_INDEX}",
        "docs_exported": status["docs_exported"],
        "persona": persona,
        "hot_sections": hot_sections,
        "splade_quant_scale": QUANT_SCALE,
        "engines": {"lexical": "tantivy 0.26 (BM25F title^3/abstract, freq-only)",
                    "sparse": f"bmp 0.2.6 (8-bit block-max impacts, "
                              f"bsize={BMP_BLOCK_SIZE}, clustered)",
                    "dense": "usearch b1 + int8 rescore (binary+rescore 32x)"},
    })
    for section in SUBSECTION_NAMES:
        entry = manifest["sections"].setdefault(section, {})
        if "state" not in entry:
            entry["state"] = "live" if (root / "sections" / section).is_dir() else "absent"
        if section in status.get("section_info", {}):
            entry["docs"] = status["section_info"][section]["docs"]
            entry["hot"] = section in hot_sections
    _save_manifest(root, manifest)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index-root", default=str(REPO_ROOT / "data/thinclient_index"))
    parser.add_argument("--persona", default="political_science",
                        choices=sorted(PERSONAS))
    parser.add_argument("--limit", type=int, default=None,
                        help="cap exported docs (validation builds)")
    parser.add_argument("--slices", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--bmp-shard-docs", type=int, default=2_000_000)
    parser.add_argument("--build-workers", type=int, default=2,
                        help="parallel section builds (RAM-bound: ~6-8 GB each — "
                             "each worker runs TWO era builders; 3 workers "
                             "OOM-killed the 150M build on a 31 GB host)")
    parser.add_argument("--keep-spool", action="store_true")
    args = parser.parse_args()

    if args.limit:
        args.slices = 1
        args.workers = 1

    root = Path(args.index_root)
    root.mkdir(parents=True, exist_ok=True)
    hot = PERSONAS[args.persona]
    status = load_status(root)
    logger.info("index root %s | persona %s (hot: %s) | resuming at phase=%s",
                root, args.persona, ",".join(hot), status["phase"])

    phase_export(root, status, args.slices, args.workers, args.limit)
    phase_build(root, status, hot, args.keep_spool, args.bmp_shard_docs,
                args.build_workers)
    phase_pack(root, status, hot)
    phase_dense(root, status)
    write_manifest(root, args.persona, hot, status)

    logger.info("DONE. %s docs across sections: %s", f"{status['docs_exported']:,}",
                json.dumps(status.get("section_info", {}), indent=2))


if __name__ == "__main__":
    main()
