#!/usr/bin/env python3
"""Sectioned ("hot/cold") index eval: section the corpus by content domain, keep
only the user's home section(s) live, store the rest as aggressively compressed
archives, and unpack-on-demand for off-domain requests.

Scenario tested (from the thin-client discussion): a political-science student
keeps `social_sciences` unpacked; `med_bio`, `phys_eng`, `cs_math`, `other` live
as zstd-19 long-range-matched JSONL archives. When they make an off-domain
request (here: a med/bio query), the section is unpacked = decompressed +
bulk-rebuilt into a live index, then can be dropped and re-packed afterwards
(repack cost ≈ pack cost; the archive is kept, so "repack" is just deleting the
live index — re-export is only needed if the section received writes).

Method
──────
1. Reuse the 1M-doc `lexcomp_base` subset (built by eval_lexical_lossless.py).
2. Partition into DISJOINT sections by priority-ordered concept matching:
   a doc goes to the first section whose concept phrases match it.
3. Build each section index with the proven lossless "combined" config
   (zstd6 + _source minus sparse_field + freqs + noid), force_merge, measure.
4. Pack: scroll each section's docs (WITH sparse_field, from lexcomp_base) to
   NDJSON → zstd level 19 + long-distance matching. Measure size + time.
5. Unpack drill: rebuild `med_bio` from its archive into a fresh index; time
   decompress / bulk / refresh separately. Parity: 10 queries × {bm25f, splade}
   against the original section index — expect identical scores.
6. Report the disk math for the political-science-user scenario at subset scale
   and extrapolated to the 150.4M full corpus and a 15M laptop tier.

Usage
─────
    SFU_OPENSEARCH_URL=http://...:9200 \
    .venv/bin/python3 scripts/eval_sectioned_index.py [--keep-indices]
"""
import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

import requests
import zstandard

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("eval_sectioned")

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))

from eval_lexical_lossless import (  # noqa: E402
    OS_URL, req, store_bytes, force_merge, wait_task, variant_config, SPLADE_ONNX,
)

REPO_ROOT = Path(__file__).parent.parent
ARCHIVE_DIR = REPO_ROOT / "data/sectioned"
DEFAULT_OUTPUT = REPO_ROOT / "data/eval_results/sectioned_index_eval.json"
DEFAULT_DIVERSE = REPO_ROOT / "data/eval_results/diverse_queries.json"

BASE_INDEX = "lexcomp_base"
SUBSET_DOCS = 1_000_000
FULL_SCALE_DOCS = 150_413_098
LAPTOP_TIER_DOCS = 15_000_000

# Priority-ordered: a doc is assigned to the FIRST section whose phrases match
# its `concepts` field; `other` catches the rest. Phrases are OpenAlex concept
# names (concepts is a text field on lexcomp_base, which retains positions).
SECTIONS: list[tuple[str, list[str]]] = [
    ("social_sciences", ["political science", "sociology", "economics", "law",
                         "education", "psychology", "history", "philosophy",
                         "business", "geography"]),
    ("med_bio", ["medicine", "biology", "biochemistry", "genetics",
                 "neuroscience", "immunology", "microbiology"]),
    ("phys_eng", ["physics", "engineering", "materials science", "chemistry",
                  "environmental science", "geology"]),
    ("cs_math", ["computer science", "mathematics"]),
]
HOME_SECTION = "social_sciences"   # the political-science student's hot section
OFFREQ_SECTION = "med_bio"         # the off-domain request drill
PARITY_QUERIES = 10
PARITY_TOP_K = 20


def section_query(idx: int) -> dict:
    """bool query assigning docs to SECTIONS[idx] disjointly (or `other` if
    idx == len(SECTIONS): matches none of the section phrases)."""
    def phrases(terms: list[str]) -> list[dict]:
        return [{"match_phrase": {"concepts": t}} for t in terms]

    if idx < len(SECTIONS):
        _, terms = SECTIONS[idx]
        must_not = [p for i in range(idx) for p in phrases(SECTIONS[i][1])]
        return {"bool": {"should": phrases(terms), "minimum_should_match": 1,
                         "must_not": must_not}}
    all_phrases = [p for _, terms in SECTIONS for p in phrases(terms)]
    return {"bool": {"must_not": all_phrases}}


def wait_for_base() -> None:
    for _ in range(240):
        try:
            if req("GET", f"{BASE_INDEX}/_count?ignore_unavailable=true",
                   timeout=30).get("count", 0) >= SUBSET_DOCS:
                return
        except Exception:
            pass
        logger.info("waiting for %s (built by eval_lexical_lossless.py)…", BASE_INDEX)
        time.sleep(30)
    raise RuntimeError(f"{BASE_INDEX} not available with {SUBSET_DOCS} docs")


def build_sections() -> dict[str, dict]:
    cfg = variant_config("combined")
    names = [n for n, _ in SECTIONS] + ["other"]
    out: dict[str, dict] = {}
    for i, name in enumerate(names):
        index = f"lexsec_{name}"
        if requests.head(f"{OS_URL}/{index}", timeout=30).status_code != 200:
            logger.info("Building section %s …", index)
            req("PUT", index, cfg, timeout=60)
            body = {"source": {"index": BASE_INDEX, "query": section_query(i)},
                    "dest": {"index": index}}
            resp = req("POST", "_reindex?wait_for_completion=false&slices=2&refresh=true",
                       body, timeout=120)
            wait_task(resp["task"], f"reindex->{index}")
        req("POST", f"{index}/_refresh", timeout=120)
        if store_bytes(index)[1] > 1:
            force_merge(index)
        size, _ = store_bytes(index)
        docs = req("GET", f"{index}/_count", timeout=60)["count"]
        out[name] = {"index": index, "docs": docs, "live_mb": round(size / 1e6, 1)}
        logger.info("section %-16s %8d docs  %8.1f MB live", name, docs, size / 1e6)
    return out


def pack_section(name: str, idx: int) -> dict:
    """Scroll the section's docs (with sparse_field) from BASE and write a
    zstd-19 long-distance-matched NDJSON archive."""
    ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
    path = ARCHIVE_DIR / f"{name}.jsonl.zst"
    if path.exists():
        return {"packed_mb": round(path.stat().st_size / 1e6, 1), "pack_secs": None,
                "path": str(path)}
    try:
        params = zstandard.ZstdCompressionParameters.from_level(
            19, enable_ldm=True, window_log=27, threads=os.cpu_count() or 8)
        cctx = zstandard.ZstdCompressor(compression_params=params)
    except Exception:
        cctx = zstandard.ZstdCompressor(level=19, threads=os.cpu_count() or 8)

    t0 = time.perf_counter()
    n = 0
    body = {"size": 2000, "query": section_query(idx)}
    data = req("POST", f"{BASE_INDEX}/_search?scroll=5m", body, timeout=120)
    with open(path, "wb") as fh, cctx.stream_writer(fh) as writer:
        while True:
            hits = data["hits"]["hits"]
            if not hits:
                break
            for h in hits:
                doc = h["_source"]
                doc["_id"] = h["_id"]
                writer.write((json.dumps(doc) + "\n").encode())
                n += 1
            data = req("POST", "_search/scroll",
                       {"scroll": "5m", "scroll_id": data["_scroll_id"]}, timeout=120)
    secs = time.perf_counter() - t0
    logger.info("packed %-16s %8d docs -> %8.1f MB in %.0fs",
                name, n, path.stat().st_size / 1e6, secs)
    return {"packed_mb": round(path.stat().st_size / 1e6, 1),
            "pack_secs": round(secs, 1), "path": str(path), "docs": n}


def unpack_section(name: str) -> dict:
    """The off-domain request: decompress archive + bulk-build a live index."""
    path = ARCHIVE_DIR / f"{name}.jsonl.zst"
    index = f"lexsec_{name}_restored"
    if requests.head(f"{OS_URL}/{index}", timeout=30).status_code == 200:
        req("DELETE", index, timeout=120)
    cfg = variant_config("combined")
    cfg["settings"]["index"]["refresh_interval"] = "-1"  # bulk-load mode
    req("PUT", index, cfg, timeout=60)

    dctx = zstandard.ZstdDecompressor(max_window_size=2 ** 27)
    t0 = time.perf_counter()
    lines: list[bytes] = []
    with open(path, "rb") as fh, dctx.stream_reader(fh) as reader:
        buf = b""
        while True:
            chunk = reader.read(8 << 20)
            if not chunk:
                break
            buf += chunk
            parts = buf.split(b"\n")
            buf = parts.pop()
            lines.extend(parts)
    decompress_secs = time.perf_counter() - t0

    t0 = time.perf_counter()
    batch: list[bytes] = []
    n = 0

    def flush(batch: list[bytes]) -> None:
        if not batch:
            return
        resp = requests.post(f"{OS_URL}/_bulk", data=b"\n".join(batch) + b"\n",
                             headers={"Content-Type": "application/x-ndjson"},
                             timeout=300)
        resp.raise_for_status()
        if resp.json().get("errors"):
            items = [i for i in resp.json()["items"] if i["index"].get("error")]
            raise RuntimeError(f"bulk errors: {items[:2]}")

    for ln in lines:
        doc = json.loads(ln)
        _id = doc.pop("_id")
        batch.append(json.dumps({"index": {"_index": index, "_id": _id}}).encode())
        batch.append(json.dumps(doc).encode())
        n += 1
        if len(batch) >= 1000:  # 500 docs
            flush(batch)
            batch = []
    flush(batch)
    bulk_secs = time.perf_counter() - t0

    t0 = time.perf_counter()
    req("PUT", f"{index}/_settings", {"index": {"refresh_interval": "30s"}}, timeout=60)
    req("POST", f"{index}/_refresh", timeout=300)
    refresh_secs = time.perf_counter() - t0
    total = decompress_secs + bulk_secs + refresh_secs
    logger.info("unpacked %s: %d docs in %.0fs (decompress %.0fs, bulk %.0fs, refresh %.0fs)",
                name, n, total, decompress_secs, bulk_secs, refresh_secs)
    return {"index": index, "docs": n, "decompress_secs": round(decompress_secs, 1),
            "bulk_secs": round(bulk_secs, 1), "refresh_secs": round(refresh_secs, 1),
            "unpack_total_secs": round(total, 1)}


def parity(original: str, restored: str) -> dict:
    from lib.opensearch_retriever import OpenSearchRetriever
    records = json.loads((REPO_ROOT / "data/eval_results/diverse_queries.json").read_text())
    queries = []
    seen: set[str] = set()
    for rec in records:
        if rec["paraphrase"] not in seen:
            seen.add(rec["paraphrase"])
            queries.append(rec["paraphrase"])
        if len(queries) >= PARITY_QUERIES:
            break
    r_orig = OpenSearchRetriever(url=OS_URL, index=original,
                                 splade_model_path=SPLADE_ONNX, timeout=30)
    r_rest = OpenSearchRetriever(url=OS_URL, index=restored,
                                 splade_model_path=SPLADE_ONNX, timeout=30)
    stats = {"runs": 0, "same_id_set": 0, "same_score_multiset": 0, "max_score_diff": 0.0}
    for q in queries:
        for mode in ("bm25f", "splade"):
            a = [(d["openalex_id"], round(d["score"], 4))
                 for d in r_orig.search(q, top_k=PARITY_TOP_K, mode=mode)]
            b = [(d["openalex_id"], round(d["score"], 4))
                 for d in r_rest.search(q, top_k=PARITY_TOP_K, mode=mode)]
            stats["runs"] += 1
            if {i for i, _ in a} == {i for i, _ in b}:
                stats["same_id_set"] += 1
            if sorted(s for _, s in a) == sorted(s for _, s in b):
                stats["same_score_multiset"] += 1
            for (_, sa), (_, sb) in zip(a, b):
                stats["max_score_diff"] = max(stats["max_score_diff"], abs(sa - sb))
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Sectioned hot/cold index eval")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--keep-indices", action="store_true")
    args = parser.parse_args()

    wait_for_base()
    sections = build_sections()
    total_docs = sum(s["docs"] for s in sections.values())
    logger.info("partition covers %d/%d docs", total_docs, SUBSET_DOCS)

    names = [n for n, _ in SECTIONS] + ["other"]
    for i, name in enumerate(names):
        sections[name].update(pack_section(name, i))

    unpack = unpack_section(OFFREQ_SECTION)
    par = parity(f"lexsec_{OFFREQ_SECTION}", unpack["index"])
    logger.info("parity restored vs original: %s", par)

    # ── Scenario math: political-science user, home = social_sciences ──
    live_mb = sections[HOME_SECTION]["live_mb"]
    packed_others_mb = sum(s["packed_mb"] for n, s in sections.items() if n != HOME_SECTION)
    all_live_mb = sum(s["live_mb"] for s in sections.values())
    scenario_mb = live_mb + packed_others_mb
    docs_per_sec = unpack["docs"] / unpack["unpack_total_secs"]

    def scale(mb: float, docs: int) -> float:
        return round(mb / 1e3 * docs / SUBSET_DOCS, 1)  # → GB at target corpus size

    out = {
        "config": {
            "base_index": BASE_INDEX, "subset_docs": SUBSET_DOCS,
            "section_config": "combined (zstd6 + nosrc_splade + freqs + noid)",
            "archive": "NDJSON + zstd-19 long-distance (window 128MB), incl. sparse_field",
            "home_section": HOME_SECTION, "offrequest_section": OFFREQ_SECTION,
        },
        "sections": sections,
        "unpack_drill": unpack,
        "parity_restored_vs_original": par,
        "scenario_polisci_user": {
            "live_home_mb": live_mb,
            "packed_others_mb": round(packed_others_mb, 1),
            "scenario_total_mb": round(scenario_mb, 1),
            "all_sections_live_mb": round(all_live_mb, 1),
            "saved_vs_all_live_pct": round((1 - scenario_mb / all_live_mb) * 100, 1),
            "unpack_docs_per_sec": round(docs_per_sec),
            "scaled_gb": {
                "full_150M": {"scenario": scale(scenario_mb, FULL_SCALE_DOCS),
                              "all_live": scale(all_live_mb, FULL_SCALE_DOCS)},
                "laptop_15M": {"scenario": scale(scenario_mb, LAPTOP_TIER_DOCS),
                               "all_live": scale(all_live_mb, LAPTOP_TIER_DOCS)},
            },
        },
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(out, indent=2))

    print("\n" + "=" * 100)
    print("SECTIONED INDEX EVAL — hot/cold sections, 1M-doc subset, combined lossless config")
    print("=" * 100)
    print(f"{'section':<18} {'docs':>9} {'live MB':>9} {'packed MB':>10} {'pack ratio':>10}")
    print("-" * 100)
    for name in names:
        s = sections[name]
        ratio = s["live_mb"] / s["packed_mb"] if s["packed_mb"] else 0
        print(f"{name:<18} {s['docs']:>9} {s['live_mb']:>9.1f} {s['packed_mb']:>10.1f} {ratio:>9.1f}x")
    print("-" * 100)
    sc = out["scenario_polisci_user"]
    print(f"Political-science user: home '{HOME_SECTION}' live + others packed = "
          f"{sc['scenario_total_mb']:.0f} MB vs {sc['all_sections_live_mb']:.0f} MB all-live "
          f"({sc['saved_vs_all_live_pct']:.1f}% saved)")
    print(f"Off-domain request ({OFFREQ_SECTION}): unpack {unpack['docs']:,} docs in "
          f"{unpack['unpack_total_secs']:.0f}s (decompress {unpack['decompress_secs']:.0f}s + "
          f"bulk {unpack['bulk_secs']:.0f}s + refresh {unpack['refresh_secs']:.0f}s) "
          f"= {sc['unpack_docs_per_sec']:,} docs/s")
    print(f"Parity restored vs original: id-set {par['same_id_set']}/{par['runs']}, "
          f"score-multiset {par['same_score_multiset']}/{par['runs']}, "
          f"maxΔ {par['max_score_diff']:.4f}")
    print(f"Scaled: laptop 15M tier {sc['scaled_gb']['laptop_15M']['scenario']} GB vs "
          f"{sc['scaled_gb']['laptop_15M']['all_live']} GB all-live; "
          f"full 150M {sc['scaled_gb']['full_150M']['scenario']} GB vs "
          f"{sc['scaled_gb']['full_150M']['all_live']} GB")
    print("=" * 100)

    if not args.keep_indices:
        for name in names:
            req("DELETE", f"lexsec_{name}", timeout=120)
        req("DELETE", unpack["index"], timeout=120)
        logger.info("Deleted lexsec_* indices (archives kept in %s)", ARCHIVE_DIR)
    logger.info("Wrote results -> %s", args.output)


if __name__ == "__main__":
    main()
