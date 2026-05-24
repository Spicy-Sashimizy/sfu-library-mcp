#!/usr/bin/env python3
"""Dense-ANN index builder (PROOF OF CONCEPT, subset only).

Builds the `openalex_works_dense` k-NN index from a SUBSET of the existing
`openalex_works` index — it scrolls docs straight out of OpenSearch (it does NOT
re-read the OpenAlex snapshots). Each doc's "{title}. {abstract}" text is encoded
with the production v5 bi-encoder (models/sfu-academic-embed-v5, bge-small,
384-dim, L2-normalized) on GPU in fp16, and the 384-dim vector is bulk-indexed
into a lucene-HNSW `embedding` field.

FAIRNESS REQUIREMENT
────────────────────
A dense leg that simply lacked the judged docs would look artificially weak. So
the subset is built as the UNION of:
  1. every doc_id that appears in data/eval_results/llm_judge_cache.json
     (the judged set — guaranteed present so dense is judged on the same docs as
     BM25F/SPLADE), PLUS
  2. a background distractor sample scrolled from openalex_works, up to a total
     of --max-docs (default 600k). Distractors give the ANN graph realistic
     neighbours so a dense hit isn't trivially the only candidate.

This is still a SUBSET of the full ~150M-doc lexical index, so dense recall here
is a LOWER BOUND on what a full dense index would deliver.

Usage
─────
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
    SFU_OPENSEARCH_URL=http://...:9200 \
    python scripts/dense_indexer_poc.py \
        --judge-cache data/eval_results/llm_judge_cache.json \
        --max-docs 600000 \
        --model models/sfu-academic-embed-v5
"""
import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("dense_indexer_poc")

REPO_ROOT = Path(__file__).parent.parent
SOURCE_INDEX = "openalex_works"
DENSE_INDEX = "openalex_works_dense"
DEFAULT_JUDGE_CACHE = REPO_ROOT / "data/eval_results/llm_judge_cache.json"
DEFAULT_MODEL = str(REPO_ROOT / "models/sfu-academic-embed-v5")
DENSE_TEMPLATE = REPO_ROOT / "docker/opensearch/index_template_dense.json"

SCROLL_BATCH = 1000
SCROLL_TIME = "5m"
ENCODE_BATCH = 256
BULK_BATCH = 500
MGET_BATCH = 200
SOURCE_FIELDS = ["title", "abstract", "doi", "openalex_id", "publication_year", "type", "is_oa"]


def opensearch_url() -> str:
    return os.environ.get(
        "SFU_OPENSEARCH_URL",
        "http://claudebox-sfu-library-mcp-training-opensearch:9200",
    ).rstrip("/")


def doc_text(title: str, abstract: str) -> str:
    """Build doc text as "{title}. {abstract}" — mirrors eval_embedder.py:81-88."""
    title = (title or "").strip()
    abstract = (abstract or "").strip()
    if title and abstract:
        sep = " " if title.endswith((".", "?", "!")) else ". "
        return f"{title}{sep}{abstract}"
    return title or abstract


def judged_doc_ids(path: Path) -> set[str]:
    cache = json.loads(path.read_text())
    ids = set()
    for key in cache:
        _query, doc_id = key.rsplit("||", 1)
        ids.add(doc_id)
    return ids


def register_template(session: requests.Session, url: str) -> None:
    body = json.loads(DENSE_TEMPLATE.read_text())
    resp = session.put(f"{url}/_index_template/openalex_works_dense_template", json=body, timeout=30)
    resp.raise_for_status()
    logger.info("Registered dense index template (priority %s)", body.get("priority"))


def recreate_index(session: requests.Session, url: str) -> None:
    session.delete(f"{url}/{DENSE_INDEX}", timeout=30)
    resp = session.put(f"{url}/{DENSE_INDEX}", timeout=60)
    if resp.status_code >= 300:
        raise RuntimeError(f"Failed to create {DENSE_INDEX}: {resp.status_code} {resp.text}")
    # Confirm knn + dim came from the template, not the catch-all openalex_works_template.
    mapping = session.get(f"{url}/{DENSE_INDEX}/_mapping", timeout=30).json()
    props = mapping[DENSE_INDEX]["mappings"]["properties"]
    emb = props.get("embedding", {})
    if emb.get("type") != "knn_vector" or emb.get("dimension") != 384:
        raise RuntimeError(
            f"{DENSE_INDEX} did not pick up the dense template (embedding={emb}). "
            "Check that index_template_dense.json priority outranks openalex_works_template."
        )
    settings = session.get(f"{url}/{DENSE_INDEX}/_settings", timeout=30).json()
    knn = settings[DENSE_INDEX]["settings"]["index"].get("knn")
    logger.info("Created %s  (index.knn=%s, embedding dim=%d)", DENSE_INDEX, knn, emb["dimension"])


def mget_judged(session: requests.Session, url: str, ids: list[str]) -> dict[str, dict]:
    """Fetch judged docs by _id so they are guaranteed in the subset."""
    fetched: dict[str, dict] = {}
    missing = 0
    for i in range(0, len(ids), MGET_BATCH):
        batch = ids[i : i + MGET_BATCH]
        resp = session.post(
            f"{url}/{SOURCE_INDEX}/_mget",
            json={"ids": batch},
            params={"_source_includes": ",".join(SOURCE_FIELDS)},
            timeout=60,
        )
        resp.raise_for_status()
        for doc in resp.json().get("docs", []):
            if doc.get("found"):
                fetched[doc["_id"]] = doc.get("_source", {})
            else:
                missing += 1
    logger.info("Judged docs: fetched %d, missing from source index %d", len(fetched), missing)
    return fetched


def scroll_distractors(session: requests.Session, url: str, need: int, exclude: set[str]):
    """Yield (doc_id, _source) scrolled from openalex_works until `need` new docs."""
    if need <= 0:
        return
    body = {
        "size": SCROLL_BATCH,
        "query": {"match_all": {}},
        "_source": SOURCE_FIELDS,
    }
    resp = session.post(
        f"{url}/{SOURCE_INDEX}/_search", params={"scroll": SCROLL_TIME}, json=body, timeout=60
    )
    resp.raise_for_status()
    data = resp.json()
    scroll_id = data.get("_scroll_id")
    yielded = 0
    try:
        while True:
            hits = (data.get("hits") or {}).get("hits") or []
            if not hits:
                break
            for hit in hits:
                doc_id = hit["_id"]
                if doc_id in exclude:
                    continue
                yield doc_id, hit.get("_source", {})
                yielded += 1
                if yielded >= need:
                    return
            resp = session.post(
                f"{url}/_search/scroll",
                json={"scroll": SCROLL_TIME, "scroll_id": scroll_id},
                timeout=60,
            )
            resp.raise_for_status()
            data = resp.json()
            scroll_id = data.get("_scroll_id")
    finally:
        if scroll_id:
            session.delete(f"{url}/_search/scroll", json={"scroll_id": [scroll_id]}, timeout=30)


def bulk_index(session: requests.Session, url: str, rows: list[dict]) -> int:
    """Bulk-index rows; each row is a _source dict that already contains 'embedding'.

    Returns count of successfully indexed docs.
    """
    lines = []
    for r in rows:
        lines.append(json.dumps({"index": {"_index": DENSE_INDEX, "_id": r["openalex_id"]}}))
        lines.append(json.dumps(r))
    payload = "\n".join(lines) + "\n"
    resp = session.post(
        f"{url}/_bulk", data=payload, headers={"Content-Type": "application/x-ndjson"}, timeout=180
    )
    resp.raise_for_status()
    result = resp.json()
    ok = 0
    if result.get("errors"):
        for item in result.get("items", []):
            idx = item.get("index", {})
            if idx.get("status", 500) < 300:
                ok += 1
            else:
                logger.warning("Bulk item error: %s", idx.get("error"))
    else:
        ok = len(rows)
    return ok


def normalize_source(src: dict) -> dict:
    """Project a source doc to the dense index schema (drop sparse_field etc.)."""
    is_oa = src.get("is_oa")
    py = src.get("publication_year")
    try:
        py = int(py) if py not in (None, "", "None") else None
    except (ValueError, TypeError):
        py = None
    return {
        "openalex_id": src.get("openalex_id") or "",
        "doi": src.get("doi") if src.get("doi") not in (None, "None") else None,
        "title": src.get("title") or "",
        "abstract": src.get("abstract") or "",
        "publication_year": py,
        "type": src.get("type") or "",
        "is_oa": bool(is_oa) if isinstance(is_oa, bool) else False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Dense-ANN subset index builder (POC)")
    parser.add_argument("--judge-cache", default=str(DEFAULT_JUDGE_CACHE))
    parser.add_argument("--max-docs", type=int, default=600_000,
                        help="Total subset size (judged docs + distractors)")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--keep-index", action="store_true",
                        help="Do not delete an existing dense index first")
    args = parser.parse_args()

    import numpy as np
    import torch
    from sentence_transformers import SentenceTransformer

    url = opensearch_url()
    session = requests.Session()

    judged = judged_doc_ids(Path(args.judge_cache))
    logger.info("Judge cache has %d unique judged doc ids", len(judged))

    register_template(session, url)
    if not args.keep_index:
        recreate_index(session, url)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Loading embedder %s on %s", args.model, device)
    model = SentenceTransformer(args.model, device=device)
    if device == "cuda":
        model = model.half()  # fp16 encode

    # ── Build the doc pool: judged first (guaranteed), then distractors ──
    judged_docs = mget_judged(session, url, sorted(judged))
    found_judged_ids = set(judged_docs)

    pool: list[tuple[str, dict]] = [(doc_id, src) for doc_id, src in judged_docs.items()]
    distractor_need = max(0, args.max_docs - len(pool))
    logger.info("Scrolling up to %d distractor docs from %s ...", distractor_need, SOURCE_INDEX)
    n_distract = 0
    for doc_id, src in scroll_distractors(session, url, distractor_need, found_judged_ids):
        pool.append((doc_id, src))
        n_distract += 1
        if n_distract % 50000 == 0:
            logger.info("  scrolled %d distractors (pool=%d)", n_distract, len(pool))
    logger.info("Pool ready: %d docs (%d judged + %d distractors)",
                len(pool), len(found_judged_ids), n_distract)

    # ── Encode + bulk index in streaming batches ──
    t0 = time.time()
    indexed = 0
    buf_ids: list[str] = []
    buf_src: list[dict] = []
    buf_text: list[str] = []

    def flush() -> None:
        nonlocal indexed
        if not buf_text:
            return
        embs = model.encode(
            buf_text, batch_size=ENCODE_BATCH, normalize_embeddings=True,
            show_progress_bar=False, convert_to_numpy=True,
        )
        embs = np.asarray(embs, dtype=np.float32)
        rows = []
        for j, doc_id in enumerate(buf_ids):
            row = normalize_source(buf_src[j])
            row["openalex_id"] = doc_id
            row["embedding"] = embs[j].tolist()
            rows.append(row)
        for bi in range(0, len(rows), BULK_BATCH):
            indexed += bulk_index(session, url, rows[bi : bi + BULK_BATCH])
        buf_ids.clear()
        buf_src.clear()
        buf_text.clear()

    skipped_empty = 0
    for doc_id, src in pool:
        text = doc_text(src.get("title", ""), src.get("abstract", ""))
        if not text:
            skipped_empty += 1
            continue
        buf_ids.append(doc_id)
        buf_src.append(src)
        buf_text.append(text)
        if len(buf_text) >= ENCODE_BATCH * 8:  # encode ~2k docs at a time
            flush()
            if indexed % 50000 < (ENCODE_BATCH * 8):
                rate = indexed / max(1e-9, time.time() - t0)
                logger.info("  indexed %d docs (%.0f docs/s)", indexed, rate)
    flush()

    session.post(f"{url}/{DENSE_INDEX}/_refresh", timeout=60)
    count = session.get(f"{url}/{DENSE_INDEX}/_count", timeout=30).json().get("count")
    elapsed = time.time() - t0

    # Verify judged-doc coverage in the dense index.
    present_judged = 0
    judged_list = sorted(found_judged_ids)
    for i in range(0, len(judged_list), MGET_BATCH):
        batch = judged_list[i : i + MGET_BATCH]
        r = session.post(f"{url}/{DENSE_INDEX}/_mget", json={"ids": batch},
                         params={"_source": "false"}, timeout=60).json()
        present_judged += sum(1 for d in r.get("docs", []) if d.get("found"))

    logger.info("=" * 70)
    logger.info("DENSE INDEX BUILD COMPLETE")
    logger.info("  index:                 %s", DENSE_INDEX)
    logger.info("  docs indexed:          %d (count=%s)", indexed, count)
    logger.info("  skipped (empty text):  %d", skipped_empty)
    logger.info("  judged docs in cache:  %d", len(judged))
    logger.info("  judged found in source:%d", len(found_judged_ids))
    logger.info("  judged in dense index: %d", present_judged)
    logger.info("  encode+index time:     %.1fs (%.0f docs/s)", elapsed, indexed / max(1e-9, elapsed))
    logger.info("=" * 70)

    if device == "cuda":
        torch.cuda.empty_cache()


if __name__ == "__main__":
    sys.exit(main())
