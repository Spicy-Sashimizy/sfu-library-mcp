#!/usr/bin/env python3
"""Q3.1: Mine hard negatives from the LOCAL OpenSearch index.

For each eval query, retrieve the top-K candidates from one retrieval leg
(SPLADE or BM25F) against the local `openalex_works` index. These are
*candidate* hard negatives — documents a retriever ranks highly. The
false-negative filter (papers the LLM judge scored >=2 are actually relevant)
is applied later by `scripts/merge_negatives.py`, which pools both legs.

The retrieval DSL is kept byte-for-byte aligned with production
(`src/lib/opensearch_retriever.py`) and the benchmark
(`scripts/benchmark_llm_judge.py`) so mined negatives match what the live
system actually surfaces:

  - BM25F : multi_match most_fields, tie_breaker=0.5, title^3 / abstract / concepts^2
  - SPLADE: bool.should rank_feature per term, log scaling_factor=4, top-64 terms

SPLADE query encoding uses the same ONNX weights + bert-base-uncased WordPiece
tokenizer the SPLADE indexer used. This was verified to reproduce indexed
`sparse_field` weights to 4 decimal places, so query-side encoding matches the
document-side encoding stored in the index.

Usage (local-index mining — the Q3.1 path):
    SFU_OPENSEARCH_URL=http://...:9200 \
    python scripts/mine_hard_negatives.py \
        --queries data/sfu_eval_queries.json \
        --retriever splade --top-k 50 \
        --output data/training/hard_negatives_splade.jsonl

    python scripts/mine_hard_negatives.py \
        --queries data/sfu_eval_queries.json \
        --retriever bm25f --top-k 50 \
        --output data/training/hard_negatives_bm25.jsonl

This supersedes the original Phase K behavior, which mined negatives via the
live OpenAlex BM25 API (see git history of this file). Mining from the local
index instead lets the negatives match production retrieval DSL exactly and
removes the live-API rate-limit dependency.
"""
import argparse
import json
import logging
import os
import sys
from pathlib import Path

import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).parent.parent
DEFAULT_QUERIES = REPO_ROOT / "data/sfu_eval_queries.json"
INDEX = "openalex_works"

# SPLADE encoder assets. The ONNX model holds the exact weights used by the
# SPLADE indexer; the tokenizer must be the matching bert-base-uncased
# WordPiece vocab (30522 tokens, lowercase). Verified to reproduce indexed
# sparse_field weights for spot-checked docs.
SPLADE_ONNX_PATH = REPO_ROOT / "models/splade_onnx/model.onnx"
SPLADE_ONNX_FP16_PATH = REPO_ROOT / "models/splade_onnx_fp16/model.onnx"
# Any local model dir carrying the bert-base-uncased tokenizer.json works; these
# SFU embed models all ship the identical 30522-token uncased WordPiece vocab.
SPLADE_TOKENIZER_CANDIDATES = [
    REPO_ROOT / "models/sfu-academic-embed-v1",
    REPO_ROOT / "models/sfu-academic-embed-v4-bge",
    REPO_ROOT / "models/sfu-academic-embed-v4-mini",
]

SPLADE_MAX_TERMS = 64  # matches production _build_splade_query / benchmark


def opensearch_url() -> str:
    return os.environ.get(
        "SFU_OPENSEARCH_URL",
        "http://claudebox-sfu-library-mcp-training-opensearch:9200",
    ).rstrip("/")


# ── SPLADE query encoder (ONNX, offline, index-aligned) ───────────────────────

class SpladeOnnxEncoder:
    """Encode a query into a sparse {token: weight} dict via the indexer's ONNX model.

    log1p(relu(logits)) max-pooled over the sequence — identical to the indexer
    and to src/lib/opensearch_retriever.encode_splade. Special "[...]" tokens are
    dropped so they never become rank_feature fields.
    """

    def __init__(self) -> None:
        import numpy as np  # noqa: F401  (used in encode)
        import onnxruntime as ort
        from transformers import AutoTokenizer

        onnx_path = SPLADE_ONNX_PATH if SPLADE_ONNX_PATH.exists() else SPLADE_ONNX_FP16_PATH
        if not onnx_path.exists():
            raise FileNotFoundError(
                f"No SPLADE ONNX model found at {SPLADE_ONNX_PATH} or {SPLADE_ONNX_FP16_PATH}"
            )

        tok_dir = next((p for p in SPLADE_TOKENIZER_CANDIDATES if p.exists()), None)
        if tok_dir is None:
            raise FileNotFoundError(
                "No local bert-base-uncased tokenizer found. Tried: "
                + ", ".join(str(p) for p in SPLADE_TOKENIZER_CANDIDATES)
            )

        self.tokenizer = AutoTokenizer.from_pretrained(str(tok_dir))
        vocab = self.tokenizer.get_vocab()
        if len(vocab) != 30522:
            logger.warning(
                "Tokenizer vocab size %d != 30522 (SPLADE model expects bert-base-uncased)",
                len(vocab),
            )
        self.id_to_token = {v: k for k, v in vocab.items()}
        self.session = ort.InferenceSession(
            str(onnx_path), providers=["CPUExecutionProvider"]
        )
        logger.info("SPLADE ONNX encoder loaded (%s, tokenizer %s)", onnx_path.name, tok_dir.name)

    def encode(self, text: str, top_k: int = SPLADE_MAX_TERMS) -> dict[str, float]:
        import numpy as np

        enc = self.tokenizer(
            [text], max_length=512, truncation=True, padding=True, return_tensors="np"
        )
        input_ids = enc["input_ids"].astype(np.int64)
        feeds = {
            "input_ids": input_ids,
            "attention_mask": enc["attention_mask"].astype(np.int64),
            "token_type_ids": enc.get(
                "token_type_ids", np.zeros_like(input_ids)
            ).astype(np.int64),
        }
        logits = self.session.run(["logits"], feeds)[0][0]  # (seq, vocab)
        vec = np.log1p(np.maximum(logits, 0.0)).max(axis=0)  # (vocab,)

        nonzero = np.nonzero(vec)[0]
        if nonzero.size == 0:
            return {}
        order = nonzero[np.argsort(-vec[nonzero])]
        sparse: dict[str, float] = {}
        for idx in order:
            token = self.id_to_token.get(int(idx), "")
            weight = float(vec[int(idx)])
            if not token or token.startswith("["):
                continue
            if weight <= 0.0:
                continue
            sparse[token] = round(weight, 4)
            if len(sparse) >= top_k:
                break
        return sparse


# ── Local OpenSearch retrieval (production-aligned DSL) ────────────────────────

def _post_search(session: requests.Session, url: str, body: dict, timeout: int) -> list[dict]:
    resp = session.post(f"{url}/{INDEX}/_search", json=body, timeout=timeout)
    resp.raise_for_status()
    hits = resp.json().get("hits", {}).get("hits", [])
    out = []
    for rank, h in enumerate(hits, start=1):
        src = h.get("_source", {})
        out.append({
            "doc_id": h["_id"],
            "openalex_id": src.get("openalex_id", h["_id"]),
            "title": src.get("title", ""),
            "abstract": src.get("abstract", ""),
            "doi": src.get("doi", ""),
            "publication_year": src.get("publication_year"),
            "score": h.get("_score", 0.0),
            "rank": rank,
        })
    return out


def bm25f_search(session: requests.Session, url: str, query: str, top_k: int) -> list[dict]:
    body = {
        "size": top_k,
        "query": {
            "multi_match": {
                "query": query,
                "fields": ["title^3", "abstract", "concepts^2"],
                "type": "most_fields",
                "tie_breaker": 0.5,
            }
        },
        "_source": ["openalex_id", "title", "abstract", "doi", "publication_year"],
    }
    return _post_search(session, url, body, timeout=20)


def splade_search(
    session: requests.Session, url: str, sparse: dict[str, float], top_k: int
) -> list[dict]:
    if not sparse:
        return []
    should = [
        {"rank_feature": {"field": f"sparse_field.{t}", "boost": w,
                          "log": {"scaling_factor": 4}}}
        for t, w in sorted(sparse.items(), key=lambda x: -x[1])[:SPLADE_MAX_TERMS]
    ]
    body = {
        "size": top_k,
        "query": {"bool": {"should": should}},
        "_source": ["openalex_id", "title", "abstract", "doi", "publication_year"],
    }
    return _post_search(session, url, body, timeout=30)


def load_queries(path: Path) -> list[dict]:
    data = json.loads(path.read_text())
    if isinstance(data, dict):
        data = data.get("queries", [])
    return data


def run_local_mining(
    queries: list[dict],
    retriever: str,
    top_k: int,
    output_path: Path,
) -> dict:
    """Mine candidate negatives for one leg. Returns summary stats."""
    url = opensearch_url()
    session = requests.Session()

    encoder = None
    if retriever == "splade":
        encoder = SpladeOnnxEncoder()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    total_records = 0
    queries_with_hits = 0
    empty_queries = []

    with output_path.open("w") as out:
        for i, q in enumerate(queries):
            query_text = q.get("query", "").strip()
            subject = q.get("subject", "")
            if not query_text:
                continue

            try:
                if retriever == "splade":
                    sparse = encoder.encode(query_text)
                    if not sparse:
                        logger.warning("Empty SPLADE vector for query: %r", query_text[:60])
                    hits = splade_search(session, url, sparse, top_k)
                else:
                    hits = bm25f_search(session, url, query_text, top_k)
            except Exception as exc:
                logger.error("Retrieval failed for %r: %s", query_text[:60], exc)
                hits = []

            if hits:
                queries_with_hits += 1
            else:
                empty_queries.append(query_text)

            for h in hits:
                rec = {
                    "query": query_text,
                    "subject": subject,
                    "retriever": retriever,
                    "doc_id": h["openalex_id"] or h["doc_id"],
                    "rank": h["rank"],
                    "score": h["score"],
                    "title": h["title"],
                    "abstract": h["abstract"],
                    "doi": h["doi"],
                    "publication_year": h["publication_year"],
                }
                out.write(json.dumps(rec) + "\n")
                total_records += 1

            if (i + 1) % 20 == 0:
                logger.info("  [%d/%d] queries processed, %d candidates so far",
                            i + 1, len(queries), total_records)

    stats = {
        "retriever": retriever,
        "top_k": top_k,
        "queries": len(queries),
        "queries_with_hits": queries_with_hits,
        "empty_queries": len(empty_queries),
        "total_candidates": total_records,
        "output": str(output_path),
    }
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Q3.1: mine hard negatives from the local OpenSearch index"
    )
    parser.add_argument("--queries", default=str(DEFAULT_QUERIES),
                        help="Eval query JSON (list of {query, subject, ...})")
    parser.add_argument("--retriever", choices=["splade", "bm25f"],
                        help="Which retrieval leg to mine")
    parser.add_argument("--top-k", type=int, default=50,
                        help="Candidates to retrieve per query (default 50)")
    parser.add_argument("--output", help="Output JSONL path")

    args = parser.parse_args()

    # Local-index mining path
    if not args.retriever:
        parser.error("--retriever {splade,bm25f} is required for local-index mining")
    if not args.output:
        parser.error("--output is required for local-index mining")

    queries_path = Path(args.queries)
    if not queries_path.exists():
        logger.error("Queries file not found: %s", queries_path)
        sys.exit(1)
    queries = load_queries(queries_path)
    logger.info("Loaded %d queries from %s", len(queries), queries_path)
    logger.info("OpenSearch URL: %s", opensearch_url())

    stats = run_local_mining(
        queries=queries,
        retriever=args.retriever,
        top_k=args.top_k,
        output_path=Path(args.output),
    )
    logger.info("=== %s leg done ===", stats["retriever"].upper())
    logger.info("  queries:           %d", stats["queries"])
    logger.info("  queries with hits: %d", stats["queries_with_hits"])
    logger.info("  empty queries:     %d", stats["empty_queries"])
    logger.info("  total candidates:  %d", stats["total_candidates"])
    logger.info("  output:            %s", stats["output"])


if __name__ == "__main__":
    main()
