#!/usr/bin/env python3
"""Fast new-vs-old SPLADE fine-tune validator on a SUBSET index (option "B").

The full 150M in-place re-encode is upsert-bound (~1k docs/s -> ~40h), which is
the production *deployment*, not the question "did fine-tuning sfu-splade-v1
improve retrieval over the base SPLADE?". This answers that in minutes by
mirroring the dense POC method (scripts/dense_indexer_poc.py + eval_dense_poc.py):

  PHASE 1 (build):  build ONE subset index `openalex_works_splade_eval` =
      judged docs (from the LLM-judge cache, guaranteed coverage) UNION a
      background distractor sample scrolled from openalex_works, up to --max-docs.
      Each doc's "{title}. {abstract}" is encoded with BOTH models into two
      rank_features fields on the SAME doc:
          sparse_base : naver/splade-cocondenser-ensembledistil (the fine-tune's base)
          sparse_v1   : models/sfu-splade-v1                    (the fine-tune)
      Identical docs + identical distractors => the ONLY variable is the model.

  PHASE 2 (eval):   for each diverse (paraphrased) eval query, run the production
      SPLADE retrieval DSL (bool.should per-term rank_feature, log scaling_factor=4,
      top-64 query terms, minimum_should_match=1) against each field, score the
      top-10 against the judge cache (NDCG@10 / MRR@10, gain 2^g-1, threshold 2),
      sliced keyword / natural / overall. Report base-vs-v1 deltas.

CAVEATS (printed): a subset has guaranteed judged-doc coverage + far fewer
distractors than the full 150M index, so ABSOLUTE numbers are optimistic. The
base-vs-v1 DELTA on the identical subset is the robust signal (the model is the
only thing that changes). Full-corpus recall is what the 40h re-encode confirms.

Usage:
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 SFU_OPENSEARCH_URL=http://...:9200 \
    python scripts/eval_splade_finetune_subset.py --max-docs 100000 \
        --output data/eval_results/splade_finetune_subset.json
"""
import argparse
import json
import logging
import math
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("splade_subset_eval")

REPO_ROOT = Path(__file__).parent.parent
SOURCE_INDEX = "openalex_works"
DEFAULT_EVAL_INDEX = "openalex_works_splade_eval"
DEFAULT_JUDGE_CACHE = REPO_ROOT / "data/eval_results/llm_judge_cache.json"
DEFAULT_DIVERSE = REPO_ROOT / "data/eval_results/diverse_queries.json"
DEFAULT_OUTPUT = REPO_ROOT / "data/eval_results/splade_finetune_subset.json"
BASE_MODEL = "naver/splade-cocondenser-ensembledistil"
V1_MODEL = str(REPO_ROOT / "models/sfu-splade-v1")

SOURCE_FIELDS = ["title", "abstract", "openalex_id"]
RELEVANT_THRESHOLD = 2
K_NDCG = 10
MAX_QUERY_TERMS = 64       # matches opensearch_retriever max_terms
MAX_DOC_TERMS = 256        # doc-side top-k (matches indexer)
SCALING_FACTOR = 4         # matches production rank_feature log scaling
ENCODE_BATCH = 32
BULK_BATCH = 500
JUDGE_KEY_LEN = 80


def _safe_tok(t: str) -> bool:
    """rank_features feature names can't contain '.' (the field-path separator we
    use in `sparse_base.{term}` queries); keep only [a-z0-9#] wordpiece tokens."""
    return bool(t) and all(c.isalnum() or c == "#" for c in t)


def os_url() -> str:
    return os.environ.get(
        "SFU_OPENSEARCH_URL",
        "http://claudebox-sfu-library-mcp-training-opensearch:9200",
    ).rstrip("/")


# ── SPLADE encoder (PyTorch, fp16, GPU) — log1p(relu(logits)) max-pooled ────────
class SpladeTorch:
    def __init__(self, model_path: str, device: str):
        import torch
        from transformers import AutoModelForMaskedLM, AutoTokenizer
        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(model_path)
        m = AutoModelForMaskedLM.from_pretrained(model_path).to(device).eval()
        self.model = m.half() if device == "cuda" else m
        self.device = device
        self.id2tok = {v: k for k, v in self.tok.get_vocab().items()}

    def encode(self, texts: list[str], top_k: int) -> list[dict]:
        torch = self.torch
        with torch.no_grad():
            enc = self.tok(texts, max_length=256, truncation=True, padding=True,
                           return_tensors="pt").to(self.device)
            logits = self.model(**enc).logits                       # (B,S,V)
            mask = enc["attention_mask"].unsqueeze(-1)              # (B,S,1)
            relu = torch.log1p(torch.relu(logits)) * mask
            vec = relu.max(dim=1).values.float().cpu()             # (B,V)
        out = []
        for row in vec:
            nz = torch.nonzero(row).squeeze(-1).tolist()
            d = {}
            for idx in nz:
                tok = self.id2tok.get(idx, "")
                rw = round(float(row[idx]), 4)
                # rank_features rejects non-positive values; round-to-zero (tiny
                # weights -> 0.0000) would fail the whole doc, so drop them here.
                if rw <= 0.0 or not _safe_tok(tok):
                    continue
                d[tok] = rw
            if len(d) > top_k:
                d = dict(sorted(d.items(), key=lambda x: -x[1])[:top_k])
            out.append(d)
        return out


# ── Metrics (match eval_pipeline.py) ────────────────────────────────────────────
def dcg(gains, k):
    return sum((2 ** g - 1) / math.log2(i + 2) for i, g in enumerate(gains[:k]))


def ndcg_at_k(ranked, ideal, k):
    idcg = dcg(sorted(ideal, reverse=True), k)
    return dcg(ranked, k) / idcg if idcg else 0.0


def mrr_at_k(ranked, k):
    for i, g in enumerate(ranked[:k]):
        if g >= RELEVANT_THRESHOLD:
            return 1.0 / (i + 1)
    return 0.0


def mean(xs):
    return sum(xs) / len(xs) if xs else 0.0


# ── Phase 1: build the two-field subset index ───────────────────────────────────
def create_index(url, index):
    requests.delete(f"{url}/{index}", timeout=30)
    body = {
        "settings": {"index": {"number_of_shards": 1, "number_of_replicas": 0,
                               "refresh_interval": "30s"}},
        "mappings": {"properties": {
            "openalex_id": {"type": "keyword"},
            "title": {"type": "text"},
            "abstract": {"type": "text"},
            "sparse_base": {"type": "rank_features"},
            "sparse_v1": {"type": "rank_features"},
        }},
    }
    r = requests.put(f"{url}/{index}", json=body, timeout=30)
    r.raise_for_status()
    logger.info("Created %s (rank_features: sparse_base + sparse_v1)", index)


def fetch_judged(url, ids):
    """mget judged docs by _id from openalex_works -> {id: {title,abstract,openalex_id}}."""
    out = {}
    ids = list(ids)
    for i in range(0, len(ids), 200):
        batch = ids[i:i + 200]
        r = requests.post(f"{url}/{SOURCE_INDEX}/_mget",
                          params={"_source_includes": ",".join(SOURCE_FIELDS)},
                          json={"ids": batch}, timeout=60)
        r.raise_for_status()
        for d in r.json().get("docs", []):
            if d.get("found"):
                out[d["_id"]] = d.get("_source", {})
    return out


def scroll_distractors(url, need, exclude):
    """Yield (id, _source) scrolled from openalex_works until `need` new docs."""
    body = {"size": 1000, "query": {"function_score": {"random_score": {}}},
            "_source": SOURCE_FIELDS}
    r = requests.post(f"{url}/{SOURCE_INDEX}/_search",
                      params={"scroll": "5m"}, json=body, timeout=120).json()
    sid = r.get("_scroll_id")
    seen = 0
    while True:
        hits = r.get("hits", {}).get("hits", [])
        if not hits:
            break
        for h in hits:
            if h["_id"] in exclude:
                continue
            yield h["_id"], h.get("_source", {})
            seen += 1
            if seen >= need:
                requests.delete(f"{url}/_search/scroll",
                                json={"scroll_id": [sid]}, timeout=30)
                return
        r = requests.post(f"{url}/_search/scroll",
                          json={"scroll": "5m", "scroll_id": sid}, timeout=120).json()
        sid = r.get("_scroll_id")


def build(url, index, judge_cache, max_docs, device):
    judged_ids = set()
    for k, v in json.loads(Path(judge_cache).read_text()).items():
        if isinstance(v, int):
            judged_ids.add(k.rsplit("||", 1)[1])
    logger.info("Judged doc ids: %d", len(judged_ids))

    docs = {}  # id -> source (title/abstract/openalex_id)
    docs.update(fetch_judged(url, judged_ids))
    logger.info("Judged docs found in source index: %d", len(docs))
    need = max(0, max_docs - len(docs))
    if need:
        logger.info("Scrolling %d distractors...", need)
        for did, src in scroll_distractors(url, need, set(docs)):
            docs[did] = src
    logger.info("Total subset docs: %d", len(docs))

    create_index(url, index)
    base = SpladeTorch(BASE_MODEL, device)
    v1 = SpladeTorch(V1_MODEL, device)
    logger.info("Both SPLADE models loaded on %s", device)

    items = list(docs.items())
    buf, indexed, t0 = [], 0, time.time()

    def flush(rows):
        if not rows:
            return
        payload = "".join(rows)
        rr = requests.post(f"{url}/{index}/_bulk", data=payload,
                           headers={"Content-Type": "application/x-ndjson"}, timeout=120)
        rr.raise_for_status()
        resp = rr.json()
        if resp.get("errors"):
            first = next((it["index"]["error"] for it in resp.get("items", [])
                          if it.get("index", {}).get("error")), None)
            logger.warning("bulk item errors; first: %s", first)

    for i in range(0, len(items), ENCODE_BATCH):
        chunk = items[i:i + ENCODE_BATCH]
        texts = [f"{(s.get('title') or '').strip()}. {(s.get('abstract') or '').strip()}"
                 for _, s in chunk]
        sb = base.encode(texts, MAX_DOC_TERMS)
        sv = v1.encode(texts, MAX_DOC_TERMS)
        for (did, s), b_vec, v_vec in zip(chunk, sb, sv):
            if not b_vec and not v_vec:
                continue
            doc = {"openalex_id": s.get("openalex_id", did),
                   "title": s.get("title", ""), "abstract": s.get("abstract", ""),
                   "sparse_base": b_vec, "sparse_v1": v_vec}
            buf.append(json.dumps({"index": {"_index": index, "_id": did}}) + "\n")
            buf.append(json.dumps(doc) + "\n")
            indexed += 1
        if len(buf) >= BULK_BATCH * 2:
            flush(buf); buf = []
        if (i // ENCODE_BATCH) % 50 == 0:
            logger.info("  encoded+queued %d/%d (%.0f doc/s)",
                        indexed, len(items), indexed / max(1e-9, time.time() - t0))
    flush(buf)
    requests.post(f"{url}/{index}/_refresh", timeout=60)
    cnt = requests.get(f"{url}/{index}/_count", timeout=30).json().get("count")
    logger.info("Indexed %d docs into %s in %.0fs (count=%s)",
                indexed, index, time.time() - t0, cnt)
    return cnt


# ── Phase 2: benchmark base vs v1 over the subset ────────────────────────────────
def splade_query_body(sparse: dict, field: str, top_k: int) -> dict:
    should = [{"rank_feature": {"field": f"{field}.{t}", "boost": w,
                                "log": {"scaling_factor": SCALING_FACTOR}}}
              for t, w in sorted(sparse.items(), key=lambda x: -x[1])[:MAX_QUERY_TERMS]]
    return {"size": top_k, "query": {"bool": {"should": should, "minimum_should_match": 1}},
            "_source": ["openalex_id"]}


def retrieve(url, index, sparse, field, top_k):
    if not sparse:
        return []
    r = requests.post(f"{url}/{index}/_search", json=splade_query_body(sparse, field, top_k),
                      timeout=60)
    r.raise_for_status()
    return [h["_source"].get("openalex_id", h["_id"])
            for h in r.json().get("hits", {}).get("hits", [])]


def load_grades(path):
    grades = defaultdict(dict)
    for k, v in json.loads(Path(path).read_text()).items():
        if isinstance(v, int):
            q, d = k.rsplit("||", 1)
            grades[q][d] = v
    return grades


def evaluate(url, index, judge_cache, diverse, device, out_path):
    grades = load_grades(judge_cache)
    records = json.loads(Path(diverse).read_text())
    base = SpladeTorch(BASE_MODEL, device)
    v1 = SpladeTorch(V1_MODEL, device)

    slices = ("keyword", "natural", "overall")
    acc = {m: {s: {"ndcg": [], "mrr": []} for s in slices} for m in ("base", "v1")}
    scored, skipped = 0, 0
    for i, rec in enumerate(records):
        q = rec["paraphrase"]
        qtype = rec["query_type"]
        key = rec.get("judge_key", rec["original_query"][:JUDGE_KEY_LEN])
        gb = grades.get(key, {})
        if not any(g >= RELEVANT_THRESHOLD for g in gb.values()):
            skipped += 1
            continue
        ideal = list(gb.values())
        qb = base.encode([q], MAX_QUERY_TERMS)[0]
        qv = v1.encode([q], MAX_QUERY_TERMS)[0]
        for m, sparse, field in (("base", qb, "sparse_base"), ("v1", qv, "sparse_v1")):
            ranked = retrieve(url, index, sparse, field, 50)
            g = [gb.get(d, 0) for d in ranked]
            for s in (qtype, "overall"):
                acc[m][s]["ndcg"].append(ndcg_at_k(g, ideal, K_NDCG))
                acc[m][s]["mrr"].append(mrr_at_k(g, K_NDCG))
        scored += 1
        if (i + 1) % 40 == 0:
            logger.info("  scored %d/%d", scored, len(records))

    def summ(m, s):
        a = acc[m][s]
        return {"n": len(a["ndcg"]), "ndcg@10": round(mean(a["ndcg"]), 4),
                "mrr@10": round(mean(a["mrr"]), 4)}

    summary = {m: {s: summ(m, s) for s in slices} for m in ("base", "v1")}
    deltas = {s: {"ndcg@10": round(summary["v1"][s]["ndcg@10"] - summary["base"][s]["ndcg@10"], 4),
                  "mrr@10": round(summary["v1"][s]["mrr@10"] - summary["base"][s]["mrr@10"], 4)}
              for s in slices}

    out = {
        "comparison": "SPLADE retrieval over a subset index: base (naver/splade-cocondenser-"
                      "ensembledistil) vs fine-tuned models/sfu-splade-v1, identical docs+distractors.",
        "index": index, "scored": scored, "skipped_no_gt": skipped,
        "ndcg_k": K_NDCG, "query_terms": MAX_QUERY_TERMS, "scaling_factor": SCALING_FACTOR,
        "summary": summary, "delta_v1_minus_base": deltas,
        "caveats": [
            "Subset index has guaranteed judged-doc coverage + far fewer distractors than the "
            "full ~150M index, so ABSOLUTE NDCG is optimistic. The base-vs-v1 DELTA (identical "
            "docs/distractors; model is the only variable) is the robust signal.",
            "Measures retrieval quality on the subset; full-corpus recall is confirmed by the "
            "in-place 150M re-encode + eval_pipeline.py (the ~40h production run).",
        ],
    }
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(out, indent=2))

    print("\n" + "=" * 78)
    print("SPLADE FINE-TUNE — base vs sfu-splade-v1 (subset retrieval, LLM-judge NDCG@10)")
    print("=" * 78)
    print(f"scored {scored} queries | skipped {skipped} | index {index}")
    print(f"{'slice':<9}{'base':>9}{'v1':>9}{'dNDCG':>9}   {'base_mrr':>9}{'v1_mrr':>9}{'dMRR':>8}")
    print("-" * 78)
    for s in slices:
        b, v, d = summary["base"][s], summary["v1"][s], deltas[s]
        print(f"{s:<9}{b['ndcg@10']:>9.4f}{v['ndcg@10']:>9.4f}{d['ndcg@10']:>+9.4f}   "
              f"{b['mrr@10']:>9.4f}{v['mrr@10']:>9.4f}{d['mrr@10']:>+8.4f}")
    print("-" * 78)
    print("d = v1 minus base. Subset => absolute is optimistic; the DELTA is the signal.")
    print("=" * 78)
    logger.info("Wrote %s", out_path)


def main():
    ap = argparse.ArgumentParser(description="Fast base-vs-v1 SPLADE fine-tune validator (subset)")
    ap.add_argument("--phase", choices=["build", "eval", "both"], default="both")
    ap.add_argument("--max-docs", type=int, default=100_000)
    ap.add_argument("--index", default=DEFAULT_EVAL_INDEX)
    ap.add_argument("--judge-cache", default=str(DEFAULT_JUDGE_CACHE))
    ap.add_argument("--diverse-queries", default=str(DEFAULT_DIVERSE))
    ap.add_argument("--output", default=str(DEFAULT_OUTPUT))
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    url = os_url()
    logger.info("OpenSearch: %s | index: %s | device: %s", url, args.index, args.device)
    if args.phase in ("build", "both"):
        build(url, args.index, args.judge_cache, args.max_docs, args.device)
    if args.phase in ("eval", "both"):
        evaluate(url, args.index, args.judge_cache, args.diverse_queries, args.device, args.output)


if __name__ == "__main__":
    main()
