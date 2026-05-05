#!/usr/bin/env python3
"""Project progress dashboard.

Reads eval history, checks artifact existence, and prints a concise status
table covering all phases G–O plus the deferred / out-of-scope items.

Usage:
    python scripts/show_progress.py
    python scripts/show_progress.py --brief
"""

import argparse
import json
import math
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent


# ── Phase definitions ─────────────────────────────────────────────────────────

PHASES = [
    # (label, description, completion_check: callable → bool | "manual")
    ("G",    "Production RRF wiring (MCP server)",           lambda: _file("src/lib/tools.py") and _grep("src/lib/tools.py", "_maybe_rerank")),
    ("H",    "v3 regression ablation (BGE vs MiniLM)",       "manual"),
    ("I",    "Strategy 3 round-robin + persistent cache",    lambda: _file("scripts/openalex_cache.py") or _grep("scripts/generate_sfu_training_data.py", "target_per_provider")),
    ("I-bis","Non-OpenAlex training data sources",           lambda: _file("scripts/arxiv_fetcher.py") and _file("scripts/generate_synthetic_queries.py")),
    ("J",    "Dedup + reproducibility fixes",                lambda: _grep("scripts/generate_sfu_training_data.py", "sha1")),
    ("K",    "BM25-mined hard negatives",                    lambda: _file("scripts/mine_hard_negatives.py")),
    ("L",    "Training: v4-bge + v4-mini (BGE-small base)",  lambda: _file("models/sfu-academic-embed-v4-bge")),
    ("M",    "Eval methodology (120 queries, bootstrap CI)", lambda: _eval_query_count() >= 120),
    ("O.1",  "Model path swap to v4-bge (production)",       lambda: _check_model_path()),
    ("O.2",  "CrossEncoder Tier 1.5 reranker",               lambda: _grep("src/lib/reranker.py", "rerank_with_crossencoder")),
    ("O.3",  "Query log capture (LambdaMART prerequisite)",  lambda: _grep("src/lib/tools.py", "_log_query")),
    ("O.3b", "LambdaMART learned reranker",                  lambda: _lambdamart_trained()),
    ("N",    "Real query-log model (v5 fine-tune)",          lambda: _lambdamart_log_count() >= 2000),
]

def _deferred_items() -> list[tuple[str, str, str]]:
    count = _lambdamart_log_count()
    return [
        ("ColBERT",         "Late-interaction ranker",       "deferred — architectural cost > benefit at current scale"),
        ("CrossEncoder v2", "Larger cross-encoder model",    "deferred — latency budget not established"),
        ("Full-text index", "SFU full-text scraping",        "out-of-scope — license forbids"),
        ("LambdaMART v2",   "Click-signal learned blend",    f"blocked — need {max(0, 2000 - count)} more logged queries"),
    ]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _file(rel: str) -> bool:
    return (REPO_ROOT / rel).exists()


def _grep(rel: str, pattern: str) -> bool:
    p = REPO_ROOT / rel
    if not p.exists():
        return False
    try:
        return pattern in p.read_text()
    except OSError:
        return False


def _eval_query_count() -> int:
    p = REPO_ROOT / "data/sfu_eval_queries.json"
    if not p.exists():
        return 0
    try:
        return len(json.loads(p.read_text()))
    except Exception:
        return 0


def _check_model_path() -> bool:
    """True if the embedding config points to v4-bge (not the default MiniLM)."""
    p = REPO_ROOT / "src/lib/config.py"
    if not p.exists():
        return False
    txt = p.read_text()
    return "v4-bge" in txt or "sfu-academic-embed" in txt


def _lambdamart_trained() -> bool:
    return _file("models/lambdamart_v1.txt")


def _lambdamart_log_count() -> int:
    p = REPO_ROOT / "logs/query_log.jsonl"
    if not p.exists():
        return 0
    try:
        return sum(1 for line in p.open() if line.strip())
    except OSError:
        return 0


def _resolve(check) -> tuple[bool, str]:
    if check == "manual":
        return True, "manual"
    try:
        result = check()
        return bool(result), "auto"
    except Exception:
        return False, "error"


# ── Eval history ──────────────────────────────────────────────────────────────

def _load_best_runs(n: int = 6) -> list[dict]:
    p = REPO_ROOT / "results/sfu_eval_history.jsonl"
    if not p.exists():
        return []
    entries = []
    with p.open() as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    entries.append(json.loads(line))
                except Exception:
                    pass
    by_model: dict[str, dict] = {}
    for e in entries:
        key = e.get("model", "")
        if key not in by_model or e.get("num_queries", 0) >= by_model[key].get("num_queries", 0):
            by_model[key] = e
    ranked = sorted(by_model.values(), key=lambda x: x.get("ndcg10", 0), reverse=True)
    return ranked[:n]


# ── Print helpers ─────────────────────────────────────────────────────────────

def _tick(ok: bool) -> str:
    return "✅" if ok else "🔲"


def print_progress(brief: bool = False):
    print()
    print("=" * 70)
    print("  SFU Library MCP — Project Progress")
    print("=" * 70)

    # Phases
    print()
    print("PHASES")
    print(f"  {'Phase':<8}  {'Status'}  {'Description'}")
    print("  " + "-" * 65)
    for label, desc, check in PHASES:
        ok, _ = _resolve(check)
        print(f"  {label:<8}  {_tick(ok)}      {desc}")

    # Model leaderboard
    runs = _load_best_runs()
    if runs:
        print()
        print("MODEL LEADERBOARD (best run per model)")
        print(f"  {'Model':<48} {'NDCG@10':>8} {'MRR@10':>8} {'Recall@10':>9} {'N':>5}")
        print("  " + "-" * 80)
        for r in runs:
            model = r.get("model", "")[:47]
            ndcg = r.get("ndcg10", 0.0)
            mrr = r.get("mrr10", 0.0)
            rec = r.get("recall10", 0.0)
            n = r.get("num_queries", 0)
            marker = " ← current best" if r is runs[0] else ""
            print(f"  {model:<48} {ndcg:>8.4f} {mrr:>8.4f} {rec:>9.4f} {n:>5}{marker}")

    # Data and artifact counts
    if not brief:
        print()
        print("KEY ARTIFACTS")
        artifacts = [
            ("data/sfu_training_triplets.clean.jsonl",  "Training triplets"),
            ("models/sfu-academic-embed-v4-bge",         "v4-bge model"),
            ("models/sfu-academic-embed-v4-mini",        "v4-mini model"),
            ("data/sfu_eval_queries.json",               f"Eval queries ({_eval_query_count()})"),
            ("results/sfu_eval_history.jsonl",           "Eval history"),
            ("logs/query_log.jsonl",                     f"Query log ({_lambdamart_log_count()} entries)"),
            ("models/lambdamart_v1.txt",                 "LambdaMART model"),
        ]
        for rel, label in artifacts:
            exists = _file(rel)
            print(f"  {_tick(exists)}  {label:<40}  {rel}")

        # LambdaMART readiness
        log_count = _lambdamart_log_count()
        print()
        print("LAMBDAMART TRAINING READINESS")
        bars = [
            (500,  "Citation-proxy training"),
            (2000, "Click-signal training"),
        ]
        for threshold, label in bars:
            filled = min(log_count, threshold)
            pct = filled / threshold
            bar_len = 30
            bar = "█" * int(pct * bar_len) + "░" * (bar_len - int(pct * bar_len))
            status = "✅ ready" if log_count >= threshold else f"{threshold - log_count} more needed"
            print(f"  {label:<30}  [{bar}]  {log_count}/{threshold}  {status}")

        # Deferred / out-of-scope
        print()
        print("DEFERRED / OUT-OF-SCOPE")
        for label, desc, note in _deferred_items():
            print(f"  ⏸  {label:<22}  {desc:<35}  {note}")

    print()
    print("=" * 70)
    print()


def main():
    parser = argparse.ArgumentParser(description="SFU Library MCP project progress dashboard")
    parser.add_argument("--brief", action="store_true", help="Skip artifact and deferred sections")
    args = parser.parse_args()
    print_progress(brief=args.brief)


if __name__ == "__main__":
    main()
