"""Analytics bundle for the Research Analytics GUI (Phase N tracking layer).

build_analytics_bundle() assembles the exact JSON shapes the design's dashboard
panels consume, from artifacts already on disk (read-only, no heavy compute):

  ndcg_by_subject   <- data/eval_results/benchmark_llm_judge_final.json
  reranker_signal   <- models/lambdamart_v1.feature_importance.json | heuristic weights
  position_bias     <- logs/engagement_log.jsonl  (P(click|rank) + propensity)
  model_versions    <- models/model_registry.json (via lib.model_registry)
  session_replay    <- logs/engagement_log.jsonl  (query -> result -> action tree)
  kpis              <- benchmark summary + dataset counts

Panels with no data source yet return {"status": "stub"|"awaiting_click_data"} so the
contract is explicit rather than silently empty. Served by GET /analytics.
"""

import json
import logging
from collections import defaultdict
from pathlib import Path

logger = logging.getLogger("sfu_library_mcp")

_REPO_ROOT = Path(__file__).resolve().parents[2]
_BENCHMARK = _REPO_ROOT / "data/eval_results/benchmark_llm_judge_final.json"
_LM_IMPORTANCE = _REPO_ROOT / "models/lambdamart_v1.feature_importance.json"
_JUDGE_CACHE = _REPO_ROOT / "data/eval_results/llm_judge_cache.json"
_LM_EVAL = _REPO_ROOT / "data/eval_results/lambdamart_eval.json"

# Minimum clicks before the propensity curve is considered usable rather than noise.
_MIN_CLICKS_FOR_PROPENSITY = 20


def _read_json(path: Path):
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


# ── Panels ──────────────────────────────────────────────────────────────────────

def _panel_ndcg_by_subject() -> dict:
    bench = _read_json(_BENCHMARK)
    if not bench:
        return {"status": "stub", "reason": "benchmark file missing"}
    breakdown = bench.get("subject_breakdown") or {}
    # Surface the production fusion (rrf) as the primary, plus all methods for compare.
    primary = breakdown.get("rrf") or {}
    subjects = sorted(
        ({"subject": s, "ndcg": v} for s, v in primary.items()),
        key=lambda r: r["ndcg"],
    )
    return {
        "status": "ok",
        "primary_method": "rrf",
        "subjects": subjects,
        "by_method": breakdown,
        "timestamp": bench.get("timestamp"),
    }


def _panel_reranker_signal() -> dict:
    imp = _read_json(_LM_IMPORTANCE)
    if imp and imp.get("importance_gain"):
        total = sum(imp["importance_gain"].values()) or 1.0
        axes = [
            {"signal": name, "gain": gain, "weight": round(gain / total, 4)}
            for name, gain in imp["importance_gain"].items()
        ]
        return {"status": "ok", "source": "lambdamart_importance",
                "embed_model_path": imp.get("embed_model_path"), "axes": axes}
    # Fallback: the heuristic Stage-1 weights (always available).
    try:
        from lib.reranker import _WEIGHTS_WITH_EMBEDDING
        axes = [{"signal": k, "weight": v} for k, v in _WEIGHTS_WITH_EMBEDDING.items()]
        return {"status": "ok", "source": "heuristic_weights", "axes": axes}
    except Exception:
        return {"status": "stub"}


def _panel_position_bias(events: list[dict]) -> dict:
    impressions = defaultdict(int)
    clicks = defaultdict(int)
    for e in events:
        rank = e.get("rank")
        if rank is None:
            continue
        if e.get("kind") == "impression":
            impressions[rank] += 1
        elif e.get("kind") == "result_click":
            clicks[rank] += 1

    total_clicks = sum(clicks.values())
    ranks = sorted(set(impressions) | set(clicks)) or list(range(1, 11))
    curve = []
    for r in ranks:
        imp = impressions.get(r, 0)
        clk = clicks.get(r, 0)
        curve.append({
            "rank": r,
            "impressions": imp,
            "clicks": clk,
            "click": round(clk / imp, 4) if imp else 0.0,
        })
    # Propensity (rank-based examination): normalise click-rate to rank 1 == 1.0.
    base = next((c["click"] for c in curve if c["rank"] == 1 and c["click"] > 0), 0.0)
    for c in curve:
        c["propensity"] = round(c["click"] / base, 4) if base else None

    status = "ok" if total_clicks >= _MIN_CLICKS_FOR_PROPENSITY else "awaiting_click_data"
    return {
        "status": status,
        "total_clicks": total_clicks,
        "total_impressions": sum(impressions.values()),
        "min_clicks_required": _MIN_CLICKS_FOR_PROPENSITY,
        "curve": curve,
        "note": "Propensity feeds IPS debiasing before LambdaMART click-label training.",
    }


def _panel_model_versions() -> dict:
    try:
        from lib.model_registry import list_versions, registry
        return {"status": "ok", "updated": registry().get("updated"),
                "versions": list_versions()}
    except Exception as e:
        return {"status": "stub", "reason": str(e)}


def _panel_session_replay(events: list[dict]) -> dict:
    """Reconstruct sessions as query -> result -> action trees (newest first)."""
    sessions: dict[str, dict] = {}
    for e in events:
        sid = e.get("session_id")
        if not sid:
            continue
        sess = sessions.setdefault(sid, {"session_id": sid, "start": e.get("ts"),
                                         "queries": {}, "n_queries": 0,
                                         "n_clicks": 0, "n_actions": 0})
        kind = e.get("kind")
        q = e.get("query") or ""
        if kind == "query":
            sess["queries"].setdefault(q, {"query": q, "ts": e.get("ts"), "results": {}})
            sess["n_queries"] = len(sess["queries"])
        elif kind == "result_click":
            qnode = sess["queries"].setdefault(q, {"query": q, "results": {}})
            doc = e.get("doc_id") or ""
            qnode["results"].setdefault(doc, {"doc_id": doc, "rank": e.get("rank"), "actions": []})
            sess["n_clicks"] += 1
        elif kind == "action":
            qnode = sess["queries"].setdefault(q, {"query": q, "results": {}})
            doc = e.get("doc_id") or ""
            rnode = qnode["results"].setdefault(doc, {"doc_id": doc, "rank": e.get("rank"), "actions": []})
            rnode["actions"].append(e.get("action_type"))
            sess["n_actions"] += 1

    # Materialize dict trees into ordered lists.
    out = []
    for sess in sessions.values():
        queries = []
        for qnode in sess["queries"].values():
            results = list(qnode.get("results", {}).values())
            queries.append({"query": qnode["query"], "ts": qnode.get("ts"), "results": results})
        out.append({
            "session_id": sess["session_id"], "start": sess["start"],
            "n_queries": sess["n_queries"], "n_clicks": sess["n_clicks"],
            "n_actions": sess["n_actions"], "queries": queries,
        })
    out.sort(key=lambda s: s.get("start") or "", reverse=True)
    return {"status": "ok" if out else "awaiting_click_data", "sessions": out}


def _panel_kpis() -> dict:
    bench = _read_json(_BENCHMARK) or {}
    summary = bench.get("summary") or {}
    best = max(
        (m.get("mean_ndcg", 0) for m in summary.values() if isinstance(m, dict)),
        default=None,
    )
    judge = _read_json(_JUDGE_CACHE) or {}
    n_pairs = len(judge) if isinstance(judge, dict) else 0
    n_queries = bench.get("num_queries")
    lm_eval = _read_json(_LM_EVAL)
    return {
        "status": "ok",
        "ndcg_mean_best": best,
        "eval_queries": n_queries,
        "judged_pairs": n_pairs,
        "lambdamart_eval": lm_eval,  # null until training is run
    }


# ── Bundle ──────────────────────────────────────────────────────────────────────

_PANELS = {
    "ndcg_by_subject": lambda ev: _panel_ndcg_by_subject(),
    "reranker_signal": lambda ev: _panel_reranker_signal(),
    "position_bias": _panel_position_bias,
    "model_versions": lambda ev: _panel_model_versions(),
    "session_replay": _panel_session_replay,
    "kpis": lambda ev: _panel_kpis(),
    # Not part of the LambdaMART tracking layer — declared explicitly as stubs.
    "embedding_drift": lambda ev: {"status": "stub", "reason": "UMAP export not wired"},
    "access_funnel": lambda ev: {"status": "stub", "reason": "access-resolver logging not wired"},
    "oa_equity": lambda ev: {"status": "stub", "reason": "OA-by-subject export not wired"},
}


def build_analytics_bundle(panel: str | None = None) -> dict:
    """Assemble the analytics bundle, or a single panel when `panel` is given."""
    from lib.engagement import load_events
    events = load_events()

    if panel:
        fn = _PANELS.get(panel)
        if fn is None:
            return {"error": f"unknown panel: {panel}", "panels": sorted(_PANELS)}
        return {"panel": panel, "data": fn(events)}

    return {
        "generated_for": "research_analytics_dashboard",
        "panels": {name: fn(events) for name, fn in _PANELS.items()},
    }
