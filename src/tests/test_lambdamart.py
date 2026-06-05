"""Tests for the LambdaMART Tier 2 reranker, shared features, and GUI tracking.

These run WITHOUT lightgbm installed: the inference path and the feature module are
exercised via the unavailable-model fallback and a stub booster. Only tests that
genuinely need lightgbm are guarded with importorskip.
"""

import json

import pytest

import lib.reranker as reranker
from lib.lambdamart_features import FEATURE_NAMES, N_FEATURES, doc_features


# ── Shared feature module ───────────────────────────────────────────────────────

def _norm(year=2024, doc_type="article", doi="10.1/x", abstract="a" * 80):
    """A _normalize_for_rerank-shaped view."""
    return {"title": "t", "abstract": abstract, "year": year, "type": doc_type,
            "doi": doi, "authors": ["A"], "has_fulltext": True}


def test_feature_vector_length_and_order():
    feats = doc_features(_norm(), 10, 0.5)
    assert len(feats) == N_FEATURES == 6
    assert FEATURE_NAMES == [
        "log_citations", "recency", "has_doi", "type_score",
        "abstract_present", "embed_cosine",
    ]


def test_feature_determinism():
    n = _norm()
    assert doc_features(n, 42, 0.7) == doc_features(n, 42, 0.7)


def test_known_feature_values():
    import math
    feats = doc_features(_norm(year=reranker._CURRENT_YEAR, doc_type="article",
                               doi="10.1/x", abstract="a" * 80), 100, 0.9)
    log_cit, recency, has_doi, type_score, abstract_present, embed = feats
    assert log_cit == pytest.approx(math.log1p(100))
    assert recency == pytest.approx(1.0)          # current year -> no decay
    assert has_doi == 1.0
    assert type_score == 1.0                       # article
    assert abstract_present == 1.0                 # > 50 chars
    assert embed == pytest.approx(0.9)


def test_has_doi_and_abstract_flags_off():
    feats = doc_features(_norm(doi="", abstract="short"), 0, 0.0)
    assert feats[2] == 0.0   # has_doi
    assert feats[4] == 0.0   # abstract_present (<=50 chars)
    assert feats[0] == 0.0   # log_citations of 0


def test_recency_uses_shared_current_year():
    # Feature recency must track reranker._CURRENT_YEAR, not a hardcoded constant.
    cur = reranker._CURRENT_YEAR
    recent = doc_features(_norm(year=cur), 0, 0.0)[1]
    older = doc_features(_norm(year=cur - 10), 0, 0.0)[1]
    assert recent == pytest.approx(1.0)
    assert older == pytest.approx(0.5)             # 1 - 0.05*10


def test_feature_parity_raw_vs_normalized_shape():
    # The builder constructs normalize_work-shaped docs; production passes the same.
    # A raw-OpenAlex-shaped doc and a normalized-shaped doc for the same logical work
    # must yield the same feature vector (guards _normalize_for_rerank divergence).
    raw = {  # raw OpenAlex shape (has "authorships")
        "title": "t", "abstract": "a" * 80, "publication_year": 2024,
        "type": "article", "doi": "https://doi.org/10.1/x",
        "authorships": [{"author": {"display_name": "A"}}],
        "open_access": {"is_oa": True},
    }
    normalized = {  # normalize_work shape
        "title": "t", "abstract": "a" * 80, "date": "2024",
        "type": "article", "doi": "10.1/x", "authors": ["A"], "is_oa": True, "oa_url": "",
    }
    fr = doc_features(reranker._normalize_for_rerank(raw), 5, 0.3)
    fn = doc_features(reranker._normalize_for_rerank(normalized), 5, 0.3)
    assert fr == fn


# ── Inference fallback (no lightgbm / no model) ─────────────────────────────────

def _docs(n=5):
    return [{"cited_by_count": i, "title": f"t{i}", "abstract": "x" * 80,
             "type": "article", "doi": "10.1/x", "date": "2020"} for i in range(n)]


def test_falls_back_when_model_unavailable(monkeypatch):
    monkeypatch.setattr(reranker, "_lambdamart", False)
    docs = _docs(5)
    out = reranker.rerank_with_lambdamart(docs, "q", 3)
    assert out == docs[:3]


def test_respects_limit_on_fallback(monkeypatch):
    monkeypatch.setattr(reranker, "_lambdamart", False)
    assert len(reranker.rerank_with_lambdamart(_docs(5), "q", 2)) == 2


def test_get_lambdamart_missing_file(monkeypatch, tmp_path):
    monkeypatch.setattr(reranker, "_lambdamart", None)
    # Point at a path that doesn't exist -> unavailable, never raises.
    missing = tmp_path / "nope.txt"
    assert reranker._get_lambdamart(str(missing)) is None


def test_score_and_resort_with_stub_model(monkeypatch):
    # Stub booster whose predict() reverses order (higher score for later docs),
    # so we can assert re-sorting without lightgbm. Returns a numpy array to match
    # a real lgb.Booster (the inference path calls .tolist() on the result).
    import numpy as np

    class StubBooster:
        def predict(self, X):
            return np.arange(len(X), dtype=float)  # ascending -> last doc scores highest

    monkeypatch.setattr(reranker, "_lambdamart", StubBooster())
    # Avoid the embedding model in CI: force the cosine path to None -> zeros.
    monkeypatch.setattr(reranker, "_compute_semantic_scores", lambda *a, **k: None)

    docs = _docs(4)
    out = reranker.rerank_with_lambdamart(docs, "q", 4)
    # StubBooster gives doc[3] the top score, doc[0] the lowest.
    assert [d["cited_by_count"] for d in out] == [3, 2, 1, 0]


def test_stub_model_respects_limit(monkeypatch):
    import numpy as np

    class StubBooster:
        def predict(self, X):
            return np.zeros(len(X), dtype=float)
    monkeypatch.setattr(reranker, "_lambdamart", StubBooster())
    monkeypatch.setattr(reranker, "_compute_semantic_scores", lambda *a, **k: None)
    assert len(reranker.rerank_with_lambdamart(_docs(6), "q", 3)) == 3


def test_scoring_error_falls_back(monkeypatch):
    class BrokenBooster:
        def predict(self, X):
            raise RuntimeError("boom")
    monkeypatch.setattr(reranker, "_lambdamart", BrokenBooster())
    monkeypatch.setattr(reranker, "_compute_semantic_scores", lambda *a, **k: None)
    docs = _docs(5)
    assert reranker.rerank_with_lambdamart(docs, "q", 3) == docs[:3]


# ── Config wiring ───────────────────────────────────────────────────────────────

def test_config_flag_default_off():
    from lib.config import load_config
    cfg = load_config()
    assert cfg.features.get("lambdamart_enabled") is False
    assert hasattr(cfg, "lambdamart_model_path")


# ── Engagement + analytics tracking ─────────────────────────────────────────────

def test_engagement_validation_and_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setenv("SFU_ENGAGEMENT_LOG_PATH", str(tmp_path / "eng.jsonl"))
    import importlib
    import lib.engagement as eng
    importlib.reload(eng)

    # Valid single + batch; one invalid event is skipped, not fatal.
    assert eng.record_engagement({"session_id": "s1", "kind": "result_click",
                                  "query": "q", "rank": 1, "doc_id": "d1"}) == 1
    n = eng.record_engagement({"events": [
        {"session_id": "s1", "kind": "action", "query": "q", "doc_id": "d1", "action_type": "pdf"},
        {"session_id": "s1", "kind": "action", "doc_id": "d1"},   # missing action_type -> invalid
        {"kind": "query"},                                         # missing session_id -> invalid
    ]})
    assert n == 1
    assert len(eng.load_events()) == 2


def test_engagement_rejects_bad_kind(tmp_path, monkeypatch):
    monkeypatch.setenv("SFU_ENGAGEMENT_LOG_PATH", str(tmp_path / "eng.jsonl"))
    import importlib
    import lib.engagement as eng
    importlib.reload(eng)
    assert eng.log_engagement("s1", "not_a_kind") is False


def test_analytics_bundle_shape(tmp_path, monkeypatch):
    monkeypatch.setenv("SFU_ENGAGEMENT_LOG_PATH", str(tmp_path / "eng.jsonl"))
    import importlib
    import lib.engagement as eng
    importlib.reload(eng)
    import lib.analytics as analytics
    importlib.reload(analytics)

    # Seed impressions (denominators) + a couple clicks.
    evs = []
    for s in range(3):
        for rank in range(1, 4):
            evs.append({"session_id": f"s{s}", "kind": "impression",
                        "query": "q", "rank": rank, "doc_id": f"d{rank}"})
    evs.append({"session_id": "s0", "kind": "result_click", "query": "q", "rank": 1, "doc_id": "d1"})
    evs.append({"session_id": "s0", "kind": "query", "query": "q"})
    eng.record_engagement({"events": evs})

    bundle = analytics.build_analytics_bundle()
    panels = bundle["panels"]
    # All declared panels present.
    for name in ("ndcg_by_subject", "reranker_signal", "position_bias",
                 "model_versions", "session_replay", "kpis"):
        assert name in panels
    # Position bias computes P(click|rank) from real impressions/clicks.
    pb = panels["position_bias"]
    rank1 = next(c for c in pb["curve"] if c["rank"] == 1)
    assert rank1["impressions"] == 3 and rank1["clicks"] == 1
    assert rank1["propensity"] == 1.0
    # Reranker signal falls back to heuristic weights when no trained importance file.
    assert panels["reranker_signal"]["status"] == "ok"
    # Model-versions table is populated from the registry.
    assert panels["model_versions"]["status"] == "ok"
    assert len(panels["model_versions"]["versions"]) >= 1


def test_analytics_single_panel():
    import lib.analytics as analytics
    out = analytics.build_analytics_bundle(panel="model_versions")
    assert out["panel"] == "model_versions"
    assert "data" in out


def test_model_registry_marks_availability():
    from lib.model_registry import list_versions
    rows = list_versions()
    # lambdamart_v1 is pending and its file is absent -> available False.
    lm = [r for r in rows if r["component"] == "lambdamart"]
    assert lm and lm[0]["status"] == "pending"
    assert lm[0].get("available") is False
