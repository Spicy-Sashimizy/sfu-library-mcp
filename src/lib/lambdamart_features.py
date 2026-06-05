"""Shared feature extraction for the LambdaMART learned reranker (Phase N / Tier 2).

This module is the SINGLE SOURCE OF TRUTH for the feature vector. Both the
training script (scripts/train_lambdamart.py via scripts/build_lambdamart_dataset.py)
and the inference path (reranker.rerank_with_lambdamart) call doc_features() so the
features a model is trained on are byte-for-byte the features it scores at serve time.

Design notes
------------
* Features are computed from the output of ``reranker._normalize_for_rerank(doc)``
  (the canonical, shape-stable doc view) plus two side inputs the normalized view
  intentionally drops: ``cited_by_count`` (lives at the top level of a normalize_work
  dict) and ``embed_cosine`` (computed by the embedding model, not stored on the doc).
* Scorers are imported from reranker, not re-implemented, so train/infer never drift.
  ``_CURRENT_YEAR`` is likewise shared — recency decay is anchored to the same year.
* ``bm25_rank_score`` from the original Phase O.3 draft is deliberately EXCLUDED: it is
  a positional/retrieval-order signal that has no equivalent in the unordered
  judge-cache training set. Including ``1/(1+rank)`` would teach the model the
  arbitrary ordering of the training cache and mispredict in production. If a
  positional signal is wanted later it must come from a real retrieval pass over the
  candidates at train time (out of scope while OpenSearch is offline).
"""

import math

from lib.reranker import _CURRENT_YEAR, _score_recency, _score_type_match  # noqa: F401

# Order matters — this is the column order of the trained model's feature matrix and
# is passed to lgb.Dataset(feature_name=...). Do not reorder without retraining.
FEATURE_NAMES = [
    "log_citations",     # log1p(cited_by_count)
    "recency",           # reranker._score_recency: 1.0 at current year, -0.05/yr
    "has_doi",           # 1.0 if a DOI is present
    "type_score",        # reranker._score_type_match: article=1.0 .. other=0.5
    "abstract_present",  # 1.0 if abstract longer than 50 chars
    "embed_cosine",      # cosine(query, title+abstract) from the SFU embedding model
]

N_FEATURES = len(FEATURE_NAMES)


def doc_features(norm: dict, cited_by_count: int, embed_cosine: float) -> list[float]:
    """Build the LambdaMART feature row for one (query, doc) pair.

    Args:
        norm: output of ``reranker._normalize_for_rerank(doc)`` — provides
            ``year``, ``type``, ``doi``, ``abstract``.
        cited_by_count: raw citation count (top-level on a normalize_work dict;
            not carried by ``norm``). 0 when unknown.
        embed_cosine: cosine similarity between the query embedding and the
            doc's title+abstract embedding (0.0 when the embedder is unavailable).

    Returns:
        A list of ``N_FEATURES`` floats in ``FEATURE_NAMES`` order.
    """
    log_citations = math.log1p(max(0, cited_by_count or 0))
    recency = _score_recency(norm)
    has_doi = 1.0 if norm.get("doi") else 0.0
    type_score = _score_type_match(norm)
    abstract_present = 1.0 if len(norm.get("abstract") or "") > 50 else 0.0
    embed = float(embed_cosine or 0.0)
    return [log_citations, recency, has_doi, type_score, abstract_present, embed]
