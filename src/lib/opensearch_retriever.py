"""OpenSearch retriever — BM25F and SPLADE sparse-vector search.

Two search modes, selected by the `splade_enabled` config flag:
  - BM25F (default, no GPU required): multi_match across title^3 / abstract / concepts^2.
  - SPLADE (splade_enabled=True): encode query → sparse term weights → rank_features query.
    Requires transformers + torch; loaded lazily so BM25F deployments don't need them.

Index schema is defined in docker/opensearch/index_template.json.
"""

import logging
import os
import threading
from typing import Any

logger = logging.getLogger("sfu_library_mcp")

_SPLADE_MODEL: Any = None
_SPLADE_TOKENIZER: Any = None
_SPLADE_LOCK = threading.Lock()


def _get_splade_model(model_path: str):
    """Load SPLADE model + tokenizer once (singleton, thread-safe)."""
    global _SPLADE_MODEL, _SPLADE_TOKENIZER
    with _SPLADE_LOCK:
        if _SPLADE_MODEL is None:
            try:
                import torch
                from transformers import AutoModelForMaskedLM, AutoTokenizer
            except ImportError as exc:
                raise ImportError(
                    "transformers and torch are required for SPLADE encoding. "
                    "Install them or disable splade_enabled."
                ) from exc
            logger.info("Loading SPLADE model from %s", model_path)
            _SPLADE_TOKENIZER = AutoTokenizer.from_pretrained(model_path)
            _SPLADE_MODEL = AutoModelForMaskedLM.from_pretrained(model_path)
            device = "cuda" if torch.cuda.is_available() else "cpu"
            _SPLADE_MODEL = _SPLADE_MODEL.to(device)
            _SPLADE_MODEL.eval()
            logger.info("SPLADE model loaded on %s", device)
        return _SPLADE_MODEL, _SPLADE_TOKENIZER


def encode_splade(text: str, model_path: str) -> dict[str, float]:
    """Encode text to a sparse SPLADE term-weight dict."""
    import torch

    model, tokenizer = _get_splade_model(model_path)
    device = next(model.parameters()).device
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=512,
        padding=True,
    ).to(device)
    with torch.no_grad():
        logits = model(**inputs).logits
    # ReLU + log(1 + x) → sparse weights; max-pool over sequence dim
    sparse = torch.log1p(torch.relu(logits)).max(dim=1).values.squeeze(0)
    vocab = tokenizer.get_vocab()
    id_to_token = {v: k for k, v in vocab.items()}
    sparse_dict: dict[str, float] = {}
    nonzero = sparse.nonzero(as_tuple=True)[0].tolist()
    for idx in nonzero:
        weight = sparse[idx].item()
        if weight > 0:
            token = id_to_token.get(idx, f"__unk_{idx}__")
            sparse_dict[token] = weight
    return sparse_dict


class OpenSearchRetriever:
    """Retriever wrapping an OpenSearch index.

    Supports BM25F (no extra deps) and SPLADE (requires torch + transformers).
    The caller controls the mode via `splade_enabled`.
    """

    def __init__(
        self,
        url: str = "",
        index: str = "openalex_works",
        splade_enabled: bool = False,
        splade_model_path: str = "naver/splade-cocondenser-distil",
        timeout: int = 10,
    ):
        self.url = url or os.environ.get("SFU_OPENSEARCH_URL", "http://localhost:9200")
        self.index = index
        self.splade_enabled = splade_enabled
        self.splade_model_path = splade_model_path
        self.timeout = timeout

    def _http(self, method: str, path: str, body: dict | None = None) -> dict | None:
        """Execute an HTTP request against OpenSearch; return parsed JSON or None.

        Narrowed to network/transport and JSON-decode errors so genuine bugs
        (e.g. a malformed query body raising TypeError) propagate instead of being
        silently swallowed into an empty result set.
        """
        import requests as _requests
        try:
            url = f"{self.url.rstrip('/')}/{path}"
            resp = _requests.request(
                method,
                url,
                json=body,
                timeout=self.timeout,
                headers={"Content-Type": "application/json"},
            )
            resp.raise_for_status()
            return resp.json()
        except (_requests.exceptions.RequestException, ValueError) as exc:
            logger.warning("OpenSearch request failed (%s %s): %s", method, path, exc)
            return None

    @staticmethod
    def _build_filter_clauses(filters: dict | None) -> list[dict]:
        """Translate OpenAlex-style filter dict → OpenSearch bool.filter clauses.

        Accepts the same filter keys the live OpenAlex path uses (built in
        tools._handle_search_academic) plus a couple of plain aliases:
          - publication_year: "YYYY-YYYY" range, or a single "YYYY"
          - from_publication_date / to_publication_date: ISO dates → year range
          - type: term filter on the keyword `type` field
          - open_access.is_oa / is_oa: term filter on the boolean `is_oa` field
        Unknown keys are ignored so callers can pass a superset safely.
        """
        if not filters:
            return []

        clauses: list[dict] = []

        def _to_year(value: str) -> int | None:
            try:
                return int(str(value)[:4])
            except (ValueError, TypeError):
                return None

        # Explicit year range or single year
        year_range: dict[str, int] = {}
        py = filters.get("publication_year")
        if py:
            text = str(py)
            if "-" in text:
                lo, _, hi = text.partition("-")
                lo_y, hi_y = _to_year(lo), _to_year(hi)
                if lo_y is not None:
                    year_range["gte"] = lo_y
                if hi_y is not None:
                    year_range["lte"] = hi_y
            else:
                single = _to_year(text)
                if single is not None:
                    year_range["gte"] = single
                    year_range["lte"] = single

        # from/to publication dates → year bounds (index stores integer year)
        from_date = filters.get("from_publication_date")
        if from_date:
            y = _to_year(from_date)
            if y is not None:
                year_range.setdefault("gte", y)
        to_date = filters.get("to_publication_date")
        if to_date:
            y = _to_year(to_date)
            if y is not None:
                year_range.setdefault("lte", y)

        if year_range:
            clauses.append({"range": {"publication_year": year_range}})

        work_type = filters.get("type")
        if work_type:
            clauses.append({"term": {"type": work_type}})

        # OA flag may arrive as the OpenAlex-style "open_access.is_oa" key or a
        # plain "is_oa". Treat the string "true"/bool True as the OA filter.
        oa_value = filters.get("open_access.is_oa", filters.get("is_oa"))
        if oa_value not in (None, "", False, "false"):
            clauses.append({"term": {"is_oa": True}})

        return clauses

    def _build_bm25f_query(self, query: str, top_k: int, filters: dict | None = None) -> dict:
        body: dict = {
            "size": top_k,
            "query": {
                "bool": {
                    "must": [
                        {
                            "multi_match": {
                                "query": query,
                                "fields": ["title^3", "abstract", "concepts^2"],
                                "type": "most_fields",
                                "tie_breaker": 0.5,
                            }
                        }
                    ],
                }
            },
            "_source": ["doi", "title", "abstract", "publication_year", "type", "is_oa"],
        }
        filter_clauses = self._build_filter_clauses(filters)
        if filter_clauses:
            body["query"]["bool"]["filter"] = filter_clauses
        return body

    def _build_splade_query(self, query: str, top_k: int, max_terms: int = 64,
                            filters: dict | None = None) -> dict:
        sparse = encode_splade(query, self.splade_model_path)
        if not sparse:
            logger.warning("SPLADE encoding produced empty vector; falling back to BM25F")
            return self._build_bm25f_query(query, top_k, filters=filters)
        # Use bool.should with per-term rank_feature + log scaling (SPLADE paper recommendation).
        # scaling_factor=4 boosts rare term weights; top max_terms by weight controls latency.
        should = [
            {"rank_feature": {"field": f"sparse_field.{t}", "boost": w,
                              "log": {"scaling_factor": 4}}}
            for t, w in sorted(sparse.items(), key=lambda x: -x[1])[:max_terms]
        ]
        bool_query: dict = {"should": should, "minimum_should_match": 1}
        filter_clauses = self._build_filter_clauses(filters)
        if filter_clauses:
            bool_query["filter"] = filter_clauses
        return {
            "size": top_k,
            "query": {"bool": bool_query},
            "_source": ["doi", "title", "abstract", "publication_year", "type", "is_oa"],
        }

    def search(self, query: str, top_k: int = 50, mode: str | None = None,
               filters: dict | None = None) -> list[dict]:
        """Run a search and return normalized result dicts.

        Args:
            query: raw user query
            top_k: max hits to return
            mode: "bm25f" or "splade" to force a mode for this call.
                None (default) uses self.splade_enabled. Per-call selection lets
                FederatedSearchRouter run both modes for RRF fusion regardless of
                the instance flag.
            filters: OpenAlex-style filter dict (publication_year / from/to dates /
                type / is_oa). Translated to an OpenSearch bool.filter so the local
                legs honour the same year/type/OA constraints as the live API path.

        Returns: list of {doi, title, abstract, publication_year, year, score, source, type, is_oa}
        """
        if mode is None:
            use_splade = self.splade_enabled
        else:
            use_splade = mode == "splade"

        if use_splade:
            body = self._build_splade_query(query, top_k, filters=filters)
        else:
            body = self._build_bm25f_query(query, top_k, filters=filters)

        data = self._http("POST", f"{self.index}/_search", body)
        if not data:
            return []

        hits = (data.get("hits") or {}).get("hits") or []
        results: list[dict] = []
        for hit in hits:
            src = hit.get("_source") or {}
            pub_year = src.get("publication_year")
            results.append({
                "doi": src.get("doi", ""),
                "title": src.get("title", ""),
                "abstract": src.get("abstract", ""),
                # Emit the keys the downstream reranker and query logger expect:
                #   reranker._normalize_for_rerank reads "date" (OpenAlex-normalized shape)
                #   tools._log_query reads "publication_year".
                # "year" is retained for back-compat with existing callers/tests.
                "publication_year": pub_year,
                "date": str(pub_year) if pub_year is not None else "",
                "year": pub_year,
                "score": hit.get("_score", 0.0),
                "source": "opensearch",
                "type": src.get("type", ""),
                "is_oa": src.get("is_oa", False),
            })
        return results

    def is_available(self) -> bool:
        """Return True if the OpenSearch cluster is reachable and healthy."""
        data = self._http("GET", "_cluster/health")
        if not data:
            return False
        return data.get("status") in ("green", "yellow")
