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
from pathlib import Path
from typing import Any

logger = logging.getLogger("sfu_library_mcp")

_SPLADE_SESSION: Any = None  # onnxruntime.InferenceSession
_SPLADE_TOKENIZER: Any = None
_SPLADE_ID_TO_TOKEN: dict[int, str] | None = None
_SPLADE_LOCK = threading.Lock()

# Repo root = .../sfu-library-mcp(-training); this file is at src/lib/.
_REPO_ROOT = Path(__file__).resolve().parents[2]
# The SPLADE ONNX export carries the exact indexer weights. The matching
# bert-base-uncased WordPiece tokenizer (30522 lowercase tokens) lives in the
# model dir itself or in any of the SFU embed model dirs (identical vocab).
# This mirrors scripts/mine_hard_negatives.py::SpladeOnnxEncoder, which was
# verified to reproduce indexed sparse_field weights.
_TOKENIZER_FALLBACK_DIRS = [
    _REPO_ROOT / "models" / "sfu-academic-embed-v1",
    _REPO_ROOT / "models" / "sfu-academic-embed-v4-bge",
    _REPO_ROOT / "models" / "sfu-academic-embed-v4-mini",
]


def _resolve_onnx_path(model_path: str) -> Path:
    """Resolve model_path (a dir or a direct .onnx file) to the ONNX file.

    Raises a clear error for a bare HF hub id (e.g. "naver/splade-...") which
    cannot be loaded offline in the serving container.
    """
    p = Path(model_path)
    if p.is_file() and p.suffix == ".onnx":
        return p
    if p.is_dir():
        candidate = p / "model.onnx"
        if candidate.is_file():
            return candidate
        raise FileNotFoundError(
            f"No model.onnx found in SPLADE model dir {p}. "
            "Point splade_model_path at the local ONNX export "
            "(e.g. models/splade_onnx)."
        )
    # Looks like a bare hub id or a missing path — can't load offline.
    raise FileNotFoundError(
        f"SPLADE model path {model_path!r} is not a local ONNX export dir/file. "
        "Offline serving requires the local ONNX model (e.g. models/splade_onnx); "
        "bare HuggingFace hub ids cannot be downloaded in the serving container."
    )


def _resolve_tokenizer_dir(model_path: str) -> Path:
    """Find a bert-base-uncased tokenizer dir (model dir first, then fallbacks)."""
    p = Path(model_path)
    search = []
    if p.is_dir():
        search.append(p)
    elif p.is_file():
        search.append(p.parent)
    search.extend(_TOKENIZER_FALLBACK_DIRS)
    for d in search:
        if (d / "tokenizer.json").is_file() or (d / "tokenizer_config.json").is_file():
            return d
    raise FileNotFoundError(
        "No local bert-base-uncased tokenizer found for SPLADE. Tried: "
        + ", ".join(str(d) for d in search)
    )


def _get_splade_model(model_path: str):
    """Load SPLADE ONNX session + tokenizer once (singleton, thread-safe).

    Returns (session, tokenizer, id_to_token). Loads the local ONNX export via
    onnxruntime (CPU) — no torch, no network — matching the indexer's weights.
    """
    global _SPLADE_SESSION, _SPLADE_TOKENIZER, _SPLADE_ID_TO_TOKEN
    with _SPLADE_LOCK:
        if _SPLADE_SESSION is None:
            try:
                import onnxruntime as ort
                from transformers import AutoTokenizer
            except ImportError as exc:
                raise ImportError(
                    "onnxruntime and transformers are required for SPLADE encoding. "
                    "Install them or disable splade_enabled."
                ) from exc

            onnx_path = _resolve_onnx_path(model_path)
            tok_dir = _resolve_tokenizer_dir(model_path)
            logger.info("Loading SPLADE ONNX model from %s (tokenizer %s)",
                        onnx_path, tok_dir.name)
            tokenizer = AutoTokenizer.from_pretrained(str(tok_dir))
            vocab = tokenizer.get_vocab()
            if len(vocab) != 30522:
                logger.warning(
                    "SPLADE tokenizer vocab size %d != 30522 (expected bert-base-uncased)",
                    len(vocab),
                )
            _SPLADE_TOKENIZER = tokenizer
            _SPLADE_ID_TO_TOKEN = {v: k for k, v in vocab.items()}
            _SPLADE_SESSION = ort.InferenceSession(
                str(onnx_path), providers=["CPUExecutionProvider"]
            )
            logger.info("SPLADE ONNX model loaded (CPU)")
        return _SPLADE_SESSION, _SPLADE_TOKENIZER, _SPLADE_ID_TO_TOKEN


def encode_splade(text: str, model_path: str) -> dict[str, float]:
    """Encode text to a sparse SPLADE term-weight dict via the local ONNX model.

    log1p(relu(logits)) max-pooled over the sequence dim — identical to the
    indexer and to scripts/mine_hard_negatives.py::SpladeOnnxEncoder. Special
    "[...]" tokens are dropped so they never become rank_feature fields.
    """
    import numpy as np

    session, tokenizer, id_to_token = _get_splade_model(model_path)
    enc = tokenizer(
        [text], max_length=512, truncation=True, padding=True, return_tensors="np"
    )
    input_ids = enc["input_ids"].astype(np.int64)
    feeds = {
        "input_ids": input_ids,
        "attention_mask": enc["attention_mask"].astype(np.int64),
        "token_type_ids": enc.get("token_type_ids", np.zeros_like(input_ids)).astype(np.int64),
    }
    logits = session.run(["logits"], feeds)[0][0]  # (seq, vocab)
    vec = np.log1p(np.maximum(logits, 0.0)).max(axis=0)  # (vocab,)

    nonzero = np.nonzero(vec)[0]
    if nonzero.size == 0:
        return {}
    order = nonzero[np.argsort(-vec[nonzero])]
    sparse_dict: dict[str, float] = {}
    for idx in order:
        token = id_to_token.get(int(idx), "")
        weight = float(vec[int(idx)])
        if not token or token.startswith("["):
            continue
        if weight <= 0.0:
            continue
        sparse_dict[token] = round(weight, 4)
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
        splade_model_path: str = "",
        timeout: int = 10,
    ):
        # Empty → local ONNX export (offline-loadable). Bare HF hub ids no longer
        # work in the serving container, so we no longer default to one.
        if not splade_model_path:
            splade_model_path = str(_REPO_ROOT / "models" / "splade_onnx")
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
