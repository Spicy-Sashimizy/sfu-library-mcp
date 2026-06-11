"""Server configuration from environment variables and Docker secrets."""

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

SERVER_VERSION = "1.1.0-phase-g"

logger = logging.getLogger("sfu_library_mcp")


def _int_env(name: str, default: int) -> int:
    """Parse an int env var, falling back to the default on missing/invalid values."""
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        return int(raw)
    except ValueError:
        logger.warning("Invalid int for %s=%r; using default %s", name, raw, default)
        return default


def _float_env(name: str, default: float) -> float:
    """Parse a float env var, falling back to the default on missing/invalid values."""
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        return float(raw)
    except ValueError:
        logger.warning("Invalid float for %s=%r; using default %s", name, raw, default)
        return default

# Repo root = .../sfu-library-mcp(-training); this file is at src/lib/config.py.
_REPO_ROOT = Path(__file__).resolve().parents[2]
# Local SPLADE ONNX export produced by the indexer. Loading this offline via
# onnxruntime reproduces the indexed sparse weights exactly; the previous default
# (a bare HF hub id) could not be fetched in the air-gapped serving container.
_DEFAULT_SPLADE_MODEL_PATH = str(_REPO_ROOT / "models" / "splade_onnx")


@dataclass
class ServerConfig:
    """Configuration for the SFU Library MCP server.

    Values come from environment variables with hardcoded fallbacks.
    """

    # Timeouts
    search_timeout: int = 30  # requests timeout for API calls

    # Retry settings
    max_retries: int = 3
    retry_base_delay: float = 1.0  # seconds
    retry_max_delay: float = 60.0  # seconds

    # Circuit breaker
    circuit_breaker_threshold: int = 5
    circuit_breaker_timeout: float = 60.0

    # Cache settings
    cache_ttl: int = 300  # seconds
    cache_max_size: int = 100  # max entries
    cache_max_memory_mb: int = 50  # max memory in MB

    # Logging
    log_level: str = "INFO"
    log_file: str = ""

    # Feature flags
    features: dict[str, bool] = field(default_factory=lambda: {
        "cache_enabled": True,
        "retry_enabled": True,
        "circuit_breaker_enabled": True,
        "metrics_enabled": True,
        "rerank_enabled": True,
        "zotero_enabled": True,
        "semantic_scholar_enabled": True,
        "europe_pmc_enabled": False,
        "local_embedding_enabled": True,
        # When True, reranker fuses original Primo doc order (lexical/BM25 proxy)
        # with embedding cosine ranks via Reciprocal Rank Fusion. Off by default
        # until the SFU NDCG benchmark confirms it beats single-ranker scoring.
        "rrf_enabled": False,
        # CrossEncoder second-pass reranker on top-20 RRF candidates (~50-100ms).
        # Enabled 2026-05-16 after Q1.4 latency smoke test confirmed acceptable budget.
        "crossencoder_enabled": True,
        # Structured query log for LambdaMART training data collection (Phase O.3).
        # Writes (query, ranked docs, latency) to query_log_path as JSONL.
        "query_log_enabled": False,
        # Tier 2 LambdaMART learned reranker (Phase N). Off by default until the
        # NDCG eval (data/eval_results/lambdamart_eval.json) confirms a win — and a
        # no-op anyway unless lightgbm + models/lambdamart_v1.txt are both present.
        "lambdamart_enabled": False,
        # Emit per-rank result impressions to the engagement log on each search
        # (the propensity denominators for the analytics position-bias panel).
        # Click/action events are ingested via /engagement regardless of this flag.
        "engagement_log_enabled": False,
        # Phase P: OpenSearch / SPLADE feature flags.
        # Master switch — OpenSearch path (P.1 container must be running).
        # Default-on as of 2026-05-16 after the 120-query Phase P.11 eval; the
        # federated router falls back gracefully on connection failure.
        "local_opensearch_enabled": True,
        # Single-mode SPLADE on OpenSearch queries (BM25F otherwise). Retained
        # for diagnostics; production traffic goes through local_rrf_enabled
        # which dispatches both modes and RRF-fuses (+5.1% NDCG@10 vs BM25).
        "splade_enabled": False,
        # Route queries through FederatedSearchRouter (live API + local index).
        # Default-on as of 2026-05-16; falls through to live API on failure.
        "federated_search_enabled": True,
        # Dispatch BM25F + SPLADE on the local index and RRF-fuse.
        # Eval (2026-05-15, 120 queries): NDCG@10 0.2723 vs BM25 0.259 / SPLADE 0.2586.
        # Avg overlap BM25↔SPLADE = 7.5% — they retrieve near-orthogonal sets.
        "local_rrf_enabled": True,
    })

    # Local embedding model
    embedding_model_path: str = ""
    embedding_default_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    # Zotero integration
    zotero_api_key: str = ""
    zotero_user_id: str = ""

    # OpenAlex — api_key gives 100 req/s; mailto polite pool gives 10 req/s; neither = 1 req/s
    openalex_api_key: str = ""
    openalex_mailto: str = ""
    # Daily call budget (free plan = ~1,000 searches/day; default 900 leaves 10% headroom)
    openalex_daily_call_limit: int = 900
    openalex_tracker_path: str = "/tmp/openalex_calls.json"

    # Unpaywall (email required for access)
    unpaywall_email: str = ""

    # SFU Database Registry (Solr)
    sfu_db_registry_cache_ttl: int = 86400   # 24 h
    sfu_db_registry_cache_file: str = "/tmp/sfu_databases_cache.json"

    # Local search backend: "thinclient" (tantivy+BMP+usearch, no JVM — the
    # validated laptop stack) or "opensearch" (legacy, external cluster).
    search_backend: str = "thinclient"
    # Thin-client index root (built by scripts/build_thinclient_index.py)
    thinclient_index_root: str = ""

    # OpenSearch / SPLADE (Phase P — legacy backend)
    opensearch_url: str = "http://localhost:9200"
    opensearch_index: str = "openalex_works"
    # Local SPLADE ONNX export dir (default) or a HF model id; used when
    # splade_enabled = True. The default loads offline via onnxruntime.
    splade_model_path: str = _DEFAULT_SPLADE_MODEL_PATH
    # Queries within this many days of today route to the live API (federated router)
    federated_recency_days: int = 30

    # Query log for LambdaMART training data (O.3); empty = disabled
    query_log_path: str = ""

    # Trained LambdaMART model (Phase N); empty = use reranker default
    # (models/lambdamart_v1.txt). Only consulted when lambdamart_enabled.
    lambdamart_model_path: str = ""

    # Tool metrics log; empty = disabled (in-memory only, resets on restart)
    metrics_log_path: str = ""

    # EZProxy
    sfu_ezproxy_base: str = "https://proxy.lib.sfu.ca/login?url="

    # Semantic Scholar API
    semantic_scholar_api_key: str = ""

    # Multi-profile support
    active_profile: str = "default"


def _read_secret(name: str, env_var: str, default: str = "") -> str:
    """Read a secret from Docker secret file, falling back to env var.

    Priority: /run/secrets/{name} -> os.environ[env_var] -> default
    """
    secret_path = Path(f"/run/secrets/{name}")
    try:
        if secret_path.is_file():
            value = secret_path.read_text().strip()
            if value:
                logger.debug("Loaded secret '%s' from Docker secret file", name)
                return value
    except OSError:
        pass

    env_value = os.environ.get(env_var, "")
    if env_value:
        logger.debug("Loaded secret '%s' from env var %s", name, env_var)
        return env_value

    if default:
        return default

    logger.debug("Secret '%s' not found in Docker secrets or env var %s", name, env_var)
    return ""


def _load_dotenv() -> None:
    """Load .env file from project root if it exists.

    Only sets vars that aren't already in the environment.
    """
    for candidate in [
        Path(__file__).parent.parent.parent / ".env",
        Path.cwd() / ".env",
    ]:
        if candidate.is_file():
            try:
                for line in candidate.read_text().splitlines():
                    line = line.strip()
                    if not line or line.startswith("#") or "=" not in line:
                        continue
                    key, _, value = line.partition("=")
                    key = key.strip()
                    value = value.strip()
                    if key and key not in os.environ:
                        os.environ[key] = value
                logger.debug("Loaded .env from %s", candidate)
                return
            except OSError:
                pass


def load_config() -> ServerConfig:
    """Load configuration from Docker secrets and environment variables."""
    _load_dotenv()

    def _bool(val: str) -> bool:
        return val.lower() in ("1", "true", "yes", "on")

    # Parse feature flags from env
    default_features = {
        "cache_enabled": True,
        "retry_enabled": True,
        "circuit_breaker_enabled": True,
        "metrics_enabled": True,
        "rerank_enabled": True,
        "zotero_enabled": True,
        "semantic_scholar_enabled": True,
        "europe_pmc_enabled": False,
        "local_embedding_enabled": True,
        "rrf_enabled": False,
        "crossencoder_enabled": True,
        "query_log_enabled": False,
        "lambdamart_enabled": False,
        "engagement_log_enabled": False,
        "local_opensearch_enabled": True,
        "splade_enabled": False,
        "federated_search_enabled": True,
        "local_rrf_enabled": True,
    }
    for key in default_features:
        env_key = f"SFU_FEATURE_{key.upper()}"
        env_val = os.environ.get(env_key)
        if env_val is not None:
            default_features[key] = _bool(env_val)

    return ServerConfig(
        search_timeout=_int_env("SFU_SEARCH_TIMEOUT", 30),
        max_retries=_int_env("SFU_MAX_RETRIES", 3),
        retry_base_delay=_float_env("SFU_RETRY_BASE_DELAY", 1.0),
        retry_max_delay=_float_env("SFU_RETRY_MAX_DELAY", 60.0),
        circuit_breaker_threshold=_int_env("SFU_CB_THRESHOLD", 5),
        circuit_breaker_timeout=_float_env("SFU_CB_TIMEOUT", 60.0),
        cache_ttl=_int_env("SFU_CACHE_TTL", 300),
        cache_max_size=_int_env("SFU_CACHE_MAX_SIZE", 100),
        cache_max_memory_mb=_int_env("SFU_CACHE_MAX_MEMORY_MB", 50),
        log_level=os.environ.get("SFU_LOG_LEVEL", "INFO"),
        log_file=os.environ.get("SFU_LOG_FILE", "/tmp/sfu-library-mcp.log"),
        features=default_features,
        active_profile=os.environ.get("SFU_ACTIVE_PROFILE", "default"),
        zotero_api_key=_read_secret("zotero_api_key", "SFU_ZOTERO_API_KEY"),
        zotero_user_id=_read_secret("zotero_user_id", "SFU_ZOTERO_USER_ID"),
        openalex_api_key=_read_secret("openalex_api_key", "OPENALEX_API_KEY"),
        openalex_mailto=_read_secret("openalex_mailto", "OPENALEX_MAILTO"),
        openalex_daily_call_limit=_int_env("OPENALEX_DAILY_CALL_LIMIT", 900),
        openalex_tracker_path=os.environ.get("OPENALEX_TRACKER_PATH", "/tmp/openalex_calls.json"),
        unpaywall_email=_read_secret("unpaywall_email", "UNPAYWALL_EMAIL"),
        sfu_db_registry_cache_ttl=_int_env("SFU_DB_REGISTRY_CACHE_TTL", 86400),
        sfu_db_registry_cache_file=os.environ.get(
            "SFU_DB_REGISTRY_CACHE_FILE", "/tmp/sfu_databases_cache.json"
        ),
        sfu_ezproxy_base=os.environ.get(
            "SFU_EZPROXY_BASE", "https://proxy.lib.sfu.ca/login?url="
        ),
        semantic_scholar_api_key=_read_secret(
            "semantic_scholar_api_key", "SFU_SEMANTIC_SCHOLAR_API_KEY"
        ),
        embedding_model_path=os.environ.get("SFU_EMBEDDING_MODEL_PATH", ""),
        embedding_default_model=os.environ.get(
            "SFU_EMBEDDING_DEFAULT_MODEL",
            "sentence-transformers/all-MiniLM-L6-v2",
        ),
        query_log_path=os.environ.get("SFU_QUERY_LOG_PATH", ""),
        lambdamart_model_path=os.environ.get("SFU_LAMBDAMART_MODEL_PATH", ""),
        metrics_log_path=os.environ.get("SFU_METRICS_LOG_PATH", ""),
        search_backend=os.environ.get("SFU_SEARCH_BACKEND", "thinclient"),
        thinclient_index_root=os.environ.get("SFU_THINCLIENT_INDEX_ROOT", ""),
        opensearch_url=os.environ.get("SFU_OPENSEARCH_URL", "http://localhost:9200"),
        opensearch_index=os.environ.get("SFU_OPENSEARCH_INDEX", "openalex_works"),
        splade_model_path=os.environ.get(
            "SFU_SPLADE_MODEL_PATH", _DEFAULT_SPLADE_MODEL_PATH
        ),
        federated_recency_days=_int_env("SFU_FEDERATED_RECENCY_DAYS", 30),
    )


def validate_config(config: ServerConfig) -> list[str]:
    """Validate configuration and return a list of warnings."""
    warnings: list[str] = []

    if config.search_timeout < 1:
        warnings.append(f"search_timeout too low: {config.search_timeout}")
    if config.max_retries < 0:
        warnings.append(f"max_retries cannot be negative: {config.max_retries}")
    if config.cache_ttl < 0:
        warnings.append(f"cache_ttl cannot be negative: {config.cache_ttl}")
    if config.cache_max_size < 1:
        warnings.append(f"cache_max_size must be positive: {config.cache_max_size}")

    valid_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
    if config.log_level.upper() not in valid_levels:
        warnings.append(f"Invalid log_level: {config.log_level}")

    # OpenAlex rate limit tier
    if not config.openalex_api_key and not config.openalex_mailto:
        warnings.append("Neither OPENALEX_API_KEY nor OPENALEX_MAILTO set — OpenAlex limited to 1 req/s")

    # OpenAlex tracker path — /tmp is cleared on container/host restart
    if config.openalex_tracker_path.startswith("/tmp"):
        warnings.append(
            "OPENALEX_TRACKER_PATH is in /tmp — the daily call counter will reset on container restart. "
            "Set OPENALEX_TRACKER_PATH to a persistent volume path to preserve budget across restarts."
        )

    # Unpaywall (required for OA fallback step)
    if not config.unpaywall_email:
        warnings.append("UNPAYWALL_EMAIL is unset — Unpaywall OA resolution disabled")

    # Semantic Scholar (optional API key for higher rate limits)
    if not config.semantic_scholar_api_key:
        warnings.append("SFU_SEMANTIC_SCHOLAR_API_KEY is empty — S2 requests will be unauthenticated (lower rate limit)")

    # Zotero credential validation
    if not config.zotero_api_key:
        warnings.append("SFU_ZOTERO_API_KEY is empty — Zotero tools will fail")
    if not config.zotero_user_id:
        warnings.append("SFU_ZOTERO_USER_ID is empty — Zotero tools will fail")

    return warnings
