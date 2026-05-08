"""Server configuration from environment variables and Docker secrets."""

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

SERVER_VERSION = "1.1.0-phase-g"

logger = logging.getLogger("sfu_library_mcp")


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
        # Ship after confirming latency budget is acceptable (Phase O.2).
        "crossencoder_enabled": False,
        # Structured query log for LambdaMART training data collection (Phase O.3).
        # Writes (query, ranked docs, latency) to query_log_path as JSONL.
        "query_log_enabled": False,
        # TODO(Phase P): Add local_opensearch_enabled, splade_enabled,
        # federated_search_enabled, opensearch_url, opensearch_index,
        # splade_model_path, federated_recency_days — see docs/SPLADE_OPENSEARCH_INTEGRATION_PLAN.md §P.8
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

    # Query log for LambdaMART training data (O.3); empty = disabled
    query_log_path: str = ""

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
        "crossencoder_enabled": False,
        "query_log_enabled": False,
    }
    for key in default_features:
        env_key = f"SFU_FEATURE_{key.upper()}"
        env_val = os.environ.get(env_key)
        if env_val is not None:
            default_features[key] = _bool(env_val)

    return ServerConfig(
        search_timeout=int(os.environ.get("SFU_SEARCH_TIMEOUT", "30")),
        max_retries=int(os.environ.get("SFU_MAX_RETRIES", "3")),
        retry_base_delay=float(os.environ.get("SFU_RETRY_BASE_DELAY", "1.0")),
        retry_max_delay=float(os.environ.get("SFU_RETRY_MAX_DELAY", "60.0")),
        circuit_breaker_threshold=int(os.environ.get("SFU_CB_THRESHOLD", "5")),
        circuit_breaker_timeout=float(os.environ.get("SFU_CB_TIMEOUT", "60.0")),
        cache_ttl=int(os.environ.get("SFU_CACHE_TTL", "300")),
        cache_max_size=int(os.environ.get("SFU_CACHE_MAX_SIZE", "100")),
        cache_max_memory_mb=int(os.environ.get("SFU_CACHE_MAX_MEMORY_MB", "50")),
        log_level=os.environ.get("SFU_LOG_LEVEL", "INFO"),
        log_file=os.environ.get("SFU_LOG_FILE", "/tmp/sfu-library-mcp.log"),
        features=default_features,
        active_profile=os.environ.get("SFU_ACTIVE_PROFILE", "default"),
        zotero_api_key=_read_secret("zotero_api_key", "SFU_ZOTERO_API_KEY"),
        zotero_user_id=_read_secret("zotero_user_id", "SFU_ZOTERO_USER_ID"),
        openalex_api_key=_read_secret("openalex_api_key", "OPENALEX_API_KEY"),
        openalex_mailto=_read_secret("openalex_mailto", "OPENALEX_MAILTO"),
        openalex_daily_call_limit=int(os.environ.get("OPENALEX_DAILY_CALL_LIMIT", "900")),
        openalex_tracker_path=os.environ.get("OPENALEX_TRACKER_PATH", "/tmp/openalex_calls.json"),
        unpaywall_email=_read_secret("unpaywall_email", "UNPAYWALL_EMAIL"),
        sfu_db_registry_cache_ttl=int(
            os.environ.get("SFU_DB_REGISTRY_CACHE_TTL", "86400")
        ),
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
        metrics_log_path=os.environ.get("SFU_METRICS_LOG_PATH", ""),
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
