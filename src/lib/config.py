"""Server configuration from environment variables and Docker secrets."""

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

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
        "fusion_enabled": True,
        "rerank_enabled": True,
        "zotero_enabled": True,
    })

    # Zotero integration
    zotero_api_key: str = ""
    zotero_user_id: str = ""

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
        "fusion_enabled": True,
        "rerank_enabled": True,
        "zotero_enabled": True,
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
        semantic_scholar_api_key=_read_secret(
            "semantic_scholar_api_key", "SFU_SEMANTIC_SCHOLAR_API_KEY"
        ),
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

    # Semantic Scholar credential validation
    if not config.semantic_scholar_api_key:
        warnings.append("SFU_SEMANTIC_SCHOLAR_API_KEY is empty — S2 requests will be unauthenticated (lower rate limit)")

    # Zotero credential validation
    if not config.zotero_api_key:
        warnings.append("SFU_ZOTERO_API_KEY is empty — Zotero tools will fail")
    if not config.zotero_user_id:
        warnings.append("SFU_ZOTERO_USER_ID is empty — Zotero tools will fail")

    return warnings
