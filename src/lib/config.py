"""Server configuration from environment variables and Docker secrets.

Implements CONFIG-001 through CONFIG-004, ERR-005, AUTH-008.
Secret loading priority: Docker secret file (/run/secrets/X) -> env var (SFU_X) -> empty string.
"""

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger("sfu_library_mcp")


@dataclass
class ServerConfig:
    """Configuration for the SFU Library MCP server.

    Values come from environment variables with hardcoded fallbacks to
    preserve backwards compatibility with the original monolith.
    """

    # Authentication (CONFIG-001: externalize credentials)
    sfu_username: str = ""
    sfu_password: str = ""
    mfa_secret: str = ""
    mfa_device_name: str = ""  # Duo device name to select

    # Token management
    token_cache_file: str = ""
    token_refresh_buffer: int = 300  # seconds before expiry to refresh (AUTH-001)

    # Timeouts (ERR-005, AUTH-008: configurable timeouts)
    auth_timeout: int = 10  # WebDriverWait timeout
    search_timeout: int = 30  # requests timeout for API calls
    auth_sleep_after_login: int = 5  # sleep after page loads

    # Retry settings
    max_retries: int = 3
    retry_base_delay: float = 1.0  # seconds
    retry_max_delay: float = 60.0  # seconds

    # Circuit breaker
    circuit_breaker_threshold: int = 5
    circuit_breaker_timeout: float = 60.0

    # Cache settings (PERF-001, PERF-005)
    cache_ttl: int = 300  # seconds
    cache_max_size: int = 100  # max entries
    cache_max_memory_mb: int = 50  # max memory in MB

    # Logging
    log_level: str = "INFO"
    log_file: str = ""

    # Feature flags (CONFIG-003)
    features: dict[str, bool] = field(default_factory=lambda: {
        "cache_enabled": True,
        "retry_enabled": True,
        "circuit_breaker_enabled": True,
        "token_encryption_enabled": False,
        "screenshot_on_failure": False,
        "metrics_enabled": True,
        "fusion_enabled": True,
        "rerank_enabled": True,
        "pdf_download_enabled": True,
        "zotero_enabled": True,
        "host_download_enabled": True,
    })

    # Zotero integration
    zotero_api_key: str = ""
    zotero_user_id: str = ""

    # PDF download
    download_dir: str = "/tmp/sfu-library-downloads"
    host_download_dir: str = "/mnt/host-downloads"
    download_timeout: int = 60
    max_pdf_text_chars: int = 100_000

    # EZProxy
    ezproxy_prefix: str = "https://proxy.lib.sfu.ca/login?url="  # deprecated, kept for backwards compat
    ezproxy_login_url: str = "https://login.proxy.lib.sfu.ca/login?qurl="
    ezproxy_proxy_base: str = "proxy.lib.sfu.ca"

    # Tiered download strategy
    download_tiers: list[str] = field(default_factory=lambda: ["curl_cffi", "playwright", "requests"])
    playwright_timeout: int = 30  # seconds for Playwright page load

    # Capture server
    capture_server_port: int = 8787
    capture_server_enabled: bool = False

    # Anti-detection rate limiting
    download_session_budget: int = 15
    download_hourly_limit: int = 20
    download_per_domain_hourly_limit: int = 5
    download_min_delay: float = 3.0   # seconds, minimum inter-download delay
    download_max_delay: float = 8.0   # seconds, maximum inter-download delay
    download_backfill_cap: int = 10   # max items per backfill run
    download_budget_seconds: float = 90.0  # total time budget for all URL attempts per article

    # Multi-profile support (CONFIG-004)
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


def load_config() -> ServerConfig:
    """Load configuration from Docker secrets and environment variables.

    Secret loading priority: Docker secret file -> env var -> empty string.
    """
    script_dir = Path(__file__).parent.parent  # src/
    default_cache = str(script_dir / "token_cache.json")

    def _bool(val: str) -> bool:
        return val.lower() in ("1", "true", "yes", "on")

    # Parse feature flags from env
    default_features = {
        "cache_enabled": True,
        "retry_enabled": True,
        "circuit_breaker_enabled": True,
        "token_encryption_enabled": False,
        "screenshot_on_failure": False,
        "metrics_enabled": True,
        "fusion_enabled": True,
        "rerank_enabled": True,
        "pdf_download_enabled": True,
        "zotero_enabled": True,
        "host_download_enabled": True,
    }
    for key in default_features:
        env_key = f"SFU_FEATURE_{key.upper()}"
        env_val = os.environ.get(env_key)
        if env_val is not None:
            default_features[key] = _bool(env_val)

    return ServerConfig(
        sfu_username=_read_secret("sfu_username", "SFU_USERNAME"),
        sfu_password=_read_secret("sfu_password", "SFU_PASSWORD"),
        mfa_secret=_read_secret("sfu_mfa_secret", "SFU_MFA_SECRET"),
        mfa_device_name=_read_secret("sfu_mfa_device_name", "SFU_MFA_DEVICE_NAME"),
        token_cache_file=os.environ.get("SFU_TOKEN_CACHE_FILE", default_cache),
        token_refresh_buffer=int(os.environ.get("SFU_TOKEN_REFRESH_BUFFER", "300")),
        auth_timeout=int(os.environ.get("SFU_AUTH_TIMEOUT", "10")),
        search_timeout=int(os.environ.get("SFU_SEARCH_TIMEOUT", "30")),
        auth_sleep_after_login=int(os.environ.get("SFU_AUTH_SLEEP", "5")),
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
        download_dir=os.environ.get("SFU_DOWNLOAD_DIR", "/tmp/sfu-library-downloads"),
        host_download_dir=os.environ.get("SFU_HOST_DOWNLOAD_DIR", "/mnt/host-downloads"),
        download_timeout=int(os.environ.get("SFU_DOWNLOAD_TIMEOUT", "60")),
        max_pdf_text_chars=int(os.environ.get("SFU_MAX_PDF_TEXT_CHARS", "100000")),
        ezproxy_prefix=os.environ.get("SFU_EZPROXY_PREFIX", "https://proxy.lib.sfu.ca/login?url="),
        ezproxy_login_url=os.environ.get("SFU_EZPROXY_LOGIN_URL", "https://login.proxy.lib.sfu.ca/login?qurl="),
        ezproxy_proxy_base=os.environ.get("SFU_EZPROXY_PROXY_BASE", "proxy.lib.sfu.ca"),
        download_tiers=[t.strip() for t in os.environ.get("SFU_DOWNLOAD_TIERS", "curl_cffi,playwright,requests").split(",")],
        playwright_timeout=int(os.environ.get("SFU_PLAYWRIGHT_TIMEOUT", "30")),
        capture_server_port=int(os.environ.get("SFU_CAPTURE_SERVER_PORT", "8787")),
        capture_server_enabled=_bool(os.environ.get("SFU_CAPTURE_SERVER_ENABLED", "false")),
        download_session_budget=int(os.environ.get("SFU_DOWNLOAD_SESSION_BUDGET", "15")),
        download_hourly_limit=int(os.environ.get("SFU_DOWNLOAD_HOURLY_LIMIT", "20")),
        download_per_domain_hourly_limit=int(os.environ.get("SFU_DOWNLOAD_PER_DOMAIN_HOURLY_LIMIT", "5")),
        download_min_delay=float(os.environ.get("SFU_DOWNLOAD_MIN_DELAY", "3.0")),
        download_max_delay=float(os.environ.get("SFU_DOWNLOAD_MAX_DELAY", "8.0")),
        download_backfill_cap=int(os.environ.get("SFU_DOWNLOAD_BACKFILL_CAP", "10")),
        download_budget_seconds=float(os.environ.get("SFU_DOWNLOAD_BUDGET_SECONDS", "90")),
    )


def validate_config(config: ServerConfig) -> list[str]:
    """Validate configuration and return a list of warnings.

    Returns:
        List of warning strings. Empty list means configuration is valid.
    """
    warnings: list[str] = []

    if not config.sfu_username:
        warnings.append("SFU_USERNAME is empty")
    if not config.sfu_password:
        warnings.append("SFU_PASSWORD is empty")
    if not config.mfa_secret:
        warnings.append("SFU_MFA_SECRET is empty")

    if config.auth_timeout < 1:
        warnings.append(f"auth_timeout too low: {config.auth_timeout}")
    if config.search_timeout < 1:
        warnings.append(f"search_timeout too low: {config.search_timeout}")
    if config.max_retries < 0:
        warnings.append(f"max_retries cannot be negative: {config.max_retries}")
    if config.token_refresh_buffer < 0:
        warnings.append(f"token_refresh_buffer cannot be negative: {config.token_refresh_buffer}")
    if config.cache_ttl < 0:
        warnings.append(f"cache_ttl cannot be negative: {config.cache_ttl}")
    if config.cache_max_size < 1:
        warnings.append(f"cache_max_size must be positive: {config.cache_max_size}")

    valid_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
    if config.log_level.upper() not in valid_levels:
        warnings.append(f"Invalid log_level: {config.log_level}")

    # Zotero credential validation
    if not config.zotero_api_key:
        warnings.append("SFU_ZOTERO_API_KEY is empty — Zotero tools will fail")
    if not config.zotero_user_id:
        warnings.append("SFU_ZOTERO_USER_ID is empty — Zotero tools will fail")

    return warnings
