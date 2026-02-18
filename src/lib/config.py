"""Server configuration from environment variables with hardcoded fallbacks.

Implements CONFIG-001 through CONFIG-004, ERR-005, AUTH-008.
"""

import os
from dataclasses import dataclass, field
from typing import Any


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
    mfa_device_name: str = ""  # Duo device name to select (e.g. "mfa-device")

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
    ezproxy_prefix: str = "https://proxy.lib.sfu.ca/login?url="

    # Multi-profile support (CONFIG-004)
    active_profile: str = "default"


def load_config() -> ServerConfig:
    """Load configuration from environment variables with hardcoded fallbacks.

    The hardcoded fallbacks match the original monolith values so the server
    continues to work without any env vars set.
    """
    from pathlib import Path

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
        sfu_username=os.environ.get("SFU_USERNAME", "REDACTED_SFU_USERNAME"),
        sfu_password=os.environ.get("SFU_PASSWORD", "REDACTED_SFU_PASSWORD"),
        mfa_secret=os.environ.get("SFU_MFA_SECRET", "REDACTED_MFA_SECRET"),
        mfa_device_name=os.environ.get("SFU_MFA_DEVICE_NAME", "mfa-device"),
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
        zotero_api_key=os.environ.get("SFU_ZOTERO_API_KEY", "REDACTED_ZOTERO_API_KEY"),
        zotero_user_id=os.environ.get("SFU_ZOTERO_USER_ID", "16321308"),
        download_dir=os.environ.get("SFU_DOWNLOAD_DIR", "/tmp/sfu-library-downloads"),
        host_download_dir=os.environ.get("SFU_HOST_DOWNLOAD_DIR", "/mnt/host-downloads"),
        download_timeout=int(os.environ.get("SFU_DOWNLOAD_TIMEOUT", "60")),
        max_pdf_text_chars=int(os.environ.get("SFU_MAX_PDF_TEXT_CHARS", "100000")),
        ezproxy_prefix=os.environ.get("SFU_EZPROXY_PREFIX", "https://proxy.lib.sfu.ca/login?url="),
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

    return warnings
