"""EZProxy URL transformation utilities.

SFU's EZProxy uses hostname-based proxying with dot-to-hyphen rewriting:
  https://onlinelibrary.wiley.com/doi/123
  -> https://onlinelibrary-wiley-com.proxy.lib.sfu.ca/doi/123

These functions live in their own module to avoid circular imports
between downloader, publisher_router, and rate_limiter.
"""

from urllib.parse import urlparse, urlunparse


def make_proxied_url(url: str, proxy_base: str = "proxy.lib.sfu.ca") -> str:
    """Convert URL to SFU hostname-based EZProxy format.

    Dots in hostname are replaced with hyphens per SFU's EZProxy config:
    https://onlinelibrary.wiley.com/doi/123
    -> https://onlinelibrary-wiley-com.proxy.lib.sfu.ca/doi/123
    """
    parsed = urlparse(url)
    if not parsed.hostname:
        return url
    # Already proxied?
    if proxy_base in (parsed.hostname or ""):
        return url
    proxied_host = f"{parsed.hostname.replace('.', '-')}.{proxy_base}"
    proxied = parsed._replace(netloc=proxied_host)
    return urlunparse(proxied)


def unwrap_proxied_hostname(hostname: str, proxy_base: str = "proxy.lib.sfu.ca") -> str:
    """Reverse the EZProxy hostname rewrite.

    journals-sagepub-com.proxy.lib.sfu.ca -> journals.sagepub.com
    """
    suffix = f".{proxy_base}"
    if not hostname.endswith(suffix):
        return hostname
    original_part = hostname[: -len(suffix)]  # e.g. "journals-sagepub-com"
    return original_part.replace("-", ".")     # e.g. "journals.sagepub.com"
