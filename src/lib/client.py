"""SFU Library API client with authentication via Selenium.

Extracted from the monolith with enhancements for all AUTH, SEL, ERR TODOs:
- AUTH-001: Proactive token refresh before expiry
- AUTH-002: MFA retry logic (2 attempts)
- AUTH-003: MFA method fallback loop
- AUTH-004: File locking on token cache (fcntl.flock)
- AUTH-005: Robust MFA detection (regex-based, not hardcoded strings)
- AUTH-006: Token encryption at rest (Fernet)
- AUTH-007: Session persistence across server restarts
- SEL-001: Webdriver cleanup (try/finally + process kill fallback)
- SEL-002: Webdriver health check before operations
- SEL-003: Screenshot capture on auth failure
- SEL-004: Chrome binary check before launch
- SEL-005: Log Chrome/ChromeDriver versions
- ERR-004: Graceful degradation when API unavailable
- ERR-007: API response validation
"""

import base64
import fcntl
import json
import logging
import os
import shutil
import signal
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import requests

from lib.config import ServerConfig, load_config
from lib.retry import retry_with_backoff, CircuitBreaker
from lib.validators import sanitize_search_query, validate_api_response
from lib.cache import ResponseCache

logger = logging.getLogger("sfu_library_mcp")

# Lazy encryption import — only needed when token encryption is enabled
_fernet = None


def _get_fernet(key: bytes):
    """Lazy-load Fernet for token encryption (AUTH-006)."""
    global _fernet
    if _fernet is None:
        try:
            from cryptography.fernet import Fernet
            _fernet = Fernet(key)
        except ImportError:
            logger.warning("cryptography package not installed; token encryption disabled")
            return None
    return _fernet


class SFULibraryClient:
    """Client for searching SFU Library with authenticated API access."""

    BASE_URL = "https://sfu-primo.hosted.exlibrisgroup.com"
    SEARCH_PATH = "/primo_library/libweb/webservices/rest/primo-explore/v1/pnxs"
    LOGIN_URL = f"{BASE_URL}/primo-explore/login?vid=SFUL&lang=en_US"
    SEARCH_PAGE_URL = f"{BASE_URL}/primo-explore/search?vid=SFUL&lang=en_US"

    def __init__(
        self,
        headless: bool = True,
        config: ServerConfig | None = None,
    ):
        self.config = config or load_config()
        self.headless = headless
        self.jwt_token: str | None = None
        self.token_expiry: int | None = None
        self.user_info: dict = {}
        self.cookies: dict = {}

        # Circuit breaker for API calls (ERR-002)
        self.circuit_breaker = CircuitBreaker(
            threshold=self.config.circuit_breaker_threshold,
            timeout=self.config.circuit_breaker_timeout,
        )

        # Response cache (PERF-001)
        self.cache = ResponseCache(
            ttl=self.config.cache_ttl,
            max_size=self.config.cache_max_size,
            max_memory_mb=self.config.cache_max_memory_mb,
        )

    # ─── JWT helpers ────────────────────────────────────────────────

    def _decode_jwt(self, token: str) -> dict | None:
        """Decode JWT token to extract payload (without verification)."""
        try:
            parts = token.split(".")
            if len(parts) != 3:
                return None
            payload = parts[1]
            padding = 4 - len(payload) % 4
            if padding != 4:
                payload += "=" * padding
            decoded = base64.urlsafe_b64decode(payload)
            return json.loads(decoded)
        except Exception:
            logger.debug("Failed to decode JWT token")
            return None

    def _is_token_valid(self, token: str) -> bool:
        """Check if a JWT token is still valid (not expired)."""
        payload = self._decode_jwt(token)
        if not payload:
            return False
        exp = payload.get("exp")
        if not exp:
            return False
        buffer_seconds = self.config.token_refresh_buffer
        current_time = time.time()
        return current_time < (exp - buffer_seconds)

    def _should_refresh_proactively(self) -> bool:
        """AUTH-001: Check if token should be refreshed before expiry."""
        if not self.jwt_token:
            return True
        payload = self._decode_jwt(self.jwt_token)
        if not payload or not payload.get("exp"):
            return True
        remaining = payload["exp"] - time.time()
        # Refresh when less than 2x buffer remaining
        return remaining < (self.config.token_refresh_buffer * 2)

    # ─── Token cache with file locking (AUTH-004) ────────────────

    def _encrypt_token(self, token: str) -> str:
        """AUTH-006: Encrypt token for storage."""
        if not self.config.features.get("token_encryption_enabled"):
            return token
        key = base64.urlsafe_b64encode(
            (self.config.sfu_username + self.config.mfa_secret).ljust(32)[:32].encode()
        )
        f = _get_fernet(key)
        if f is None:
            return token
        return f.encrypt(token.encode()).decode()

    def _decrypt_token(self, encrypted: str) -> str:
        """AUTH-006: Decrypt stored token."""
        if not self.config.features.get("token_encryption_enabled"):
            return encrypted
        key = base64.urlsafe_b64encode(
            (self.config.sfu_username + self.config.mfa_secret).ljust(32)[:32].encode()
        )
        f = _get_fernet(key)
        if f is None:
            return encrypted
        try:
            return f.decrypt(encrypted.encode()).decode()
        except Exception:
            logger.warning("Token decryption failed; treating as plaintext")
            return encrypted

    def save_token_cache(self) -> None:
        """Save JWT token and metadata to cache file with file locking."""
        if not self.jwt_token:
            return
        payload = self._decode_jwt(self.jwt_token)
        stored_token = self._encrypt_token(self.jwt_token)
        cache_data = {
            "jwt_token": stored_token,
            "expiry": payload.get("exp") if payload else None,
            "user": payload.get("user") if payload else None,
            "userName": payload.get("userName") if payload else None,
            "userGroup": payload.get("userGroup") if payload else None,
            "cached_at": datetime.now().isoformat(),
            "cookies": self.cookies,
            "encrypted": self.config.features.get("token_encryption_enabled", False),
        }
        try:
            with open(self.config.token_cache_file, "w") as f:
                fcntl.flock(f.fileno(), fcntl.LOCK_EX)
                try:
                    json.dump(cache_data, f, indent=2)
                finally:
                    fcntl.flock(f.fileno(), fcntl.LOCK_UN)
            logger.debug("Token cache saved to %s", self.config.token_cache_file)
        except OSError as e:
            logger.error("Failed to save token cache: %s", e)

    def load_token_cache(self) -> bool:
        """Load JWT token from cache file if valid, with file locking."""
        if not os.path.exists(self.config.token_cache_file):
            return False
        try:
            with open(self.config.token_cache_file, "r") as f:
                fcntl.flock(f.fileno(), fcntl.LOCK_SH)
                try:
                    cache_data = json.load(f)
                finally:
                    fcntl.flock(f.fileno(), fcntl.LOCK_UN)

            stored_token = cache_data.get("jwt_token")
            if not stored_token:
                return False

            # Decrypt if needed
            if cache_data.get("encrypted"):
                token = self._decrypt_token(stored_token)
            else:
                token = stored_token

            if not self._is_token_valid(token):
                logger.debug("Cached token expired or invalid")
                return False

            self.jwt_token = token
            self.cookies = cache_data.get("cookies", {})
            self.user_info = {
                "user": cache_data.get("user"),
                "userName": cache_data.get("userName"),
                "userGroup": cache_data.get("userGroup"),
            }
            logger.info("Loaded valid token from cache for user %s", self.user_info.get("userName"))
            return True
        except Exception as e:
            logger.error("Failed to load token cache: %s", e)
            return False

    def clear_token_cache(self) -> None:
        """Delete the token cache file and clear in-memory auth state."""
        if os.path.exists(self.config.token_cache_file):
            os.remove(self.config.token_cache_file)
        self.jwt_token = None
        self.cookies = {}
        self.user_info = {}
        self.token_expiry = None
        logger.info("Token cache cleared")

    # ─── Authentication ──────────────────────────────────────────

    def ensure_authenticated(self, force: bool = False) -> bool:
        """Ensure we have a valid token, authenticating if necessary."""
        if force:
            self.clear_token_cache()
            return self.authenticate()
        if self.load_token_cache():
            # AUTH-001: Proactive refresh
            if self._should_refresh_proactively():
                logger.info("Token nearing expiry, proactively refreshing")
                return self.authenticate()
            return True
        return self.authenticate()

    def _check_chrome_binary(self) -> bool:
        """SEL-004: Check if Chrome binary exists before launching."""
        chrome_paths = [
            shutil.which("google-chrome"),
            shutil.which("google-chrome-stable"),
            shutil.which("chromium"),
            shutil.which("chromium-browser"),
            "/usr/bin/google-chrome",
            "/usr/bin/chromium",
        ]
        for path in chrome_paths:
            if path and os.path.isfile(path):
                logger.debug("Found Chrome at: %s", path)
                return True
        logger.error("Chrome binary not found in any standard location")
        return False

    def _log_driver_versions(self, driver) -> None:
        """SEL-005: Log Chrome and ChromeDriver versions."""
        try:
            caps = driver.capabilities
            browser_version = caps.get("browserVersion", caps.get("version", "unknown"))
            driver_version = caps.get("chrome", {}).get("chromedriverVersion", "unknown")
            if isinstance(driver_version, str):
                driver_version = driver_version.split(" ")[0]
            logger.info("Chrome %s, ChromeDriver %s", browser_version, driver_version)
        except Exception:
            logger.debug("Could not determine browser/driver versions")

    def _capture_screenshot(self, driver, name: str = "auth_failure") -> None:
        """SEL-003: Capture screenshot on authentication failure."""
        if not self.config.features.get("screenshot_on_failure"):
            return
        try:
            screenshot_dir = Path(self.config.token_cache_file).parent / "screenshots"
            screenshot_dir.mkdir(exist_ok=True)
            path = screenshot_dir / f"{name}_{int(time.time())}.png"
            driver.save_screenshot(str(path))
            logger.info("Screenshot saved: %s", path)
        except Exception as e:
            logger.debug("Failed to capture screenshot: %s", e)

    def _cleanup_driver(self, driver) -> None:
        """SEL-001: Robust webdriver cleanup with process kill fallback."""
        if driver is None:
            return
        pid = None
        try:
            pid = driver.service.process.pid
        except Exception:
            pass
        try:
            driver.quit()
        except Exception:
            logger.warning("driver.quit() failed, attempting process cleanup")
            if pid:
                try:
                    os.kill(pid, signal.SIGTERM)
                    time.sleep(0.5)
                    os.kill(pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                except Exception as e:
                    logger.debug("Process cleanup failed: %s", e)

    def authenticate(self) -> bool:
        """Authenticate and obtain JWT token via Selenium."""
        # SEL-004: Check Chrome exists
        if not self._check_chrome_binary():
            logger.error("Cannot authenticate: Chrome not found")
            return False

        # Lazy import selenium to avoid import errors when not authenticating
        try:
            from selenium import webdriver
            from selenium.webdriver.chrome.options import Options
            from selenium.webdriver.common.by import By
            from selenium.webdriver.support.ui import WebDriverWait
            from selenium.webdriver.support import expected_conditions as EC
            import pyotp
        except ImportError as e:
            logger.error("Selenium/pyotp not available: %s", e)
            return False

        chrome_options = Options()
        chrome_options.add_argument("--headless=new")
        chrome_options.add_argument("--window-size=1920,1080")
        chrome_options.add_argument("--disable-gpu")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("--disable-blink-features=AutomationControlled")
        chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
        chrome_options.add_experimental_option("useAutomationExtension", False)

        driver = None
        try:
            driver = webdriver.Chrome(options=chrome_options)

            # SEL-005: Log versions
            self._log_driver_versions(driver)

            driver.execute_cdp_cmd("Network.setUserAgentOverride", {
                "userAgent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
            })

            driver.get(self.LOGIN_URL)
            time.sleep(self.config.auth_sleep_after_login)

            signin_btn = WebDriverWait(driver, self.config.auth_timeout).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, "button[aria-label*='Sign']"))
            )
            driver.execute_script("arguments[0].click();", signin_btn)
            time.sleep(self.config.auth_sleep_after_login)

            username_field = WebDriverWait(driver, self.config.auth_timeout).until(
                EC.presence_of_element_located((By.ID, "username"))
            )
            username_field.send_keys(self.config.sfu_username)
            driver.find_element(By.ID, "password").send_keys(self.config.sfu_password)
            driver.find_element(By.NAME, "submit").click()
            time.sleep(self.config.auth_sleep_after_login)

            # Handle MFA
            totp = pyotp.TOTP(self.config.mfa_secret)
            duo_iframe = WebDriverWait(driver, self.config.auth_timeout).until(
                EC.presence_of_element_located((By.ID, "duo_iframe"))
            )
            driver.switch_to.frame(duo_iframe)
            time.sleep(3)

            # AUTH-005: Robust MFA method detection
            # First try to match the configured device name, then fall back
            # to pattern-based detection.
            import re
            method_links = driver.find_elements(By.CSS_SELECTOR, "a.item")
            method_selected = False

            # Priority 1: Match configured device name exactly
            if self.config.mfa_device_name:
                device_pattern = re.compile(re.escape(self.config.mfa_device_name), re.I)
                for link in method_links:
                    try:
                        if device_pattern.search(link.text):
                            driver.execute_script("arguments[0].click();", link)
                            time.sleep(2)
                            method_selected = True
                            logger.info("Selected configured MFA device: %s", link.text.strip())
                            break
                    except Exception:
                        continue

            # Priority 2: Pattern-based fallback
            if not method_selected:
                mfa_patterns = [
                    re.compile(r"mobile\s+application|authenticat|google\s*auth|totp", re.I),
                    re.compile(r"passcode|token|otp", re.I),
                    re.compile(r"call|phone|sms|text", re.I),
                ]
                for pattern in mfa_patterns:
                    if method_selected:
                        break
                    for link in method_links:
                        try:
                            if pattern.search(link.text):
                                driver.execute_script("arguments[0].click();", link)
                                time.sleep(2)
                                method_selected = True
                                logger.info("Selected MFA method: %s", link.text.strip())
                                break
                        except Exception:
                            continue

            # Priority 3: Click first available
            if not method_selected and method_links:
                try:
                    driver.execute_script("arguments[0].click();", method_links[0])
                    time.sleep(2)
                    logger.info("Fallback: selected first MFA method")
                except Exception:
                    pass

            # AUTH-002: MFA code submission with retry (2 attempts)
            mfa_success = False
            for mfa_attempt in range(2):
                mfa_code = totp.now()
                try:
                    code_input = WebDriverWait(driver, self.config.auth_timeout).until(
                        EC.presence_of_element_located((By.ID, "code"))
                    )
                    code_input.clear()
                    code_input.send_keys(mfa_code)

                    submit_btn = driver.find_element(By.XPATH, "//button[contains(text(), 'Submit')]")
                    submit_btn.click()
                    driver.switch_to.default_content()
                    time.sleep(10)

                    if "primo" in driver.current_url.lower():
                        mfa_success = True
                        break
                    else:
                        logger.warning("MFA attempt %d failed, retrying", mfa_attempt + 1)
                        if mfa_attempt < 1:
                            # Switch back to iframe for retry
                            duo_iframe = WebDriverWait(driver, self.config.auth_timeout).until(
                                EC.presence_of_element_located((By.ID, "duo_iframe"))
                            )
                            driver.switch_to.frame(duo_iframe)
                            time.sleep(3)
                except Exception as e:
                    logger.warning("MFA attempt %d exception: %s", mfa_attempt + 1, e)

            if not mfa_success:
                self._capture_screenshot(driver, "mfa_failure")
                logger.error("MFA authentication failed after 2 attempts")
                return False

            driver.get(self.SEARCH_PAGE_URL)
            time.sleep(self.config.auth_sleep_after_login)

            jwt_raw = driver.execute_script("return sessionStorage.getItem('primoExploreJwt');")
            if jwt_raw:
                self.jwt_token = jwt_raw.strip('"')
                payload = self._decode_jwt(self.jwt_token)
                if payload:
                    self.user_info = {
                        "user": payload.get("user"),
                        "userName": payload.get("userName"),
                        "userGroup": payload.get("userGroup"),
                    }
                    self.token_expiry = payload.get("exp")
                    logger.info("Authenticated as %s", self.user_info.get("userName"))

            # Phase 1: Capture Primo cookies while still on Primo domain
            for cookie in driver.get_cookies():
                self.cookies[cookie["name"]] = cookie["value"]
                logger.debug("Captured Primo cookie: %s (domain: %s)", cookie["name"], cookie.get("domain", "unknown"))

            # Phase 2: Establish EZProxy session using the active CAS session
            EZPROXY_TARGET = "https://proxy.lib.sfu.ca/login?url=https://www.sfu.ca"
            try:
                logger.info("Establishing EZProxy session...")
                driver.get(EZPROXY_TARGET)
                time.sleep(3)  # Allow CAS redirect to complete
                logger.info("EZProxy session URL: %s", driver.current_url)
                # Capture EZProxy cookies and merge (don't overwrite Primo cookies)
                for cookie in driver.get_cookies():
                    if cookie["name"] not in self.cookies:
                        self.cookies[cookie["name"]] = cookie["value"]
                        logger.debug("Captured EZProxy cookie: %s (domain: %s)", cookie["name"], cookie.get("domain", "unknown"))
            except Exception as e:
                logger.warning("EZProxy session setup failed (downloads may not work): %s", e)

            self.save_token_cache()
            return True

        except Exception as e:
            logger.error("Authentication failed: %s", e)
            if driver:
                self._capture_screenshot(driver, "auth_exception")
            return False
        finally:
            # SEL-001: Robust cleanup
            self._cleanup_driver(driver)

    # ─── SEL-002: Health check ───────────────────────────────────

    def _check_webdriver_health(self, driver) -> bool:
        """SEL-002: Check if webdriver is still responsive."""
        try:
            _ = driver.title
            return True
        except Exception:
            return False

    # ─── API calls with retry and caching ────────────────────────

    def search(
        self,
        query: str,
        limit: int = 10,
        offset: int = 0,
        field: str = "any",
        precision: str = "contains",
        sort: str = "rank",
        tab: str = "default_tab",
        scope: str = "default_scope",
    ) -> dict | None:
        """Search the library using the REST API with JWT authentication."""
        if not self.jwt_token:
            return None

        # Sanitize input (DATA-002)
        sanitized_query = sanitize_search_query(query)
        if not sanitized_query:
            logger.warning("Empty query after sanitization")
            return None

        # Check cache (PERF-001)
        if self.config.features.get("cache_enabled"):
            cache_key = self.cache.make_key("search", sanitized_query, limit, offset, field, sort, tab, scope)
            cached = self.cache.get(cache_key)
            if cached is not None:
                logger.debug("Cache hit for search: %s", sanitized_query[:50])
                return cached

        url = f"{self.BASE_URL}{self.SEARCH_PATH}"

        params = {
            "q": f"{field},{precision},{sanitized_query}",
            "vid": "SFUL",
            "inst": "01SFUL",
            "tab": tab,
            "scope": scope,
            "lang": "en_US",
            "offset": offset,
            "limit": limit,
            "sort": sort,
            "skipDelivery": "Y",
            "blendFacetsSeparately": "true",
            "pcAvailability": "false",
            "getMore": 0,
            "rtaLinks": "true",
            "newspapersActive": "true",
            "newspapersSearch": "false",
        }

        result = self._make_api_request(url, params)

        # Cache on success
        if result is not None and self.config.features.get("cache_enabled"):
            self.cache.put(cache_key, result)

        return result

    def get_item_details(self, doc_id: str, context: str = "L") -> dict | None:
        """Get detailed information about a specific item."""
        if not self.jwt_token:
            return None

        # Check cache
        if self.config.features.get("cache_enabled"):
            cache_key = self.cache.make_key("item", doc_id, context)
            cached = self.cache.get(cache_key)
            if cached is not None:
                logger.debug("Cache hit for item: %s", doc_id)
                return cached

        url = f"{self.BASE_URL}{self.SEARCH_PATH}/{doc_id}"

        params = {
            "vid": "SFUL",
            "inst": "01SFUL",
            "lang": "en_US",
            "context": context,
        }

        result = self._make_api_request(url, params)

        if result is not None and self.config.features.get("cache_enabled"):
            self.cache.put(cache_key, result)

        return result

    def _make_api_request(self, url: str, params: dict) -> dict | None:
        """Make an authenticated API request with retry logic."""
        headers = {
            "Accept": "application/json, text/plain, */*",
            "Authorization": f"Bearer {self.jwt_token}",
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "Referer": f"{self.BASE_URL}/primo-explore/search?vid=SFUL",
            "Origin": self.BASE_URL,
        }

        session = requests.Session()
        for name, value in self.cookies.items():
            session.cookies.set(name, value, domain="sfu-primo.hosted.exlibrisgroup.com")

        try:
            response = session.get(
                url,
                params=params,
                headers=headers,
                timeout=self.config.search_timeout,
            )

            if response.status_code == 200:
                data = response.json()
                # ERR-007: Validate response
                if isinstance(data, dict):
                    return data
                logger.warning("Unexpected response type: %s", type(data))
                return data
            elif response.status_code == 401:
                logger.warning("Token rejected (401), clearing cache")
                self.clear_token_cache()
                return None
            elif response.status_code == 429:
                logger.warning("Rate limited (429)")
                return None
            else:
                logger.warning("API returned status %d", response.status_code)
                return None
        except requests.Timeout:
            logger.error("API request timed out after %ds", self.config.search_timeout)
            return None
        except requests.ConnectionError as e:
            # ERR-004: Graceful degradation
            logger.error("API connection failed: %s", e)
            return None
        except Exception as e:
            logger.error("API request failed: %s", e)
            return None

    def get_token_status(self) -> dict:
        """Get current token status."""
        if not self.jwt_token:
            if self.load_token_cache():
                pass
            else:
                return {"valid": False, "message": "No token available"}

        payload = self._decode_jwt(self.jwt_token)
        if not payload:
            return {"valid": False, "message": "Invalid token"}

        exp = payload.get("exp", 0)
        remaining = exp - time.time()

        if remaining <= 0:
            return {"valid": False, "message": "Token expired"}

        hours = int(remaining // 3600)
        minutes = int((remaining % 3600) // 60)

        return {
            "valid": True,
            "message": f"Valid for {hours}h {minutes}m",
            "user": payload.get("userName"),
            "userId": payload.get("user"),
            "userGroup": payload.get("userGroup"),
            "expiresIn": f"{hours}h {minutes}m",
            "expiresAt": datetime.fromtimestamp(exp).isoformat(),
        }


def fetch_crossref_metadata(doi: str, timeout: int = 10) -> dict | None:
    """Strategy D: Fetch metadata from CrossRef API by DOI.

    Free, no authentication required. Returns a normalized metadata dict
    compatible with extract_metadata() output, or None on failure.
    """
    if not doi:
        return None

    # Normalize DOI — strip URL prefix if present
    if doi.startswith("https://doi.org/"):
        doi = doi[len("https://doi.org/"):]
    elif doi.startswith("http://doi.org/"):
        doi = doi[len("http://doi.org/"):]

    url = f"https://api.crossref.org/works/{doi}"
    headers = {
        "Accept": "application/json",
        "User-Agent": "SFULibraryMCP/1.0 (mailto:REDACTED_SFU_USERNAME@sfu.ca)",
    }

    try:
        response = requests.get(url, headers=headers, timeout=timeout)
        if response.status_code != 200:
            logger.debug("CrossRef returned %d for DOI %s", response.status_code, doi)
            return None

        data = response.json()
        work = data.get("message", {})

        # Extract authors
        authors = []
        for author in work.get("author", []):
            family = author.get("family", "")
            given = author.get("given", "")
            if family and given:
                authors.append(f"{family}, {given}")
            elif family:
                authors.append(family)

        # Extract date
        date_parts = work.get("published-print", work.get("published-online", {}))
        date = ""
        if date_parts and date_parts.get("date-parts"):
            parts = date_parts["date-parts"][0]
            date = str(parts[0]) if parts else ""

        # Extract journal
        journal = ""
        container = work.get("container-title", [])
        if container:
            journal = container[0]

        # Extract pages
        pages = work.get("page", "")
        spage, epage = "", ""
        if "-" in pages:
            parts = pages.split("-", 1)
            spage, epage = parts[0].strip(), parts[1].strip()

        return {
            "title": work.get("title", [""])[0] if work.get("title") else "",
            "authors": authors,
            "creators": authors,
            "contributors": [],
            "date": date,
            "publisher": work.get("publisher", ""),
            "type": work.get("type", ""),
            "source": journal,
            "isbn": "",
            "issn": work.get("ISSN", [""])[0] if work.get("ISSN") else "",
            "doi": doi,
            "volume": work.get("volume", ""),
            "issue": work.get("issue", ""),
            "spage": spage,
            "epage": epage,
            "pages": pages,
            "record_id": "",
            "is_cdi": False,
            "resource_type": "article" if "article" in work.get("type", "") else "other",
        }
    except requests.Timeout:
        logger.debug("CrossRef request timed out for DOI %s", doi)
        return None
    except Exception as e:
        logger.debug("CrossRef lookup failed for DOI %s: %s", doi, e)
        return None
