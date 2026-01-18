"""
SFU Library MCP Server
Provides Claude Desktop access to the SFU Library database through the Primo API.
Supports JWT token caching, authenticated searches, and detailed item retrieval.

================================================================================
TODO: RELIABILITY & REDUNDANCY IMPROVEMENT TRACKING
================================================================================
SECTION: AUTHENTICATION & TOKEN MANAGEMENT
-------------------------------------------
[ ] TODO-AUTH-001: Implement token refresh BEFORE expiry (currently only checks after expiry)
[ ] TODO-AUTH-002: Add retry logic for MFA code submission (currently fails silently on first attempt)
[ ] TODO-AUTH-003: Implement multiple authentication method fallbacks (MFA alternatives)
[ ] TODO-AUTH-004: Add concurrent request handling for token cache (file locking)
[ ] TODO-AUTH-005: Validate MFA method detection - hardcoded 'fuck'/'higjacking' is fragile
[ ] TODO-AUTH-006: Add token encryption at rest for security
[ ] TODO-AUTH-007: Implement session persistence across server restarts
[ ] TODO-AUTH-008: Add authentication timeout configuration (currently hardcoded 10s waits)

SECTION: ERROR HANDLING & RESILIENCE
------------------------------------
[ ] TODO-ERR-001: Add exponential backoff for API requests (currently no retry logic)
[ ] TODO-ERR-002: Implement circuit breaker pattern for repeated API failures
[ ] TODO-ERR-003: Add structured logging (currently using print statements)
[ ] TODO-ERR-004: Implement graceful degradation when API is unavailable
[ ] TODO-ERR-005: Add request timeout configuration (currently hardcoded)
[ ] TODO-ERR-006: Handle network interruptions during long-running operations
[ ] TODO-ERR-007: Add validation for API response schemas
[ ] TODO-ERR-008: Implement rate limiting detection and backoff

SECTION: SELENIUM WEBDRIVER RELIABILITY
---------------------------------------
[ ] TODO-SEL-001: Add webdriver process cleanup on crash (zombie processes)
[ ] TODO-SEL-002: Implement webdriver health checks before operations
[ ] TODO-SEL-003: Add screenshot capture on authentication failure for debugging
[ ] TODO-SEL-004: Handle Chrome binary not found scenarios
[ ] TODO-SEL-005: Add webdriver version compatibility checks
[ ] TODO-SEL-006: Implement alternative to Selenium (playwright/puppeteer) as fallback

SECTION: DATA INTEGRITY & VALIDATION
------------------------------------
[ ] TODO-DATA-001: Validate ISBN/ISSN format before API calls
[ ] TODO-DATA-002: Add input sanitization for search queries (XSS prevention)
[ ] TODO-DATA-003: Implement response data validation and error recovery
[ ] TODO-DATA-004: Add character encoding handling for non-ASCII content
[ ] TODO-DATA-005: Validate citation format outputs against standards

SECTION: PERFORMANCE & CACHING
-------------------------------
[ ] TODO-PERF-001: Implement response caching for frequently accessed items
[ ] TODO-PERF-002: Add async/concurrent request handling for batch operations
[ ] TODO-PERF-003: Implement request queuing to prevent throttling
[ ] TODO-PERF-004: Add metrics collection for performance monitoring
[ ] TODO-PERF-005: Optimize memory usage for large result sets

SECTION: TESTING & MONITORING
------------------------------
[ ] TODO-TEST-001: Add unit tests for authentication flow
[ ] TODO-TEST-002: Add integration tests for API calls
[ ] TODO-TEST-003: Implement health check endpoint
[ ] TODO-TEST-004: Add load testing for batch operations
[ ] TODO-TEST-005: Implement error scenario testing

SECTION: CONFIGURATION & DEPLOYMENT
-----------------------------------
[ ] TODO-CONFIG-001: Externalize hardcoded credentials to environment variables/secrets
[ ] TODO-CONFIG-002: Add configuration validation at startup
[ ] TODO-CONFIG-003: Implement feature flags for new functionality
[ ] TODO-CONFIG-004: Add support for multiple user profiles

================================================================================
END OF TODO LIST - Last updated: 2026-01-11
================================================================================
"""

import json
import time
import base64
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any
import asyncio

import requests
import pyotp
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent

# Credentials
USERNAME = "REDACTED_SFU_USERNAME"
PASSWORD = "REDACTED_SFU_PASSWORD"
MFA_SECRET = "REDACTED_MFA_SECRET"

# Token cache file - store in the MCP server directory
SCRIPT_DIR = Path(__file__).parent
TOKEN_CACHE_FILE = SCRIPT_DIR / "token_cache.json"


class SFULibraryClient:
    """Client for searching SFU Library with authenticated API access."""

    def __init__(self, headless=True, token_cache_file=TOKEN_CACHE_FILE):
        self.headless = headless
        self.jwt_token = None
        self.token_expiry = None
        self.user_info = {}
        self.cookies = {}
        self.token_cache_file = token_cache_file

    def _decode_jwt(self, token):
        """Decode JWT token to extract payload (without verification)."""
        try:
            parts = token.split('.')
            if len(parts) != 3:
                return None
            payload = parts[1]
            padding = 4 - len(payload) % 4
            if padding != 4:
                payload += '=' * padding
            decoded = base64.urlsafe_b64decode(payload)
            return json.loads(decoded)
        except Exception as e:
            return None

    # TODO-AUTH-001: This only validates tokens - consider proactive refresh
    # TODO-ERR-005: Hardcoded 300 second buffer should be configurable
    def _is_token_valid(self, token):
        """Check if a JWT token is still valid (not expired)."""
        payload = self._decode_jwt(token)
        if not payload:
            return False
        exp = payload.get('exp')
        if not exp:
            return False
        # Add 5 minute buffer before expiry
        buffer_seconds = 300
        current_time = time.time()
        if current_time >= (exp - buffer_seconds):
            return False
        return True

    # TODO-AUTH-004: No file locking - concurrent processes may corrupt cache
    # TODO-AUTH-006: Token stored in plaintext - should be encrypted at rest
    # TODO-ERR-003: Should use proper logging instead of silent failure
    def save_token_cache(self):
        """Save JWT token and metadata to cache file."""
        if not self.jwt_token:
            return
        payload = self._decode_jwt(self.jwt_token)
        cache_data = {
            'jwt_token': self.jwt_token,
            'expiry': payload.get('exp') if payload else None,
            'user': payload.get('user') if payload else None,
            'userName': payload.get('userName') if payload else None,
            'userGroup': payload.get('userGroup') if payload else None,
            'cached_at': datetime.now().isoformat(),
            'cookies': self.cookies
        }
        with open(self.token_cache_file, 'w') as f:
            json.dump(cache_data, f, indent=2)

    # TODO-AUTH-004: No file locking - race condition with save_token_cache
    # TODO-ERR-003: Generic exception catch hides errors - should log specifics
    def load_token_cache(self):
        """Load JWT token from cache file if valid."""
        if not os.path.exists(self.token_cache_file):
            return False
        try:
            with open(self.token_cache_file, 'r') as f:
                cache_data = json.load(f)
            token = cache_data.get('jwt_token')
            if not token:
                return False
            if not self._is_token_valid(token):
                return False
            self.jwt_token = token
            self.cookies = cache_data.get('cookies', {})
            self.user_info = {
                'user': cache_data.get('user'),
                'userName': cache_data.get('userName'),
                'userGroup': cache_data.get('userGroup')
            }
            return True
        except Exception as e:
            return False

    def clear_token_cache(self):
        """Delete the token cache file."""
        if os.path.exists(self.token_cache_file):
            os.remove(self.token_cache_file)

    def ensure_authenticated(self, force=False):
        """Ensure we have a valid token, authenticating if necessary."""
        if force:
            self.clear_token_cache()
            return self.authenticate()
        if self.load_token_cache():
            return True
        return self.authenticate()

    # TODO-SEL-001: Webdriver process may not be cleaned up on exceptions
    # TODO-SEL-003: Add screenshot capture on failure for debugging
    # TODO-SEL-004: No check if Chrome binary exists before launching
    # TODO-AUTH-002: No retry logic for MFA submission failures
    # TODO-AUTH-005: MFA method detection is fragile (hardcoded strings)
    # TODO-AUTH-008: All timeouts hardcoded (WebDriverWait 10s, time.sleep arbitrary)
    # TODO-CONFIG-001: Credentials hardcoded - should use environment variables
    def authenticate(self):
        """Authenticate and obtain JWT token."""
        chrome_options = Options()
        chrome_options.add_argument('--headless=new')
        chrome_options.add_argument('--window-size=1920,1080')
        chrome_options.add_argument('--disable-gpu')
        chrome_options.add_argument('--no-sandbox')
        chrome_options.add_argument('--disable-dev-shm-usage')
        chrome_options.add_argument('--disable-blink-features=AutomationControlled')
        chrome_options.add_experimental_option('excludeSwitches', ['enable-automation'])
        chrome_options.add_experimental_option('useAutomationExtension', False)

        driver = webdriver.Chrome(options=chrome_options)
        driver.execute_cdp_cmd('Network.setUserAgentOverride', {
            "userAgent": 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        })

        try:
            driver.get('https://sfu-primo.hosted.exlibrisgroup.com/primo-explore/login?vid=SFUL&lang=en_US')
            time.sleep(5)

            signin_btn = WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, "button[aria-label*='Sign']"))
            )
            driver.execute_script("arguments[0].click();", signin_btn)
            time.sleep(5)

            username_field = WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.ID, "username"))
            )
            username_field.send_keys(USERNAME)
            driver.find_element(By.ID, "password").send_keys(PASSWORD)
            driver.find_element(By.NAME, "submit").click()
            time.sleep(5)

            # Handle MFA
            totp = pyotp.TOTP(MFA_SECRET)
            duo_iframe = WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.ID, "duo_iframe"))
            )
            driver.switch_to.frame(duo_iframe)
            time.sleep(3)

            # Select MFA method
            method_links = driver.find_elements(By.CSS_SELECTOR, "a.item")
            for link in method_links:
                if 'fuck' in link.text.lower() or 'higjacking' in link.text.lower():
                    driver.execute_script("arguments[0].click();", link)
                    time.sleep(2)
                    break

            mfa_code = totp.now()
            code_input = WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.ID, "code"))
            )
            code_input.clear()
            code_input.send_keys(mfa_code)

            submit_btn = driver.find_element(By.XPATH, "//button[contains(text(), 'Submit')]")
            submit_btn.click()
            driver.switch_to.default_content()

            time.sleep(10)

            if 'primo' not in driver.current_url.lower():
                return False

            driver.get('https://sfu-primo.hosted.exlibrisgroup.com/primo-explore/search?vid=SFUL&lang=en_US')
            time.sleep(5)

            jwt_raw = driver.execute_script("return sessionStorage.getItem('primoExploreJwt');")
            if jwt_raw:
                self.jwt_token = jwt_raw.strip('"')
                payload = self._decode_jwt(self.jwt_token)
                if payload:
                    self.user_info = {
                        'user': payload.get('user'),
                        'userName': payload.get('userName'),
                        'userGroup': payload.get('userGroup')
                    }
                    self.token_expiry = payload.get('exp')

            for cookie in driver.get_cookies():
                self.cookies[cookie['name']] = cookie['value']

            self.save_token_cache()
            return True

        except Exception as e:
            return False
        finally:
            driver.quit()

    # TODO-ERR-001: No retry logic on network failures
    # TODO-ERR-005: No timeout configuration (uses requests default)
    # TODO-ERR-008: No rate limiting detection
    # TODO-DATA-002: No input sanitization for search query
    # TODO-PERF-001: No caching of frequent searches
    def search(self, query, limit=10, offset=0, field='any', precision='contains',
               sort='rank', tab='default_tab', scope='default_scope'):
        """Search the library using the REST API with JWT authentication."""
        if not self.jwt_token:
            return None

        url = 'https://sfu-primo.hosted.exlibrisgroup.com/primo_library/libweb/webservices/rest/primo-explore/v1/pnxs'

        params = {
            'q': f'{field},{precision},{query}',
            'vid': 'SFUL',
            'inst': '01SFUL',
            'tab': tab,
            'scope': scope,
            'lang': 'en_US',
            'offset': offset,
            'limit': limit,
            'sort': sort,
            'skipDelivery': 'Y',
            'blendFacetsSeparately': 'true',
            'pcAvailability': 'false',
            'getMore': 0,
            'rtaLinks': 'true',
            'newspapersActive': 'true',
            'newspapersSearch': 'false',
        }

        headers = {
            'Accept': 'application/json, text/plain, */*',
            'Authorization': f'Bearer {self.jwt_token}',
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Referer': 'https://sfu-primo.hosted.exlibrisgroup.com/primo-explore/search?vid=SFUL',
            'Origin': 'https://sfu-primo.hosted.exlibrisgroup.com',
        }

        session = requests.Session()
        for name, value in self.cookies.items():
            session.cookies.set(name, value, domain='sfu-primo.hosted.exlibrisgroup.com')

        response = session.get(url, params=params, headers=headers)

        if response.status_code == 200:
            return response.json()
        elif response.status_code == 401:
            self.clear_token_cache()
            return None
        else:
            return None

    # TODO-ERR-001: No retry logic on network failures
    # TODO-DATA-003: No response validation schema
    def get_item_details(self, doc_id, context='L'):
        """Get detailed information about a specific item."""
        if not self.jwt_token:
            return None

        url = f'https://sfu-primo.hosted.exlibrisgroup.com/primo_library/libweb/webservices/rest/primo-explore/v1/pnxs/{doc_id}'

        params = {
            'vid': 'SFUL',
            'inst': '01SFUL',
            'lang': 'en_US',
            'context': context,
        }

        headers = {
            'Accept': 'application/json, text/plain, */*',
            'Authorization': f'Bearer {self.jwt_token}',
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
        }

        session = requests.Session()
        for name, value in self.cookies.items():
            session.cookies.set(name, value, domain='sfu-primo.hosted.exlibrisgroup.com')

        response = session.get(url, params=params, headers=headers)

        if response.status_code == 200:
            return response.json()
        return None

    def get_token_status(self):
        """Get current token status."""
        if not self.jwt_token:
            if self.load_token_cache():
                pass
            else:
                return {"valid": False, "message": "No token available"}

        payload = self._decode_jwt(self.jwt_token)
        if not payload:
            return {"valid": False, "message": "Invalid token"}

        exp = payload.get('exp', 0)
        remaining = exp - time.time()

        if remaining <= 0:
            return {"valid": False, "message": "Token expired"}

        hours = int(remaining // 3600)
        minutes = int((remaining % 3600) // 60)

        return {
            "valid": True,
            "message": f"Valid for {hours}h {minutes}m",
            "user": payload.get('userName'),
            "userId": payload.get('user'),
            "userGroup": payload.get('userGroup'),
            "expiresIn": f"{hours}h {minutes}m",
            "expiresAt": datetime.fromtimestamp(exp).isoformat()
        }


# =============================================================================
# CITATION FORMATTING FUNCTIONS
# =============================================================================

def extract_metadata(item):
    """Extract metadata from a PNX record for citation generation."""
    if not item:
        return None

    pnx = item.get('pnx', {})
    display = pnx.get('display', {})
    addata = pnx.get('addata', {})
    control = pnx.get('control', {})

    # Extract all relevant fields
    metadata = {
        'title': display.get('title', [''])[0],
        'creators': display.get('creator', []),
        'contributors': display.get('contributor', []),
        'date': display.get('creationdate', [''])[0],
        'publisher': display.get('publisher', [''])[0],
        'type': display.get('type', [''])[0].lower(),
        'source': display.get('source', [''])[0],  # Journal name
        'isbn': addata.get('isbn', [''])[0] if addata.get('isbn') else '',
        'issn': addata.get('issn', [''])[0] if addata.get('issn') else '',
        'doi': addata.get('doi', [''])[0] if addata.get('doi') else '',
        'volume': addata.get('volume', [''])[0] if addata.get('volume') else '',
        'issue': addata.get('issue', [''])[0] if addata.get('issue') else '',
        'spage': addata.get('spage', [''])[0] if addata.get('spage') else '',
        'epage': addata.get('epage', [''])[0] if addata.get('epage') else '',
        'pages': addata.get('pages', [''])[0] if addata.get('pages') else '',
        'record_id': control.get('recordid', [''])[0] if control.get('recordid') else '',
    }

    # Get authors - combine creators and contributors
    authors = metadata['creators'] if metadata['creators'] else metadata['contributors']
    metadata['authors'] = authors

    # Determine resource type
    doc_type = metadata['type']
    if 'article' in doc_type or 'journal' in doc_type:
        metadata['resource_type'] = 'article'
    elif 'book' in doc_type:
        metadata['resource_type'] = 'book'
    else:
        metadata['resource_type'] = 'other'

    return metadata


def format_author_apa(author_string):
    """Format a single author for APA style (Last, F. M.)."""
    if not author_string:
        return ''
    # Clean up author string (remove $$Q suffixes etc)
    author = author_string.split('$$')[0].strip()

    # Try to parse "Last, First" format
    if ',' in author:
        parts = author.split(',', 1)
        last = parts[0].strip()
        first = parts[1].strip() if len(parts) > 1 else ''
        # Get initials
        initials = ' '.join([n[0] + '.' for n in first.split() if n])
        return f"{last}, {initials}" if initials else last
    else:
        # Assume "First Last" format
        parts = author.split()
        if len(parts) >= 2:
            last = parts[-1]
            initials = ' '.join([n[0] + '.' for n in parts[:-1] if n])
            return f"{last}, {initials}"
        return author


def format_author_mla(author_string):
    """Format a single author for MLA style (Last, First Middle)."""
    if not author_string:
        return ''
    author = author_string.split('$$')[0].strip()
    return author


def format_apa_citation(metadata):
    """Generate APA 7th edition citation."""
    if not metadata:
        return "Unable to generate citation: no metadata available."

    parts = []

    # Authors
    authors = metadata.get('authors', [])
    if authors:
        if len(authors) == 1:
            parts.append(format_author_apa(authors[0]))
        elif len(authors) == 2:
            parts.append(f"{format_author_apa(authors[0])} & {format_author_apa(authors[1])}")
        elif len(authors) <= 20:
            author_list = ', '.join([format_author_apa(a) for a in authors[:-1]])
            parts.append(f"{author_list}, & {format_author_apa(authors[-1])}")
        else:
            author_list = ', '.join([format_author_apa(a) for a in authors[:19]])
            parts.append(f"{author_list}, ... {format_author_apa(authors[-1])}")

    # Year
    year = metadata.get('date', 'n.d.')
    if year:
        # Extract just the year
        year = year[:4] if len(year) >= 4 else year
    parts.append(f"({year}).")

    # Title
    title = metadata.get('title', 'Untitled')
    resource_type = metadata.get('resource_type', 'other')

    if resource_type == 'article':
        # Article title (not italicized in plain text)
        parts.append(f"{title}.")
        # Journal name (would be italicized)
        source = metadata.get('source', '')
        if source:
            journal_part = source
            vol = metadata.get('volume', '')
            issue = metadata.get('issue', '')
            if vol:
                journal_part += f", {vol}"
                if issue:
                    journal_part += f"({issue})"
            pages = metadata.get('pages', '') or f"{metadata.get('spage', '')}-{metadata.get('epage', '')}".strip('-')
            if pages:
                journal_part += f", {pages}"
            parts.append(f"{journal_part}.")
    else:
        # Book title (would be italicized)
        parts.append(f"{title}.")
        publisher = metadata.get('publisher', '')
        if publisher:
            parts.append(publisher + '.')

    # DOI
    doi = metadata.get('doi', '')
    if doi:
        if not doi.startswith('http'):
            doi = f"https://doi.org/{doi}"
        parts.append(doi)

    return ' '.join(parts)


def format_mla_citation(metadata):
    """Generate MLA 9th edition citation."""
    if not metadata:
        return "Unable to generate citation: no metadata available."

    parts = []

    # Authors
    authors = metadata.get('authors', [])
    if authors:
        if len(authors) == 1:
            parts.append(format_author_mla(authors[0]) + '.')
        elif len(authors) == 2:
            parts.append(f"{format_author_mla(authors[0])}, and {format_author_mla(authors[1])}.")
        else:
            parts.append(f"{format_author_mla(authors[0])}, et al.")

    # Title
    title = metadata.get('title', 'Untitled')
    resource_type = metadata.get('resource_type', 'other')

    if resource_type == 'article':
        parts.append(f'"{title}."')
        source = metadata.get('source', '')
        if source:
            journal_part = source
            vol = metadata.get('volume', '')
            issue = metadata.get('issue', '')
            if vol:
                journal_part += f", vol. {vol}"
            if issue:
                journal_part += f", no. {issue}"
            year = metadata.get('date', '')[:4] if metadata.get('date') else ''
            if year:
                journal_part += f", {year}"
            pages = metadata.get('pages', '') or f"{metadata.get('spage', '')}-{metadata.get('epage', '')}".strip('-')
            if pages:
                journal_part += f", pp. {pages}"
            parts.append(journal_part + '.')
    else:
        parts.append(f"{title}.")
        publisher = metadata.get('publisher', '')
        year = metadata.get('date', '')[:4] if metadata.get('date') else ''
        if publisher and year:
            parts.append(f"{publisher}, {year}.")
        elif publisher:
            parts.append(f"{publisher}.")
        elif year:
            parts.append(f"{year}.")

    # DOI
    doi = metadata.get('doi', '')
    if doi:
        if not doi.startswith('http'):
            doi = f"https://doi.org/{doi}"
        parts.append(doi)

    return ' '.join(parts)


def format_chicago_citation(metadata):
    """Generate Chicago 17th edition citation (notes-bibliography style)."""
    if not metadata:
        return "Unable to generate citation: no metadata available."

    parts = []

    # Authors
    authors = metadata.get('authors', [])
    if authors:
        if len(authors) == 1:
            parts.append(format_author_mla(authors[0]) + '.')
        elif len(authors) <= 3:
            author_list = ', '.join([format_author_mla(a) for a in authors[:-1]])
            parts.append(f"{author_list}, and {format_author_mla(authors[-1])}.")
        else:
            parts.append(f"{format_author_mla(authors[0])}, et al.")

    # Title
    title = metadata.get('title', 'Untitled')
    resource_type = metadata.get('resource_type', 'other')

    if resource_type == 'article':
        parts.append(f'"{title}."')
        source = metadata.get('source', '')
        if source:
            journal_part = source
            vol = metadata.get('volume', '')
            issue = metadata.get('issue', '')
            if vol:
                journal_part += f" {vol}"
            if issue:
                journal_part += f", no. {issue}"
            year = metadata.get('date', '')[:4] if metadata.get('date') else ''
            if year:
                journal_part += f" ({year})"
            pages = metadata.get('pages', '') or f"{metadata.get('spage', '')}-{metadata.get('epage', '')}".strip('-')
            if pages:
                journal_part += f": {pages}"
            parts.append(journal_part + '.')
    else:
        parts.append(f"{title}.")
        publisher = metadata.get('publisher', '')
        year = metadata.get('date', '')[:4] if metadata.get('date') else ''
        if publisher:
            parts.append(f"{publisher}, {year}." if year else f"{publisher}.")

    # DOI
    doi = metadata.get('doi', '')
    if doi:
        if not doi.startswith('http'):
            doi = f"https://doi.org/{doi}"
        parts.append(doi)

    return ' '.join(parts)


def format_bibtex_entry(metadata):
    """Generate BibTeX entry."""
    if not metadata:
        return "% Unable to generate citation: no metadata available."

    resource_type = metadata.get('resource_type', 'other')

    # Generate citation key
    authors = metadata.get('authors', [])
    first_author = authors[0].split('$$')[0].split(',')[0].strip() if authors else 'unknown'
    first_author = ''.join(c for c in first_author if c.isalnum())
    year = metadata.get('date', '')[:4] if metadata.get('date') else 'YYYY'
    title_word = metadata.get('title', 'untitled').split()[0] if metadata.get('title') else 'untitled'
    title_word = ''.join(c for c in title_word if c.isalnum())
    key = f"{first_author.lower()}{year}{title_word.lower()}"

    # Entry type
    if resource_type == 'article':
        entry_type = 'article'
    else:
        entry_type = 'book'

    lines = [f"@{entry_type}{{{key},"]

    # Authors
    if authors:
        author_str = ' and '.join([a.split('$$')[0].strip() for a in authors])
        lines.append(f"  author = {{{author_str}}},")

    # Title
    title = metadata.get('title', '')
    if title:
        lines.append(f"  title = {{{title}}},")

    # Year
    if year and year != 'YYYY':
        lines.append(f"  year = {{{year}}},")

    if resource_type == 'article':
        source = metadata.get('source', '')
        if source:
            lines.append(f"  journal = {{{source}}},")
        vol = metadata.get('volume', '')
        if vol:
            lines.append(f"  volume = {{{vol}}},")
        issue = metadata.get('issue', '')
        if issue:
            lines.append(f"  number = {{{issue}}},")
        pages = metadata.get('pages', '') or f"{metadata.get('spage', '')}-{metadata.get('epage', '')}".strip('-')
        if pages:
            lines.append(f"  pages = {{{pages}}},")
    else:
        publisher = metadata.get('publisher', '')
        if publisher:
            lines.append(f"  publisher = {{{publisher}}},")

    # Identifiers
    isbn = metadata.get('isbn', '')
    if isbn:
        lines.append(f"  isbn = {{{isbn}}},")

    issn = metadata.get('issn', '')
    if issn:
        lines.append(f"  issn = {{{issn}}},")

    doi = metadata.get('doi', '')
    if doi:
        lines.append(f"  doi = {{{doi}}},")

    lines.append("}")

    return '\n'.join(lines)


def format_ris_entry(metadata):
    """Generate RIS format entry (for EndNote, Zotero, etc.)."""
    if not metadata:
        return "TY  - GEN\nER  -"

    resource_type = metadata.get('resource_type', 'other')

    lines = []

    # Type
    if resource_type == 'article':
        lines.append("TY  - JOUR")
    else:
        lines.append("TY  - BOOK")

    # Authors
    for author in metadata.get('authors', []):
        author_clean = author.split('$$')[0].strip()
        lines.append(f"AU  - {author_clean}")

    # Title
    title = metadata.get('title', '')
    if title:
        lines.append(f"TI  - {title}")

    # Year
    year = metadata.get('date', '')[:4] if metadata.get('date') else ''
    if year:
        lines.append(f"PY  - {year}")

    if resource_type == 'article':
        source = metadata.get('source', '')
        if source:
            lines.append(f"JO  - {source}")
        vol = metadata.get('volume', '')
        if vol:
            lines.append(f"VL  - {vol}")
        issue = metadata.get('issue', '')
        if issue:
            lines.append(f"IS  - {issue}")
        spage = metadata.get('spage', '')
        if spage:
            lines.append(f"SP  - {spage}")
        epage = metadata.get('epage', '')
        if epage:
            lines.append(f"EP  - {epage}")
    else:
        publisher = metadata.get('publisher', '')
        if publisher:
            lines.append(f"PB  - {publisher}")

    # Identifiers
    isbn = metadata.get('isbn', '')
    if isbn:
        lines.append(f"SN  - {isbn}")

    issn = metadata.get('issn', '')
    if issn:
        lines.append(f"SN  - {issn}")

    doi = metadata.get('doi', '')
    if doi:
        lines.append(f"DO  - {doi}")

    lines.append("ER  -")

    return '\n'.join(lines)


def extract_full_text_links(item):
    """Extract all full-text access links from a record."""
    if not item:
        return None

    pnx = item.get('pnx', {})
    links = pnx.get('links', {})
    addata = pnx.get('addata', {})
    delivery = pnx.get('delivery', {})

    result = {
        'html_links': [],
        'pdf_links': [],
        'source_links': [],
        'doi_url': None,
        'open_access': False
    }

    # Extract HTML links
    for link in links.get('linktohtml', []):
        if isinstance(link, str):
            result['html_links'].append(link)

    # Extract PDF links
    for link in links.get('linktopdf', []):
        if isinstance(link, str):
            result['pdf_links'].append(link)

    # Extract source links
    for link in links.get('linktorsrc', []):
        if isinstance(link, str):
            result['source_links'].append(link)

    # DOI URL
    doi = addata.get('doi', [''])[0] if addata.get('doi') else ''
    if doi:
        if doi.startswith('http'):
            result['doi_url'] = doi
        else:
            result['doi_url'] = f"https://doi.org/{doi}"

    # Check for open access indicators
    oa_indicators = links.get('openaccess', []) or links.get('openurl', [])
    result['open_access'] = len(oa_indicators) > 0

    return result


# =============================================================================
# RESULT FORMATTING FUNCTIONS
# =============================================================================

def format_search_results(results):
    """Format search results for display."""
    if not results:
        return "No results found or search failed."

    docs = results.get('docs', [])
    info = results.get('info', {})
    total = info.get('total', 0)

    output = [f"Found {total:,} total results\n"]
    output.append("=" * 60 + "\n")

    for i, doc in enumerate(docs, 1):
        pnx = doc.get('pnx', {})
        display = pnx.get('display', {})
        control = pnx.get('control', {})
        addata = pnx.get('addata', {})
        links = pnx.get('links', {})
        delivery = pnx.get('delivery', {})

        title = display.get('title', ['No title'])[0]
        creators = display.get('creator', display.get('contributor', []))
        creator = creators[0] if creators else 'Unknown'
        pub_date = display.get('creationdate', ['N/A'])[0]
        doc_type = display.get('type', ['N/A'])[0]
        description = display.get('description', [''])[0][:300]
        source = display.get('source', [''])[0] if display.get('source') else ''
        publisher = display.get('publisher', [''])[0] if display.get('publisher') else ''

        # Get identifiers
        doc_id = control.get('recordid', [''])[0] if control.get('recordid') else ''
        isbn = addata.get('isbn', [''])[0] if addata.get('isbn') else ''
        issn = addata.get('issn', [''])[0] if addata.get('issn') else ''
        doi = addata.get('doi', [''])[0] if addata.get('doi') else ''

        # Get access links
        fulltext_links = links.get('linktorsrc', []) or links.get('linktohtml', [])

        # Availability
        availability = delivery.get('availability', [''])[0] if delivery.get('availability') else ''

        output.append(f"{i}. {title}\n")
        output.append(f"   Author: {creator}\n")
        output.append(f"   Date: {pub_date}\n")
        output.append(f"   Type: {doc_type}\n")

        if source:
            output.append(f"   Source: {source}\n")
        if publisher:
            output.append(f"   Publisher: {publisher}\n")
        if isbn:
            output.append(f"   ISBN: {isbn}\n")
        if issn:
            output.append(f"   ISSN: {issn}\n")
        if doi:
            output.append(f"   DOI: {doi}\n")
        if doc_id:
            output.append(f"   Record ID: {doc_id}\n")
        if availability:
            output.append(f"   Availability: {availability}\n")
        if description:
            output.append(f"   Description: {description}...\n")
        if fulltext_links:
            output.append(f"   Full Text: Available\n")

        output.append("\n")

    return "".join(output)


def format_item_details(item):
    """Format detailed item information."""
    if not item:
        return "Could not retrieve item details."

    pnx = item.get('pnx', {})
    display = pnx.get('display', {})
    control = pnx.get('control', {})
    addata = pnx.get('addata', {})
    links = pnx.get('links', {})
    delivery = pnx.get('delivery', {})

    output = ["=" * 60 + "\n"]
    output.append("ITEM DETAILS\n")
    output.append("=" * 60 + "\n\n")

    # Basic info
    title = display.get('title', ['No title'])[0]
    output.append(f"Title: {title}\n\n")

    creators = display.get('creator', [])
    if creators:
        output.append(f"Author(s): {', '.join(creators)}\n")

    contributors = display.get('contributor', [])
    if contributors:
        output.append(f"Contributor(s): {', '.join(contributors)}\n")

    pub_date = display.get('creationdate', [''])[0]
    if pub_date:
        output.append(f"Publication Date: {pub_date}\n")

    doc_type = display.get('type', [''])[0]
    if doc_type:
        output.append(f"Type: {doc_type}\n")

    publisher = display.get('publisher', [''])[0]
    if publisher:
        output.append(f"Publisher: {publisher}\n")

    source = display.get('source', [''])[0]
    if source:
        output.append(f"Source/Journal: {source}\n")

    # Identifiers
    output.append("\n--- Identifiers ---\n")
    isbn = addata.get('isbn', [])
    if isbn:
        output.append(f"ISBN: {', '.join(isbn)}\n")

    issn = addata.get('issn', [])
    if issn:
        output.append(f"ISSN: {', '.join(issn)}\n")

    doi = addata.get('doi', [])
    if doi:
        output.append(f"DOI: {', '.join(doi)}\n")

    doc_id = control.get('recordid', [''])[0]
    if doc_id:
        output.append(f"Record ID: {doc_id}\n")

    # Description
    description = display.get('description', [])
    if description:
        output.append("\n--- Description ---\n")
        for desc in description:
            output.append(f"{desc}\n")

    # Subjects
    subjects = display.get('subject', [])
    if subjects:
        output.append("\n--- Subjects ---\n")
        for subj in subjects[:10]:
            output.append(f"- {subj}\n")

    # Links
    output.append("\n--- Access Links ---\n")

    linktohtml = links.get('linktohtml', [])
    if linktohtml:
        output.append(f"HTML Link: Available\n")

    linktorsrc = links.get('linktorsrc', [])
    if linktorsrc:
        output.append(f"Full Text Link: Available\n")

    linktopdf = links.get('linktopdf', [])
    if linktopdf:
        output.append(f"PDF Link: Available\n")

    # Availability
    availability = delivery.get('availability', [])
    if availability:
        output.append(f"\nAvailability: {', '.join(availability)}\n")

    holding = delivery.get('holding', [])
    if holding:
        output.append("\n--- Holdings ---\n")
        for h in holding[:5]:
            lib = h.get('libraryCode', '')
            location = h.get('subLocationCode', '')
            call = h.get('callNumber', '')
            output.append(f"  {lib} - {location}: {call}\n")

    return "".join(output)


# Initialize the MCP server
server = Server("sfu-library")

# Global client instance
client = None


def get_client():
    """Get or create the library client."""
    global client
    if client is None:
        client = SFULibraryClient(headless=True)
    return client


@server.list_tools()
async def list_tools() -> list[Tool]:
    """List available tools."""
    return [
        Tool(
            name="search_library",
            description="Search the SFU Library database for books, articles, journals, and other academic resources. Returns detailed information including titles, authors, publication dates, and availability.",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query (e.g., 'machine learning', 'climate change')"
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of results to return (default: 10, max: 50)",
                        "default": 10
                    },
                    "offset": {
                        "type": "integer",
                        "description": "Starting offset for pagination (default: 0)",
                        "default": 0
                    },
                    "field": {
                        "type": "string",
                        "description": "Field to search in: 'any' (all fields), 'title', 'creator' (author), 'sub' (subject), 'isbn', 'issn'",
                        "enum": ["any", "title", "creator", "sub", "isbn", "issn"],
                        "default": "any"
                    },
                    "sort": {
                        "type": "string",
                        "description": "Sort order: 'rank' (relevance), 'date' (newest first), 'author', 'title'",
                        "enum": ["rank", "date", "author", "title"],
                        "default": "rank"
                    },
                    "resource_type": {
                        "type": "string",
                        "description": "Type of resources: 'all' (everything), 'electronic' (online only), 'courses' (course reserves)",
                        "enum": ["all", "electronic", "courses"],
                        "default": "all"
                    }
                },
                "required": ["query"]
            }
        ),
        Tool(
            name="get_item_details",
            description="Get detailed information about a specific library item using its record ID. Returns full metadata including description, subjects, availability, and access links.",
            inputSchema={
                "type": "object",
                "properties": {
                    "record_id": {
                        "type": "string",
                        "description": "The record ID of the item (obtained from search results)"
                    }
                },
                "required": ["record_id"]
            }
        ),
        Tool(
            name="get_token_status",
            description="Check the current authentication token status. Shows if authenticated, time remaining, and user information.",
            inputSchema={
                "type": "object",
                "properties": {}
            }
        ),
        Tool(
            name="authenticate",
            description="Authenticate with the SFU Library system. Usually done automatically, but can be called manually to refresh authentication or force re-authentication.",
            inputSchema={
                "type": "object",
                "properties": {
                    "force": {
                        "type": "boolean",
                        "description": "Force re-authentication even if a valid token exists",
                        "default": False
                    }
                }
            }
        ),
        Tool(
            name="search_by_author",
            description="Search for works by a specific author in the SFU Library.",
            inputSchema={
                "type": "object",
                "properties": {
                    "author": {
                        "type": "string",
                        "description": "Author name to search for"
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of results (default: 10)",
                        "default": 10
                    }
                },
                "required": ["author"]
            }
        ),
        Tool(
            name="search_by_subject",
            description="Search for resources on a specific subject/topic in the SFU Library.",
            inputSchema={
                "type": "object",
                "properties": {
                    "subject": {
                        "type": "string",
                        "description": "Subject/topic to search for"
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of results (default: 10)",
                        "default": 10
                    }
                },
                "required": ["subject"]
            }
        ),
        Tool(
            name="search_by_isbn",
            description="Look up a specific book by its ISBN number.",
            inputSchema={
                "type": "object",
                "properties": {
                    "isbn": {
                        "type": "string",
                        "description": "ISBN number (10 or 13 digits)"
                    }
                },
                "required": ["isbn"]
            }
        ),
        Tool(
            name="search_electronic_resources",
            description="Search specifically for electronic/online resources available through SFU Library (e-books, online journals, databases).",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query"
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of results (default: 10)",
                        "default": 10
                    }
                },
                "required": ["query"]
            }
        ),
        Tool(
            name="clear_cache",
            description="Clear the cached authentication token. Useful if experiencing authentication issues.",
            inputSchema={
                "type": "object",
                "properties": {}
            }
        ),
        # NEW TOOLS
        Tool(
            name="get_full_text_links",
            description="Extract all full-text access URLs (HTML, PDF, DOI) for a specific library item. Use this to get direct links to articles and documents.",
            inputSchema={
                "type": "object",
                "properties": {
                    "record_id": {
                        "type": "string",
                        "description": "The record ID of the item (obtained from search results)"
                    }
                },
                "required": ["record_id"]
            }
        ),
        Tool(
            name="generate_citation",
            description="Generate a citation for a library item in various formats (APA, MLA, Chicago, BibTeX).",
            inputSchema={
                "type": "object",
                "properties": {
                    "record_id": {
                        "type": "string",
                        "description": "The record ID of the item to cite"
                    },
                    "format": {
                        "type": "string",
                        "description": "Citation format: 'apa' (APA 7th), 'mla' (MLA 9th), 'chicago' (Chicago 17th), 'bibtex'",
                        "enum": ["apa", "mla", "chicago", "bibtex"],
                        "default": "apa"
                    }
                },
                "required": ["record_id"]
            }
        ),
        Tool(
            name="batch_generate_citations",
            description="Generate citations for multiple library items at once. Returns all citations in the specified format.",
            inputSchema={
                "type": "object",
                "properties": {
                    "record_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of record IDs to generate citations for (max 20)"
                    },
                    "format": {
                        "type": "string",
                        "description": "Citation format: 'apa', 'mla', 'chicago', 'bibtex'",
                        "enum": ["apa", "mla", "chicago", "bibtex"],
                        "default": "apa"
                    }
                },
                "required": ["record_ids"]
            }
        ),
        Tool(
            name="export_search_results",
            description="Search and export results in various formats (JSON, CSV, BibTeX, RIS). Useful for importing into reference managers or spreadsheets.",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query"
                    },
                    "format": {
                        "type": "string",
                        "description": "Export format: 'json', 'csv', 'bibtex', 'ris' (EndNote/Zotero compatible)",
                        "enum": ["json", "csv", "bibtex", "ris"],
                        "default": "bibtex"
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of results to export (default: 10, max: 100)",
                        "default": 10
                    }
                },
                "required": ["query"]
            }
        ),
        Tool(
            name="batch_isbn_lookup",
            description="Look up multiple books by their ISBN numbers in a single request. Returns basic info and availability for each.",
            inputSchema={
                "type": "object",
                "properties": {
                    "isbn_list": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of ISBN numbers to look up (max 20)"
                    }
                },
                "required": ["isbn_list"]
            }
        )
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent]:
    """Handle tool calls."""
    lib_client = get_client()

    if name == "search_library":
        query = arguments.get("query", "")
        limit = min(arguments.get("limit", 10), 50)
        offset = arguments.get("offset", 0)
        field = arguments.get("field", "any")
        sort = arguments.get("sort", "rank")
        resource_type = arguments.get("resource_type", "all")

        # Map resource type to tab/scope
        tab = "default_tab"
        scope = "default_scope"
        if resource_type == "electronic":
            tab = "online_only_tab"
            scope = "ElectronicOnly_scope"
        elif resource_type == "courses":
            tab = "course_tab"
            scope = "course_scope"

        # Ensure authenticated
        if not lib_client.ensure_authenticated():
            return [TextContent(type="text", text="Authentication failed. Please try again or check your credentials.")]

        results = lib_client.search(
            query=query,
            limit=limit,
            offset=offset,
            field=field,
            sort=sort,
            tab=tab,
            scope=scope
        )

        if results is None:
            # Token might have expired, try re-authenticating
            if lib_client.ensure_authenticated(force=True):
                results = lib_client.search(
                    query=query,
                    limit=limit,
                    offset=offset,
                    field=field,
                    sort=sort,
                    tab=tab,
                    scope=scope
                )

        formatted = format_search_results(results)
        return [TextContent(type="text", text=formatted)]

    elif name == "get_item_details":
        record_id = arguments.get("record_id", "")

        if not lib_client.ensure_authenticated():
            return [TextContent(type="text", text="Authentication failed.")]

        item = lib_client.get_item_details(record_id)
        formatted = format_item_details(item)
        return [TextContent(type="text", text=formatted)]

    elif name == "get_token_status":
        status = lib_client.get_token_status()
        if status.get("valid"):
            text = f"""Token Status: VALID
User: {status.get('user', 'Unknown')} ({status.get('userId', '')})
User Group: {status.get('userGroup', '')}
Expires In: {status.get('expiresIn', '')}
Expires At: {status.get('expiresAt', '')}"""
        else:
            text = f"Token Status: INVALID\nReason: {status.get('message', 'Unknown')}"
        return [TextContent(type="text", text=text)]

    elif name == "authenticate":
        force = arguments.get("force", False)
        success = lib_client.ensure_authenticated(force=force)
        if success:
            status = lib_client.get_token_status()
            text = f"""Authentication successful!
User: {status.get('user', 'Unknown')} ({status.get('userId', '')})
User Group: {status.get('userGroup', '')}
Token valid for: {status.get('expiresIn', '')}"""
        else:
            text = "Authentication failed. Please check your credentials and network connection."
        return [TextContent(type="text", text=text)]

    elif name == "search_by_author":
        author = arguments.get("author", "")
        limit = arguments.get("limit", 10)

        if not lib_client.ensure_authenticated():
            return [TextContent(type="text", text="Authentication failed.")]

        results = lib_client.search(query=author, limit=limit, field='creator')
        formatted = format_search_results(results)
        return [TextContent(type="text", text=formatted)]

    elif name == "search_by_subject":
        subject = arguments.get("subject", "")
        limit = arguments.get("limit", 10)

        if not lib_client.ensure_authenticated():
            return [TextContent(type="text", text="Authentication failed.")]

        results = lib_client.search(query=subject, limit=limit, field='sub')
        formatted = format_search_results(results)
        return [TextContent(type="text", text=formatted)]

    elif name == "search_by_isbn":
        isbn = arguments.get("isbn", "")

        if not lib_client.ensure_authenticated():
            return [TextContent(type="text", text="Authentication failed.")]

        results = lib_client.search(query=isbn, limit=5, field='isbn')
        formatted = format_search_results(results)
        return [TextContent(type="text", text=formatted)]

    elif name == "search_electronic_resources":
        query = arguments.get("query", "")
        limit = arguments.get("limit", 10)

        if not lib_client.ensure_authenticated():
            return [TextContent(type="text", text="Authentication failed.")]

        results = lib_client.search(
            query=query,
            limit=limit,
            tab="online_only_tab",
            scope="ElectronicOnly_scope"
        )
        formatted = format_search_results(results)
        return [TextContent(type="text", text=formatted)]

    elif name == "clear_cache":
        lib_client.clear_token_cache()
        return [TextContent(type="text", text="Token cache cleared successfully.")]

    # NEW TOOL HANDLERS

    elif name == "get_full_text_links":
        record_id = arguments.get("record_id", "")

        if not lib_client.ensure_authenticated():
            return [TextContent(type="text", text="Authentication failed.")]

        # Try direct lookup first, then fallback to search
        item = lib_client.get_item_details(record_id)
        if not item:
            # Fallback: search by record ID
            results = lib_client.search(query=record_id, limit=1)
            if results and results.get('docs'):
                item = results['docs'][0]

        if not item:
            return [TextContent(type="text", text=f"Could not find item with record ID: {record_id}")]

        links = extract_full_text_links(item)
        if not links:
            return [TextContent(type="text", text="No access links found for this item.")]

        # Format output
        output = ["=" * 50]
        output.append("FULL TEXT ACCESS LINKS")
        output.append("=" * 50)

        if links['doi_url']:
            output.append(f"\nDOI: {links['doi_url']}")

        if links['open_access']:
            output.append("\nOpen Access: Yes")

        if links['html_links']:
            output.append(f"\nHTML Links ({len(links['html_links'])}):")
            for link in links['html_links'][:5]:
                output.append(f"  - {link}")

        if links['pdf_links']:
            output.append(f"\nPDF Links ({len(links['pdf_links'])}):")
            for link in links['pdf_links'][:5]:
                output.append(f"  - {link}")

        if links['source_links']:
            output.append(f"\nSource Links ({len(links['source_links'])}):")
            for link in links['source_links'][:5]:
                output.append(f"  - {link}")

        if not any([links['html_links'], links['pdf_links'], links['source_links'], links['doi_url']]):
            output.append("\nNo direct access links available for this item.")
            output.append("Try searching for it on the library website.")

        return [TextContent(type="text", text="\n".join(output))]

    elif name == "generate_citation":
        record_id = arguments.get("record_id", "")
        citation_format = arguments.get("format", "apa").lower()

        if not lib_client.ensure_authenticated():
            return [TextContent(type="text", text="Authentication failed.")]

        # Try direct lookup first, then fallback to search
        item = lib_client.get_item_details(record_id)
        if not item:
            # Fallback: search by record ID
            results = lib_client.search(query=record_id, limit=1)
            if results and results.get('docs'):
                item = results['docs'][0]

        if not item:
            return [TextContent(type="text", text=f"Could not find item with record ID: {record_id}")]

        metadata = extract_metadata(item)

        if citation_format == "apa":
            citation = format_apa_citation(metadata)
            format_name = "APA 7th Edition"
        elif citation_format == "mla":
            citation = format_mla_citation(metadata)
            format_name = "MLA 9th Edition"
        elif citation_format == "chicago":
            citation = format_chicago_citation(metadata)
            format_name = "Chicago 17th Edition"
        elif citation_format == "bibtex":
            citation = format_bibtex_entry(metadata)
            format_name = "BibTeX"
        else:
            citation = format_apa_citation(metadata)
            format_name = "APA 7th Edition (default)"

        output = f"--- {format_name} ---\n\n{citation}"
        return [TextContent(type="text", text=output)]

    elif name == "batch_generate_citations":
        record_ids = arguments.get("record_ids", [])[:20]  # Limit to 20
        citation_format = arguments.get("format", "apa").lower()

        if not record_ids:
            return [TextContent(type="text", text="No record IDs provided.")]

        if not lib_client.ensure_authenticated():
            return [TextContent(type="text", text="Authentication failed.")]

        format_names = {
            "apa": "APA 7th Edition",
            "mla": "MLA 9th Edition",
            "chicago": "Chicago 17th Edition",
            "bibtex": "BibTeX"
        }
        format_name = format_names.get(citation_format, "APA 7th Edition")

        output = [f"--- {format_name} Citations ({len(record_ids)} items) ---\n"]

        for i, record_id in enumerate(record_ids, 1):
            # Try direct lookup first, then fallback to search
            item = lib_client.get_item_details(record_id)
            if not item:
                results = lib_client.search(query=record_id, limit=1)
                if results and results.get('docs'):
                    item = results['docs'][0]

            if not item:
                output.append(f"{i}. [Error: Could not find record {record_id}]\n")
                continue

            metadata = extract_metadata(item)

            if citation_format == "apa":
                citation = format_apa_citation(metadata)
            elif citation_format == "mla":
                citation = format_mla_citation(metadata)
            elif citation_format == "chicago":
                citation = format_chicago_citation(metadata)
            elif citation_format == "bibtex":
                citation = format_bibtex_entry(metadata)
            else:
                citation = format_apa_citation(metadata)

            if citation_format == "bibtex":
                output.append(f"{citation}\n")
            else:
                output.append(f"{i}. {citation}\n")

        return [TextContent(type="text", text="\n".join(output))]

    elif name == "export_search_results":
        query = arguments.get("query", "")
        export_format = arguments.get("format", "bibtex").lower()
        limit = min(arguments.get("limit", 10), 100)

        if not lib_client.ensure_authenticated():
            return [TextContent(type="text", text="Authentication failed.")]

        results = lib_client.search(query=query, limit=limit)
        if not results or not results.get('docs'):
            return [TextContent(type="text", text="No results found to export.")]

        docs = results.get('docs', [])
        total = results.get('info', {}).get('total', 0)

        if export_format == "json":
            # Simplified JSON export
            export_data = []
            for doc in docs:
                pnx = doc.get('pnx', {})
                display = pnx.get('display', {})
                addata = pnx.get('addata', {})
                control = pnx.get('control', {})

                export_data.append({
                    'record_id': control.get('recordid', [''])[0] if control.get('recordid') else '',
                    'title': display.get('title', [''])[0],
                    'authors': display.get('creator', []),
                    'date': display.get('creationdate', [''])[0],
                    'type': display.get('type', [''])[0],
                    'publisher': display.get('publisher', [''])[0],
                    'isbn': addata.get('isbn', [''])[0] if addata.get('isbn') else '',
                    'issn': addata.get('issn', [''])[0] if addata.get('issn') else '',
                    'doi': addata.get('doi', [''])[0] if addata.get('doi') else '',
                })

            output = json.dumps(export_data, indent=2)
            header = f"// JSON Export: {len(docs)} of {total:,} results for '{query}'\n\n"
            return [TextContent(type="text", text=header + output)]

        elif export_format == "csv":
            # CSV export
            lines = ["record_id,title,authors,date,type,publisher,isbn,issn,doi"]
            for doc in docs:
                pnx = doc.get('pnx', {})
                display = pnx.get('display', {})
                addata = pnx.get('addata', {})
                control = pnx.get('control', {})

                # Escape quotes and commas
                def escape_csv(val):
                    if not val:
                        return ''
                    val = str(val).replace('"', '""')
                    if ',' in val or '"' in val or '\n' in val:
                        return f'"{val}"'
                    return val

                row = [
                    escape_csv(control.get('recordid', [''])[0] if control.get('recordid') else ''),
                    escape_csv(display.get('title', [''])[0]),
                    escape_csv('; '.join(display.get('creator', []))),
                    escape_csv(display.get('creationdate', [''])[0]),
                    escape_csv(display.get('type', [''])[0]),
                    escape_csv(display.get('publisher', [''])[0]),
                    escape_csv(addata.get('isbn', [''])[0] if addata.get('isbn') else ''),
                    escape_csv(addata.get('issn', [''])[0] if addata.get('issn') else ''),
                    escape_csv(addata.get('doi', [''])[0] if addata.get('doi') else ''),
                ]
                lines.append(','.join(row))

            header = f"# CSV Export: {len(docs)} of {total:,} results for '{query}'\n"
            return [TextContent(type="text", text=header + '\n'.join(lines))]

        elif export_format == "bibtex":
            # BibTeX export
            entries = []
            for doc in docs:
                metadata = extract_metadata(doc)
                entries.append(format_bibtex_entry(metadata))

            header = f"% BibTeX Export: {len(docs)} of {total:,} results for '{query}'\n\n"
            return [TextContent(type="text", text=header + '\n\n'.join(entries))]

        elif export_format == "ris":
            # RIS export (EndNote/Zotero compatible)
            entries = []
            for doc in docs:
                metadata = extract_metadata(doc)
                entries.append(format_ris_entry(metadata))

            header = f"# RIS Export: {len(docs)} of {total:,} results for '{query}'\n\n"
            return [TextContent(type="text", text=header + '\n\n'.join(entries))]

        else:
            return [TextContent(type="text", text=f"Unknown export format: {export_format}")]

    elif name == "batch_isbn_lookup":
        isbn_list = arguments.get("isbn_list", [])[:20]  # Limit to 20

        if not isbn_list:
            return [TextContent(type="text", text="No ISBN numbers provided.")]

        if not lib_client.ensure_authenticated():
            return [TextContent(type="text", text="Authentication failed.")]

        output = ["=" * 50]
        output.append(f"BATCH ISBN LOOKUP ({len(isbn_list)} items)")
        output.append("=" * 50 + "\n")

        found_count = 0
        not_found_count = 0

        for isbn in isbn_list:
            isbn_clean = isbn.replace('-', '').replace(' ', '')
            results = lib_client.search(query=isbn_clean, limit=1, field='isbn')

            if results and results.get('docs'):
                doc = results['docs'][0]
                pnx = doc.get('pnx', {})
                display = pnx.get('display', {})
                delivery = pnx.get('delivery', {})

                title = display.get('title', ['No title'])[0][:60]
                authors = display.get('creator', ['Unknown'])
                author = authors[0].split('$$')[0] if authors else 'Unknown'
                date = display.get('creationdate', ['N/A'])[0]
                availability = delivery.get('availability', ['Unknown'])[0] if delivery.get('availability') else 'Unknown'

                output.append(f"ISBN: {isbn}")
                output.append(f"  Status: FOUND")
                output.append(f"  Title: {title}")
                output.append(f"  Author: {author}")
                output.append(f"  Date: {date}")
                output.append(f"  Availability: {availability}")
                output.append("")
                found_count += 1
            else:
                output.append(f"ISBN: {isbn}")
                output.append(f"  Status: NOT FOUND")
                output.append("")
                not_found_count += 1

        output.append("-" * 50)
        output.append(f"Summary: {found_count} found, {not_found_count} not found")

        return [TextContent(type="text", text="\n".join(output))]

    else:
        return [TextContent(type="text", text=f"Unknown tool: {name}")]


async def main():
    """Run the MCP server."""
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


if __name__ == "__main__":
    asyncio.run(main())
