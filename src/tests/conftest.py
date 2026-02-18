"""Shared test fixtures for SFU Library MCP server tests."""

import json
import os
import tempfile

import pytest


@pytest.fixture
def sample_pnx_record():
    """Realistic PNX JSON record for citation/formatting tests."""
    return {
        "pnx": {
            "display": {
                "title": ["Machine Learning: A Probabilistic Perspective"],
                "creator": [
                    "Murphy, Kevin P.$$QMurphy, Kevin P."
                ],
                "contributor": [],
                "creationdate": ["2012"],
                "publisher": ["MIT Press"],
                "type": ["book"],
                "source": [""],
                "description": [
                    "A comprehensive introduction to machine learning that uses probabilistic models."
                ],
                "subject": [
                    "Machine learning",
                    "Probabilities",
                    "Artificial intelligence"
                ],
            },
            "addata": {
                "isbn": ["9780262018029"],
                "issn": [],
                "doi": [],
                "volume": [],
                "issue": [],
                "spage": [],
                "epage": [],
                "pages": [],
            },
            "control": {
                "recordid": ["alma991234567890"]
            },
            "links": {
                "linktohtml": [],
                "linktopdf": [],
                "linktorsrc": ["https://example.com/fulltext"],
                "openaccess": [],
            },
            "delivery": {
                "availability": ["available"],
                "holding": [
                    {
                        "libraryCode": "SFUL",
                        "subLocationCode": "STACKS",
                        "callNumber": "Q325.5 .M87 2012",
                    }
                ],
            },
        }
    }


@pytest.fixture
def sample_article_record():
    """Realistic PNX JSON record for a journal article."""
    return {
        "pnx": {
            "display": {
                "title": ["Deep Learning for Natural Language Processing"],
                "creator": [
                    "Smith, John A.$$QSmith, John A.",
                    "Doe, Jane B.$$QDoe, Jane B."
                ],
                "contributor": [],
                "creationdate": ["2023"],
                "publisher": [""],
                "type": ["article"],
                "source": ["Journal of Artificial Intelligence Research"],
                "description": [
                    "A survey of deep learning techniques for NLP tasks."
                ],
                "subject": ["Deep learning", "Natural language processing"],
            },
            "addata": {
                "isbn": [],
                "issn": ["1076-9757"],
                "doi": ["10.1613/jair.1.12345"],
                "volume": ["76"],
                "issue": ["3"],
                "spage": ["1"],
                "epage": ["45"],
                "pages": ["1-45"],
            },
            "control": {
                "recordid": ["alma999888777666"]
            },
            "links": {
                "linktohtml": ["https://example.com/article.html"],
                "linktopdf": ["https://example.com/article.pdf"],
                "linktorsrc": [],
                "openaccess": ["https://example.com/oa"],
            },
            "delivery": {
                "availability": ["available_online"],
            },
        }
    }


@pytest.fixture
def mock_search_response(sample_pnx_record, sample_article_record):
    """Realistic search API response."""
    return {
        "docs": [sample_pnx_record, sample_article_record],
        "info": {
            "total": 2,
            "first": 0,
            "last": 1,
        },
    }


@pytest.fixture
def mock_config():
    """ServerConfig with test values."""
    # Deferred import to avoid circular dependency during collection
    from lib.config import ServerConfig

    return ServerConfig(
        sfu_username="testuser",
        sfu_password="testpass",
        mfa_secret="TESTSECRETBASE32A",
        token_cache_file="/tmp/test_token_cache.json",
        auth_timeout=5,
        search_timeout=10,
        token_refresh_buffer=60,
        max_retries=2,
        log_level="DEBUG",
        features={"cache_enabled": True, "retry_enabled": True},
    )


@pytest.fixture
def tmp_cache_file():
    """Temporary file for token cache tests."""
    fd, path = tempfile.mkstemp(suffix=".json")
    os.close(fd)
    yield path
    if os.path.exists(path):
        os.remove(path)


@pytest.fixture
def sample_jwt_payload():
    """Sample decoded JWT payload."""
    import time

    return {
        "user": "testuser",
        "userName": "Test User",
        "userGroup": "STUDENT",
        "exp": int(time.time()) + 3600,
        "iat": int(time.time()),
    }


@pytest.fixture
def make_jwt_token():
    """Factory fixture to create JWT tokens with custom payloads."""
    import base64

    def _make(payload: dict) -> str:
        header = base64.urlsafe_b64encode(
            json.dumps({"alg": "HS256", "typ": "JWT"}).encode()
        ).rstrip(b"=").decode()
        body = base64.urlsafe_b64encode(
            json.dumps(payload).encode()
        ).rstrip(b"=").decode()
        sig = base64.urlsafe_b64encode(b"fakesig").rstrip(b"=").decode()
        return f"{header}.{body}.{sig}"

    return _make


@pytest.fixture
def dl_config_tiered():
    """ServerConfig with all download tiers enabled."""
    from lib.config import ServerConfig

    return ServerConfig(
        download_dir="/tmp/test-dl-cache-tiered",
        host_download_dir="/tmp/test-host-downloads",
        download_timeout=30,
        max_pdf_text_chars=1000,
        ezproxy_prefix="https://proxy.lib.sfu.ca/login?url=",
        ezproxy_login_url="https://login.proxy.lib.sfu.ca/login?qurl=",
        ezproxy_proxy_base="proxy.lib.sfu.ca",
        download_tiers=["curl_cffi", "playwright", "requests"],
        playwright_timeout=10,
    )
