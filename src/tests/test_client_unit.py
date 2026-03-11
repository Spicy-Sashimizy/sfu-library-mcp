"""Unit tests for the SFU Library client module."""

import json
import os
import time

import pytest

from lib.client import SFULibraryClient
from lib.config import ServerConfig


@pytest.fixture
def test_config(tmp_cache_file):
    """Config for testing with temp cache file."""
    return ServerConfig(
        sfu_username="testuser",
        sfu_password="testpass",
        mfa_secret="TESTSECRETBASE32A",
        token_cache_file=tmp_cache_file,
        auth_timeout=5,
        search_timeout=10,
        token_refresh_buffer=60,
        max_retries=1,
        log_level="DEBUG",
        features={
            "cache_enabled": True,
            "retry_enabled": True,
            "circuit_breaker_enabled": True,
            "token_encryption_enabled": False,
            "screenshot_on_failure": False,
            "metrics_enabled": True,
        },
    )


@pytest.fixture
def client(test_config):
    """Client instance with test config."""
    return SFULibraryClient(config=test_config)


class TestJWTDecode:
    def test_valid_token(self, client, make_jwt_token, sample_jwt_payload):
        token = make_jwt_token(sample_jwt_payload)
        payload = client._decode_jwt(token)
        assert payload is not None
        assert payload["user"] == "testuser"
        assert payload["userName"] == "Test User"

    def test_invalid_token_not_three_parts(self, client):
        assert client._decode_jwt("invalid") is None
        assert client._decode_jwt("two.parts") is None

    def test_malformed_base64(self, client):
        assert client._decode_jwt("a.!!!invalid!!!.c") is None

    def test_empty_token(self, client):
        assert client._decode_jwt("") is None


class TestTokenValidation:
    def test_valid_token(self, client, make_jwt_token):
        payload = {"exp": int(time.time()) + 7200, "user": "test"}
        token = make_jwt_token(payload)
        assert client._is_token_valid(token) is True

    def test_expired_token(self, client, make_jwt_token):
        payload = {"exp": int(time.time()) - 100, "user": "test"}
        token = make_jwt_token(payload)
        assert client._is_token_valid(token) is False

    def test_no_expiry(self, client, make_jwt_token):
        payload = {"user": "test"}
        token = make_jwt_token(payload)
        assert client._is_token_valid(token) is False

    def test_within_buffer(self, client, make_jwt_token):
        # Token expires in 30 seconds, but buffer is 60 seconds
        payload = {"exp": int(time.time()) + 30, "user": "test"}
        token = make_jwt_token(payload)
        assert client._is_token_valid(token) is False


class TestProactiveRefresh:
    def test_no_token_needs_refresh(self, client):
        assert client._should_refresh_proactively() is True

    def test_fresh_token_no_refresh(self, client, make_jwt_token):
        payload = {"exp": int(time.time()) + 7200}
        client.jwt_token = make_jwt_token(payload)
        assert client._should_refresh_proactively() is False

    def test_near_expiry_triggers_refresh(self, client, make_jwt_token):
        # Expires in 90s, buffer is 60s, threshold is 2*buffer=120s
        payload = {"exp": int(time.time()) + 90}
        client.jwt_token = make_jwt_token(payload)
        assert client._should_refresh_proactively() is True


class TestTokenCache:
    def test_save_and_load(self, client, make_jwt_token, tmp_cache_file):
        payload = {"exp": int(time.time()) + 7200, "user": "test", "userName": "Test", "userGroup": "STUDENT"}
        client.jwt_token = make_jwt_token(payload)
        client.cookies = {"session": "abc123"}

        client.save_token_cache()
        assert os.path.exists(tmp_cache_file)

        # Create fresh client to load
        new_client = SFULibraryClient(config=client.config)
        assert new_client.load_token_cache() is True
        assert new_client.jwt_token is not None
        assert new_client.cookies == {"session": "abc123"}

    def test_load_nonexistent_file(self, client):
        client.config.token_cache_file = "/tmp/nonexistent_cache_test.json"
        assert client.load_token_cache() is False

    def test_load_expired_token(self, client, make_jwt_token, tmp_cache_file):
        payload = {"exp": int(time.time()) - 100, "user": "test"}
        client.jwt_token = make_jwt_token(payload)
        client.save_token_cache()

        new_client = SFULibraryClient(config=client.config)
        assert new_client.load_token_cache() is False

    def test_clear_cache(self, client, make_jwt_token, tmp_cache_file):
        payload = {"exp": int(time.time()) + 7200, "user": "test"}
        client.jwt_token = make_jwt_token(payload)
        client.save_token_cache()
        assert os.path.exists(tmp_cache_file)

        client.clear_token_cache()
        assert not os.path.exists(tmp_cache_file)

    def test_file_locking(self, client, make_jwt_token, tmp_cache_file):
        """Verify file locking doesn't deadlock on save/load cycle."""
        payload = {"exp": int(time.time()) + 7200, "user": "test", "userName": "Test", "userGroup": "G"}
        client.jwt_token = make_jwt_token(payload)
        client.save_token_cache()
        # Load immediately after save (same process)
        assert client.load_token_cache() is True


class TestTokenEncryption:
    def test_encryption_disabled_passthrough(self, client):
        client.config.features["token_encryption_enabled"] = False
        token = "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.test.sig"
        assert client._encrypt_token(token) == token
        assert client._decrypt_token(token) == token


class TestTokenStatus:
    def test_no_token(self, client):
        status = client.get_token_status()
        assert status["valid"] is False

    def test_valid_token_status(self, client, make_jwt_token):
        payload = {
            "exp": int(time.time()) + 7200,
            "user": "testid",
            "userName": "Test User",
            "userGroup": "STUDENT",
        }
        client.jwt_token = make_jwt_token(payload)
        status = client.get_token_status()
        assert status["valid"] is True
        assert status["user"] == "Test User"
        assert status["userId"] == "testid"

    def test_expired_token_status(self, client, make_jwt_token):
        payload = {"exp": int(time.time()) - 100, "user": "test"}
        client.jwt_token = make_jwt_token(payload)
        status = client.get_token_status()
        assert status["valid"] is False


class TestSearchWithMockedSession:
    def test_search_without_token_returns_none(self, client):
        assert client.search("test") is None

    def test_sanitizes_input(self, client, make_jwt_token):
        payload = {"exp": int(time.time()) + 7200, "user": "test"}
        client.jwt_token = make_jwt_token(payload)
        # Empty query after sanitization returns None
        result = client.search("")
        assert result is None

    def test_cache_hit(self, client, make_jwt_token):
        payload = {"exp": int(time.time()) + 7200, "user": "test"}
        client.jwt_token = make_jwt_token(payload)

        # Manually populate cache
        cache_key = client.cache.make_key("search", "cached query", 10, 0, "any", "rank", "default_tab", "default_scope")
        expected = {"docs": [], "info": {"total": 0}}
        client.cache.put(cache_key, expected)

        result = client.search("cached query")
        assert result == expected
