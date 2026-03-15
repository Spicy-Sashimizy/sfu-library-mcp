"""Unit tests for the SFU Library client module."""

import json
import os
import time

import pytest

from lib.client import SFULibraryClient
from lib.config import ServerConfig


@pytest.fixture
def test_config():
    """Config for testing."""
    return ServerConfig(
        search_timeout=10,
        max_retries=1,
        log_level="DEBUG",
        features={
            "cache_enabled": True,
            "retry_enabled": True,
            "circuit_breaker_enabled": True,
            "metrics_enabled": True,
        },
    )


@pytest.fixture
def client(test_config):
    """Client instance with test config."""
    return SFULibraryClient(config=test_config)


class TestSearch:
    def test_empty_query_returns_none(self, client):
        result = client.search("")
        assert result is None

    def test_cache_hit(self, client):
        # Manually populate cache
        cache_key = client.cache.make_key("search", "cached query", 10, 0, "any", "rank", "default_tab", "default_scope")
        expected = {"docs": [], "info": {"total": 0}}
        client.cache.put(cache_key, expected)

        result = client.search("cached query")
        assert result == expected


class TestClientInit:
    def test_creates_with_defaults(self):
        client = SFULibraryClient()
        assert client.config is not None
        assert client.cache is not None
        assert client.circuit_breaker is not None

    def test_creates_with_custom_config(self, test_config):
        client = SFULibraryClient(config=test_config)
        assert client.config.search_timeout == 10
