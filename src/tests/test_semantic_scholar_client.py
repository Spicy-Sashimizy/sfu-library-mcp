"""Tests for SemanticScholarClient retry logic and circuit breaker."""

import time
from unittest.mock import MagicMock, patch

import pytest
import requests

from lib.semantic_scholar import SemanticScholarClient
from lib.retry import CircuitBreaker


def _make_client(**kwargs) -> SemanticScholarClient:
    defaults = dict(max_retries=2, retry_base_delay=0.0,
                    circuit_breaker_threshold=3, circuit_breaker_timeout=5.0)
    defaults.update(kwargs)
    return SemanticScholarClient(**defaults)


def _ok(data: dict) -> MagicMock:
    resp = MagicMock()
    resp.status_code = 200
    resp.json.return_value = data
    resp.raise_for_status.return_value = None
    return resp


def _err(status: int) -> MagicMock:
    resp = MagicMock()
    resp.status_code = status
    resp.raise_for_status.side_effect = requests.HTTPError(response=resp)
    return resp


class TestS2Retry:
    def test_success_returns_data(self):
        client = _make_client()
        payload = {"paperId": "abc", "title": "Test"}
        with patch("requests.get", return_value=_ok(payload)):
            result = client._get("/paper/abc", {})
        assert result == payload

    def test_retries_on_timeout_then_succeeds(self):
        client = _make_client(max_retries=2)
        payload = {"paperId": "abc"}
        with patch("requests.get", side_effect=[requests.Timeout(), _ok(payload)]):
            result = client._get("/paper/abc", {})
        assert result == payload

    def test_retries_on_http_error_then_succeeds(self):
        client = _make_client(max_retries=2)
        payload = {"paperId": "abc"}
        with patch("requests.get", side_effect=[_err(503), _ok(payload)]):
            result = client._get("/paper/abc", {})
        assert result == payload

    def test_returns_none_after_all_retries_exhausted(self):
        client = _make_client(max_retries=2)
        with patch("requests.get", side_effect=requests.Timeout()):
            result = client._get("/paper/abc", {})
        assert result is None

    def test_retries_on_429(self):
        client = _make_client(max_retries=1)
        payload = {"paperId": "abc"}
        rate_limited = _ok({})
        rate_limited.status_code = 429
        with patch("requests.get", side_effect=[rate_limited, _ok(payload)]):
            result = client._get("/paper/abc", {})
        assert result == payload

    def test_returns_none_when_429_all_retries(self):
        client = _make_client(max_retries=1)
        rate_limited = _ok({})
        rate_limited.status_code = 429
        with patch("requests.get", return_value=rate_limited):
            result = client._get("/paper/abc", {})
        assert result is None

    def test_no_retry_on_unexpected_exception(self):
        client = _make_client(max_retries=3)
        with patch("requests.get", side_effect=ValueError("unexpected")):
            result = client._get("/paper/abc", {})
        assert result is None

    def test_attempt_count_matches_retries(self):
        client = _make_client(max_retries=2)
        with patch("requests.get", side_effect=requests.Timeout()) as mock_get:
            client._get("/paper/abc", {})
        assert mock_get.call_count == 3


class TestS2CircuitBreaker:
    def test_circuit_opens_after_threshold_failures(self):
        client = _make_client(circuit_breaker_threshold=3, max_retries=0)
        with patch("requests.get", side_effect=requests.Timeout()):
            for _ in range(3):
                client._get("/paper/abc", {})
        assert client._breaker.state == CircuitBreaker.OPEN

    def test_circuit_open_blocks_request(self):
        client = _make_client(circuit_breaker_threshold=1, max_retries=0)
        with patch("requests.get", side_effect=requests.Timeout()):
            client._get("/paper/abc", {})
        with patch("requests.get") as mock_get:
            result = client._get("/paper/abc", {})
        assert result is None
        mock_get.assert_not_called()

    def test_circuit_resets_after_successful_request(self):
        client = _make_client(circuit_breaker_threshold=3, max_retries=0)
        with patch("requests.get", side_effect=requests.Timeout()):
            for _ in range(3):
                client._get("/paper/abc", {})
        assert client._breaker.state == CircuitBreaker.OPEN
        client._breaker.state = CircuitBreaker.HALF_OPEN
        with patch("requests.get", return_value=_ok({"paperId": "abc"})):
            client._get("/paper/abc", {})
        assert client._breaker.state == CircuitBreaker.CLOSED

    def test_circuit_half_open_after_timeout(self):
        client = _make_client(circuit_breaker_threshold=1, circuit_breaker_timeout=0.05, max_retries=0)
        with patch("requests.get", side_effect=requests.Timeout()):
            client._get("/paper/abc", {})
        time.sleep(0.1)
        assert client._breaker.can_proceed() is True
        assert client._breaker.state == CircuitBreaker.HALF_OPEN
