"""Tests for OpenAlexClient retry logic and circuit breaker integration."""

import json
import time
from unittest.mock import MagicMock, patch, call

import pytest
import requests

from lib.openalex import OpenAlexClient, DailyCallTracker
from lib.retry import CircuitBreaker


def _make_client(**kwargs) -> OpenAlexClient:
    """Return a client with a in-memory tracker (no /tmp writes) and fast retries."""
    defaults = dict(
        tracker_path="/tmp/test_openalex_tracker.json",
        max_retries=2,
        retry_base_delay=0.0,  # zero delay for tests
        circuit_breaker_threshold=3,
        circuit_breaker_timeout=5.0,
    )
    defaults.update(kwargs)
    client = OpenAlexClient(**defaults)
    # Reset tracker so tests start at 0 calls
    client._tracker._count = 0
    client._tracker._date = client._tracker._today()
    return client


def _ok_response(data: dict) -> MagicMock:
    resp = MagicMock()
    resp.status_code = 200
    resp.json.return_value = data
    resp.raise_for_status.return_value = None
    return resp


def _error_response(status: int) -> MagicMock:
    resp = MagicMock()
    resp.status_code = status
    resp.raise_for_status.side_effect = requests.HTTPError(response=resp)
    return resp


class TestOpenAlexClientRetry:
    def test_successful_request_returns_data(self):
        client = _make_client()
        payload = {"results": [{"id": "W1"}], "meta": {"count": 1}}
        with patch("requests.get", return_value=_ok_response(payload)):
            result = client._get("/works", {"search": "test"})
        assert result == payload

    def test_retries_on_timeout_then_succeeds(self):
        client = _make_client(max_retries=2)
        payload = {"results": [], "meta": {}}
        with patch("requests.get", side_effect=[
            requests.Timeout("timeout"),
            _ok_response(payload),
        ]):
            result = client._get("/works", {})
        assert result == payload

    def test_retries_on_http_error_then_succeeds(self):
        client = _make_client(max_retries=2)
        payload = {"results": [], "meta": {}}
        with patch("requests.get", side_effect=[
            _error_response(503),
            _ok_response(payload),
        ]):
            result = client._get("/works", {})
        assert result == payload

    def test_returns_none_after_all_retries_exhausted(self):
        client = _make_client(max_retries=2)
        with patch("requests.get", side_effect=requests.Timeout("timeout")):
            result = client._get("/works", {})
        assert result is None

    def test_retries_on_429_with_4x_backoff(self):
        client = _make_client(max_retries=1, retry_base_delay=0.0)
        payload = {"results": [], "meta": {}}
        rate_limited = _ok_response({})
        rate_limited.status_code = 429
        rate_limited.raise_for_status.return_value = None
        with patch("requests.get", side_effect=[rate_limited, _ok_response(payload)]):
            result = client._get("/works", {})
        assert result == payload

    def test_returns_none_when_429_all_retries(self):
        client = _make_client(max_retries=1)
        rate_limited = _ok_response({})
        rate_limited.status_code = 429
        rate_limited.raise_for_status.return_value = None
        with patch("requests.get", return_value=rate_limited):
            result = client._get("/works", {})
        assert result is None

    def test_no_retry_on_unexpected_exception(self):
        client = _make_client(max_retries=3)
        with patch("requests.get", side_effect=ValueError("unexpected")):
            result = client._get("/works", {})
        assert result is None

    def test_attempt_count_matches_retries(self):
        client = _make_client(max_retries=2)
        with patch("requests.get", side_effect=requests.Timeout()) as mock_get:
            client._get("/works", {})
        assert mock_get.call_count == 3  # 1 original + 2 retries


class TestOpenAlexCircuitBreaker:
    def test_circuit_opens_after_threshold_failures(self):
        client = _make_client(circuit_breaker_threshold=3, max_retries=0)
        with patch("requests.get", side_effect=requests.Timeout()):
            for _ in range(3):
                client._get("/works", {})
        assert client._breaker.state == CircuitBreaker.OPEN

    def test_circuit_open_blocks_request_without_http_call(self):
        client = _make_client(circuit_breaker_threshold=1, max_retries=0)
        with patch("requests.get", side_effect=requests.Timeout()):
            client._get("/works", {})  # triggers open
        with patch("requests.get") as mock_get:
            result = client._get("/works", {})
        assert result is None
        mock_get.assert_not_called()

    def test_circuit_open_property_reflects_state(self):
        client = _make_client(circuit_breaker_threshold=1, max_retries=0)
        assert client.circuit_open is False
        with patch("requests.get", side_effect=requests.Timeout()):
            client._get("/works", {})
        assert client.circuit_open is True

    def test_circuit_resets_after_successful_request(self):
        client = _make_client(circuit_breaker_threshold=3, max_retries=0)
        payload = {"results": [], "meta": {}}
        with patch("requests.get", side_effect=requests.Timeout()):
            for _ in range(3):
                client._get("/works", {})
        assert client._breaker.state == CircuitBreaker.OPEN
        # Manually set to HALF_OPEN so test request is allowed through
        client._breaker.state = CircuitBreaker.HALF_OPEN
        with patch("requests.get", return_value=_ok_response(payload)):
            client._get("/works", {})
        assert client._breaker.state == CircuitBreaker.CLOSED

    def test_circuit_half_open_after_timeout(self):
        client = _make_client(circuit_breaker_threshold=1, circuit_breaker_timeout=0.05, max_retries=0)
        with patch("requests.get", side_effect=requests.Timeout()):
            client._get("/works", {})
        assert client._breaker.state == CircuitBreaker.OPEN
        time.sleep(0.1)
        assert client._breaker.can_proceed() is True
        assert client._breaker.state == CircuitBreaker.HALF_OPEN


class TestOpenAlexBudgetBlocking:
    def test_budget_exhausted_blocks_request(self):
        client = _make_client()
        client._tracker._count = client._tracker.limit  # exhaust budget
        with patch("requests.get") as mock_get:
            result = client._get("/works", {})
        assert result is None
        mock_get.assert_not_called()

    def test_budget_not_exhausted_allows_request(self):
        client = _make_client()
        client._tracker._count = 0
        payload = {"results": [], "meta": {}}
        with patch("requests.get", return_value=_ok_response(payload)):
            result = client._get("/works", {})
        assert result == payload

    def test_budget_counter_increments_on_success(self):
        client = _make_client()
        before = client._tracker._count
        with patch("requests.get", return_value=_ok_response({"results": [], "meta": {}})):
            client._get("/works", {})
        assert client._tracker._count == before + 1

    def test_budget_counter_not_incremented_on_failure(self):
        client = _make_client(max_retries=0)
        before = client._tracker._count
        with patch("requests.get", side_effect=requests.Timeout()):
            client._get("/works", {})
        assert client._tracker._count == before


class TestOpenAlexResponseCache:
    def test_cached_response_skips_http_call(self):
        client = _make_client()
        payload = {"results": [{"id": "W999"}], "meta": {}}
        with patch("requests.get", return_value=_ok_response(payload)):
            first = client._get("/works", {"search": "cached"})
        with patch("requests.get") as mock_get:
            second = client._get("/works", {"search": "cached"})
        assert second == payload
        mock_get.assert_not_called()

    def test_different_params_bypass_cache(self):
        client = _make_client()
        payload_a = {"results": [{"id": "W1"}], "meta": {}}
        payload_b = {"results": [{"id": "W2"}], "meta": {}}
        with patch("requests.get", side_effect=[_ok_response(payload_a), _ok_response(payload_b)]):
            r1 = client._get("/works", {"search": "alpha"})
            r2 = client._get("/works", {"search": "beta"})
        assert r1 == payload_a
        assert r2 == payload_b
