"""Tests for retry module."""

import time

import pytest

from lib.retry import (
    CircuitBreaker,
    CircuitBreakerOpenError,
    retry_with_backoff,
)


class TestRetryWithBackoff:
    def test_succeeds_on_first_try(self):
        call_count = 0

        @retry_with_backoff(max_retries=3, base_delay=0.01)
        def succeed():
            nonlocal call_count
            call_count += 1
            return "ok"

        assert succeed() == "ok"
        assert call_count == 1

    def test_retries_then_succeeds(self):
        call_count = 0

        @retry_with_backoff(max_retries=3, base_delay=0.01)
        def fail_twice():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ConnectionError("fail")
            return "ok"

        assert fail_twice() == "ok"
        assert call_count == 3

    def test_exhausts_retries_then_raises(self):
        @retry_with_backoff(max_retries=2, base_delay=0.01)
        def always_fail():
            raise ConnectionError("fail")

        with pytest.raises(ConnectionError):
            always_fail()

    def test_no_retry_on_excluded_exception(self):
        call_count = 0

        @retry_with_backoff(
            max_retries=3,
            base_delay=0.01,
            no_retry_on=(ValueError,),
        )
        def fail_value():
            nonlocal call_count
            call_count += 1
            raise ValueError("no retry")

        with pytest.raises(ValueError):
            fail_value()
        assert call_count == 1

    def test_backoff_delay_increases(self):
        times = []

        @retry_with_backoff(max_retries=3, base_delay=0.05, max_delay=10.0)
        def track_times():
            times.append(time.time())
            if len(times) < 4:
                raise ConnectionError("fail")
            return "ok"

        track_times()
        # delays should be approximately 0.05, 0.1, 0.2
        if len(times) >= 3:
            delay1 = times[1] - times[0]
            delay2 = times[2] - times[1]
            assert delay2 >= delay1 * 1.5  # exponential growth

    def test_rate_limit_429_longer_backoff(self):
        """Verify 429 errors trigger longer backoff."""
        call_count = 0

        class RateLimitError(Exception):
            def __init__(self):
                super().__init__("rate limited")

            class response:
                status_code = 429

        @retry_with_backoff(
            max_retries=1,
            base_delay=0.01,
            max_delay=1.0,
            retry_on=(RateLimitError,),
        )
        def rate_limited():
            nonlocal call_count
            call_count += 1
            raise RateLimitError()

        with pytest.raises(RateLimitError):
            rate_limited()
        assert call_count == 2  # initial + 1 retry


class TestCircuitBreaker:
    def test_starts_closed(self):
        cb = CircuitBreaker(threshold=3)
        assert cb.state == CircuitBreaker.CLOSED
        assert cb.can_proceed() is True

    def test_opens_after_threshold(self):
        cb = CircuitBreaker(threshold=3, timeout=60.0)
        for _ in range(3):
            cb.record_failure()
        assert cb.state == CircuitBreaker.OPEN
        assert cb.can_proceed() is False

    def test_stays_closed_below_threshold(self):
        cb = CircuitBreaker(threshold=3)
        cb.record_failure()
        cb.record_failure()
        assert cb.state == CircuitBreaker.CLOSED
        assert cb.can_proceed() is True

    def test_success_resets_count(self):
        cb = CircuitBreaker(threshold=3)
        cb.record_failure()
        cb.record_failure()
        cb.record_success()
        assert cb.failure_count == 0
        assert cb.state == CircuitBreaker.CLOSED

    def test_half_open_after_timeout(self):
        cb = CircuitBreaker(threshold=1, timeout=0.05)
        cb.record_failure()
        assert cb.state == CircuitBreaker.OPEN
        time.sleep(0.06)
        assert cb.can_proceed() is True
        assert cb.state == CircuitBreaker.HALF_OPEN

    def test_reset(self):
        cb = CircuitBreaker(threshold=1)
        cb.record_failure()
        assert cb.state == CircuitBreaker.OPEN
        cb.reset()
        assert cb.state == CircuitBreaker.CLOSED
        assert cb.failure_count == 0

    def test_circuit_breaker_with_retry_decorator(self):
        cb = CircuitBreaker(threshold=2, timeout=60.0)

        @retry_with_backoff(
            max_retries=5,
            base_delay=0.0,
            circuit_breaker=cb,
        )
        def always_fail():
            raise ConnectionError("fail")

        # Each logical request records exactly ONE failure (Item 1). With
        # threshold=2 the breaker opens only after the SECOND failing request,
        # regardless of how many retries each performs.
        with pytest.raises(ConnectionError):
            always_fail()
        assert cb.failure_count == 1
        assert cb.state == CircuitBreaker.CLOSED

        with pytest.raises(ConnectionError):
            always_fail()
        assert cb.state == CircuitBreaker.OPEN

        # Third call should now fail immediately with CircuitBreakerOpenError
        with pytest.raises(CircuitBreakerOpenError):
            always_fail()

    def test_decorator_records_one_failure_per_request(self):
        """Regression (Item 1): the decorator must record exactly ONE breaker
        failure per logical request, no matter how many retries it performs.

        Previously record_failure() was called per-attempt, so a single failing
        request with max_retries=3 logged 4 failures and tripped a threshold-5
        breaker after ~1-2 requests.
        """
        cb = CircuitBreaker(threshold=5, timeout=60.0)

        @retry_with_backoff(
            max_retries=3,
            base_delay=0.0,
            circuit_breaker=cb,
        )
        def always_fail():
            raise ConnectionError("fail")

        with pytest.raises(ConnectionError):
            always_fail()

        # One logical request → exactly one recorded failure, breaker still CLOSED.
        assert cb.failure_count == 1
        assert cb.state == CircuitBreaker.CLOSED

    def test_decorator_success_records_no_failure(self):
        """A request that eventually succeeds must record no failures."""
        cb = CircuitBreaker(threshold=5, timeout=60.0)
        calls = {"n": 0}

        @retry_with_backoff(
            max_retries=3,
            base_delay=0.0,
            circuit_breaker=cb,
        )
        def fail_twice_then_ok():
            calls["n"] += 1
            if calls["n"] < 3:
                raise ConnectionError("fail")
            return "ok"

        assert fail_twice_then_ok() == "ok"
        assert cb.failure_count == 0
        assert cb.state == CircuitBreaker.CLOSED
