"""Retry logic with exponential backoff and circuit breaker."""

import asyncio
import functools
import logging
import threading
import time
from typing import Callable, TypeVar

logger = logging.getLogger("sfu_library_mcp")

F = TypeVar("F", bound=Callable)


class CircuitBreaker:
    """Circuit breaker to prevent repeated calls to a failing service.

    States:
        CLOSED: Normal operation, requests pass through.
        OPEN: Service is considered down, requests fail immediately.
        HALF_OPEN: After timeout, allow one test request through.
    """

    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"

    def __init__(self, threshold: int = 5, timeout: float = 60.0):
        self.threshold = threshold
        self.timeout = timeout
        self.failure_count = 0
        self.state = self.CLOSED
        self.last_failure_time: float = 0.0
        self._probe_in_flight = False
        self._lock = threading.Lock()

    def record_success(self) -> None:
        """Record a successful call."""
        with self._lock:
            self.failure_count = 0
            self.state = self.CLOSED
            self._probe_in_flight = False

    def record_failure(self) -> None:
        """Record a failed call."""
        with self._lock:
            self.failure_count += 1
            self.last_failure_time = time.time()
            self._probe_in_flight = False
            if self.failure_count >= self.threshold:
                self.state = self.OPEN
                logger.warning(
                    "Circuit breaker OPEN after %d failures", self.failure_count
                )

    def can_proceed(self) -> bool:
        """Check if a request should be allowed through."""
        with self._lock:
            if self.state == self.CLOSED:
                return True

            if self.state == self.OPEN:
                elapsed = time.time() - self.last_failure_time
                if elapsed >= self.timeout:
                    self.state = self.HALF_OPEN
                    self._probe_in_flight = True
                    logger.info("Circuit breaker HALF_OPEN, allowing test request")
                    return True
                return False

            # HALF_OPEN: allow only a single test request through.
            if not self._probe_in_flight:
                self._probe_in_flight = True
                return True
            return False

    def reset(self) -> None:
        """Reset the circuit breaker to closed state."""
        with self._lock:
            self.failure_count = 0
            self.state = self.CLOSED
            self.last_failure_time = 0.0
            self._probe_in_flight = False


class CircuitBreakerOpenError(Exception):
    """Raised when circuit breaker is open and blocking requests."""
    pass


def retry_with_backoff(
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    retry_on: tuple = (Exception,),
    no_retry_on: tuple = (),
    circuit_breaker: CircuitBreaker | None = None,
):
    """Decorator that retries a function with exponential backoff.

    Args:
        max_retries: Maximum number of retry attempts.
        base_delay: Initial delay between retries in seconds.
        max_delay: Maximum delay between retries in seconds.
        retry_on: Tuple of exception types to retry on.
        no_retry_on: Tuple of exception types to never retry on (takes priority).
        circuit_breaker: Optional CircuitBreaker instance.
    """

    def decorator(func: F) -> F:
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            if circuit_breaker and not circuit_breaker.can_proceed():
                raise CircuitBreakerOpenError(
                    f"Circuit breaker is open for {func.__name__}"
                )

            last_exception = None
            for attempt in range(max_retries + 1):
                try:
                    result = func(*args, **kwargs)
                    if circuit_breaker:
                        circuit_breaker.record_success()
                    return result
                except no_retry_on:
                    # Never retry these
                    raise
                except retry_on as e:
                    last_exception = e

                    if attempt < max_retries:
                        delay = min(base_delay * (2 ** attempt), max_delay)
                        # Check for rate limiting (429)
                        status_code = getattr(e, "status_code", None)
                        if hasattr(e, "response"):
                            status_code = getattr(e.response, "status_code", status_code)
                        if status_code == 429:
                            delay = min(delay * 4, max_delay)
                            logger.warning(
                                "Rate limited (429), backing off %0.1fs", delay
                            )

                        logger.info(
                            "Retry %d/%d for %s after %.1fs: %s",
                            attempt + 1,
                            max_retries,
                            func.__name__,
                            delay,
                            str(e),
                        )
                        time.sleep(delay)

            # Record at most ONE failure per logical request: only after all
            # retries are exhausted. Recording per-attempt would over-count and
            # trip the breaker after a single failing request (e.g. max_retries=3
            # would log 4 failures, opening a threshold-5 breaker prematurely).
            if circuit_breaker:
                circuit_breaker.record_failure()
            raise last_exception  # type: ignore[misc]

        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            if circuit_breaker and not circuit_breaker.can_proceed():
                raise CircuitBreakerOpenError(
                    f"Circuit breaker is open for {func.__name__}"
                )

            last_exception = None
            for attempt in range(max_retries + 1):
                try:
                    result = await func(*args, **kwargs)
                    if circuit_breaker:
                        circuit_breaker.record_success()
                    return result
                except no_retry_on:
                    raise
                except retry_on as e:
                    last_exception = e

                    if attempt < max_retries:
                        delay = min(base_delay * (2 ** attempt), max_delay)
                        status_code = getattr(e, "status_code", None)
                        if hasattr(e, "response"):
                            status_code = getattr(e.response, "status_code", status_code)
                        if status_code == 429:
                            delay = min(delay * 4, max_delay)

                        logger.info(
                            "Retry %d/%d for %s after %.1fs: %s",
                            attempt + 1,
                            max_retries,
                            func.__name__,
                            delay,
                            str(e),
                        )
                        await asyncio.sleep(delay)

            # Record at most ONE failure per logical request (see sync_wrapper).
            if circuit_breaker:
                circuit_breaker.record_failure()
            raise last_exception  # type: ignore[misc]

        if asyncio.iscoroutinefunction(func):
            return async_wrapper  # type: ignore[return-value]
        return sync_wrapper  # type: ignore[return-value]

    return decorator
