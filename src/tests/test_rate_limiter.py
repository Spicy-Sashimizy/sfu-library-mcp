"""Tests for the download rate limiter."""

import threading
import time
from unittest.mock import patch

import pytest

from lib.config import ServerConfig
from lib.rate_limiter import DownloadRateLimiter, RateLimitExceeded


@pytest.fixture
def rl_config():
    return ServerConfig(
        download_session_budget=15,
        download_hourly_limit=20,
        download_per_domain_hourly_limit=5,
        download_min_delay=0.0,  # No delay in tests
        download_max_delay=0.0,
    )


@pytest.fixture
def fast_config():
    """Config with minimal limits for quick limit-exceeded tests."""
    return ServerConfig(
        download_session_budget=3,
        download_hourly_limit=5,
        download_per_domain_hourly_limit=2,
        download_min_delay=0.0,
        download_max_delay=0.0,
    )


@pytest.fixture
def limiter(rl_config):
    return DownloadRateLimiter(rl_config)


@pytest.fixture
def fast_limiter(fast_config):
    return DownloadRateLimiter(fast_config)


class TestExtractDomain:
    def test_simple_domain(self):
        assert DownloadRateLimiter._extract_domain("https://sciencedirect.com/article/123") == "sciencedirect.com"

    def test_www_prefix(self):
        assert DownloadRateLimiter._extract_domain("https://www.nature.com/articles/123") == "nature.com"

    def test_subdomain(self):
        assert DownloadRateLimiter._extract_domain("https://link.springer.com/chapter/10") == "springer.com"

    def test_invalid_url(self):
        assert DownloadRateLimiter._extract_domain("not-a-url") in ("unknown", "not-a-url")


class TestAcquireWithinBudget:
    @patch("lib.rate_limiter.time.sleep")
    def test_acquire_succeeds_within_budget(self, mock_sleep, limiter):
        """acquire() should succeed when under all limits."""
        limiter.acquire("https://example.com/paper.pdf")
        # Should have called sleep (with 0 delay in test config)
        mock_sleep.assert_called_once()

    @patch("lib.rate_limiter.time.sleep")
    def test_acquire_multiple_within_budget(self, mock_sleep, limiter):
        """Multiple acquires within budget should succeed."""
        for i in range(5):
            limiter.acquire(f"https://example{i}.com/paper.pdf")
            limiter.record_download(f"https://example{i}.com/paper.pdf")
        assert mock_sleep.call_count == 5


class TestSessionBudgetExceeded:
    @patch("lib.rate_limiter.time.sleep")
    def test_session_budget_exceeded(self, mock_sleep, fast_limiter):
        """Should raise RateLimitExceeded after session budget exhausted."""
        for i in range(3):
            fast_limiter.acquire(f"https://example{i}.com/paper.pdf")
            fast_limiter.record_download(f"https://example{i}.com/paper.pdf")

        with pytest.raises(RateLimitExceeded, match="Session download budget exceeded"):
            fast_limiter.acquire("https://example99.com/paper.pdf")


class TestHourlyLimitExceeded:
    @patch("lib.rate_limiter.time.sleep")
    def test_hourly_limit_exceeded(self, mock_sleep):
        """Should raise after hourly limit is hit."""
        config = ServerConfig(
            download_session_budget=100,  # High session budget
            download_hourly_limit=5,
            download_per_domain_hourly_limit=100,  # High per-domain
            download_min_delay=0.0,
            download_max_delay=0.0,
        )
        limiter = DownloadRateLimiter(config)
        for i in range(5):
            limiter.acquire(f"https://example{i}.com/paper.pdf")
            limiter.record_download(f"https://example{i}.com/paper.pdf")

        with pytest.raises(RateLimitExceeded, match="Hourly download limit exceeded"):
            limiter.acquire("https://example99.com/paper.pdf")


class TestPerDomainLimit:
    @patch("lib.rate_limiter.time.sleep")
    def test_per_domain_limit(self, mock_sleep, fast_limiter):
        """Should raise after per-domain limit from same domain."""
        for i in range(2):
            fast_limiter.acquire(f"https://sciencedirect.com/article/{i}")
            fast_limiter.record_download(f"https://sciencedirect.com/article/{i}")

        with pytest.raises(RateLimitExceeded, match="Per-domain hourly limit exceeded"):
            fast_limiter.acquire("https://sciencedirect.com/article/99")

    @patch("lib.rate_limiter.time.sleep")
    def test_different_domains_independent(self, mock_sleep, fast_limiter):
        """Downloads from domain A should not affect domain B's limit."""
        for i in range(2):
            fast_limiter.acquire(f"https://sciencedirect.com/article/{i}")
            fast_limiter.record_download(f"https://sciencedirect.com/article/{i}")

        # Different domain should still work
        fast_limiter.acquire("https://springer.com/chapter/1")
        fast_limiter.record_download("https://springer.com/chapter/1")
        # No exception raised


class TestHumanizedDelay:
    def test_acquire_sleeps_within_range(self):
        """acquire() should sleep between min_delay and max_delay."""
        config = ServerConfig(
            download_session_budget=15,
            download_hourly_limit=20,
            download_per_domain_hourly_limit=5,
            download_min_delay=3.0,
            download_max_delay=8.0,
        )
        limiter = DownloadRateLimiter(config)
        with patch("lib.rate_limiter.time.sleep") as mock_sleep:
            limiter.acquire("https://example.com/paper.pdf")
            mock_sleep.assert_called_once()
            delay = mock_sleep.call_args[0][0]
            assert 3.0 <= delay <= 8.0


class TestGetBudgetStatus:
    @patch("lib.rate_limiter.time.sleep")
    def test_initial_budget_status(self, mock_sleep, limiter):
        """Fresh limiter should show full budget."""
        status = limiter.get_budget_status()
        assert status["session_remaining"] == 15
        assert status["session_used"] == 0
        assert status["hourly_remaining"] == 20
        assert status["hourly_used"] == 0
        assert status["per_domain"] == {}

    @patch("lib.rate_limiter.time.sleep")
    def test_budget_after_downloads(self, mock_sleep, limiter):
        """Budget should decrease after downloads."""
        limiter.acquire("https://example.com/paper1.pdf")
        limiter.record_download("https://example.com/paper1.pdf")
        limiter.acquire("https://example.com/paper2.pdf")
        limiter.record_download("https://example.com/paper2.pdf")

        status = limiter.get_budget_status()
        assert status["session_remaining"] == 13
        assert status["session_used"] == 2
        assert status["hourly_used"] == 2
        assert "example.com" in status["per_domain"]
        assert status["per_domain"]["example.com"]["used"] == 2


class TestThreadSafety:
    @patch("lib.rate_limiter.time.sleep")
    def test_concurrent_acquires(self, mock_sleep):
        """Concurrent acquire() calls should not corrupt state."""
        config = ServerConfig(
            download_session_budget=100,
            download_hourly_limit=100,
            download_per_domain_hourly_limit=100,
            download_min_delay=0.0,
            download_max_delay=0.0,
        )
        limiter = DownloadRateLimiter(config)
        errors = []

        def worker(thread_id):
            try:
                for i in range(10):
                    url = f"https://example{thread_id}.com/paper{i}.pdf"
                    limiter.acquire(url)
                    limiter.record_download(url)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(t,)) for t in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        status = limiter.get_budget_status()
        assert status["session_used"] == 50  # 5 threads * 10 downloads
