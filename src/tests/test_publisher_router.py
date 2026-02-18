"""Unit tests for the publisher_router module."""

import threading

import pytest

from lib.publisher_router import DomainClass, DIRECT_TIERS, PublisherRouter


class TestExtractDomain:
    """Test domain extraction from URLs."""

    def test_simple_domain(self):
        assert PublisherRouter.extract_domain("https://example.com/page") == "example.com"

    def test_subdomain_stripped(self):
        assert PublisherRouter.extract_domain("https://onlinelibrary.wiley.com/doi/123") == "wiley.com"

    def test_www_stripped(self):
        assert PublisherRouter.extract_domain("https://www.springer.com/article") == "springer.com"

    def test_deep_subdomain(self):
        assert PublisherRouter.extract_domain("https://journals.sagepub.com/doi/full/10.1177/123") == "sagepub.com"

    def test_proxy_url_hostname_based(self):
        """Hostname-based proxy URLs should extract the original publisher domain."""
        assert PublisherRouter.extract_domain("https://onlinelibrary-wiley-com.proxy.lib.sfu.ca/doi/123") == "wiley.com"

    def test_proxy_url_sagepub(self):
        """Hostname-based proxy URLs for SAGE should extract sagepub.com."""
        assert PublisherRouter.extract_domain("https://journals-sagepub-com.proxy.lib.sfu.ca/doi/pdf/10.1177/123") == "sagepub.com"

    def test_proxy_url_sciencedirect(self):
        """Hostname-based proxy URLs for ScienceDirect should extract sciencedirect.com."""
        assert PublisherRouter.extract_domain("https://www-sciencedirect-com.proxy.lib.sfu.ca/science/article/123") == "sciencedirect.com"

    def test_legacy_prefix_proxy_url(self):
        """Legacy prefix-mode proxy URLs should extract the proxy domain (sfu.ca)."""
        assert PublisherRouter.extract_domain("https://proxy.lib.sfu.ca/login?url=https://example.com") == "sfu.ca"

    def test_empty_url(self):
        assert PublisherRouter.extract_domain("") == "unknown"

    def test_invalid_url(self):
        assert PublisherRouter.extract_domain("not-a-url") == "unknown"

    def test_ip_address(self):
        result = PublisherRouter.extract_domain("http://192.168.1.1/page")
        # IP addresses have dots — should return last two parts
        assert result == "168.1.1" or result == "1.1"  # depends on split logic

    def test_single_part_hostname(self):
        assert PublisherRouter.extract_domain("http://localhost/page") == "localhost"


class TestTierOrdering:
    """Test get_tier_order for different domain classifications."""

    @pytest.fixture
    def router(self):
        return PublisherRouter()

    @pytest.fixture
    def base_tiers(self):
        return ["curl_cffi", "playwright", "requests"]

    def test_unknown_domain_default_order(self, router, base_tiers):
        """Unknown domains should use the default tier order."""
        result = router.get_tier_order("https://unknown.com/pdf", base_tiers)
        assert result == base_tiers

    def test_direct_ok_domain(self, router, base_tiers):
        """DIRECT_OK domains should have direct tiers first, browser last."""
        router.record_success("https://journals.sagepub.com/doi/pdf/10.1177/123", "curl_cffi")
        result = router.get_tier_order("https://journals.sagepub.com/other", base_tiers)
        # Direct tiers first
        assert result[0] in DIRECT_TIERS
        assert result[-1] == "playwright"

    def test_browser_only_domain(self, router, base_tiers):
        """BROWSER_ONLY domains should have playwright first, direct tiers last."""
        router.record_failure("https://onlinelibrary.wiley.com/doi/123", "curl_cffi")
        result = router.get_tier_order("https://onlinelibrary.wiley.com/other", base_tiers)
        # Browser first
        assert result[0] == "playwright"
        # Direct tiers still included as fallback
        assert "curl_cffi" in result
        assert "requests" in result

    def test_all_tiers_always_included(self, router, base_tiers):
        """All configured tiers should always be present regardless of classification."""
        router.record_failure("https://example.com/pdf", "curl_cffi")
        result = router.get_tier_order("https://example.com/pdf", base_tiers)
        assert set(result) == set(base_tiers)

    def test_two_tier_config(self, router):
        """Should work with configs that have fewer tiers."""
        two_tiers = ["curl_cffi", "requests"]
        result = router.get_tier_order("https://unknown.com/pdf", two_tiers)
        assert result == two_tiers

    def test_browser_only_with_no_playwright(self, router):
        """If config has no playwright, BROWSER_ONLY just returns available tiers."""
        router.record_failure("https://example.com/pdf", "curl_cffi")
        result = router.get_tier_order("https://example.com/pdf", ["curl_cffi", "requests"])
        # No playwright in config, so direct tiers are all that's available
        assert result == ["curl_cffi", "requests"]


class TestClassificationTransitions:
    """Test how domains transition between classifications."""

    @pytest.fixture
    def router(self):
        return PublisherRouter()

    def test_initial_state_is_unknown(self, router):
        assert router.get_domain_class("https://example.com/pdf") == DomainClass.UNKNOWN

    def test_direct_failure_classifies_browser_only(self, router):
        """A single direct tier failure should classify as BROWSER_ONLY."""
        router.record_failure("https://wiley.com/doi/123", "curl_cffi")
        assert router.get_domain_class("https://wiley.com/doi/123") == DomainClass.BROWSER_ONLY

    def test_direct_success_classifies_direct_ok(self, router):
        """A direct tier success should classify as DIRECT_OK."""
        router.record_success("https://sagepub.com/doi/123", "curl_cffi")
        assert router.get_domain_class("https://sagepub.com/doi/123") == DomainClass.DIRECT_OK

    def test_browser_failure_does_not_reclassify(self, router):
        """Browser tier failure should NOT change classification."""
        router.record_failure("https://example.com/pdf", "playwright")
        assert router.get_domain_class("https://example.com/pdf") == DomainClass.UNKNOWN

    def test_success_overrides_failure(self, router):
        """A direct success after a failure should upgrade to DIRECT_OK."""
        router.record_failure("https://example.com/pdf", "curl_cffi")
        assert router.get_domain_class("https://example.com/pdf") == DomainClass.BROWSER_ONLY
        router.record_success("https://example.com/pdf", "requests")
        assert router.get_domain_class("https://example.com/pdf") == DomainClass.DIRECT_OK

    def test_failure_does_not_override_success(self, router):
        """A direct failure after a success should NOT downgrade from DIRECT_OK."""
        router.record_success("https://example.com/pdf", "curl_cffi")
        assert router.get_domain_class("https://example.com/pdf") == DomainClass.DIRECT_OK
        router.record_failure("https://example.com/other", "requests")
        # Still DIRECT_OK because there was a prior direct success
        assert router.get_domain_class("https://example.com/pdf") == DomainClass.DIRECT_OK

    def test_different_subdomains_same_domain(self, router):
        """Subdomains of the same registrable domain should share classification."""
        router.record_failure("https://onlinelibrary.wiley.com/doi/123", "curl_cffi")
        assert router.get_domain_class("https://api.wiley.com/other") == DomainClass.BROWSER_ONLY

    def test_browser_success_does_not_set_direct_ok(self, router):
        """A playwright success should not set DIRECT_OK."""
        router.record_success("https://example.com/pdf", "playwright")
        assert router.get_domain_class("https://example.com/pdf") == DomainClass.UNKNOWN


class TestGetStats:
    """Test diagnostic stats output."""

    def test_empty_stats(self):
        router = PublisherRouter()
        stats = router.get_stats()
        assert stats["classifications"] == {}
        assert stats["successes"] == {}
        assert stats["failures"] == {}

    def test_stats_after_activity(self):
        router = PublisherRouter()
        router.record_success("https://sagepub.com/pdf", "curl_cffi")
        router.record_failure("https://wiley.com/pdf", "curl_cffi")
        stats = router.get_stats()
        assert stats["classifications"]["sagepub.com"] == "direct_ok"
        assert stats["classifications"]["wiley.com"] == "browser_only"
        assert "curl_cffi" in stats["successes"]["sagepub.com"]
        assert "curl_cffi" in stats["failures"]["wiley.com"]


class TestThreadSafety:
    """Test concurrent access to PublisherRouter."""

    def test_concurrent_record_and_query(self):
        """Multiple threads recording and querying should not crash."""
        router = PublisherRouter()
        errors = []

        def worker(domain_suffix: int):
            try:
                url = f"https://publisher{domain_suffix}.com/pdf"
                for _ in range(50):
                    router.record_failure(url, "curl_cffi")
                    router.get_tier_order(url, ["curl_cffi", "playwright", "requests"])
                    router.record_success(url, "requests")
                    router.get_domain_class(url)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == [], f"Thread safety errors: {errors}"

    def test_concurrent_same_domain(self):
        """Multiple threads hitting the same domain should not corrupt state."""
        router = PublisherRouter()
        errors = []

        def recorder(tier_name: str):
            try:
                for _ in range(100):
                    router.record_failure("https://example.com/pdf", tier_name)
                    router.get_tier_order("https://example.com/pdf", ["curl_cffi", "playwright", "requests"])
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=recorder, args=("curl_cffi",)),
            threading.Thread(target=recorder, args=("requests",)),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == [], f"Thread safety errors: {errors}"
        # Should be BROWSER_ONLY since only failures were recorded for direct tiers
        assert router.get_domain_class("https://example.com/pdf") == DomainClass.BROWSER_ONLY
