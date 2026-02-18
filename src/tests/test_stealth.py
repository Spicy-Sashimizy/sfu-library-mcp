"""Unit tests for the stealth evasion module."""

from unittest.mock import MagicMock

from lib.stealth import (
    CURL_IMPERSONATE_VERSION,
    STEALTH_LAUNCH_ARGS,
    STEALTH_SCRIPTS,
    apply_stealth,
    get_curl_extra_fingerprints,
    get_stealth_context_options,
)


class TestStealthScripts:
    def test_stealth_scripts_is_nonempty_string(self):
        assert isinstance(STEALTH_SCRIPTS, str)
        assert len(STEALTH_SCRIPTS) > 100

    def test_stealth_covers_webdriver(self):
        assert "navigator.webdriver" in STEALTH_SCRIPTS
        # Should also delete from prototype
        assert "__proto__.webdriver" in STEALTH_SCRIPTS

    def test_stealth_covers_chrome_object(self):
        assert "window.chrome" in STEALTH_SCRIPTS
        assert "chrome.runtime" in STEALTH_SCRIPTS or "runtime" in STEALTH_SCRIPTS

    def test_stealth_covers_plugins(self):
        assert "navigator.plugins" in STEALTH_SCRIPTS
        assert "Chrome PDF Plugin" in STEALTH_SCRIPTS

    def test_stealth_covers_languages(self):
        assert "navigator.languages" in STEALTH_SCRIPTS
        assert "en-US" in STEALTH_SCRIPTS

    def test_stealth_covers_permissions(self):
        assert "permissions.query" in STEALTH_SCRIPTS

    def test_stealth_covers_webgl(self):
        assert "WEBGL_debug_renderer_info" in STEALTH_SCRIPTS or "UNMASKED_VENDOR_WEBGL" in STEALTH_SCRIPTS
        assert "Intel" in STEALTH_SCRIPTS

    def test_stealth_covers_hardware_concurrency(self):
        assert "hardwareConcurrency" in STEALTH_SCRIPTS


class TestStealthLaunchArgs:
    def test_launch_args_has_key_flags(self):
        assert "--disable-blink-features=AutomationControlled" in STEALTH_LAUNCH_ARGS
        assert "--no-sandbox" in STEALTH_LAUNCH_ARGS
        assert "--disable-dev-shm-usage" in STEALTH_LAUNCH_ARGS


class TestStealthContextOptions:
    def test_context_options_viewport(self):
        opts = get_stealth_context_options("Mozilla/5.0 Test")
        assert opts["viewport"] == {"width": 1920, "height": 1080}

    def test_context_options_locale_and_timezone(self):
        opts = get_stealth_context_options("Mozilla/5.0 Test")
        assert opts["locale"] == "en-US"
        assert opts["timezone_id"] == "America/Vancouver"

    def test_context_options_passes_user_agent(self):
        ua = "Mozilla/5.0 Custom Agent"
        opts = get_stealth_context_options(ua)
        assert opts["user_agent"] == ua

    def test_context_options_has_extra_headers(self):
        opts = get_stealth_context_options("test")
        assert "Accept-Language" in opts["extra_http_headers"]
        assert "DNT" in opts["extra_http_headers"]


class TestApplyStealth:
    def test_apply_stealth_calls_add_init_script(self):
        mock_page = MagicMock()
        apply_stealth(mock_page)
        mock_page.add_init_script.assert_called_once_with(STEALTH_SCRIPTS)


class TestCurlCffiStealth:
    def test_curl_impersonate_version_matches_user_agent(self):
        """Version string should contain '131' to match Chrome/131 User-Agent."""
        assert "131" in CURL_IMPERSONATE_VERSION
        assert CURL_IMPERSONATE_VERSION == "chrome131"

    def test_curl_extra_fingerprints_has_grease(self):
        fp = get_curl_extra_fingerprints()
        assert fp.tls_grease is True

    def test_curl_extra_fingerprints_has_permute(self):
        fp = get_curl_extra_fingerprints()
        assert fp.tls_permute_extensions is True
