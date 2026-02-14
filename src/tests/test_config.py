"""Tests for config module."""

import os

import pytest

from lib.config import ServerConfig, load_config, validate_config


class TestServerConfig:
    def test_defaults_match_original_hardcoded_values(self):
        """Verify defaults preserve backwards compatibility."""
        config = load_config()
        assert config.sfu_username == "REDACTED_SFU_USERNAME"
        assert config.sfu_password == "REDACTED_SFU_PASSWORD"
        assert config.mfa_secret == "REDACTED_MFA_SECRET"
        assert config.token_refresh_buffer == 300
        assert config.auth_timeout == 10

    def test_env_var_overrides(self, monkeypatch):
        monkeypatch.setenv("SFU_USERNAME", "override_user")
        monkeypatch.setenv("SFU_PASSWORD", "override_pass")
        monkeypatch.setenv("SFU_MFA_SECRET", "OVERRIDESECRET32")
        monkeypatch.setenv("SFU_AUTH_TIMEOUT", "20")
        monkeypatch.setenv("SFU_SEARCH_TIMEOUT", "60")
        monkeypatch.setenv("SFU_LOG_LEVEL", "DEBUG")

        config = load_config()
        assert config.sfu_username == "override_user"
        assert config.sfu_password == "override_pass"
        assert config.mfa_secret == "OVERRIDESECRET32"
        assert config.auth_timeout == 20
        assert config.search_timeout == 60
        assert config.log_level == "DEBUG"

    def test_feature_flag_env_override(self, monkeypatch):
        monkeypatch.setenv("SFU_FEATURE_CACHE_ENABLED", "false")
        monkeypatch.setenv("SFU_FEATURE_RETRY_ENABLED", "0")

        config = load_config()
        assert config.features["cache_enabled"] is False
        assert config.features["retry_enabled"] is False

    def test_feature_flag_defaults(self):
        config = load_config()
        assert config.features["cache_enabled"] is True
        assert config.features["retry_enabled"] is True
        assert config.features["token_encryption_enabled"] is False


class TestValidateConfig:
    def test_valid_config_no_warnings(self):
        config = load_config()
        warnings = validate_config(config)
        assert len(warnings) == 0

    def test_empty_username_warning(self):
        config = ServerConfig(sfu_username="", sfu_password="pass", mfa_secret="secret")
        warnings = validate_config(config)
        assert any("SFU_USERNAME" in w for w in warnings)

    def test_empty_password_warning(self):
        config = ServerConfig(sfu_username="user", sfu_password="", mfa_secret="secret")
        warnings = validate_config(config)
        assert any("SFU_PASSWORD" in w for w in warnings)

    def test_low_timeout_warning(self):
        config = ServerConfig(
            sfu_username="u", sfu_password="p", mfa_secret="s",
            auth_timeout=0
        )
        warnings = validate_config(config)
        assert any("auth_timeout" in w for w in warnings)

    def test_negative_retries_warning(self):
        config = ServerConfig(
            sfu_username="u", sfu_password="p", mfa_secret="s",
            max_retries=-1
        )
        warnings = validate_config(config)
        assert any("max_retries" in w for w in warnings)

    def test_invalid_log_level_warning(self):
        config = ServerConfig(
            sfu_username="u", sfu_password="p", mfa_secret="s",
            log_level="INVALID"
        )
        warnings = validate_config(config)
        assert any("log_level" in w for w in warnings)

    def test_negative_cache_ttl_warning(self):
        config = ServerConfig(
            sfu_username="u", sfu_password="p", mfa_secret="s",
            cache_ttl=-1
        )
        warnings = validate_config(config)
        assert any("cache_ttl" in w for w in warnings)
