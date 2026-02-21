"""Tests for config module."""

import os
import tempfile

import pytest

from lib.config import ServerConfig, load_config, validate_config, _read_secret


class TestReadSecret:
    """Tests for the _read_secret() Docker secret helper."""

    def test_reads_from_docker_secret_file(self, tmp_path):
        """Docker secret file takes priority over env var."""
        secret_file = tmp_path / "test_secret"
        secret_file.write_text("secret_from_file")

        # Monkey-patch the path to use tmp_path
        import lib.config as config_mod
        original_path = config_mod.Path

        class MockPath(original_path):
            def __new__(cls, *args, **kwargs):
                if args and args[0] == "/run/secrets/test_secret":
                    return original_path.__new__(cls, str(secret_file))
                return original_path.__new__(cls, *args, **kwargs)

        # Simpler approach: write a real file and read it
        with tempfile.NamedTemporaryFile(mode='w', suffix='_secret', delete=False) as f:
            f.write("docker_secret_value")
            f.flush()
            secret_path = f.name

        try:
            from pathlib import Path
            from unittest.mock import patch
            mock_path = Path(secret_path)
            with patch("lib.config.Path") as mock_cls:
                mock_cls.return_value = mock_path
                result = _read_secret("test_secret", "NONEXISTENT_ENV_VAR_12345")
                assert result == "docker_secret_value"
        finally:
            os.unlink(secret_path)

    def test_falls_back_to_env_var(self, monkeypatch):
        """Falls back to env var when no Docker secret file exists."""
        monkeypatch.setenv("TEST_SECRET_VAR", "env_value")
        result = _read_secret("nonexistent_secret_file", "TEST_SECRET_VAR")
        assert result == "env_value"

    def test_returns_default_when_nothing_found(self):
        """Returns default when neither secret file nor env var exists."""
        result = _read_secret("nonexistent", "NONEXISTENT_ENV_VAR_99999", default="fallback")
        assert result == "fallback"

    def test_returns_empty_when_nothing_found_no_default(self):
        """Returns empty string when nothing found and no default."""
        result = _read_secret("nonexistent", "NONEXISTENT_ENV_VAR_99999")
        assert result == ""


class TestServerConfig:
    def test_defaults_are_empty_without_env_vars(self, monkeypatch):
        """Without env vars or Docker secrets, credentials default to empty."""
        # Clear any credential env vars that might be set
        for var in ["SFU_USERNAME", "SFU_PASSWORD", "SFU_MFA_SECRET",
                     "SFU_MFA_DEVICE_NAME", "SFU_ZOTERO_API_KEY", "SFU_ZOTERO_USER_ID"]:
            monkeypatch.delenv(var, raising=False)
        config = load_config()
        assert config.sfu_username == ""
        assert config.sfu_password == ""
        assert config.mfa_secret == ""
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
        """A fully populated config produces no warnings."""
        config = ServerConfig(
            sfu_username="testuser",
            sfu_password="testpass",
            mfa_secret="TESTSECRET",
            zotero_api_key="testkey",
            zotero_user_id="12345",
        )
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
