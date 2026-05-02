"""Tests for config module."""

import os
import tempfile

import pytest

from lib.config import ServerConfig, load_config, validate_config, _read_secret


class TestReadSecret:
    """Tests for the _read_secret() Docker secret helper."""

    def test_reads_from_docker_secret_file(self, tmp_path):
        """Docker secret file takes priority over env var."""
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
        monkeypatch.setenv("TEST_SECRET_VAR", "env_value")
        result = _read_secret("nonexistent_secret_file", "TEST_SECRET_VAR")
        assert result == "env_value"

    def test_returns_default_when_nothing_found(self):
        result = _read_secret("nonexistent", "NONEXISTENT_ENV_VAR_99999", default="fallback")
        assert result == "fallback"

    def test_returns_empty_when_nothing_found_no_default(self):
        result = _read_secret("nonexistent", "NONEXISTENT_ENV_VAR_99999")
        assert result == ""


class TestServerConfig:
    def test_defaults_without_env_vars(self, monkeypatch):
        """Without env vars or Docker secrets, Zotero creds default to empty."""
        for var in ["SFU_ZOTERO_API_KEY", "SFU_ZOTERO_USER_ID"]:
            monkeypatch.delenv(var, raising=False)
        monkeypatch.setattr("lib.config._load_dotenv", lambda: None)
        config = load_config()
        assert config.zotero_api_key == ""
        assert config.zotero_user_id == ""
        assert config.search_timeout == 30

    def test_env_var_overrides(self, monkeypatch):
        monkeypatch.setenv("SFU_SEARCH_TIMEOUT", "60")
        monkeypatch.setenv("SFU_LOG_LEVEL", "DEBUG")
        config = load_config()
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


class TestValidateConfig:
    def test_valid_config_no_warnings(self):
        config = ServerConfig(
            zotero_api_key="testkey",
            zotero_user_id="12345",
            openalex_mailto="test@sfu.ca",
            unpaywall_email="test@sfu.ca",
            semantic_scholar_api_key="testkey",
        )
        warnings = validate_config(config)
        assert len(warnings) == 0

    def test_low_timeout_warning(self):
        config = ServerConfig(search_timeout=0)
        warnings = validate_config(config)
        assert any("search_timeout" in w for w in warnings)

    def test_negative_retries_warning(self):
        config = ServerConfig(max_retries=-1)
        warnings = validate_config(config)
        assert any("max_retries" in w for w in warnings)

    def test_invalid_log_level_warning(self):
        config = ServerConfig(log_level="INVALID")
        warnings = validate_config(config)
        assert any("log_level" in w for w in warnings)

    def test_negative_cache_ttl_warning(self):
        config = ServerConfig(cache_ttl=-1)
        warnings = validate_config(config)
        assert any("cache_ttl" in w for w in warnings)

    def test_empty_zotero_key_warning(self):
        config = ServerConfig(zotero_api_key="", zotero_user_id="123")
        warnings = validate_config(config)
        assert any("ZOTERO_API_KEY" in w for w in warnings)

    def test_empty_zotero_user_id_warning(self):
        config = ServerConfig(zotero_api_key="key", zotero_user_id="")
        warnings = validate_config(config)
        assert any("ZOTERO_USER_ID" in w for w in warnings)
