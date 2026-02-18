"""Tests for the capture server."""

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def capture_client():
    """Create a test client for the capture server."""
    with patch("capture_server.load_config") as mock_config:
        from lib.config import ServerConfig
        mock_config.return_value = ServerConfig(
            download_dir="/tmp/test-capture-dl",
            capture_server_port=8787,
        )
        # Need to import after patching
        import capture_server
        capture_server._downloader = MagicMock()
        capture_server._captures = []
        yield TestClient(capture_server.app), capture_server


class TestHealthEndpoint:
    def test_health_returns_ok(self, capture_client):
        client, _ = capture_client
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert "uptime_seconds" in data
        assert "total_captures" in data


class TestCaptureEndpoint:
    def test_capture_success(self, capture_client):
        client, server = capture_client
        server._downloader.download_from_direct_url.return_value = {
            "success": True,
            "container_path": "/tmp/test.pdf",
            "size_bytes": 5000,
            "tier_used": "curl_cffi",
            "error": None,
        }
        resp = client.post("/capture", json={
            "url": "https://example.com/paper.pdf",
            "cookies": {"session": "abc"},
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is True
        assert data["size_bytes"] == 5000
        assert data["tier_used"] == "curl_cffi"

    def test_capture_failure(self, capture_client):
        client, server = capture_client
        server._downloader.download_from_direct_url.return_value = {
            "success": False,
            "container_path": None,
            "size_bytes": 0,
            "tier_used": None,
            "error": "Download failed",
        }
        resp = client.post("/capture", json={
            "url": "https://example.com/fail.pdf",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is False
        assert data["error"] == "Download failed"

    def test_capture_records_history(self, capture_client):
        client, server = capture_client
        server._downloader.download_from_direct_url.return_value = {
            "success": True,
            "container_path": "/tmp/test.pdf",
            "size_bytes": 100,
            "tier_used": "requests",
            "error": None,
        }
        client.post("/capture", json={"url": "https://example.com/a.pdf"})
        client.post("/capture", json={"url": "https://example.com/b.pdf"})
        assert len(server._captures) == 2

    def test_capture_invalid_request(self, capture_client):
        client, _ = capture_client
        resp = client.post("/capture", json={})
        assert resp.status_code == 422  # Validation error (url is required)


class TestCapturesEndpoint:
    def test_captures_empty(self, capture_client):
        client, _ = capture_client
        resp = client.get("/captures")
        assert resp.status_code == 200
        assert resp.json() == []

    def test_captures_after_capture(self, capture_client):
        client, server = capture_client
        server._downloader.download_from_direct_url.return_value = {
            "success": True,
            "container_path": "/tmp/test.pdf",
            "size_bytes": 100,
            "tier_used": "curl_cffi",
            "error": None,
        }
        client.post("/capture", json={"url": "https://example.com/test.pdf"})
        resp = client.get("/captures")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 1
        assert data[0]["success"] is True


class TestRateLimitResponse:
    def test_capture_returns_429_on_budget_exceeded(self, capture_client):
        client, server = capture_client
        server._downloader.download_from_direct_url.return_value = {
            "success": False,
            "container_path": None,
            "size_bytes": 0,
            "tier_used": None,
            "error": "Session download budget exceeded (15 downloads). Restart the server to reset.",
        }
        resp = client.post("/capture", json={
            "url": "https://example.com/paper.pdf",
        })
        assert resp.status_code == 429
        data = resp.json()
        assert "budget exceeded" in data["detail"].lower()

    def test_capture_returns_429_on_hourly_limit(self, capture_client):
        client, server = capture_client
        server._downloader.download_from_direct_url.return_value = {
            "success": False,
            "container_path": None,
            "size_bytes": 0,
            "tier_used": None,
            "error": "Hourly download limit exceeded (20 downloads/hour).",
        }
        resp = client.post("/capture", json={
            "url": "https://example.com/paper.pdf",
        })
        assert resp.status_code == 429


class TestBudgetEndpoint:
    def test_budget_endpoint(self, capture_client):
        client, server = capture_client
        from lib.rate_limiter import DownloadRateLimiter
        from lib.config import ServerConfig
        import capture_server
        capture_server._rate_limiter = DownloadRateLimiter(ServerConfig(
            download_min_delay=0.0, download_max_delay=0.0,
        ))
        resp = client.get("/budget")
        assert resp.status_code == 200
        data = resp.json()
        assert "session_remaining" in data
        assert "hourly_remaining" in data


class TestCorsHeaders:
    def test_cors_allows_chrome_extension(self, capture_client):
        client, _ = capture_client
        resp = client.options(
            "/capture",
            headers={
                "Origin": "chrome-extension://abcdef123456",
                "Access-Control-Request-Method": "POST",
            },
        )
        # CORS preflight should not return 405
        assert resp.status_code in (200, 204)
