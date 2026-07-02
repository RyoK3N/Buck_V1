"""Tests for the FastAPI backend surface (UI/backend/).

Covers the security-relevant behaviors that previously had zero coverage:
secret masking in /config, internal-error masking, bearer-token auth
(HTTP middleware and the WebSocket handshake, which the middleware can't
cover), and the rate limiter on expensive endpoints.
"""

from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from UI.backend.main import app
from UI.backend.auth import maybe_add_auth
from UI.backend.ratelimit import maybe_add_rate_limit


@pytest.fixture
def client() -> TestClient:
    # No `with` block: skips lifespan (accuracy DB / scheduler), which the
    # endpoints under test don't need.
    return TestClient(app)


# ---------------------------------------------------------------------------
# Health / config
# ---------------------------------------------------------------------------

class TestConfigEndpoint:
    def test_health(self, client):
        assert client.get("/health").status_code == 200

    def test_config_never_echoes_secret_values(self, client, monkeypatch):
        from agent_scripts.config import SETTINGS
        monkeypatch.setattr(SETTINGS, "openai_api_key", "sk-super-secret-value")
        monkeypatch.setattr(SETTINGS, "indian_api_key", "indian-secret-value")

        resp = client.get("/config")
        assert resp.status_code == 200
        body = resp.json()

        # Only boolean "configured" flags — never the values themselves.
        assert body["openai_api_key_configured"] is True
        assert body["indian_api_key_configured"] is True
        assert "sk-super-secret-value" not in resp.text
        assert "indian-secret-value" not in resp.text

    def test_config_reports_unconfigured_placeholder(self, client, monkeypatch):
        from agent_scripts.config import SETTINGS
        monkeypatch.setattr(SETTINGS, "openai_api_key", "__placeholder__")
        monkeypatch.setattr(SETTINGS, "indian_api_key", "")

        body = client.get("/config").json()
        assert body["openai_api_key_configured"] is False
        assert body["indian_api_key_configured"] is False


# ---------------------------------------------------------------------------
# Internal-error masking
# ---------------------------------------------------------------------------

class TestErrorMasking:
    def test_mcp_invoke_500_hides_exception_internals(self, client, monkeypatch):
        import mcp_server.tools as mcp_tools

        async def boom(tool, args):
            raise RuntimeError("/Users/someone/private/path.py exploded")

        monkeypatch.setattr(mcp_tools, "dispatch_async", boom)
        resp = client.post("/mcp/invoke", json={"tool": "single_analyze", "args": {}})
        assert resp.status_code == 500
        assert "private" not in resp.text
        assert "exploded" not in resp.text
        assert "see server logs" in resp.json()["detail"]

    def test_mcp_invoke_unknown_tool_is_404(self, client):
        resp = client.post("/mcp/invoke", json={"tool": "definitely_not_a_tool", "args": {}})
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# Bearer auth middleware
# ---------------------------------------------------------------------------

def _tiny_app_with_auth(token: str) -> TestClient:
    tiny = FastAPI()

    @tiny.get("/health")
    async def health():
        return {"ok": True}

    @tiny.get("/protected")
    async def protected():
        return {"ok": True}

    maybe_add_auth(tiny, token)
    return TestClient(tiny)


class TestBearerAuth:
    def test_rejects_missing_and_wrong_token(self):
        client = _tiny_app_with_auth("sekrit")
        assert client.get("/protected").status_code == 401
        assert client.get("/protected", headers={"Authorization": "Bearer wrong"}).status_code == 401

    def test_accepts_correct_token_and_exempts_health(self):
        client = _tiny_app_with_auth("sekrit")
        assert client.get("/protected", headers={"Authorization": "Bearer sekrit"}).status_code == 200
        assert client.get("/health").status_code == 200

    def test_no_token_configured_means_no_auth(self):
        tiny = FastAPI()

        @tiny.get("/protected")
        async def protected():
            return {"ok": True}

        maybe_add_auth(tiny, None)
        assert TestClient(tiny).get("/protected").status_code == 200


# ---------------------------------------------------------------------------
# WebSocket auth (the HTTP middleware can't cover WS handshakes)
# ---------------------------------------------------------------------------

class TestWebSocketAuth:
    def test_ws_rejected_without_token(self, client, monkeypatch):
        from agent_scripts.config import SETTINGS
        monkeypatch.setattr(SETTINGS, "buck_api_auth_token", "sekrit")

        with pytest.raises(WebSocketDisconnect):
            with client.websocket_connect("/accuracy/ws"):
                pass

    def test_ws_accepts_query_token(self, client, monkeypatch):
        from agent_scripts.config import SETTINGS
        monkeypatch.setattr(SETTINGS, "buck_api_auth_token", "sekrit")

        with client.websocket_connect("/accuracy/ws?token=sekrit") as ws:
            first = ws.receive_json()
            assert first["type"] == "snapshot"

    def test_ws_open_when_no_token_configured(self, client, monkeypatch):
        from agent_scripts.config import SETTINGS
        monkeypatch.setattr(SETTINGS, "buck_api_auth_token", "")

        with client.websocket_connect("/accuracy/ws") as ws:
            first = ws.receive_json()
            assert first["type"] == "snapshot"


# ---------------------------------------------------------------------------
# Rate limiting
# ---------------------------------------------------------------------------

def _tiny_app_with_limit(limit: int) -> TestClient:
    tiny = FastAPI()

    @tiny.post("/analyze")
    async def analyze():
        return {"ok": True}

    @tiny.get("/health")
    async def health():
        return {"ok": True}

    maybe_add_rate_limit(tiny, limit)
    return TestClient(tiny)


class TestRateLimit:
    def test_expensive_endpoint_throttled(self):
        client = _tiny_app_with_limit(3)
        for _ in range(3):
            assert client.post("/analyze").status_code == 200
        resp = client.post("/analyze")
        assert resp.status_code == 429
        assert "Retry-After" in resp.headers

    def test_cheap_endpoints_never_throttled(self):
        client = _tiny_app_with_limit(1)
        for _ in range(5):
            assert client.get("/health").status_code == 200

    def test_zero_disables_limiter(self):
        client = _tiny_app_with_limit(0)
        for _ in range(5):
            assert client.post("/analyze").status_code == 200
