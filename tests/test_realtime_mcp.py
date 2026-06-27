"""
Tests for the realtime MCP tools that drive / open the Buck web app.
Network-free: we point BUCK_API_URL at an unreachable port and assert the
fallback / error behaviour, and stub webbrowser for open_buck_ui.
"""

from __future__ import annotations

import asyncio

import pytest

from mcp_server import tools as T
from mcp_server.tools import BuckAppUnavailable


@pytest.fixture
def offline_api(monkeypatch):
    # Unreachable address → requests fails fast → BuckAppUnavailable / fallback.
    monkeypatch.setenv("BUCK_API_URL", "http://127.0.0.1:9")
    monkeypatch.setenv("BUCK_UI_URL", "http://localhost:5173")


def test_rt_status_falls_back_when_app_down(offline_api):
    """rt_session_status should fall back to in-process state (no live session)."""
    out = asyncio.run(T.rt_session_status("NOPE.NS"))
    assert out.get("active") is False


def test_rt_start_requires_running_app(offline_api):
    """Starting a session needs the web app; surface a clear BuckAppUnavailable."""
    with pytest.raises(BuckAppUnavailable):
        asyncio.run(T.rt_start_session(symbol="INFY.NS", model_id="m", open_ui=False))


def test_open_buck_ui_builds_deeplink(monkeypatch):
    monkeypatch.setenv("BUCK_UI_URL", "http://localhost:5173")
    captured = {}

    def fake_open(url):
        captured["url"] = url
        return True

    import webbrowser
    monkeypatch.setattr(webbrowser, "open", fake_open)

    out = asyncio.run(T.open_buck_ui(tab="realtime", symbol="INFY.NS", autostart=True))
    assert out["opened"] is True
    assert out["url"].startswith("http://localhost:5173/?")
    assert "tab=realtime" in out["url"]
    assert "symbol=INFY.NS" in out["url"]
    assert "autostart=1" in out["url"]
    assert captured["url"] == out["url"]


def test_visualize_session_handles_app_down(offline_api):
    """visualize_session must degrade gracefully (empty spec) when nothing runs."""
    out = asyncio.run(T.visualize_session(symbol="NOPE.NS", chart="equity_curve"))
    assert out["active"] is False
    assert out["spec"]["chart"] == "equity_curve"
