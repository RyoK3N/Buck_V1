"""
Detailed tests for EVERY Buck MCP tool.

Strategy
--------
* Meta tests (parametrized over all 28 tools) verify the MCP wiring: registry/impl
  parity, signature preservation (no opaque **kwargs schema), and that the live
  FastMCP server exposes real per-parameter input schemas.
* Functional tests call each tool's implementation directly:
    - network-free tools run for real;
    - network/heavy tools have their data boundary monkeypatched with synthetic
      OHLC so the tool's real logic runs without hitting the internet;
    - RL train/predict run a real 1-episode loop on synthetic data (this is the
      path that regressed with the package-shadowing import bug).
"""

from __future__ import annotations

import asyncio
import inspect
import uuid

import numpy as np
import pandas as pd
import pytest

from accuracy import db as accuracy_db
from accuracy import repository
from mcp_server.registry import BUCK_TOOLS, BUCK_TOOLS_BY_NAME
from mcp_server import tools as T
from mcp_server.tools import _IMPLS, get_mcp_wrapped, get_wrapped, BuckAppUnavailable


# ── helpers / fixtures ────────────────────────────────────────────────────────

def run(coro):
    return asyncio.run(coro)


def _ohlc(n: int = 150, start: float = 100.0) -> pd.DataFrame:
    """Deterministic synthetic OHLC frame with the columns the RL/viz code expects."""
    idx = pd.date_range("2023-01-01", periods=n, freq="D")
    rng = np.random.default_rng(7)
    close = np.maximum(1.0, start + np.cumsum(rng.normal(0, 1.0, n)))
    high = close * (1 + np.abs(rng.normal(0, 0.01, n)))
    low = close * (1 - np.abs(rng.normal(0, 0.01, n)))
    open_ = close * (1 + rng.normal(0, 0.005, n))
    vol = rng.integers(100_000, 1_000_000, n).astype(float)
    return pd.DataFrame(
        {"Open": open_, "High": high, "Low": low, "Close": close, "Volume": vol}, index=idx
    )


@pytest.fixture
def fresh_db(tmp_path):
    accuracy_db.init_db(tmp_path / "acc.db")
    yield
    accuracy_db._DB_PATH = None


@pytest.fixture
def synth_hist(monkeypatch):
    """Monkeypatch historical-data fetch (used by all RL routes) with synthetic OHLC."""
    df = _ohlc(160)
    import tools.rl.rl_tool as rt
    monkeypatch.setattr(rt, "fetch_historical_data", lambda *a, **k: df.copy())
    return df


@pytest.fixture
def rl_cleanup():
    """Yield a unique model-id prefix and remove any weight/session files created."""
    import tools.rl.ppo_continuous as ppo
    import tools.rl.dqn_agent as dqn
    import tools.rl.sessions as sess

    prefix = f"pytest_{uuid.uuid4().hex[:8]}"
    yield prefix
    for d in {ppo.WEIGHTS_DIR, dqn.WEIGHTS_DIR}:
        for p in d.glob(f"*{prefix}*"):
            p.unlink(missing_ok=True)
    if sess.SESSIONS_DIR.exists():
        for p in sess.SESSIONS_DIR.glob(f"*{prefix}*"):
            p.unlink(missing_ok=True)


# ════════════════════════════════════════════════════════════════════════════
# META TESTS — over every tool
# ════════════════════════════════════════════════════════════════════════════

ALL_TOOLS = sorted(_IMPLS.keys())


def test_registry_and_impls_match():
    assert {t["name"] for t in BUCK_TOOLS} == set(_IMPLS)


@pytest.mark.parametrize("name", ALL_TOOLS)
def test_mcp_wrapper_preserves_signature(name):
    """The MCP wrapper must expose the impl's real params (not a single kwargs)."""
    wrapped = get_mcp_wrapped(name)
    wparams = set(inspect.signature(wrapped).parameters)
    iparams = set(inspect.signature(_IMPLS[name]).parameters)
    assert wparams == iparams, f"{name}: {wparams} != {iparams}"
    assert "kwargs" not in wparams or "kwargs" in iparams


@pytest.mark.parametrize("name", ALL_TOOLS)
def test_registry_schema_properties_subset_of_params(name):
    """Every declared input_schema property must be a real impl parameter."""
    props = set(BUCK_TOOLS_BY_NAME[name]["input_schema"].get("properties", {}))
    params = set(inspect.signature(_IMPLS[name]).parameters)
    assert props <= params, f"{name}: schema props {props - params} not in impl params"


def test_live_server_exposes_real_schemas():
    """Build the FastMCP server and confirm no tool collapses to a `kwargs` schema."""
    import mcp_server.server as server
    tools = {t.name: t for t in server.mcp._tool_manager.list_tools()}
    assert set(tools) == set(_IMPLS)
    for name, tool in tools.items():
        props = set((tool.parameters or {}).get("properties", {}))
        impl_params = set(inspect.signature(_IMPLS[name]).parameters)
        # tools with no params legitimately have empty properties
        if impl_params:
            assert props == impl_params, f"{name}: server schema {props} != {impl_params}"
        assert props != {"kwargs"}, f"{name}: schema collapsed to kwargs"


# ════════════════════════════════════════════════════════════════════════════
# DISCOVERY / LIST TOOLS (network-free)
# ════════════════════════════════════════════════════════════════════════════

def test_list_tools_registry():
    out = run(T.list_tools_registry())
    assert "categories" in out


def test_list_available_intervals():
    out = run(T.list_available_intervals())
    assert "1d" in out["intervals"] and "1m" in out["intervals"]


def test_list_chart_types():
    out = run(T.list_chart_types())
    assert isinstance(out["chart_types"], list) and out["chart_types"]


def test_list_d3_chart_types():
    out = run(T.list_d3_chart_types())
    ids = {c["id"] for c in out["chart_types"]}
    assert {"reward_curve", "equity_curve"} <= ids


def test_list_rl_models():
    out = run(T.list_rl_models())
    assert "models" in out and isinstance(out["models"], list)


def test_list_training_sessions():
    out = run(T.list_training_sessions(limit=5))
    assert "sessions" in out and isinstance(out["sessions"], list)


# ════════════════════════════════════════════════════════════════════════════
# HEADROOM / CONTEXT ENGINEERING
# ════════════════════════════════════════════════════════════════════════════

def test_headroom_stats_and_reset():
    stats = run(T.headroom_stats())
    assert "usage" in stats and "cache" in stats
    out = run(T.headroom_reset())
    assert out["status"] == "reset"
    assert out["usage"]["calls"] == 0


# ════════════════════════════════════════════════════════════════════════════
# ACCURACY (DB-backed, network-free)
# ════════════════════════════════════════════════════════════════════════════

def _seed_prediction():
    repository.record_prediction(
        symbol="HDFCBANK.NS", model="claude",
        forecast={"date": "2026-06-25", "open": 1, "high": 2, "low": 0.5, "close": 1.5, "confidence": 0.5},
    )


def test_get_prediction_accuracy(fresh_db):
    out = run(T.get_prediction_accuracy())
    assert "summaries" in out


def test_list_recent_predictions(fresh_db):
    _seed_prediction()
    out = run(T.list_recent_predictions(limit=10))
    assert any(r["symbol"] == "HDFCBANK.NS" for r in out["predictions"])


def test_compare_predictions_vs_actual(fresh_db):
    _seed_prediction()
    out = run(T.compare_predictions_vs_actual(symbol="HDFCBANK.NS"))
    assert out["symbol"] == "HDFCBANK.NS" and "series" in out


# ════════════════════════════════════════════════════════════════════════════
# VISUALIZATION
# ════════════════════════════════════════════════════════════════════════════

def test_visualize_accuracy(fresh_db):
    out = run(T.visualize_accuracy(window_days=30, metric="mae"))
    assert out["spec"]["spec_version"] == "d3-buck/1"


def test_visualize_predictions(fresh_db):
    out = run(T.visualize_predictions(symbol="HDFCBANK.NS"))
    assert out["spec"]["spec_version"] == "d3-buck/1"


def test_visualize_session_no_session_offline(monkeypatch):
    monkeypatch.setenv("BUCK_API_URL", "http://127.0.0.1:9")
    out = run(T.visualize_session(chart="equity_curve"))
    assert out["active"] is False
    assert out["spec"]["chart"] == "equity_curve"


def test_visualize_market_chart(monkeypatch):
    """visualize builds a Plotly figure from (mocked) market data."""
    import UI.backend.visualizer as viz

    async def fake_fetch(**kwargs):
        return _ohlc(120)

    monkeypatch.setattr(viz, "fetch_df", fake_fetch)
    out = run(T.visualize(symbol="HDFCBANK.NS", start_date="2023-01-01",
                          end_date="2023-06-01", chart_type="price_ma"))
    assert out["chart_type"] == "price_ma"
    assert "chart" in out and isinstance(out["chart"], dict)


def test_visualize_compare(monkeypatch):
    import UI.backend.visualizer as viz

    async def fake_fetch(**kwargs):
        return _ohlc(60)

    monkeypatch.setattr(viz, "fetch_df", fake_fetch)
    out = run(T.visualize_compare(symbols=["TCS.NS", "INFY.NS"],
                                  start_date="2023-01-01", end_date="2023-03-01"))
    assert out["symbols"] == ["TCS.NS", "INFY.NS"]
    assert out["spec"]["mark"] == "multiline"
    assert "TCS.NS" in out["spec"]["meta"]["symbols"]


# ════════════════════════════════════════════════════════════════════════════
# ANALYSIS (mock the agent factory)
# ════════════════════════════════════════════════════════════════════════════

class _FakeBuck:
    async def analyze_and_predict(self, **kwargs):
        return {"symbol": kwargs.get("symbol"), "forecast": {"close": 123.0, "confidence": 0.5}}

    async def batch_analyze(self, **kwargs):
        return {"results": [{"symbol": s} for s in kwargs.get("symbols", [])]}


@pytest.fixture
def fake_buck(monkeypatch):
    import agent_scripts.buck as buck
    monkeypatch.setattr(buck.BuckFactory, "create_production_agent",
                        classmethod(lambda cls, **kw: _FakeBuck()))


def test_single_analyze(fake_buck):
    out = run(T.single_analyze(symbol="HDFCBANK.NS", start_date="2024-01-01", end_date="2024-03-01"))
    assert out["symbol"] == "HDFCBANK.NS" and "forecast" in out


def test_batch_analyze(fake_buck):
    out = run(T.batch_analyze(symbols=["TCS.NS", "INFY.NS"], start_date="2024-01-01", end_date="2024-03-01"))
    assert len(out["results"]) == 2


# ════════════════════════════════════════════════════════════════════════════
# REINFORCEMENT LEARNING (real 1-episode loop on synthetic data)
# ════════════════════════════════════════════════════════════════════════════

def test_rl_train_ppo_continuous(synth_hist, rl_cleanup):
    mid = f"{rl_cleanup}_ppo"
    out = run(T.rl_train(symbol="HDFCBANK.NS", start_date="2023-01-01", end_date="2023-09-01",
                         model_id=mid, algorithm="ppo_continuous", episodes=1, interval="1d"))
    assert out["status"] == "trained"
    assert out["algorithm"] == "ppo_continuous"
    assert out["model_id"] == mid


def test_rl_train_dqn(synth_hist, rl_cleanup):
    mid = f"{rl_cleanup}_dqn"
    out = run(T.rl_train(symbol="HDFCBANK.NS", start_date="2023-01-01", end_date="2023-09-01",
                         model_id=mid, algorithm="dqn", episodes=1, interval="1d"))
    assert out.get("status") in {"trained", "completed"} or out.get("model_id") == mid


def test_rl_predict_after_train(synth_hist, rl_cleanup):
    mid = f"{rl_cleanup}_ppo2"
    run(T.rl_train(symbol="HDFCBANK.NS", start_date="2023-01-01", end_date="2023-09-01",
                   model_id=mid, algorithm="ppo_continuous", episodes=1, interval="1d"))
    out = run(T.rl_predict(symbol="HDFCBANK.NS", start_date="2023-09-01", end_date="2023-12-01",
                           model_id=mid, interval="1d"))
    assert isinstance(out, dict)
    assert out.get("model_id") == mid or "signals" in out or "equity_curve" in out


def test_rl_simulate_mocked(monkeypatch):
    """rl_simulate: stub live data + a fake DQN agent to test the wiring."""
    import tools.rl.rl_tool as rt
    import tools.rl.dqn_agent as dqn

    monkeypatch.setattr(rt, "fetch_live_data", lambda *a, **k: {"price": 1500.0, "timestamp": "now"})

    class _Agent:
        algorithm = "dqn"
        def act(self, *a, **k):
            return 1  # BUY

    monkeypatch.setattr(dqn, "load_agent", lambda *a, **k: _Agent())
    out = run(T.rl_simulate(symbol="HDFCBANK.NS", model_id="whatever", interval="1m"))
    assert out["symbol"] == "HDFCBANK.NS"
    assert "action" in out


def test_rl_ensemble_predict_mocked(monkeypatch):
    import tools.rl.ensemble as ens
    monkeypatch.setattr(ens, "ensemble_predict",
                        lambda **kw: {"symbol": kw["symbol"], "aggregate_signal": "BUY", "models": kw["models"]})
    out = run(T.rl_ensemble_predict(symbol="HDFCBANK.NS", start_date="2023-01-01", end_date="2023-06-01",
                                    models=[{"model_id": "m1", "interval": "1d"}]))
    assert out["symbol"] == "HDFCBANK.NS" and out["aggregate_signal"] == "BUY"


# ════════════════════════════════════════════════════════════════════════════
# REALTIME run-control + UI (offline / stubbed)
# ════════════════════════════════════════════════════════════════════════════

@pytest.fixture
def offline_api(monkeypatch):
    monkeypatch.setenv("BUCK_API_URL", "http://127.0.0.1:9")
    monkeypatch.setenv("BUCK_UI_URL", "http://localhost:5173")


def test_rt_session_status_offline(offline_api):
    out = run(T.rt_session_status("X.NS"))
    assert out.get("active") is False


def test_rt_session_history_offline(offline_api):
    out = run(T.rt_session_history("X.NS"))
    assert "steps" in out


def test_rt_start_session_requires_app(offline_api):
    with pytest.raises(BuckAppUnavailable):
        run(T.rt_start_session(symbol="INFY.NS", model_id="m", open_ui=False))


def test_rt_stop_session_requires_app(offline_api):
    with pytest.raises(BuckAppUnavailable):
        run(T.rt_stop_session(symbol="INFY.NS"))


def test_open_buck_ui(monkeypatch):
    monkeypatch.setenv("BUCK_UI_URL", "http://localhost:5173")
    import webbrowser
    monkeypatch.setattr(webbrowser, "open", lambda url: True)
    out = run(T.open_buck_ui(tab="realtime", symbol="INFY.NS", autostart=True))
    assert out["opened"] is True
    assert "tab=realtime" in out["url"] and "symbol=INFY.NS" in out["url"]


# ── End-to-end through the wrapper (envelope + compact MCP string) ─────────────

def test_dispatch_envelope_and_mcp_string():
    env = run(get_wrapped("list_available_intervals")(**{}))
    assert "_headroom" in env and "data" in env
    s = run(get_mcp_wrapped("list_available_intervals")(**{}))
    assert isinstance(s, str) and "_headroom" not in s and "intervals" in s
