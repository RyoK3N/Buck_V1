"""
Replay-vs-backtest parity: the env-backed replay engine must produce EXACTLY the
same PnL as stepping a TradingEnvironment directly (the /rl/predict backtest path),
proving the old replay math bugs (half transaction cost, hardcoded unrealized PnL)
are gone. Network-free: data fetch and agent load are monkeypatched.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tools.rl.env import TradingEnvironment
from tools.rl.ppo_continuous import PPOContinuousAgent
from realtime.sim import IntradaySimulator


def _ohlc(n: int = 120, start: float = 100.0) -> pd.DataFrame:
    idx = pd.date_range("2023-01-01", periods=n, freq="D")
    rng = np.random.default_rng(11)
    close = np.maximum(1.0, start + np.cumsum(rng.normal(0, 1.0, n)))
    return pd.DataFrame(
        {
            "Open": close * (1 + rng.normal(0, 0.004, n)),
            "High": close * (1 + np.abs(rng.normal(0, 0.01, n))),
            "Low": close * (1 - np.abs(rng.normal(0, 0.01, n))),
            "Close": close,
            "Volume": rng.integers(1e5, 1e6, n).astype(float),
        },
        index=idx,
    )


def _backtest_equity(df: pd.DataFrame, agent: PPOContinuousAgent, capital: float):
    """Mirror UI/backend/routes.py:_rl_predict_ppo_continuous stepping exactly."""
    env = TradingEnvironment(df, initial_capital=capital)
    state = env.reset()
    agent.reset_window()
    equity = []
    while True:
        action, _info = agent.act(state, eval_mode=True)
        if env.idx >= env.n - 1:
            break
        state, _reward, done, si = env.step(action)
        equity.append(si.portfolio_value)
        if done:
            break
    env.close_position()
    return equity, env.portfolio_value()


def test_replay_matches_backtest(monkeypatch):
    df = _ohlc(120)
    capital = 100_000.0
    agent = PPOContinuousAgent()  # fresh weights; act(eval) is deterministic (z=mu)

    # Path A — direct backtest.
    eq_a, final_a = _backtest_equity(df, agent, capital)

    # Path B — env-backed replay engine (instant, no online updates).
    import tools.rl.rl_tool as rt
    monkeypatch.setattr(rt, "fetch_historical_data", lambda *a, **k: df.copy())
    monkeypatch.setattr(IntradaySimulator, "_load_agent", lambda self: agent)
    monkeypatch.setattr(IntradaySimulator, "_finalise", lambda self, steps: None)

    sim = IntradaySimulator(
        symbol="HDFCBANK.NS", model_id="x", interval="1d", replay=True,
        replay_start="2023-01-01", replay_end="2023-06-01",
        capital=capital, speed=1000.0, online_update_every=0,
    )
    sim.run()
    eq_b = [s["equity"] for s in sim.state.steps]

    # Same number of steps and matching equity at every bar (B is rounded to 2dp).
    assert len(eq_a) == len(eq_b), f"step count {len(eq_a)} != {len(eq_b)}"
    for a, b in zip(eq_a, eq_b):
        assert abs(round(a, 2) - b) <= 0.01, f"equity mismatch {a} vs {b}"
    # Final equity matches to the cent.
    assert abs(final_a - sim.equity) <= 0.01
    # Online updates were OFF → pure inference.
    assert sim.state.n_updates == 0


def test_replay_requires_min_data(monkeypatch):
    import tools.rl.rl_tool as rt
    monkeypatch.setattr(rt, "fetch_historical_data", lambda *a, **k: None)
    monkeypatch.setattr(IntradaySimulator, "_load_agent", lambda self: PPOContinuousAgent())
    monkeypatch.setattr(IntradaySimulator, "_finalise", lambda self, steps: None)
    sim = IntradaySimulator(symbol="X.NS", model_id="x", replay=True,
                            replay_start="2023-01-01", replay_end="2023-02-01")
    with pytest.raises(RuntimeError):
        sim._run_replay([])


def test_replay_delay_and_bar_period():
    s = IntradaySimulator(symbol="X.NS", model_id="m", interval="1h", replay=True, speed=60)
    assert s._bar_period_seconds() == 3600.0
    assert s._replay_delay() == 60.0           # 3600 / 60
    assert IntradaySimulator(symbol="X.NS", model_id="m", interval="1h",
                             replay=True, speed=1000)._replay_delay() == 0.0  # instant
