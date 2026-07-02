"""Tests for the walk-forward validation harness (accuracy/backtest.py)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from accuracy.backtest import FAST_TOOLS, walk_forward


def _ohlcv(close: np.ndarray) -> pd.DataFrame:
    n = len(close)
    rng = np.random.RandomState(7)
    return pd.DataFrame(
        {
            "Open": close - 0.1,
            "High": close + 0.2,
            "Low": close - 0.2,
            "Close": close,
            "Volume": rng.randint(1_000, 10_000, n).astype(float),
        },
        index=pd.date_range("2024-01-01", periods=n, freq="D"),
    )


class TestWalkForward:
    def test_report_shape_and_anchor_count(self):
        rng = np.random.RandomState(0)
        df = _ohlcv(100.0 + np.cumsum(rng.randn(100) * 0.5))
        report = walk_forward(df, symbol="TEST.NS", min_history=60, step=5, horizon=1)

        summary = report["summary"]
        expected_anchors = len(range(60, len(df) - 1, 5))
        assert summary["anchors"] == expected_anchors
        assert len(report["records"]) == expected_anchors
        assert set(summary["by_signal"]) == {"BUY", "SELL", "HOLD"}

        for rec in report["records"]:
            assert rec["signal"] in ("BUY", "SELL", "HOLD")
            assert 0.0 <= rec["confidence"] <= 1.0
            # HOLD is not scored; BUY/SELL are.
            assert (rec["hit"] is None) == (rec["signal"] == "HOLD")

    def test_signals_scored_against_forward_returns(self):
        # Steady uptrend: any BUY anchors must be counted as hits, any SELL
        # anchors as misses — the scorer must compare against the *forward*
        # bar, not the anchor bar.
        df = _ohlcv(100.0 + np.arange(120) * 1.0)
        report = walk_forward(df, symbol="UP.NS", min_history=60, horizon=1)

        for rec in report["records"]:
            if rec["signal"] == "BUY":
                assert rec["hit"] is True
            elif rec["signal"] == "SELL":
                assert rec["hit"] is False
            assert rec["forward_return"] > 0

    def test_too_little_data_raises(self):
        df = _ohlcv(100.0 + np.arange(30) * 0.1)
        with pytest.raises(ValueError):
            walk_forward(df, min_history=60)

    def test_fast_tools_exist_in_registry(self):
        from agent_scripts.tools import ToolFactory
        available = set(ToolFactory.get_available_tools())
        assert set(FAST_TOOLS) <= available
