"""Tests for agent_scripts.tools — BaseTool, ToolFactory, and individual tools."""

from __future__ import annotations

import pandas as pd
import pytest

from agent_scripts.tools import BaseTool, ToolFactory, get_stock_data, set_stock_data


# ---------------------------------------------------------------------------
# ToolFactory discovery
# ---------------------------------------------------------------------------

class TestToolFactory:
    def test_discovers_tools(self):
        """ToolFactory should discover all implemented tools."""
        tools = ToolFactory.get_available_tools()
        assert isinstance(tools, list)
        assert len(tools) >= 6  # at least the 6 maths tools

        expected = {"rsi", "macd", "moving_average", "obv", "candlestick_patterns", "support_resistance"}
        assert expected.issubset(set(tools))

    def test_create_tool(self):
        """create_tool should return a BaseTool instance."""
        tool = ToolFactory.create_tool("rsi")
        assert isinstance(tool, BaseTool)
        assert tool.name == "rsi"

    def test_unknown_tool_raises(self):
        """create_tool should raise ValueError for an unknown name."""
        with pytest.raises(ValueError, match="Unknown tool"):
            ToolFactory.create_tool("nonexistent_tool_xyz")

    def test_create_all_tools(self):
        """create_all_tools should return a dict of all tools."""
        tools = ToolFactory.create_all_tools()
        assert isinstance(tools, dict)
        assert len(tools) >= 6
        for name, tool in tools.items():
            assert isinstance(tool, BaseTool)
            assert tool.name == name

    def test_get_langchain_tools(self):
        """get_langchain_tools should return callable tool functions."""
        lc_tools = ToolFactory.get_langchain_tools()
        assert isinstance(lc_tools, list)
        assert len(lc_tools) >= 1
        # Each langchain tool should have a name attribute
        for t in lc_tools:
            assert hasattr(t, "name")


# ---------------------------------------------------------------------------
# Shared data context (set_stock_data / get_stock_data)
# ---------------------------------------------------------------------------

class TestStockDataContext:
    def test_set_get_round_trip(self, sample_ohlcv_df):
        set_stock_data(sample_ohlcv_df)
        retrieved = get_stock_data()
        assert retrieved is not None
        pd.testing.assert_frame_equal(retrieved, sample_ohlcv_df)

    def test_get_returns_none_in_fresh_context(self):
        # The context is a ContextVar: a new thread starts with an empty
        # context, so it must see the default (None) — not whatever another
        # thread set.
        import threading

        seen: dict = {}
        t = threading.Thread(target=lambda: seen.setdefault("value", get_stock_data()))
        t.start()
        t.join()
        assert seen["value"] is None

    def test_contexts_are_isolated_across_threads(self, sample_ohlcv_df):
        # Two threads set different DataFrames concurrently; each must read
        # back its own. With a plain module global (the old implementation)
        # the last writer would win in both threads.
        import threading

        df_a = sample_ohlcv_df
        df_b = sample_ohlcv_df * 2.0
        barrier = threading.Barrier(2)
        results: dict = {}

        def worker(name, df):
            set_stock_data(df)
            barrier.wait()  # both threads have written before either reads
            results[name] = get_stock_data()

        threads = [
            threading.Thread(target=worker, args=("a", df_a)),
            threading.Thread(target=worker, args=("b", df_b)),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        pd.testing.assert_frame_equal(results["a"], df_a)
        pd.testing.assert_frame_equal(results["b"], df_b)


# ---------------------------------------------------------------------------
# Individual tool execution
# ---------------------------------------------------------------------------

def _run_tool(name: str, df: pd.DataFrame) -> dict:
    tool = ToolFactory.create_tool(name)
    return tool.execute(df)


class TestRSI:
    def test_execute(self, sample_ohlcv_df):
        result = _run_tool("rsi", sample_ohlcv_df)
        assert "signal" in result
        assert result["signal"] in ("BUY", "SELL", "HOLD")
        assert "strength" in result
        assert 0.0 <= result["strength"] <= 1.0
        assert "rsi" in result


class TestMACD:
    def test_execute(self, sample_ohlcv_df):
        result = _run_tool("macd", sample_ohlcv_df)
        assert result["signal"] in ("BUY", "SELL", "HOLD")
        assert "strength" in result
        assert "macd" in result
        assert "histogram" in result


class TestMovingAverage:
    def test_execute(self, sample_ohlcv_df):
        result = _run_tool("moving_average", sample_ohlcv_df)
        assert result["signal"] in ("BUY", "SELL", "HOLD")
        assert "strength" in result
        assert "short_ma" in result
        assert "long_ma" in result


class TestOBV:
    def test_execute(self, sample_ohlcv_df):
        result = _run_tool("obv", sample_ohlcv_df)
        assert result["signal"] in ("BUY", "SELL", "HOLD")
        assert "strength" in result
        assert "obv" in result


class TestCandlestickPatterns:
    def test_execute(self, sample_ohlcv_df):
        result = _run_tool("candlestick_patterns", sample_ohlcv_df)
        assert result["signal"] in ("BUY", "SELL", "HOLD")
        assert "strength" in result
        assert "pattern_count" in result


class TestSupportResistance:
    def test_execute(self, sample_ohlcv_df):
        result = _run_tool("support_resistance", sample_ohlcv_df)
        assert result["signal"] in ("BUY", "SELL", "HOLD")
        assert "strength" in result
        assert "nearest_support" in result
        assert "nearest_resistance" in result


# ---------------------------------------------------------------------------
# Tool with invalid data
# ---------------------------------------------------------------------------

class TestToolInvalidData:
    def test_missing_columns_raises(self):
        """A tool given a DataFrame without OHLCV columns should raise or handle gracefully."""
        bad_df = pd.DataFrame({"A": [1, 2, 3]})
        tool = ToolFactory.create_tool("rsi")
        with pytest.raises((ValueError, KeyError)):
            tool.execute(bad_df)
