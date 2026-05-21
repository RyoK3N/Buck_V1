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

    def test_get_returns_none_initially(self):
        import agent_scripts.tools as tools_mod
        old = tools_mod._stock_data
        try:
            tools_mod._stock_data = None
            assert get_stock_data() is None
        finally:
            tools_mod._stock_data = old


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
