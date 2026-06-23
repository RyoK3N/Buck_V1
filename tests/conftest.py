"""Shared pytest fixtures for the Buck test suite."""

from __future__ import annotations

import os
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from agent_scripts.interfaces import AnalysisResult, NewsData, StockData


# ---------------------------------------------------------------------------
# Auto-use fixture: ensure config.py never calls sys.exit(1) during tests
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _patch_settings(monkeypatch):
    """Set a dummy OPENAI_API_KEY and clear the LRU cache so
    ``get_settings()`` doesn't ``sys.exit(1)`` when the real key is absent."""
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("LOG_LEVEL", "WARNING")

    from agent_scripts.config import get_settings, get_logger
    get_settings.cache_clear()
    get_logger.cache_clear()

    yield

    get_settings.cache_clear()
    get_logger.cache_clear()


# ---------------------------------------------------------------------------
# Sample OHLCV DataFrame
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_ohlcv_df() -> pd.DataFrame:
    """200-row OHLCV DataFrame with a realistic random walk."""
    np.random.seed(42)
    n = 200
    close = 100.0 + np.cumsum(np.random.randn(n) * 0.5)
    df = pd.DataFrame(
        {
            "Open": close - np.abs(np.random.randn(n) * 0.25),
            "High": close + np.abs(np.random.randn(n) * 0.5),
            "Low": close - np.abs(np.random.randn(n) * 0.5),
            "Close": close,
            "Volume": np.random.randint(1_000, 10_000, n).astype(float),
        },
        index=pd.date_range("2024-01-01", periods=n, freq="h"),
    )
    return df


# ---------------------------------------------------------------------------
# StockData TypedDict wrapping the DataFrame
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_stock_data(sample_ohlcv_df) -> StockData:
    return StockData(
        symbol="TEST.NS",
        data=sample_ohlcv_df,
        interval="1h",
        start_date="2024-01-01",
        end_date="2024-01-09",
    )


# ---------------------------------------------------------------------------
# NewsData TypedDict with mixed-sentiment headlines
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_news_data() -> NewsData:
    now = datetime.now()
    news_items = [
        {
            "title": "Company reports record profit and strong revenue growth",
            "summary": "Quarterly earnings beat expectations with a 20% surge in revenue.",
            "source": "The Economic Times",
            "pub_date": (now - timedelta(hours=6)).isoformat(),
            "topics": ["Financial Results"],
        },
        {
            "title": "Stock rallies on positive earnings momentum",
            "summary": "Shares surge as market reacts to strong quarterly results.",
            "source": "Business Standard",
            "pub_date": (now - timedelta(hours=12)).isoformat(),
            "topics": ["Financial Results", "Capital Investment"],
        },
        {
            "title": "Company faces major loss and debt restructuring",
            "summary": "Analysts warn of potential layoff and decline in operations.",
            "source": "CNBC TV18",
            "pub_date": (now - timedelta(days=1)).isoformat(),
            "topics": ["Regulatory and Legal"],
        },
        {
            "title": "New technology partnership announced for innovation",
            "summary": "Strategic deal to expand into renewable energy markets.",
            "source": "Moneycontrol",
            "pub_date": (now - timedelta(days=2)).isoformat(),
            "topics": ["New Offerings", "Artificial Intelligence"],
        },
        {
            "title": "Market outlook remains cautious amid weak global cues",
            "summary": "Investors adopt a bearish stance on the index.",
            "source": "Zee Business",
            "pub_date": (now - timedelta(days=3)).isoformat(),
            "topics": ["Big Data/Analytics"],
        },
    ]
    return NewsData(
        symbol="TEST.NS",
        news=news_items,
        source="test",
        retrieved_at=now,
    )


# ---------------------------------------------------------------------------
# Sample AnalysisResult (for predictor tests)
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_analysis_result(sample_ohlcv_df) -> AnalysisResult:
    return AnalysisResult(
        symbol="TEST.NS",
        analysis_type="technical_analysis",
        data={
            "symbol": "TEST.NS",
            "data_info": {
                "interval": "1h",
                "start_date": "2024-01-01",
                "end_date": "2024-01-09",
                "data_points": len(sample_ohlcv_df),
            },
            "tools_used": ["rsi", "macd"],
            "tools_results": {
                "rsi": {
                    "signal": "HOLD",
                    "strength": 0.5,
                    "rsi": 55.0,
                    "condition": "neutral",
                },
                "macd": {
                    "signal": "BUY",
                    "strength": 0.7,
                    "macd": 0.5,
                    "signal_line": 0.3,
                    "histogram": 0.2,
                },
            },
            "summary": {
                "overall_signal": "BUY",
                "signal_strength": 0.6,
                "signals_breakdown": {"BUY": 0.7, "SELL": 0.0, "HOLD": 0.5},
                "key_metrics": {
                    "rsi": 55.0,
                    "rsi_condition": "neutral",
                    "macd": 0.5,
                    "macd_signal": 0.3,
                    "macd_histogram": 0.2,
                },
                "tools_processed": 2,
                "tools_failed": 0,
            },
        },
        timestamp=datetime.now(),
        confidence=0.75,
    )
