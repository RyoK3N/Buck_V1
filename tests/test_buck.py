"""Tests for agent_scripts.buck — Buck orchestrator and BuckFactory."""

from __future__ import annotations

import asyncio
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pandas as pd
import pytest

from agent_scripts.analyzers import TechnicalAnalyzer
from agent_scripts.buck import Buck, BuckFactory
from agent_scripts.interfaces import (
    AnalysisResult,
    BuckConfig,
    Forecast,
    StockData,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mock_config() -> BuckConfig:
    return BuckConfig(
        openai_api_key="test-key",
        openai_base_url=None,
        chat_model="gpt-4o",
        temperature=0.0,
        max_tokens=1500,
        news_items=8,
        log_level="WARNING",
    )


def _mock_data_provider(stock_data):
    provider = AsyncMock()
    provider.get_stock_data.return_value = stock_data
    provider.get_news_data.return_value = None
    return provider


def _mock_analyzer():
    analyzer = MagicMock()
    analyzer.analysis_type = "composite_analysis"
    analyzer.analyze.return_value = AnalysisResult(
        symbol="TEST.NS",
        analysis_type="composite_analysis",
        data={"summary": {"overall_signal": "BUY"}},
        timestamp=datetime.now(),
        confidence=0.75,
    )
    return analyzer


def _mock_predictor():
    predictor = AsyncMock()
    predictor.predict.return_value = Forecast(
        date="2024-06-10",
        open=100.0,
        high=105.0,
        low=98.0,
        close=103.0,
        confidence=0.8,
        reasoning="Test forecast.",
    )
    return predictor


# ---------------------------------------------------------------------------
# Buck initialisation
# ---------------------------------------------------------------------------

class TestBuckInit:
    def test_init_with_defaults(self):
        """Buck() should initialise with default components."""
        buck = Buck()
        assert buck.config is not None
        assert buck.data_provider is not None
        assert buck.analyzer is not None
        assert buck.predictor is not None

    def test_init_with_custom_components(self, sample_stock_data):
        buck = Buck(
            config=_mock_config(),
            data_provider=_mock_data_provider(sample_stock_data),
            analyzer=_mock_analyzer(),
            predictor=_mock_predictor(),
        )
        assert buck.config["chat_model"] == "gpt-4o"


# ---------------------------------------------------------------------------
# Cache operations
# ---------------------------------------------------------------------------

class TestBuckCache:
    def test_set_get_clear(self, sample_stock_data):
        buck = Buck(
            config=_mock_config(),
            data_provider=_mock_data_provider(sample_stock_data),
            analyzer=_mock_analyzer(),
            predictor=_mock_predictor(),
        )

        # Initially empty
        assert buck.get_cached_analysis("TEST.NS") is None
        assert buck.get_cached_forecast("TEST.NS") is None

        # Manually populate cache
        result = AnalysisResult(
            symbol="TEST.NS",
            analysis_type="test",
            data={},
            timestamp=datetime.now(),
            confidence=0.5,
        )
        buck._analysis_cache["TEST.NS"] = [result]

        forecast = Forecast(
            date="2024-06-10", open=100.0, high=105.0, low=98.0,
            close=103.0, confidence=0.8, reasoning="cached",
        )
        buck._forecast_cache["TEST.NS"] = forecast

        assert buck.get_cached_analysis("TEST.NS") is not None
        assert buck.get_cached_forecast("TEST.NS") is not None

        buck.clear_cache()
        assert buck.get_cached_analysis("TEST.NS") is None
        assert buck.get_cached_forecast("TEST.NS") is None


# ---------------------------------------------------------------------------
# Confidence calculation
# ---------------------------------------------------------------------------

class TestBuckConfidence:
    def test_calculate_confidence(self, sample_stock_data):
        buck = Buck(
            config=_mock_config(),
            data_provider=_mock_data_provider(sample_stock_data),
            analyzer=_mock_analyzer(),
            predictor=_mock_predictor(),
        )
        results = [
            AnalysisResult(symbol="X", analysis_type="a", data={},
                           timestamp=datetime.now(), confidence=0.6),
            AnalysisResult(symbol="X", analysis_type="b", data={},
                           timestamp=datetime.now(), confidence=0.8),
        ]
        avg = buck._calculate_overall_confidence(results)
        assert abs(avg - 0.7) < 1e-9

    def test_empty_results(self, sample_stock_data):
        buck = Buck(
            config=_mock_config(),
            data_provider=_mock_data_provider(sample_stock_data),
            analyzer=_mock_analyzer(),
            predictor=_mock_predictor(),
        )
        assert buck._calculate_overall_confidence([]) == 0.0


# ---------------------------------------------------------------------------
# Async context manager
# ---------------------------------------------------------------------------

class TestBuckContextManager:
    @pytest.mark.asyncio
    async def test_context_manager(self, sample_stock_data):
        async with Buck(
            config=_mock_config(),
            data_provider=_mock_data_provider(sample_stock_data),
            analyzer=_mock_analyzer(),
            predictor=_mock_predictor(),
        ) as buck:
            assert isinstance(buck, Buck)


# ---------------------------------------------------------------------------
# BuckFactory
# ---------------------------------------------------------------------------

class TestBuckFactory:
    def test_create_default_agent(self):
        agent = BuckFactory.create_default_agent()
        assert isinstance(agent, Buck)

    def test_create_custom_agent(self):
        agent = BuckFactory.create_custom_agent(config=_mock_config())
        assert isinstance(agent, Buck)


# ---------------------------------------------------------------------------
# Batch concurrency — per-symbol data isolation
# ---------------------------------------------------------------------------
# agent_scripts/tools.py also exposes a module-level set_stock_data() /
# get_stock_data() context that LangChain @tool-decorated functions read from
# (not exercised by the request path tested here, since TechnicalAnalyzer
# passes each symbol's DataFrame to tool.execute() directly — see that
# module's docstring). This test locks in that the request path Buck.
# batch_analyze() actually uses stays correct under real concurrent
# execution: _perform_analysis runs each symbol's analysis in a thread pool
# executor (agent_scripts/buck.py's run_in_executor call), so this exercises
# genuine multi-threaded concurrency, not just interleaved coroutines.

def _trending_ohlcv_df(direction: str, seed: int, n: int = 80) -> pd.DataFrame:
    """A strongly trending OHLCV series so each symbol's technical signal is
    unambiguous (BUY for 'up', SELL for 'down') and any cross-symbol data
    contamination during concurrent batch analysis is easy to detect. Each
    caller passes a distinct `seed` — two symbols sharing a seed would
    produce byte-identical data (and, incidentally, the same LSTM tool
    checkpoint-cache fingerprint), which defeats the point of testing
    cross-symbol isolation."""
    rng = np.random.RandomState(seed)
    step = 1.0 if direction == "up" else -1.0
    close = 100.0 + np.cumsum(np.full(n, step) + rng.randn(n) * 0.05)
    return pd.DataFrame(
        {
            "Open": close - 0.1,
            "High": close + 0.2,
            "Low": close - 0.2,
            "Close": close,
            "Volume": rng.randint(1_000, 10_000, n).astype(float),
        },
        index=pd.date_range("2024-01-01", periods=n, freq="h"),
    )


class TestBuckBatchConcurrency:
    @pytest.mark.asyncio
    async def test_batch_analyze_does_not_cross_contaminate_symbols(self):
        """Run several distinctly-trending symbols through batch_analyze()
        concurrently and assert each symbol's own analysis reflects its own
        data — not another symbol's, which is what global mutable state
        shared across concurrent requests would produce."""
        symbols_data = {
            "UP_A.NS": _trending_ohlcv_df("up", seed=1),
            "DOWN_A.NS": _trending_ohlcv_df("down", seed=2),
            "UP_B.NS": _trending_ohlcv_df("up", seed=3),
            "DOWN_B.NS": _trending_ohlcv_df("down", seed=4),
        }

        async def fake_get_stock_data(symbol, start_date, end_date, interval):
            # Stagger so the four analyses are genuinely interleaved instead
            # of completing in submission order.
            await asyncio.sleep(0.01 if "A" in symbol else 0.05)
            df = symbols_data[symbol]
            return StockData(
                symbol=symbol, data=df, interval=interval,
                start_date=start_date, end_date=end_date,
            )

        provider = AsyncMock()
        provider.get_stock_data.side_effect = fake_get_stock_data
        provider.get_news_data.return_value = None

        async def fake_predict(analysis_results, **kwargs):
            # Forecast 'close' echoes the overall technical signal so the
            # assertion below can tell which symbol's data actually drove
            # each result.
            signal = analysis_results[0]["data"]["summary"]["overall_signal"]
            return Forecast(
                date="2024-06-10", open=100.0, high=105.0, low=95.0,
                close=111.0 if signal == "BUY" else 99.0 if signal == "SELL" else 100.0,
                confidence=0.5, reasoning=f"signal={signal}",
            )

        predictor = AsyncMock()
        predictor.predict.side_effect = fake_predict

        buck = Buck(
            config=_mock_config(),
            data_provider=provider,
            analyzer=TechnicalAnalyzer(),
            predictor=predictor,
        )

        result = await buck.batch_analyze(
            symbols=list(symbols_data.keys()),
            start_date="2024-01-01",
            end_date="2024-01-04",
            interval="1h",
            max_concurrent=4,
        )

        assert result["summary"]["successful"] == 4
        for symbol, analysis in result["results"].items():
            expected_signal = "BUY" if symbol.startswith("UP") else "SELL"
            actual_signal = analysis["analysis_results"][0]["data"]["summary"]["overall_signal"]
            assert actual_signal == expected_signal, (
                f"{symbol} (expected {expected_signal}) got {actual_signal} — "
                "cross-symbol data contamination in concurrent batch_analyze"
            )
