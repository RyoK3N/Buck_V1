"""Tests for agent_scripts.buck — Buck orchestrator and BuckFactory."""

from __future__ import annotations

from datetime import datetime
from unittest.mock import AsyncMock, MagicMock

import pytest

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
