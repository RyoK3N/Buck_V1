"""Tests for agent_scripts.predictors — OpenAIPredictor, EnsemblePredictor,
PredictorFactory, and forecast validation helpers."""

from __future__ import annotations

from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agent_scripts.interfaces import AnalysisResult, Forecast
from agent_scripts.predictors import (
    EnsemblePredictor,
    OpenAIPredictor,
    PredictorFactory,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_predictor() -> OpenAIPredictor:
    """Create an OpenAIPredictor with a dummy key (no real API calls)."""
    return OpenAIPredictor(api_key="test-key", model="gpt-4o", temperature=0.0)


# ---------------------------------------------------------------------------
# OpenAIPredictor._validate_and_format_forecast
# ---------------------------------------------------------------------------

class TestValidateAndFormatForecast:
    def test_valid_data(self):
        predictor = _make_predictor()
        raw = {
            "date": "2024-06-10",
            "open": 100.0,
            "high": 105.0,
            "low": 98.0,
            "close": 103.0,
            "confidence": 0.85,
            "reasoning": "Strong momentum signals.",
        }
        forecast = predictor._validate_and_format_forecast(raw, "TEST.NS")
        assert forecast["date"] == "2024-06-10"
        assert forecast["open"] == 100.0
        assert forecast["high"] == 105.0
        assert forecast["low"] == 98.0
        assert forecast["close"] == 103.0
        assert forecast["confidence"] == 0.85

    def test_missing_fields_raises(self):
        predictor = _make_predictor()
        raw = {"date": "2024-06-10", "open": 100.0}  # missing several fields
        with pytest.raises(ValueError, match="Missing required fields"):
            predictor._validate_and_format_forecast(raw, "TEST.NS")

    def test_price_adjustment_high(self):
        """high < max(open, close) should be auto-adjusted."""
        predictor = _make_predictor()
        raw = {
            "date": "2024-06-10",
            "open": 105.0,
            "high": 100.0,  # too low
            "low": 98.0,
            "close": 103.0,
            "confidence": 0.7,
            "reasoning": "Test",
        }
        forecast = predictor._validate_and_format_forecast(raw, "TEST.NS")
        assert forecast["high"] >= max(forecast["open"], forecast["close"])

    def test_price_adjustment_low(self):
        """low > min(open, close) should be auto-adjusted."""
        predictor = _make_predictor()
        raw = {
            "date": "2024-06-10",
            "open": 100.0,
            "high": 105.0,
            "low": 102.0,  # too high
            "close": 99.0,
            "confidence": 0.7,
            "reasoning": "Test",
        }
        forecast = predictor._validate_and_format_forecast(raw, "TEST.NS")
        assert forecast["low"] <= min(forecast["open"], forecast["close"])

    def test_confidence_clamping(self):
        """confidence > 1.0 should be clamped to 1.0."""
        predictor = _make_predictor()
        raw = {
            "date": "2024-06-10",
            "open": 100.0,
            "high": 105.0,
            "low": 98.0,
            "close": 103.0,
            "confidence": 1.5,
            "reasoning": "Test",
        }
        forecast = predictor._validate_and_format_forecast(raw, "TEST.NS")
        assert forecast["confidence"] <= 1.0


# ---------------------------------------------------------------------------
# OpenAIPredictor._build_prompts
# ---------------------------------------------------------------------------

class TestBuildPrompts:
    def test_includes_symbol_and_data(self, sample_analysis_result):
        predictor = _make_predictor()
        system, user = predictor._build_prompts([sample_analysis_result])
        assert "TEST.NS" in user
        assert "RSI" in system or "technical" in system.lower()


# ---------------------------------------------------------------------------
# OpenAIPredictor._preprocess_openai_response
# ---------------------------------------------------------------------------

class TestPreprocessResponse:
    def test_trailing_backslash_fix(self):
        predictor = _make_predictor()
        bad_json = '{"date":"2024-06-10","open":100,"high":105,"low":98,"close":103,"confidence":0.8,"reasoning":"test\\'
        result = predictor._preprocess_openai_response(bad_json)
        assert result.rstrip().endswith("}")


# ---------------------------------------------------------------------------
# EnsemblePredictor
# ---------------------------------------------------------------------------

class TestEnsemblePredictor:
    @pytest.mark.asyncio
    async def test_combines_predictions(self, sample_analysis_result):
        """Two mock predictors should produce a weighted average."""
        forecast_a = Forecast(
            date="2024-06-10", open=100.0, high=105.0, low=98.0,
            close=103.0, confidence=0.8, reasoning="Model A says buy.",
        )
        forecast_b = Forecast(
            date="2024-06-10", open=102.0, high=107.0, low=99.0,
            close=104.0, confidence=0.6, reasoning="Model B says hold.",
        )

        pred_a = AsyncMock()
        pred_a.predict.return_value = forecast_a
        pred_b = AsyncMock()
        pred_b.predict.return_value = forecast_b

        ensemble = EnsemblePredictor([pred_a, pred_b], weights=[0.7, 0.3])
        result = await ensemble.predict([sample_analysis_result])

        # Weighted averages (weights are normalized: 0.7, 0.3 already sum to 1)
        assert result["date"] == "2024-06-10"
        assert abs(result["open"] - (100.0 * 0.7 + 102.0 * 0.3)) < 0.1
        assert abs(result["close"] - (103.0 * 0.7 + 104.0 * 0.3)) < 0.1

    def test_weight_mismatch_raises(self):
        """Mismatched predictor/weight counts should raise ValueError."""
        pred = AsyncMock()
        with pytest.raises(ValueError, match="weights must match"):
            EnsemblePredictor([pred], weights=[0.5, 0.5])


# ---------------------------------------------------------------------------
# PredictorFactory
# ---------------------------------------------------------------------------

class TestPredictorFactory:
    def test_create_openai_predictor(self):
        p = PredictorFactory.create_openai_predictor(api_key="test-key")
        assert isinstance(p, OpenAIPredictor)

    def test_create_ensemble_predictor(self):
        pred = AsyncMock()
        ep = PredictorFactory.create_ensemble_predictor([pred, pred])
        assert isinstance(ep, EnsemblePredictor)

    def test_create_default_predictor(self):
        p = PredictorFactory.create_default_predictor()
        assert isinstance(p, OpenAIPredictor)
