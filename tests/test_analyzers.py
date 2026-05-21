"""Tests for agent_scripts.analyzers — TechnicalAnalyzer, SentimentAnalyzer,
CompositeAnalyzer, and AnalyzerFactory."""

from __future__ import annotations

from datetime import datetime
from unittest.mock import MagicMock

import pytest

from agent_scripts.analyzers import (
    AnalyzerFactory,
    CompositeAnalyzer,
    SentimentAnalyzer,
    TechnicalAnalyzer,
)
from agent_scripts.interfaces import AnalysisResult


# ---------------------------------------------------------------------------
# TechnicalAnalyzer
# ---------------------------------------------------------------------------

class TestTechnicalAnalyzer:
    def _make_mock_tools(self):
        """Return a dict of mock tools with known signals."""
        buy_tool = MagicMock()
        buy_tool.execute.return_value = {"signal": "BUY", "strength": 0.8, "rsi": 30}

        sell_tool = MagicMock()
        sell_tool.execute.return_value = {"signal": "SELL", "strength": 0.6, "macd": -0.3}

        hold_tool = MagicMock()
        hold_tool.execute.return_value = {"signal": "HOLD", "strength": 0.5}

        return {"rsi": buy_tool, "macd": sell_tool, "obv": hold_tool}

    def test_with_mock_tools(self, sample_stock_data):
        """TechnicalAnalyzer returns a well-formed AnalysisResult."""
        tools = self._make_mock_tools()
        analyzer = TechnicalAnalyzer(tools=tools)
        result = analyzer.analyze(sample_stock_data)

        assert result["symbol"] == "TEST.NS"
        assert result["analysis_type"] == "technical_analysis"
        assert 0.0 <= result["confidence"] <= 1.0
        assert "tools_results" in result["data"]
        assert set(result["data"]["tools_results"].keys()) == {"rsi", "macd", "obv"}

    def test_summary_signals(self, sample_stock_data):
        """Summary should reflect aggregate BUY/SELL/HOLD counts."""
        tools = self._make_mock_tools()
        analyzer = TechnicalAnalyzer(tools=tools)
        result = analyzer.analyze(sample_stock_data)
        summary = result["data"]["summary"]

        assert "overall_signal" in summary
        assert summary["overall_signal"] in ("BUY", "SELL", "HOLD")
        assert "signals_breakdown" in summary
        assert summary["tools_processed"] == 3
        assert summary["tools_failed"] == 0

    def test_confidence_calculation(self, sample_stock_data):
        """Confidence should be between 0 and 1 and reflect tool success."""
        tools = self._make_mock_tools()
        analyzer = TechnicalAnalyzer(tools=tools)
        result = analyzer.analyze(sample_stock_data)
        assert 0.0 <= result["confidence"] <= 1.0

    def test_handles_tool_error(self, sample_stock_data):
        """If a tool raises, the error is recorded but analysis continues."""
        error_tool = MagicMock()
        error_tool.execute.side_effect = RuntimeError("boom")

        ok_tool = MagicMock()
        ok_tool.execute.return_value = {"signal": "BUY", "strength": 0.7}

        tools = {"broken": error_tool, "ok": ok_tool}
        analyzer = TechnicalAnalyzer(tools=tools)
        result = analyzer.analyze(sample_stock_data)

        assert "error" in result["data"]["tools_results"]["broken"]
        assert result["data"]["tools_results"]["ok"]["signal"] == "BUY"
        assert result["data"]["summary"]["tools_failed"] == 1


# ---------------------------------------------------------------------------
# SentimentAnalyzer
# ---------------------------------------------------------------------------

class TestSentimentAnalyzer:
    def test_positive_news(self, sample_stock_data, sample_news_data):
        """News dominated by positive keywords should yield POSITIVE label."""
        # Keep only the positive items
        positive_news = {
            **sample_news_data,
            "news": [n for n in sample_news_data["news"] if "profit" in n["title"] or "rallies" in n["title"]],
        }
        analyzer = SentimentAnalyzer()
        result = analyzer.analyze(sample_stock_data, news_data=positive_news)

        assert result["analysis_type"] == "sentiment_analysis"
        assert result["data"]["sentiment_label"] == "POSITIVE"

    def test_negative_news(self, sample_stock_data, sample_news_data):
        """News dominated by negative keywords should yield NEGATIVE label."""
        negative_news = {
            **sample_news_data,
            "news": [n for n in sample_news_data["news"] if "loss" in n["title"]],
        }
        analyzer = SentimentAnalyzer()
        result = analyzer.analyze(sample_stock_data, news_data=negative_news)

        assert result["data"]["sentiment_label"] == "NEGATIVE"

    def test_no_news(self, sample_stock_data):
        """No news_data should produce NEUTRAL with low confidence."""
        analyzer = SentimentAnalyzer()
        result = analyzer.analyze(sample_stock_data, news_data=None)

        assert result["data"]["sentiment_label"] == "NEUTRAL"
        assert result["confidence"] <= 0.2

    def test_topic_weighting(self, sample_stock_data, sample_news_data):
        """Topic weights should affect the final sentiment score."""
        analyzer = SentimentAnalyzer()
        result = analyzer.analyze(sample_stock_data, news_data=sample_news_data)
        # "Financial Results" has weight 1.5 — the result should include
        # topic analysis data
        assert "topics_analysis" in result["data"]
        topic_dist = result["data"]["topics_analysis"].get("topic_distribution", {})
        assert "Financial Results" in topic_dist


# ---------------------------------------------------------------------------
# CompositeAnalyzer
# ---------------------------------------------------------------------------

class TestCompositeAnalyzer:
    def test_combines_results(self, sample_stock_data, sample_news_data):
        """CompositeAnalyzer combines technical + sentiment results."""
        mock_tool = MagicMock()
        mock_tool.execute.return_value = {"signal": "BUY", "strength": 0.8}

        tech = TechnicalAnalyzer(tools={"mock": mock_tool})
        sent = SentimentAnalyzer()
        composite = CompositeAnalyzer(analyzers=[tech, sent])

        result = composite.analyze(sample_stock_data, news_data=sample_news_data)

        assert result["analysis_type"] == "composite_analysis"
        assert "analysis_results" in result["data"]
        assert "technical_analysis" in result["data"]["analysis_results"]
        assert "sentiment_analysis" in result["data"]["analysis_results"]
        assert 0.0 <= result["confidence"] <= 1.0


# ---------------------------------------------------------------------------
# AnalyzerFactory
# ---------------------------------------------------------------------------

class TestAnalyzerFactory:
    def test_create_technical_analyzer(self):
        analyzer = AnalyzerFactory.create_technical_analyzer(tools={})
        assert isinstance(analyzer, TechnicalAnalyzer)

    def test_create_sentiment_analyzer(self):
        analyzer = AnalyzerFactory.create_sentiment_analyzer()
        assert isinstance(analyzer, SentimentAnalyzer)

    def test_create_composite_analyzer(self):
        analyzer = AnalyzerFactory.create_composite_analyzer()
        assert isinstance(analyzer, CompositeAnalyzer)

    def test_create_composite_analyzer_with_tools(self):
        analyzer = AnalyzerFactory.create_composite_analyzer_with_tools(tools={})
        assert isinstance(analyzer, CompositeAnalyzer)
