"""Candlestick pattern detection tool."""

from __future__ import annotations
import json
from typing import Any, Dict

import pandas as pd
from langchain_core.tools import tool

from agent_scripts.tools import BaseTool, get_stock_data


class CandlestickPatternTool(BaseTool):
    """Tool for detecting candlestick patterns."""

    def __init__(self):
        super().__init__(
            "candlestick_patterns",
            "Detect classical candlestick patterns for trend reversal signals"
        )

    def execute(self, data: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        if not self._validate_data(data):
            raise ValueError("Invalid data structure")

        patterns: Dict[str, Any] = {}

        for i in range(1, min(len(data), 20)):
            current = data.iloc[-i-1]
            prev = data.iloc[-i-2] if i+1 < len(data) else None

            if prev is None:
                continue

            o, h, l, c = current['Open'], current['High'], current['Low'], current['Close']
            po, ph, pl, pc = prev['Open'], prev['High'], prev['Low'], prev['Close']

            body = abs(c - o)
            total = h - l
            upper_shadow = h - max(o, c)
            lower_shadow = min(o, c) - l

            pattern_dict: Dict[str, float] = {}

            if body <= 0.1 * total:
                pattern_dict['Doji'] = 0.9
            if lower_shadow >= 2 * body and upper_shadow <= 0.1 * total and c > o:
                pattern_dict['Hammer'] = 0.8
            if upper_shadow >= 2 * body and lower_shadow <= 0.1 * total and c < o:
                pattern_dict['Shooting Star'] = 0.8
            if pc < po and c > o and c > po and o < pc:
                pattern_dict['Bullish Engulfing'] = 0.9
            if pc > po and c < o and c < po and o > pc:
                pattern_dict['Bearish Engulfing'] = 0.9

            if pattern_dict:
                patterns[str(data.index[-i-1])] = pattern_dict

        bullish_patterns = 0.0
        bearish_patterns = 0.0

        for timestamp, pattern_dict in patterns.items():
            for pattern_name, confidence in pattern_dict.items():
                if any(word in pattern_name for word in ['Bullish', 'Hammer']):
                    bullish_patterns += confidence
                elif any(word in pattern_name for word in ['Bearish', 'Shooting']):
                    bearish_patterns += confidence

        result: Dict[str, Any] = {
            'patterns': patterns,
            'bullish_score': bullish_patterns,
            'bearish_score': bearish_patterns,
            'pattern_count': len(patterns),
        }

        if bullish_patterns > bearish_patterns:
            result['signal'] = 'BUY'
            result['strength'] = min(bullish_patterns / (bullish_patterns + bearish_patterns + 0.1), 1.0)
        elif bearish_patterns > bullish_patterns:
            result['signal'] = 'SELL'
            result['strength'] = min(bearish_patterns / (bullish_patterns + bearish_patterns + 0.1), 1.0)
        else:
            result['signal'] = 'HOLD'
            result['strength'] = 0.5

        return result


@tool
def candlestick_patterns() -> str:
    """Detect classical candlestick patterns (Doji, Hammer, Engulfing, etc.) in the current stock data. Identifies potential trend reversals."""
    data = get_stock_data()
    if data is None:
        return json.dumps({"error": "No stock data available"})
    t = CandlestickPatternTool()
    result = t.execute(data)
    return json.dumps(result, default=str)


TOOL_CLASS = CandlestickPatternTool
TOOL_FUNC = candlestick_patterns
