"""RSI tool – Relative Strength Index calculation."""

from __future__ import annotations
import json
import math
from typing import Any, Dict

import pandas as pd
from langchain_core.tools import tool

from agent_scripts.tools import BaseTool, get_stock_data


class RSITool(BaseTool):
    """Tool for calculating Relative Strength Index."""

    def __init__(self):
        super().__init__(
            "rsi",
            "Calculate Relative Strength Index (RSI) for overbought/oversold conditions"
        )

    def execute(self, data: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        if not self._validate_data(data):
            raise ValueError("Invalid data structure")

        window = kwargs.get('window', 14)

        delta = data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - 100 / (1 + rs)

        current_rsi = float(rsi.iloc[-1])

        if math.isnan(current_rsi):
            return {'rsi': current_rsi, 'window': window, 'signal': 'HOLD',
                    'strength': 0.0, 'note': 'Insufficient data for RSI'}

        result: Dict[str, Any] = {
            'rsi': current_rsi,
            'window': window,
            'overbought_threshold': 70,
            'oversold_threshold': 30,
        }

        if current_rsi > 70:
            result['signal'] = 'SELL'
            result['condition'] = 'OVERBOUGHT'
            result['strength'] = min((current_rsi - 70) / 30, 1.0)
        elif current_rsi < 30:
            result['signal'] = 'BUY'
            result['condition'] = 'OVERSOLD'
            result['strength'] = min((30 - current_rsi) / 30, 1.0)
        else:
            result['signal'] = 'HOLD'
            result['condition'] = 'NEUTRAL'
            result['strength'] = 0.5

        return result


@tool
def rsi(window: int = 14) -> str:
    """Calculate Relative Strength Index (RSI) for the current stock data. Identifies overbought (>70) and oversold (<30) conditions."""
    data = get_stock_data()
    if data is None:
        return json.dumps({"error": "No stock data available"})
    t = RSITool()
    result = t.execute(data, window=window)
    return json.dumps(result, default=str)


TOOL_CLASS = RSITool
TOOL_FUNC = rsi
