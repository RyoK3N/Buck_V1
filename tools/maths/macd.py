"""MACD tool – Moving Average Convergence Divergence."""

from __future__ import annotations
import json
from typing import Any, Dict

import pandas as pd
from langchain_core.tools import tool

from agent_scripts.tools import BaseTool, get_stock_data


class MACDTool(BaseTool):
    """Tool for calculating MACD indicator."""

    def __init__(self):
        super().__init__(
            "macd",
            "Calculate MACD (Moving Average Convergence Divergence) indicator"
        )

    def execute(self, data: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        if not self._validate_data(data):
            raise ValueError("Invalid data structure")

        short_window = kwargs.get('short_window', 12)
        long_window = kwargs.get('long_window', 26)
        signal_window = kwargs.get('signal_window', 9)

        short_ema = data['Close'].ewm(span=short_window, adjust=False).mean()
        long_ema = data['Close'].ewm(span=long_window, adjust=False).mean()
        macd_line = short_ema - long_ema
        signal_line = macd_line.ewm(span=signal_window, adjust=False).mean()
        histogram = macd_line - signal_line

        current_macd = float(macd_line.iloc[-1])
        current_signal = float(signal_line.iloc[-1])
        current_histogram = float(histogram.iloc[-1])

        result: Dict[str, Any] = {
            'macd': current_macd,
            'signal_value': current_signal,
            'histogram': current_histogram,
            'short_window': short_window,
            'long_window': long_window,
            'signal_window': signal_window,
        }

        if current_macd > current_signal and current_histogram > 0:
            result['signal'] = 'BUY'
            result['strength'] = min(abs(current_histogram) / max(abs(current_macd), 0.0001), 1.0)
        elif current_macd < current_signal and current_histogram < 0:
            result['signal'] = 'SELL'
            result['strength'] = min(abs(current_histogram) / max(abs(current_macd), 0.0001), 1.0)
        else:
            result['signal'] = 'HOLD'
            result['strength'] = 0.5

        return result


@tool
def macd(
    short_window: int = 12,
    long_window: int = 26,
    signal_window: int = 9,
) -> str:
    """Calculate MACD (Moving Average Convergence Divergence) for the current stock data. Identifies trend momentum and potential crossover signals."""
    data = get_stock_data()
    if data is None:
        return json.dumps({"error": "No stock data available"})
    t = MACDTool()
    result = t.execute(
        data,
        short_window=short_window,
        long_window=long_window,
        signal_window=signal_window,
    )
    return json.dumps(result, default=str)


TOOL_CLASS = MACDTool
TOOL_FUNC = macd
