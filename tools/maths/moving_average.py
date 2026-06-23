"""Moving Average tool – SMA & EMA calculations."""

from __future__ import annotations
import json
import math
from typing import Any, Dict

import pandas as pd
from langchain_core.tools import tool

from agent_scripts.tools import BaseTool, get_stock_data


class MovingAverageTool(BaseTool):
    """Tool for calculating moving averages."""

    def __init__(self):
        super().__init__(
            "moving_average",
            "Calculate Simple Moving Average (SMA) and Exponential Moving Average (EMA)"
        )

    def execute(self, data: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        if not self._validate_data(data):
            raise ValueError("Invalid data structure")

        short_window = kwargs.get('short_window', 10)
        long_window = kwargs.get('long_window', 50)
        ma_type = kwargs.get('moving_average_type', 'SMA')

        result: Dict[str, Any] = {
            'short_window': short_window,
            'long_window': long_window,
            'type': ma_type,
        }

        if ma_type == 'SMA':
            result['short_ma'] = float(data['Close'].rolling(window=short_window).mean().iloc[-1])
            result['long_ma'] = float(data['Close'].rolling(window=long_window).mean().iloc[-1])
        elif ma_type == 'EMA':
            result['short_ma'] = float(data['Close'].ewm(span=short_window, adjust=False).mean().iloc[-1])
            result['long_ma'] = float(data['Close'].ewm(span=long_window, adjust=False).mean().iloc[-1])

        if math.isnan(result['short_ma']) or math.isnan(result['long_ma']):
            result['signal'] = 'HOLD'
            result['strength'] = 0.0
            result['note'] = 'Insufficient data for moving average calculation'
            return result

        current_price = float(data['Close'].iloc[-1])
        if result['short_ma'] > result['long_ma'] and current_price > result['short_ma']:
            result['signal'] = 'BUY'
            result['strength'] = min((current_price - result['long_ma']) / result['long_ma'], 1.0)
        elif result['short_ma'] < result['long_ma'] and current_price < result['short_ma']:
            result['signal'] = 'SELL'
            result['strength'] = min((result['long_ma'] - current_price) / result['long_ma'], 1.0)
        else:
            result['signal'] = 'HOLD'
            result['strength'] = 0.5

        return result


@tool
def moving_average(
    short_window: int = 10,
    long_window: int = 50,
    moving_average_type: str = "SMA",
) -> str:
    """Calculate Simple Moving Average (SMA) and Exponential Moving Average (EMA) for the current stock data. Use this to identify trend direction and potential crossover signals."""
    data = get_stock_data()
    if data is None:
        return json.dumps({"error": "No stock data available"})
    t = MovingAverageTool()
    result = t.execute(
        data,
        short_window=short_window,
        long_window=long_window,
        moving_average_type=moving_average_type,
    )
    return json.dumps(result, default=str)


TOOL_CLASS = MovingAverageTool
TOOL_FUNC = moving_average
