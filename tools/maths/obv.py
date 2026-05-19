"""OBV tool – On-Balance Volume analysis."""

from __future__ import annotations
import json
from typing import Any, Dict

import pandas as pd
from langchain_core.tools import tool

from agent_scripts.tools import BaseTool, get_stock_data


class OBVTool(BaseTool):
    """Tool for calculating On-Balance Volume."""

    def __init__(self):
        super().__init__(
            "obv",
            "Calculate On-Balance Volume (OBV) for volume flow analysis"
        )

    def execute(self, data: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        if not self._validate_data(data):
            raise ValueError("Invalid data structure")

        obv = (data['Volume'].where(data['Close'].diff() > 0, -data['Volume'])).cumsum()
        current_obv = float(obv.iloc[-1])

        obv_ma = obv.rolling(window=10).mean()
        obv_trend = 'RISING' if current_obv > float(obv_ma.iloc[-1]) else 'FALLING'

        result: Dict[str, Any] = {
            'obv': current_obv,
            'obv_trend': obv_trend,
            'obv_ma': float(obv_ma.iloc[-1]),
        }

        if obv_trend == 'RISING':
            result['signal'] = 'BUY'
            result['strength'] = 0.7
        else:
            result['signal'] = 'SELL'
            result['strength'] = 0.7

        return result


@tool
def obv() -> str:
    """Calculate On-Balance Volume (OBV) for the current stock data. Measures buying and selling pressure using volume flow."""
    data = get_stock_data()
    if data is None:
        return json.dumps({"error": "No stock data available"})
    t = OBVTool()
    result = t.execute(data)
    return json.dumps(result, default=str)


TOOL_CLASS = OBVTool
TOOL_FUNC = obv
