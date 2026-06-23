"""Support & Resistance level identification tool."""

from __future__ import annotations
import json
from typing import Any, Dict

import pandas as pd
from langchain_core.tools import tool

from agent_scripts.tools import BaseTool, get_stock_data


class SupportResistanceTool(BaseTool):
    """Tool for identifying support and resistance levels."""

    def __init__(self):
        super().__init__(
            "support_resistance",
            "Identify key support and resistance levels"
        )

    def execute(self, data: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        if not self._validate_data(data):
            raise ValueError("Invalid data structure")

        window = kwargs.get('window', 5)
        levels: Dict[str, Any] = {}

        rolling_low = data['Low'].rolling(window=window).min()
        rolling_high = data['High'].rolling(window=window).max()

        for i in range(window, len(data)):
            current_price = data['Close'].iloc[i]
            current_low = data['Low'].iloc[i]
            current_high = data['High'].iloc[i]

            prev_low = rolling_low.iloc[i-1]
            prev_high = rolling_high.iloc[i-1]

            if current_low <= prev_low and current_price > prev_low:
                levels[str(data.index[i])] = {'type': 'Support', 'level': prev_low}
            elif current_high >= prev_high and current_price < prev_high:
                levels[str(data.index[i])] = {'type': 'Resistance', 'level': prev_high}

        current_price = data['Close'].iloc[-1]
        support_levels = [v['level'] for v in levels.values() if v['type'] == 'Support']
        resistance_levels = [v['level'] for v in levels.values() if v['type'] == 'Resistance']

        nearest_support = max([s for s in support_levels if s < current_price], default=None)
        nearest_resistance = min([r for r in resistance_levels if r > current_price], default=None)

        result: Dict[str, Any] = {
            'levels': levels,
            'current_price': current_price,
            'nearest_support': nearest_support,
            'nearest_resistance': nearest_resistance,
            'support_count': len(support_levels),
            'resistance_count': len(resistance_levels),
        }

        if nearest_support and nearest_resistance:
            support_distance = (current_price - nearest_support) / current_price
            resistance_distance = (nearest_resistance - current_price) / current_price

            if support_distance < 0.02:
                result['signal'] = 'BUY'
                result['strength'] = 0.8
            elif resistance_distance < 0.02:
                result['signal'] = 'SELL'
                result['strength'] = 0.8
            else:
                result['signal'] = 'HOLD'
                result['strength'] = 0.5
        else:
            result['signal'] = 'HOLD'
            result['strength'] = 0.5

        return result


@tool
def support_resistance(window: int = 5) -> str:
    """Identify key support and resistance price levels in the current stock data. Helps determine entry/exit points based on price proximity to these levels."""
    data = get_stock_data()
    if data is None:
        return json.dumps({"error": "No stock data available"})
    t = SupportResistanceTool()
    result = t.execute(data, window=window)
    return json.dumps(result, default=str)


TOOL_CLASS = SupportResistanceTool
TOOL_FUNC = support_resistance
