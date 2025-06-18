"""
Buck_V1.tools
──────────────────────────
Concrete tool implementations for stock analysis.
"""

from __future__ import annotations
import asyncio
import json
import time
from abc import ABC
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests
import yfinance as yf
from scipy import stats

from .interfaces import ITool, AnalysisResult, StockData, NewsData
from .config import LOGGER


class BaseTool(ABC):
    """Base class for all analysis tools."""
    
    def __init__(self, name: str, description: str):
        self._name = name
        self._description = description
    
    @property
    def name(self) -> str:
        return self._name
    
    @property
    def description(self) -> str:
        return self._description
    
    def _validate_data(self, data: pd.DataFrame) -> bool:
        """Validate input data."""
        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        return all(col in data.columns for col in required_columns)


class MovingAverageTool(BaseTool):
    """Tool for calculating moving averages."""
    
    def __init__(self):
        super().__init__(
            "moving_average",
            "Calculate Simple Moving Average (SMA) and Exponential Moving Average (EMA)"
        )
    
    def execute(self, data: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        """Execute moving average calculations."""
        if not self._validate_data(data):
            raise ValueError("Invalid data structure")
        
        short_window = kwargs.get('short_window', 10)
        long_window = kwargs.get('long_window', 50)
        ma_type = kwargs.get('moving_average_type', 'SMA')
        
        result = {
            'short_window': short_window,
            'long_window': long_window,
            'type': ma_type
        }
        
        if ma_type == 'SMA':
            result['short_ma'] = float(data['Close'].rolling(window=short_window).mean().iloc[-1])
            result['long_ma'] = float(data['Close'].rolling(window=long_window).mean().iloc[-1])
        elif ma_type == 'EMA':
            result['short_ma'] = float(data['Close'].ewm(span=short_window, adjust=False).mean().iloc[-1])
            result['long_ma'] = float(data['Close'].ewm(span=long_window, adjust=False).mean().iloc[-1])
        
        # Generate signal
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


class RSITool(BaseTool):
    """Tool for calculating Relative Strength Index."""
    
    def __init__(self):
        super().__init__(
            "rsi",
            "Calculate Relative Strength Index (RSI) for overbought/oversold conditions"
        )
    
    def execute(self, data: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        """Execute RSI calculation."""
        if not self._validate_data(data):
            raise ValueError("Invalid data structure")
        
        window = kwargs.get('window', 14)
        
        delta = data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - 100 / (1 + rs)
        
        current_rsi = float(rsi.iloc[-1])
        
        result = {
            'rsi': current_rsi,
            'window': window,
            'overbought_threshold': 70,
            'oversold_threshold': 30
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


class MACDTool(BaseTool):
    """Tool for calculating MACD indicator."""
    
    def __init__(self):
        super().__init__(
            "macd",
            "Calculate MACD (Moving Average Convergence Divergence) indicator"
        )
    
    def execute(self, data: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        """Execute MACD calculation."""
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
        
        result = {
            'macd': current_macd,
            'signal': current_signal,
            'histogram': current_histogram,
            'short_window': short_window,
            'long_window': long_window,
            'signal_window': signal_window
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


class OBVTool(BaseTool):
    """Tool for calculating On-Balance Volume."""
    
    def __init__(self):
        super().__init__(
            "obv",
            "Calculate On-Balance Volume (OBV) for volume flow analysis"
        )
    
    def execute(self, data: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        """Execute OBV calculation."""
        if not self._validate_data(data):
            raise ValueError("Invalid data structure")
        
        obv = (data['Volume'].where(data['Close'].diff() > 0, -data['Volume'])).cumsum()
        current_obv = float(obv.iloc[-1])
        
        # Calculate OBV trend
        obv_ma = obv.rolling(window=10).mean()
        obv_trend = 'RISING' if current_obv > float(obv_ma.iloc[-1]) else 'FALLING'
        
        result = {
            'obv': current_obv,
            'obv_trend': obv_trend,
            'obv_ma': float(obv_ma.iloc[-1])
        }
        
        if obv_trend == 'RISING':
            result['signal'] = 'BUY'
            result['strength'] = 0.7
        else:
            result['signal'] = 'SELL'
            result['strength'] = 0.7
        
        return result


class CandlestickPatternTool(BaseTool):
    """Tool for detecting candlestick patterns."""
    
    def __init__(self):
        super().__init__(
            "candlestick_patterns",
            "Detect classical candlestick patterns for trend reversal signals"
        )
    
    def execute(self, data: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        """Execute candlestick pattern detection."""
        if not self._validate_data(data):
            raise ValueError("Invalid data structure")
        
        patterns = {}
        
        for i in range(1, min(len(data), 20)):  # Check last 20 candles
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
            
            pattern_dict = {}
            
            # Doji
            if body <= 0.1 * total:
                pattern_dict['Doji'] = 0.9
            
            # Hammer
            if lower_shadow >= 2 * body and upper_shadow <= 0.1 * total and c > o:
                pattern_dict['Hammer'] = 0.8
            
            # Shooting Star
            if upper_shadow >= 2 * body and lower_shadow <= 0.1 * total and c < o:
                pattern_dict['Shooting Star'] = 0.8
            
            # Bullish Engulfing
            if pc < po and c > o and c > po and o < pc:
                pattern_dict['Bullish Engulfing'] = 0.9
            
            # Bearish Engulfing
            if pc > po and c < o and c < po and o > pc:
                pattern_dict['Bearish Engulfing'] = 0.9
            
            if pattern_dict:
                patterns[str(data.index[-i-1])] = pattern_dict
        
        # Summarize patterns
        bullish_patterns = 0
        bearish_patterns = 0
        
        for timestamp, pattern_dict in patterns.items():
            for pattern_name, confidence in pattern_dict.items():
                if any(word in pattern_name for word in ['Bullish', 'Hammer']):
                    bullish_patterns += confidence
                elif any(word in pattern_name for word in ['Bearish', 'Shooting']):
                    bearish_patterns += confidence
        
        result = {
            'patterns': patterns,
            'bullish_score': bullish_patterns,
            'bearish_score': bearish_patterns,
            'pattern_count': len(patterns)
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


class SupportResistanceTool(BaseTool):
    """Tool for identifying support and resistance levels."""
    
    def __init__(self):
        super().__init__(
            "support_resistance",
            "Identify key support and resistance levels"
        )
    
    def execute(self, data: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        """Execute support/resistance level identification."""
        if not self._validate_data(data):
            raise ValueError("Invalid data structure")
        
        window = kwargs.get('window', 5)
        levels = {}
        
        rolling_low = data['Low'].rolling(window=window).min()
        rolling_high = data['High'].rolling(window=window).max()
        
        for i in range(window, len(data)):
            current_price = data['Close'].iloc[i]
            current_low = data['Low'].iloc[i]
            current_high = data['High'].iloc[i]
            
            prev_low = rolling_low.iloc[i-1]
            prev_high = rolling_high.iloc[i-1]
            
            # Support level
            if current_low <= prev_low and current_price > prev_low:
                levels[str(data.index[i])] = {'type': 'Support', 'level': prev_low}
            
            # Resistance level
            elif current_high >= prev_high and current_price < prev_high:
                levels[str(data.index[i])] = {'type': 'Resistance', 'level': prev_high}
        
        current_price = data['Close'].iloc[-1]
        support_levels = [v['level'] for v in levels.values() if v['type'] == 'Support']
        resistance_levels = [v['level'] for v in levels.values() if v['type'] == 'Resistance']
        
        nearest_support = max([s for s in support_levels if s < current_price], default=None)
        nearest_resistance = min([r for r in resistance_levels if r > current_price], default=None)
        
        result = {
            'levels': levels,
            'current_price': current_price,
            'nearest_support': nearest_support,
            'nearest_resistance': nearest_resistance,
            'support_count': len(support_levels),
            'resistance_count': len(resistance_levels)
        }
        
        # Generate signal based on proximity to levels
        if nearest_support and nearest_resistance:
            support_distance = (current_price - nearest_support) / current_price
            resistance_distance = (nearest_resistance - current_price) / current_price
            
            if support_distance < 0.02:  # Within 2% of support
                result['signal'] = 'BUY'
                result['strength'] = 0.8
            elif resistance_distance < 0.02:  # Within 2% of resistance
                result['signal'] = 'SELL'
                result['strength'] = 0.8
            else:
                result['signal'] = 'HOLD'
                result['strength'] = 0.5
        else:
            result['signal'] = 'HOLD'
            result['strength'] = 0.5
        
        return result


class ToolFactory:
    """Factory for creating analysis tools."""
    
    _tools = {
        'moving_average': MovingAverageTool,
        'rsi': RSITool,
        'macd': MACDTool,
        'obv': OBVTool,
        'candlestick_patterns': CandlestickPatternTool,
        'support_resistance': SupportResistanceTool,
    }
    
    @classmethod
    def create_tool(cls, tool_name: str) -> ITool:
        """Create a tool instance by name."""
        if tool_name not in cls._tools:
            raise ValueError(f"Unknown tool: {tool_name}")
        
        return cls._tools[tool_name]()
    
    @classmethod
    def get_available_tools(cls) -> List[str]:
        """Get list of available tool names."""
        return list(cls._tools.keys())
    
    @classmethod
    def create_all_tools(cls) -> Dict[str, ITool]:
        """Create instances of all available tools."""
        return {name: cls.create_tool(name) for name in cls._tools.keys()} 
