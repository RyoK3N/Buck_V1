"""
Buck - (init.py)
────────────────────
An AI-Powered Stock Analysis and Prediction Agent

A stock analysis system that combines:
- Technical analysis using multiple key point indicators
- Sentiment analysis from news data source (IndianAPI)
- AI-powered predictions using OpenAI's models
- Comprehensive reporting and caching

Copyright © 2025  Buck Analytics Pvt. Ltd.  All rights reserved.
Licensed under the Apache License 2.0 – see LICENSE file for details.
"""

from .stock_agent import StockAgent, StockAgentFactory
from .config import SETTINGS, LOGGER
from .interfaces import Forecast, AnalysisResult, StockData, NewsData

__version__ = "1.0.0"
__author__ = "Buck Dev Team"
__description__ = "AI-Powered Stock Analysis and Prediction Agent that helps you make bucks"

# Main exports
__all__ = [
    "StockAgent",
    "StockAgentFactory", 
    "SETTINGS",
    "LOGGER",
    "Forecast",
    "AnalysisResult",
    "StockData",
    "NewsData",
]

# Convenience functions
def create_agent(openai_api_key: str, indian_api_key: str = "", model: str = "gpt-4o") -> StockAgent:
    """Create a production-ready Stock Agent.
    
    Args:
        openai_api_key: OpenAI API key for predictions
        indian_api_key: Indian API key for news data (optional)
        model: OpenAI model to use (default: gpt-4o)
        
    Returns:
        Configured StockAgent instance
    """
    return StockAgentFactory.create_production_agent(
        openai_api_key=openai_api_key,
        indian_api_key=indian_api_key,
        model=model
    )

async def analyze_stock(
    symbol: str,
    start_date: str, 
    end_date: str,
    openai_api_key: str,
    interval: str = "1h",
    indian_api_key: str = ""
) -> dict:
    """Quick stock analysis function.
    
    Args:
        symbol: Stock symbol (e.g., 'BHEL.NS')
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format
        openai_api_key: OpenAI API key
        interval: Data interval (default: '1h')
        indian_api_key: Indian API key for news (optional)
        
    Returns:
        Analysis and prediction results
    """
    agent = create_agent(openai_api_key, indian_api_key)
    
    async with agent:
        return await agent.analyze_and_predict(
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
            interval=interval
        )

