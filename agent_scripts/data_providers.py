"""
Buck_V1.data_providers
───────────────────────────────────
Data provider implementations for stock and news data.
"""

from __future__ import annotations
import asyncio
import json
import time
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List

import aiohttp
import pandas as pd
import requests
import yfinance as yf

from .interfaces import IDataProvider, StockData, NewsData
from .config import LOGGER


class YahooFinanceProvider(IDataProvider):
    """Yahoo Finance data provider for stock data."""
    
    def __init__(self, request_timeout: int = 30):
        self.request_timeout = request_timeout
        self._session = None
    
    async def get_stock_data(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        interval: str = "1h"
    ) -> Optional[StockData]:
        """Get stock data from Yahoo Finance."""
        try:
            # Validate symbol format
            if not symbol.endswith('.NS'):
                LOGGER.warning(f"Symbol {symbol} doesn't end with .NS, appending it")
                symbol = f"{symbol}.NS"
            
            # Validate date range
            start_dt = datetime.strptime(start_date, '%Y-%m-%d')
            end_dt = datetime.strptime(end_date, '%Y-%m-%d')
            
            if (end_dt - start_dt).days > 60 and interval in ['30m', '1h', '4h', '6h', '8h', '12h']:
                raise ValueError("Intraday data range cannot exceed 60 days")
            
            # Download data in executor to avoid blocking
            loop = asyncio.get_event_loop()
            data = await loop.run_in_executor(
                None,
                self._download_sync,
                symbol,
                start_date,
                end_date,
                interval
            )
            
            if data is None or data.empty:
                LOGGER.error(f"No data received for {symbol}")
                return None
            
            # Process MultiIndex columns if present
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.get_level_values(0)
            
            return StockData(
                symbol=symbol,
                data=data,
                interval=interval,
                start_date=start_date,
                end_date=end_date
            )
            
        except Exception as e:
            LOGGER.error(f"Error fetching stock data for {symbol}: {e}")
            return None
    
    def _download_sync(self, symbol: str, start: str, end: str, interval: str) -> pd.DataFrame:
        """Synchronous download wrapper."""
        start_time = time.time()
        data = yf.download(symbol, start=start, end=end, interval=interval)
        elapsed = time.time() - start_time
        LOGGER.info(f"Downloaded {len(data)} rows for {symbol} in {elapsed:.2f}s")
        return data
    
    async def get_news_data(self, symbol: str) -> Optional[NewsData]:
        """Yahoo Finance doesn't provide news API, return None."""
        LOGGER.warning("Yahoo Finance provider doesn't support news data")
        return None


class IndianAPINewsProvider(IDataProvider):
    """Indian API news provider."""
    
    def __init__(self, api_key: str, request_timeout: int = 30):
        self.api_key = api_key
        self.request_timeout = request_timeout
        self.base_url = "https://stock.indianapi.in/news"
    
    async def get_stock_data(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        interval: str = "1h"
    ) -> Optional[StockData]:
        """Indian API doesn't provide stock data, return None."""
        LOGGER.warning("Indian API provider doesn't support stock data")
        return None
    
    async def get_news_data(self, symbol: str) -> Optional[NewsData]:
        """Get news data from Indian API."""
        try:
            # Remove .NS suffix for API call
            company_name = symbol.replace('.NS', '')
            
            headers = {"X-Api-Key": self.api_key}
            params = {"symbol": company_name}
            
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self.request_timeout)) as session:
                async with session.get(self.base_url, headers=headers, params=params) as response:
                    if response.status == 200:
                        news_data = await response.json()
                        
                        return NewsData(
                            symbol=symbol,
                            news=news_data if isinstance(news_data, list) else [news_data],
                            source="IndianAPI",
                            retrieved_at=datetime.now()
                        )
                    else:
                        LOGGER.error(f"News API returned status {response.status}")
                        return None
                        
        except Exception as e:
            LOGGER.error(f"Error fetching news data for {symbol}: {e}")
            return None


class CompositeDataProvider(IDataProvider):
    """Composite data provider that combines multiple providers."""
    
    def __init__(self, stock_provider: IDataProvider, news_provider: IDataProvider):
        self.stock_provider = stock_provider
        self.news_provider = news_provider
    
    async def get_stock_data(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        interval: str = "1h"
    ) -> Optional[StockData]:
        """Get stock data from stock provider."""
        return await self.stock_provider.get_stock_data(symbol, start_date, end_date, interval)
    
    async def get_news_data(self, symbol: str) -> Optional[NewsData]:
        """Get news data from news provider."""
        return await self.news_provider.get_news_data(symbol)


class DataProviderFactory:
    """Factory for creating data providers."""
    
    @staticmethod
    def create_yahoo_finance_provider(request_timeout: int = 30) -> YahooFinanceProvider:
        """Create Yahoo Finance provider."""
        return YahooFinanceProvider(request_timeout)
    
    @staticmethod
    def create_indian_api_provider(api_key: str, request_timeout: int = 30) -> IndianAPINewsProvider:
        """Create Indian API news provider."""
        return IndianAPINewsProvider(api_key, request_timeout)
    
    @staticmethod
    def create_composite_provider(
        yahoo_timeout: int = 30,
        indian_api_key: str = "",
        indian_timeout: int = 30
    ) -> CompositeDataProvider:
        """Create composite provider with both stock and news capabilities."""
        stock_provider = DataProviderFactory.create_yahoo_finance_provider(yahoo_timeout)
        news_provider = DataProviderFactory.create_indian_api_provider(indian_api_key, indian_timeout)
        return CompositeDataProvider(stock_provider, news_provider) 
