"""
market_forecaster.interfaces
──────────────────────────────
Abstract interfaces and protocols for the stock analysis system.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Protocol, TypedDict, Union
from datetime import datetime
import pandas as pd


class StockData(TypedDict):
    """Stock data structure."""
    symbol: str
    data: pd.DataFrame
    interval: str
    start_date: str
    end_date: str


class NewsData(TypedDict):
    """News data structure."""
    symbol: str
    news: List[Dict[str, Any]]
    source: str
    retrieved_at: datetime


class AnalysisResult(TypedDict):
    """Analysis result structure."""
    symbol: str
    analysis_type: str
    data: Dict[str, Any]
    timestamp: datetime
    confidence: float


class Forecast(TypedDict):
    """Forecast result structure."""
    date: str
    open: float
    high: float
    low: float
    close: float
    confidence: float
    reasoning: str


class ITool(Protocol):
    """Protocol for analysis tools."""
    
    @property
    def name(self) -> str:
        """Tool name."""
        ...
    
    @property
    def description(self) -> str:
        """Tool description."""
        ...
    
    def execute(self, data: Any, **kwargs) -> Any:
        """Execute the tool with given data."""
        ...


class IDataProvider(ABC):
    """Abstract interface for data providers."""
    
    @abstractmethod
    async def get_stock_data(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        interval: str = "1h"
    ) -> Optional[StockData]:
        """Get stock data for given parameters."""
        pass
    
    @abstractmethod
    async def get_news_data(self, symbol: str) -> Optional[NewsData]:
        """Get news data for given symbol."""
        pass


class IAnalyzer(ABC):
    """Abstract interface for stock analyzers."""
    
    @abstractmethod
    def analyze(self, data: StockData, **kwargs) -> AnalysisResult:
        """Analyze stock data."""
        pass


class IPredictor(ABC):
    """Abstract interface for stock predictors."""
    
    @abstractmethod
    async def predict(
        self,
        analysis_results: List[AnalysisResult],
        **kwargs
    ) -> Forecast:
        """Generate stock forecast based on analysis results."""
        pass


class IRepository(ABC):
    """Abstract interface for data persistence."""
    
    @abstractmethod
    async def save_stock_data(self, data: StockData) -> bool:
        """Save stock data."""
        pass
    
    @abstractmethod
    async def save_analysis_result(self, result: AnalysisResult) -> bool:
        """Save analysis result."""
        pass
    
    @abstractmethod
    async def save_forecast(self, forecast: Forecast) -> bool:
        """Save forecast."""
        pass
    
    @abstractmethod
    async def get_latest_analysis(self, symbol: str) -> Optional[List[AnalysisResult]]:
        """Get latest analysis for symbol."""
        pass


class INotificationService(ABC):
    """Abstract interface for notifications."""
    
    @abstractmethod
    async def notify(self, message: str, level: str = "info") -> bool:
        """Send notification."""
        pass


class BuckConfig(TypedDict):
    """Buck configuration."""
    openai_api_key: str
    chat_model: str
    temperature: float
    max_tokens: int
    news_items: int
    log_level: str 
