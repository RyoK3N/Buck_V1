"""
Buck_V1.buck
────────────────────────────────
Our Agent Buck that orchestrates data collection, analysis, and prediction.
"""

from __future__ import annotations
import asyncio
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any

from .interfaces import (
    IDataProvider, IAnalyzer, IPredictor, 
    StockData, NewsData, AnalysisResult, Forecast, BuckConfig
)
from .data_providers import DataProviderFactory
from .analyzers import AnalyzerFactory
from .predictors import PredictorFactory
from .config import SETTINGS, LOGGER


class Buck:
    """
    Main Stock Agent for comprehensive stock analysis and prediction.
    
    This agent orchestrates the entire workflow:
    1. Data collection (stock data + news)
    2. Technical analysis using multiple tools
    3. Sentiment analysis from news
    4. AI-powered prediction using OpenAI
    5. Result storage and reporting
    """
    
    def __init__(
        self,
        config: Optional[AgentConfig] = None,
        data_provider: Optional[IDataProvider] = None,
        analyzer: Optional[IAnalyzer] = None,
        predictor: Optional[IPredictor] = None
    ):
        """Initialize the Stock Agent with configuration and components."""
        self.config = config or self._default_config()
        
        # Initialize components
        self.data_provider = data_provider or self._create_default_data_provider()
        self.analyzer = analyzer or AnalyzerFactory.create_composite_analyzer()
        self.predictor = predictor or PredictorFactory.create_default_predictor()
        
        # State management
        self._analysis_cache: Dict[str, List[AnalysisResult]] = {}
        self._forecast_cache: Dict[str, Forecast] = {}
        
        LOGGER.info("Stock Agent initialized successfully")
    
    async def analyze_and_predict(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        interval: str = "1h",
        save_results: bool = True
    ) -> Dict[str, Any]:
        """
        Complete workflow: collect data, analyze, and predict.
        
        Args:
            symbol: Stock symbol (e.g., 'BHEL.NS')
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format  
            interval: Data interval (1h, 30m, 1d, etc.)
            save_results: Whether to save results to files
            
        Returns:
            Complete analysis and prediction results
        """
        try:
            LOGGER.info(f"Starting complete analysis for {symbol}")
            
            # Step 1: Collect data
            stock_data, news_data = await self._collect_data(symbol, start_date, end_date, interval)
            
            if not stock_data:
                raise ValueError(f"No stock data available for {symbol}")
            
            # Step 2: Perform analysis
            analysis_results = await self._perform_analysis(stock_data, news_data)
            
            # Step 3: Generate prediction
            forecast = await self._generate_prediction(analysis_results)
            
            # Step 4: Compile results
            results = {
                'symbol': symbol,
                'timestamp': datetime.now().isoformat(),
                'data_info': {
                    'start_date': start_date,
                    'end_date': end_date,
                    'interval': interval,
                    'data_points': len(stock_data['data']),
                    'news_available': news_data is not None
                },
                'analysis_results': analysis_results,
                'forecast': forecast,
                'metadata': {
                    'agent_version': '1.0.0',
                    'model_used': self.config['chat_model'],
                    'analysis_confidence': self._calculate_overall_confidence(analysis_results),
                    'prediction_confidence': forecast['confidence']
                }
            }
            
            # Step 5: Save results if requested
            if save_results:
                await self._save_results(results)
            
            LOGGER.info(f"Complete analysis finished for {symbol}")
            return results
            
        except Exception as e:
            LOGGER.error(f"Analysis and prediction failed for {symbol}: {e}")
            return {
                'symbol': symbol,
                'timestamp': datetime.now().isoformat(),
                'error': str(e),
                'status': 'failed'
            }
    
    async def batch_analyze(
        self,
        symbols: List[str],
        start_date: str,
        end_date: str,
        interval: str = "1h",
        max_concurrent: int = 3
    ) -> Dict[str, Any]:
        """
        Analyze multiple stocks concurrently.
        
        Args:
            symbols: List of stock symbols
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            interval: Data interval
            max_concurrent: Maximum concurrent analyses
            
        Returns:
            Results for all symbols
        """
        LOGGER.info(f"Starting batch analysis for {len(symbols)} symbols")
        
        # Create semaphore to limit concurrency
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def analyze_single(symbol: str) -> tuple[str, Dict[str, Any]]:
            async with semaphore:
                result = await self.analyze_and_predict(symbol, start_date, end_date, interval)
                return symbol, result
        
        # Run analyses concurrently
        tasks = [analyze_single(symbol) for symbol in symbols]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Compile results
        batch_results = {
            'batch_info': {
                'symbols': symbols,
                'start_date': start_date,
                'end_date': end_date,
                'interval': interval,
                'timestamp': datetime.now().isoformat(),
                'total_symbols': len(symbols)
            },
            'results': {},
            'summary': {
                'successful': 0,
                'failed': 0,
                'avg_confidence': 0.0
            }
        }
        
        total_confidence = 0
        for result in results:
            if isinstance(result, Exception):
                LOGGER.error(f"Batch analysis error: {result}")
                continue
                
            symbol, analysis = result
            batch_results['results'][symbol] = analysis
            
            if 'error' in analysis:
                batch_results['summary']['failed'] += 1
            else:
                batch_results['summary']['successful'] += 1
                total_confidence += analysis.get('metadata', {}).get('prediction_confidence', 0)
        
        # Calculate average confidence
        if batch_results['summary']['successful'] > 0:
            batch_results['summary']['avg_confidence'] = (
                total_confidence / batch_results['summary']['successful']
            )
        
        LOGGER.info(f"Batch analysis completed: {batch_results['summary']}")
        return batch_results
    
    async def _collect_data(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        interval: str
    ) -> tuple[Optional[StockData], Optional[NewsData]]:
        """Collect stock and news data."""
        LOGGER.info(f"Collecting data for {symbol}")
        
        # Collect data concurrently
        stock_task = self.data_provider.get_stock_data(symbol, start_date, end_date, interval)
        news_task = self.data_provider.get_news_data(symbol)
        
        stock_data, news_data = await asyncio.gather(
            stock_task, news_task, return_exceptions=True
        )
        
        # Handle exceptions
        if isinstance(stock_data, Exception):
            LOGGER.error(f"Failed to get stock data: {stock_data}")
            stock_data = None
        
        if isinstance(news_data, Exception):
            LOGGER.error(f"Failed to get news data: {news_data}")
            news_data = None
        
        return stock_data, news_data
    
    async def _perform_analysis(
        self,
        stock_data: StockData,
        news_data: Optional[NewsData]
    ) -> List[AnalysisResult]:
        """Perform comprehensive analysis."""
        LOGGER.info(f"Performing analysis for {stock_data['symbol']}")
        
        # Run analysis in executor to avoid blocking
        loop = asyncio.get_event_loop()
        
        # Define a wrapper function to properly pass keyword arguments
        def analyze_with_kwargs():
            return self.analyzer.analyze(stock_data, news_data=news_data)
        
        analysis_result = await loop.run_in_executor(
            None,
            analyze_with_kwargs
        )
        
        # Cache results
        symbol = stock_data['symbol']
        self._analysis_cache[symbol] = [analysis_result]
        
        return [analysis_result]
    
    async def _generate_prediction(self, analysis_results: List[AnalysisResult]) -> Forecast:
        """Generate AI-powered prediction."""
        symbol = analysis_results[0]['symbol']
        LOGGER.info(f"Generating prediction for {symbol}")
        
        forecast = await self.predictor.predict(analysis_results)
        
        # Cache forecast
        self._forecast_cache[symbol] = forecast
        
        return forecast
    
    async def _save_results(self, results: Dict[str, Any]) -> None:
        """Save analysis results to files."""
        try:
            symbol = results['symbol']
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            # Create output directory
            output_dir = Path('output')
            output_dir.mkdir(exist_ok=True)
            
            # Save complete results
            results_file = output_dir / f"{symbol}_{timestamp}_analysis.json"
            with results_file.open('w') as f:
                json.dump(results, f, indent=2, default=str)
            
            # Save forecast separately
            if 'forecast' in results:
                forecast_file = output_dir / f"{symbol}_{timestamp}_forecast.json"
                with forecast_file.open('w') as f:
                    json.dump(results['forecast'], f, indent=2, default=str)
            
            LOGGER.info(f"Results saved to {results_file}")
            
        except Exception as e:
            LOGGER.error(f"Failed to save results: {e}")
    
    def _calculate_overall_confidence(self, analysis_results: List[AnalysisResult]) -> float:
        """Calculate overall confidence from analysis results."""
        if not analysis_results:
            return 0.0
        
        total_confidence = sum(result['confidence'] for result in analysis_results)
        return total_confidence / len(analysis_results)
    
    def _default_config(self) -> AgentConfig:
        """Create default configuration."""
        return AgentConfig(
            openai_api_key=SETTINGS.openai_api_key,
            chat_model=SETTINGS.chat_model,
            temperature=SETTINGS.temperature,
            max_tokens=SETTINGS.max_completion_tokens,
            news_items=SETTINGS.news_items,
            log_level=SETTINGS.log_level
        )
    
    def _create_default_data_provider(self) -> IDataProvider:
        """Create default data provider."""
        # Use Indian API key if available, otherwise empty string
        indian_api_key = getattr(SETTINGS, 'indian_api_key', 'sk-live-7rblZdIQfghdIsfucGOdos5iPSXsevk0zcKbtTev')
        
        return DataProviderFactory.create_composite_provider(
            yahoo_timeout=30,
            indian_api_key=indian_api_key,
            indian_timeout=30
        )
    
    # Context manager support
    async def __aenter__(self):
        """Async context manager entry."""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        # Cleanup if needed
        pass
    
    # Additional utility methods
    def get_cached_analysis(self, symbol: str) -> Optional[List[AnalysisResult]]:
        """Get cached analysis results for a symbol."""
        return self._analysis_cache.get(symbol)
    
    def get_cached_forecast(self, symbol: str) -> Optional[Forecast]:
        """Get cached forecast for a symbol."""
        return self._forecast_cache.get(symbol)
    
    def clear_cache(self):
        """Clear all cached results."""
        self._analysis_cache.clear()
        self._forecast_cache.clear()
        LOGGER.info("Cache cleared")


class BuckFactory:
    """Factory for creating Stock Agent instances."""
    
    @staticmethod
    def create_default_agent() -> Buck:
        """Create agent with default configuration."""
        return Buck()
    
    @staticmethod
    def create_custom_agent(
        config: AgentConfig,
        data_provider: Optional[IDataProvider] = None,
        analyzer: Optional[IAnalyzer] = None,
        predictor: Optional[IPredictor] = None
    ) -> Buck:
        """Create agent with custom components."""
        return Buck(config, data_provider, analyzer, predictor)
    
    @staticmethod
    def create_production_agent(
        openai_api_key: str,
        indian_api_key: str = "",
        model: str = "gpt-4o"
    ) -> Buck:
        """Create production-ready agent."""
        config = AgentConfig(
            openai_api_key=openai_api_key,
            chat_model=model,
            temperature=0.1,
            max_tokens=500,
            news_items=10,
            log_level="INFO"
        )
        
        data_provider = DataProviderFactory.create_composite_provider(
            yahoo_timeout=30,
            indian_api_key=indian_api_key,
            indian_timeout=30
        )
        
        analyzer = AnalyzerFactory.create_composite_analyzer()
        predictor = PredictorFactory.create_openai_predictor(openai_api_key, model, 0.1)
        
        return Buck(config, data_provider, analyzer, predictor) 
