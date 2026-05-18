"""
data_provider_viz.py
────────────────────────────────────────────────────────────────
Data visualization script for downloading and storing stock/news data.
This script downloads data using agent_scripts.data_providers and stores
it in CSV files in ./buck_visualizer/data-viz/

Features:
- Async data downloading for better performance
- Comprehensive error handling and logging
- Data validation and sanitization
- Proper CSV handling with headers
- News data crawling and processing
- Type hints and documentation
"""

from __future__ import annotations

import asyncio
import csv
import json
import logging
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import nest_asyncio


# Fix imports by adding the parent directory to Python path
# This allows importing agent_scripts from the Buck_V1 directory
def fix_imports():
    """
    Fix imports by adding the parent directory to sys.path.
    Based on importfix methodology for handling relative imports.
    """
    current_dir = Path(__file__).parent.absolute()
    parent_dir = current_dir.parent
    
    # Add parent directory to sys.path if not already present
    parent_str = str(parent_dir)
    if parent_str not in sys.path:
        sys.path.insert(0, parent_str)
        
# Apply import fix before importing agent_scripts
fix_imports()

import aiohttp
import pandas as pd
import requests
from bs4 import BeautifulSoup

from agent_scripts.data_providers import (
    CompositeDataProvider,
    DataProviderFactory,
    IndianAPINewsProvider,
    YahooFinanceProvider,
)
from agent_scripts.interfaces import NewsData, StockData
from agent_scripts.config import Settings
from agent_scripts.buck import Buck

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DataVisualizationDownloader:
    """
    High-level class for downloading and storing stock and news data for visualization.
    
    This class handles:
    - Stock data download and CSV storage
    - News data download and processing
    - Error handling and retry logic
    - Data validation and sanitization
    """
    
    def __init__(
        self,
        data_dir: Optional[str] = None,
        indian_api_key: str = "",
        request_timeout: int = 30
    ):
        """
        Initialize the data downloader.
        
        Args:
            data_dir: Directory to store CSV files (defaults to ./data-viz)
            indian_api_key: API key for Indian news provider
            request_timeout: Request timeout in seconds
        """
        self.data_dir = Path(data_dir or Path(__file__).parent / 'data-viz')
        self.data_dir.mkdir(exist_ok=True)
        
        self.data_provider = DataProviderFactory.create_composite_provider(
            indian_api_key=indian_api_key,
            yahoo_timeout=request_timeout,
            indian_timeout=request_timeout
        )
        
        logger.info(f"Initialized DataVisualizationDownloader with data_dir: {self.data_dir}")
    
    async def download_stock_data(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        interval: str = "1h"
    ) -> Optional[str]:
        """
        Download stock data and save to CSV file.

    Args:
            symbol: Stock symbol (e.g., "RELIANCE")
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            interval: Data interval ("1h", "1d", etc.)
            
        Returns:
            Path to saved CSV file if successful, None otherwise
        """
        try:
            # Validate inputs
            self._validate_date_format(start_date)
            self._validate_date_format(end_date)
            self._validate_symbol(symbol)
            
            logger.info(f"Downloading stock data for {symbol} from {start_date} to {end_date}")
            
            # Download stock data
            stock_data = await self.data_provider.get_stock_data(
                symbol, start_date, end_date, interval
            )
            
            if not stock_data:
                logger.error(f"No stock data received for {symbol}")
                return None
            
            # Generate filename
            safe_symbol = self._sanitize_filename(symbol)
            filename = f"{safe_symbol}_{start_date}_{end_date}_{interval}.csv"
            csv_path = self.data_dir / filename
            
            # Save to CSV
            await self._save_stock_data_to_csv(stock_data, csv_path)
            
            logger.info(f"Stock data saved to {csv_path}")
            return str(csv_path)
            
        except Exception as e:
            logger.error(f"Error downloading stock data for {symbol}: {e}")
            return None
        
    


    # Enable nested event loops
    nest_asyncio.apply()

    
    async def download_news_data(
    self,
    symbol: str,
    include_content: bool = True,
    ) -> Optional[str]:
        """
        Download news data for `symbol`, enrich with full article text if
        desired, and save to CSV in self.data_dir.

        Returns the path to the CSV, or None on error.
        """
        try:
            self._validate_symbol(symbol)

            logger.info("Downloading news data for %s", symbol)
            buck = Buck()._create_default_data_provider()
            payload = await buck.get_news_data(symbol)
        except Exception as exc:
            logger.error("Error downloading news for %s: %s", symbol, exc)
            return None

        # ------------------------------------------------------------------
        # Build the initial DataFrame
        # ------------------------------------------------------------------
        news_df = pd.DataFrame(payload["news"])
        news_df["pub_date"] = pd.to_datetime(news_df["pub_date"])
        news_df.sort_values("pub_date", ascending=False, inplace=True)

        # ------------------------------------------------------------------
        # Optionally crawl each article’s full text (concurrently)
        # ------------------------------------------------------------------
        if include_content:
            coros = [self._crawl_article_content(url) for url in news_df["url"]]
            news_df["full_content"] = await asyncio.gather(*coros)

        # ------------------------------------------------------------------
        # Write the CSV
        # ------------------------------------------------------------------
        safe_symbol = self._sanitize_filename(symbol)
        timestamp   = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename    = f"{safe_symbol}_news_{timestamp}.csv"
        csv_path    = self.data_dir / filename

        news_df.to_csv(csv_path, index=False, encoding="utf-8")
        logger.info("News data saved to %s", csv_path)

        return str(csv_path)

    
    async def download_all_data(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        interval: str = "1h",
        include_news_content: bool = True
    ) -> Tuple[Optional[str], Optional[str]]:
        """
        Download both stock and news data concurrently.
        
        Args:
            symbol: Stock symbol
            start_date: Start date for stock data
            end_date: End date for stock data
            interval: Stock data interval
            include_news_content: Whether to crawl full news content
            
        Returns:
            Tuple of (stock_csv_path, news_csv_path)
        """
        logger.info(f"Starting concurrent download for {symbol}")
        
        # Run downloads concurrently
        stock_task = self.download_stock_data(symbol, start_date, end_date, interval)
        news_task = self.download_news_data(symbol, include_news_content)
        
        stock_path, news_path = await asyncio.gather(
            stock_task, news_task, return_exceptions=True
        )
        
        # Handle exceptions
        if isinstance(stock_path, Exception):
            logger.error(f"Stock data download failed: {stock_path}")
            stock_path = None
            
        if isinstance(news_path, Exception):
            logger.error(f"News data download failed: {news_path}")
            news_path = None
        
        return stock_path, news_path
    
    async def _save_stock_data_to_csv(
        self,
        stock_data: StockData,
        csv_path: Path
    ) -> None:
        """Save stock data to CSV file."""
        try:
            df = stock_data['data']
            
            # Add metadata columns
            df = df.copy()
            df['symbol'] = stock_data['symbol']
            df['interval'] = stock_data['interval']
            df['download_timestamp'] = datetime.now().isoformat()
            
            # Save with proper index handling
            df.to_csv(csv_path, index=True, date_format='%Y-%m-%d %H:%M:%S')
            
        except Exception as e:
            logger.error(f"Error saving stock data to CSV: {e}")
            raise
    
    async def _save_news_data_to_csv(
        self,
        news_data: NewsData,
        csv_path: Path,
        include_content: bool = True
    ) -> None:
        """Save news data to CSV file."""
        try:
            news_items = news_data['news']
            processed_items = []
            
            for item in news_items:
                processed_item = {
                    'symbol': news_data['symbol'],
                    'source': news_data['source'],
                    'retrieved_at': news_data['retrieved_at'].isoformat(),
                    'title': item.get('title', ''),
                    'date': item.get('date', ''),
                    'url': item.get('url', ''),
                    'sentiment': item.get('sentiment', ''),
                    'summary': item.get('summary', ''),
                    'tags': json.dumps(item.get('tags', [])),
                    'image_url': item.get('image_url', ''),
                    'video_url': item.get('video_url', ''),
                }
                
                # Add full content if requested
                if include_content and item.get('url'):
                    content = await self._crawl_article_content(item['url'])
                    processed_item['full_content'] = content
                else:
                    processed_item['full_content'] = item.get('content', '')
                
                processed_items.append(processed_item)
            
            # Write to CSV
            if processed_items:
                df = pd.DataFrame(processed_items)
                df.to_csv(csv_path, index=False, encoding='utf-8')
            
        except Exception as e:
            logger.error(f"Error saving news data to CSV: {e}")
            raise
    
    async def _crawl_article_content(self, url: str) -> str:
        """
        Crawl full article content from URL.
        
        Args:
            url: Article URL
            
        Returns:
            Extracted text content
        """
        try:
            async with aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=30)
            ) as session:
                async with session.get(url, headers={'User-Agent': 'Mozilla/5.0'}) as response:
                    if response.status == 200:
                        html = await response.text()
                        soup = BeautifulSoup(html, 'html.parser')
                        
                        # Remove script and style elements
                        for script in soup(["script", "style"]):
                            script.decompose()
                        
                        # Extract text
                        text = soup.get_text()
                        
                        # Clean up text
                        lines = (line.strip() for line in text.splitlines())
                        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
                        text = ' '.join(chunk for chunk in chunks if chunk)
                        
                        return text[:10000]  # Limit length
                    
        except Exception as e:
            logger.warning(f"Failed to crawl content from {url}: {e}")
        
        return ""
    
    def _validate_date_format(self, date_str: str) -> None:
        """Validate date format (YYYY-MM-DD)."""
        try:
            datetime.strptime(date_str, '%Y-%m-%d')
        except ValueError:
            raise ValueError(f"Invalid date format: {date_str}. Expected YYYY-MM-DD")
    
    def _validate_symbol(self, symbol: str) -> None:
        """Validate stock symbol."""
        if not symbol or not isinstance(symbol, str):
            raise ValueError("Symbol must be a non-empty string")
        
        # Basic symbol validation
        if not re.match(r'^[A-Z0-9.-]+$', symbol.upper()):
            raise ValueError(f"Invalid symbol format: {symbol}")
    
    def _sanitize_filename(self, filename: str) -> str:
        """Sanitize filename for safe file system usage."""
        # Remove or replace problematic characters
        sanitized = re.sub(r'[<>:"/\\|?*]', '_', filename)
        sanitized = re.sub(r'\.+$', '', sanitized)  # Remove trailing dots
        return sanitized[:100]  # Limit length


# Convenience functions for backward compatibility and ease of use
async def download_data(
    symbol: str,
    start_date: str,
    end_date: str,
    interval: str = "1h",
    indian_api_key: str = ""
) -> Tuple[Optional[str], Optional[str]]:
    """
    Convenience function to download both stock and news data.
    
    Args:
        symbol: Stock symbol
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        interval: Data interval
        indian_api_key: API key for news provider
        
    Returns:
        Tuple of (stock_csv_path, news_csv_path)
    """
    downloader = DataVisualizationDownloader(indian_api_key=indian_api_key)
    return await downloader.download_all_data(
        symbol, start_date, end_date, interval
    )


# Main execution example
async def main():
    """CLI interface for the data downloader."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Download stock and news data for visualization',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s RELIANCE 2024-01-01 2024-01-07
  %(prog)s BHEL 2025-06-10 2025-06-20 1h
  %(prog)s INFY 2024-12-01 2024-12-31 1d --api-key your_news_api_key
        """
    )
    
    parser.add_argument('symbol', help='Stock symbol (e.g., RELIANCE, BHEL)')
    parser.add_argument('start_date', help='Start date (YYYY-MM-DD)')
    parser.add_argument('end_date', help='End date (YYYY-MM-DD)')
    parser.add_argument('interval', nargs='?', default='1h', 
                       help='Data interval (default: 1h). Options: 1h, 1d, 1wk, 1mo')
    parser.add_argument('--api-key', default='', 
                       help='Indian API key for news data (optional)')
    parser.add_argument('--data-dir', 
                       help='Directory to save CSV files (default: ./data-viz)')
    parser.add_argument('--no-news-content', action='store_true',
                       help='Skip crawling full news article content')
    parser.add_argument('--timeout', type=int, default=30,
                       help='Request timeout in seconds (default: 30)')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose logging')
    
    args = parser.parse_args()
    
    # Configure logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.debug("Verbose logging enabled")
    
    try:
        logger.info(f"Starting data download for {args.symbol}")
        logger.info(f"Date range: {args.start_date} to {args.end_date}")
        logger.info(f"Interval: {args.interval}")
        
        downloader = DataVisualizationDownloader(
            data_dir=args.data_dir,
            indian_api_key=args.api_key,
            request_timeout=args.timeout
        )
        
        stock_path, news_path = await downloader.download_all_data(
            symbol=args.symbol,
            start_date=args.start_date,
            end_date=args.end_date,
            interval=args.interval,
            include_news_content=not args.no_news_content
        )
        
        # Print results
        print("\n" + "="*60)
        print("DOWNLOAD COMPLETE")
        print("="*60)
        
        if stock_path:
            print(f"✅ Stock data saved to: {stock_path}")
        else:
            print("❌ Stock data download failed")
            
        if news_path:
            print(f"✅ News data saved to: {news_path}")
        else:
            print("❌ News data download failed or no news available")
        
        print("="*60)
        
        # Exit with appropriate code
        exit_code = 0
        if not stock_path and not news_path:
            exit_code = 1
        elif not stock_path or not news_path:
            exit_code = 2  # Partial success
            
        return exit_code
        
    except KeyboardInterrupt:
        logger.info("Download interrupted by user")
        print("\n❌ Download cancelled by user")
        return 130
        
    except Exception as e:
        logger.error(f"Download failed: {e}")
        print(f"\n❌ Download failed: {e}")
        return 1


if __name__ == "__main__":
    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n❌ Interrupted")
        sys.exit(130)
















