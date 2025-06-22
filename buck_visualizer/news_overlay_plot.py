from __future__ import annotations

import argparse
import asyncio
from datetime import datetime
import matplotlib.pyplot as plt
import pandas as pd

from data_provider_viz import DataVisualizationDownloader, fix_imports

fix_imports()

async def fetch(symbol: str, start: str, end: str, interval: str, api_key: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    downloader = DataVisualizationDownloader(indian_api_key=api_key)
    stock_csv, news_csv = await downloader.download_all_data(symbol, start, end, interval)
    if not stock_csv:
        raise RuntimeError('Failed to download stock data')
    stock_df = pd.read_csv(stock_csv, parse_dates=True, index_col=0)
    news_df = pd.DataFrame()
    if news_csv:
        news_df = pd.read_csv(news_csv, parse_dates=['pub_date'])
    return stock_df, news_df


def preprocess(stock: pd.DataFrame, news: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    stock = stock.sort_index()
    times = pd.Series(dtype='datetime64[ns]')
    if not news.empty:
        times = pd.to_datetime(news['pub_date']).dt.floor('min')
    return stock, times


def plot(stock: pd.DataFrame, news_times: pd.Series, symbol: str) -> None:
    plt.figure(figsize=(10,6))
    plt.plot(stock.index, stock['Close'], label='Close')
    for t in news_times:
        plt.axvline(t, color='r', linestyle='--', alpha=0.4)
    plt.title(f'{symbol} Price with News Releases')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.tight_layout()
    plt.show()


async def main() -> None:
    parser = argparse.ArgumentParser(description='Overlay news times on price chart')
    parser.add_argument('symbol')
    parser.add_argument('start_date')
    parser.add_argument('end_date')
    parser.add_argument('--interval', default='1d')
    parser.add_argument('--api-key', default='')
    args = parser.parse_args()

    stock_df, news_df = await fetch(args.symbol, args.start_date, args.end_date, args.interval, args.api_key)
    stock_df, times = preprocess(stock_df, news_df)
    plot(stock_df, times, args.symbol)


if __name__ == '__main__':
    asyncio.run(main())
