from __future__ import annotations

import argparse
import asyncio
import matplotlib.pyplot as plt
import pandas as pd

from data_provider_viz import DataVisualizationDownloader, fix_imports

# Ensure we can import agent_scripts
fix_imports()


async def fetch_stock(symbol: str, start: str, end: str, interval: str) -> pd.DataFrame:
    downloader = DataVisualizationDownloader()
    csv_path = await downloader.download_stock_data(symbol, start, end, interval)
    if not csv_path:
        raise RuntimeError("Failed to download stock data")
    return pd.read_csv(csv_path, parse_dates=True, index_col=0)


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_index()
    df['MA20'] = df['Close'].rolling(20).mean()
    df['MA50'] = df['Close'].rolling(50).mean()
    return df


def plot(df: pd.DataFrame, symbol: str) -> None:
    plt.figure(figsize=(10, 6))
    plt.plot(df.index, df['Close'], label='Close')
    plt.plot(df.index, df['MA20'], label='MA20')
    plt.plot(df.index, df['MA50'], label='MA50')
    plt.title(f'{symbol} Price with Moving Averages')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.tight_layout()
    plt.show()


async def main() -> None:
    parser = argparse.ArgumentParser(description='Plot price with moving averages')
    parser.add_argument('symbol')
    parser.add_argument('start_date')
    parser.add_argument('end_date')
    parser.add_argument('--interval', default='1d')
    args = parser.parse_args()

    df = await fetch_stock(args.symbol, args.start_date, args.end_date, args.interval)
    df = preprocess(df)
    plot(df, args.symbol)


if __name__ == '__main__':
    asyncio.run(main())
