from __future__ import annotations

import argparse
import asyncio
import matplotlib.pyplot as plt
import pandas as pd

from data_provider_viz import DataVisualizationDownloader, fix_imports

fix_imports()

async def fetch(symbol: str, start: str, end: str, interval: str) -> pd.DataFrame:
    downloader = DataVisualizationDownloader()
    csv = await downloader.download_stock_data(symbol, start, end, interval)
    if not csv:
        raise RuntimeError('Failed to download stock data')
    return pd.read_csv(csv, parse_dates=True, index_col=0)


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_index()
    ma = df['Close'].rolling(20).mean()
    std = df['Close'].rolling(20).std()
    df['MA20'] = ma
    df['Upper'] = ma + 2 * std
    df['Lower'] = ma - 2 * std
    return df


def plot(df: pd.DataFrame, symbol: str) -> None:
    plt.figure(figsize=(10,6))
    plt.plot(df.index, df['Close'], label='Close')
    plt.plot(df.index, df['MA20'], label='MA20')
    plt.fill_between(df.index, df['Upper'], df['Lower'], color='gray', alpha=0.3, label='Bollinger Band')
    plt.title(f'{symbol} Bollinger Bands')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.tight_layout()
    plt.show()


async def main() -> None:
    parser = argparse.ArgumentParser(description='Bollinger Bands plot')
    parser.add_argument('symbol')
    parser.add_argument('start_date')
    parser.add_argument('end_date')
    parser.add_argument('--interval', default='1d')
    args = parser.parse_args()

    df = await fetch(args.symbol, args.start_date, args.end_date, args.interval)
    df = preprocess(df)
    plot(df, args.symbol)


if __name__ == '__main__':
    asyncio.run(main())
