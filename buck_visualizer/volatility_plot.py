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
    returns = df['Close'].pct_change()
    df['Volatility'] = returns.rolling(30).std() * (252 ** 0.5)
    return df


def plot(df: pd.DataFrame, symbol: str) -> None:
    fig, ax = plt.subplots(figsize=(10,6))
    ax.plot(df.index, df['Volatility'], label='30D Volatility')
    ax.set_title(f'{symbol} Rolling Volatility')
    ax.set_ylabel('Volatility (annualized)')
    ax.set_xlabel('Date')
    plt.tight_layout()
    plt.show()


async def main() -> None:
    parser = argparse.ArgumentParser(description='Historical volatility plot')
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
