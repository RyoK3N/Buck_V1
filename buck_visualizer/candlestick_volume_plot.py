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
    return df


def plot(df: pd.DataFrame, symbol: str) -> None:
    fig, ax1 = plt.subplots(figsize=(10,6))
    width = 0.6
    up = df['Close'] >= df['Open']
    down = ~up
    ax1.bar(df.index[up], df['Close'][up]-df['Open'][up], width, bottom=df['Open'][up], color='g')
    ax1.bar(df.index[down], df['Open'][down]-df['Close'][down], width, bottom=df['Close'][down], color='r')
    ax1.vlines(df.index, df['Low'], df['High'], color='k', linewidth=0.5)
    ax1.set_title(f'{symbol} Candlestick')
    ax1.set_ylabel('Price')

    ax2 = ax1.twinx()
    ax2.bar(df.index, df['Volume'], width, alpha=0.3, color='b')
    ax2.set_ylabel('Volume')

    fig.autofmt_xdate()
    plt.tight_layout()
    plt.show()


async def main() -> None:
    parser = argparse.ArgumentParser(description='Candlestick with volume plot')
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
