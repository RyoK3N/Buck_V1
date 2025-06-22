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
    delta = df['Close'].diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    roll_up = up.rolling(14).mean()
    roll_down = down.rolling(14).mean()
    rs = roll_up / roll_down
    df['RSI'] = 100 - (100 / (1 + rs))
    return df


def plot(df: pd.DataFrame, symbol: str) -> None:
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10,7), sharex=True)
    ax1.plot(df.index, df['Close'], label='Close')
    ax1.set_title(f'{symbol} Price')

    ax2.plot(df.index, df['RSI'], label='RSI')
    ax2.axhline(70, color='r', linestyle='--')
    ax2.axhline(30, color='g', linestyle='--')
    ax2.set_title('Relative Strength Index')
    ax2.legend()

    plt.tight_layout()
    plt.show()


async def main() -> None:
    parser = argparse.ArgumentParser(description='RSI plot')
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
