from __future__ import annotations

import argparse
import asyncio
import plotly.express as px
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
    df['Returns'] = df['Close'].pct_change()
    return df


DESCRIPTION = """
Histogram of daily percentage returns. The distribution's shape highlights
volatility and skewness. Use the interactive view to inspect tail events.
"""


def plot(df: pd.DataFrame, symbol: str) -> None:
    fig = px.histogram(df, x='Returns', nbins=50, title=f'{symbol} Daily Returns Distribution')
    fig.update_layout(xaxis_title='Return', yaxis_title='Frequency')
    fig.show()
    print(DESCRIPTION)


async def main() -> None:
    parser = argparse.ArgumentParser(description='Daily returns histogram')
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
