from __future__ import annotations

import argparse
import asyncio
import plotly.graph_objects as go
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


DESCRIPTION = """
Volatility measures the magnitude of price swings. This plot shows the
annualized standard deviation of daily returns over the last 30 days.
Rising volatility often signals uncertainty. Hover for precise values.
"""


def plot(df: pd.DataFrame, symbol: str) -> None:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df['Volatility'], name='30D Volatility'))
    fig.update_layout(
        title=f'{symbol} Rolling Volatility',
        xaxis_title='Date',
        yaxis_title='Volatility (annualized)',
        hovermode='x unified'
    )
    fig.show()
    print(DESCRIPTION)


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
