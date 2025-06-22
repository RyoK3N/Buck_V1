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
    ma = df['Close'].rolling(20).mean()
    std = df['Close'].rolling(20).std()
    df['MA20'] = ma
    df['Upper'] = ma + 2 * std
    df['Lower'] = ma - 2 * std
    return df


DESCRIPTION = """
Bollinger Bands consist of a moving average with upper and lower bands set two
standard deviations away. Price touching the bands can indicate overbought or
oversold conditions. Hover to explore.
"""


def plot(df: pd.DataFrame, symbol: str) -> None:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df['Close'], name='Close'))
    fig.add_trace(go.Scatter(x=df.index, y=df['MA20'], name='MA20'))
    fig.add_trace(go.Scatter(
        x=df.index, y=df['Upper'], name='Upper Band', line=dict(color='lightgrey'),
    ))
    fig.add_trace(go.Scatter(
        x=df.index, y=df['Lower'], name='Lower Band', fill='tonexty',
        line=dict(color='lightgrey')
    ))

    fig.update_layout(
        title=f'{symbol} Bollinger Bands',
        xaxis_title='Date',
        yaxis_title='Price',
        hovermode='x unified'
    )

    fig.show()
    print(DESCRIPTION)


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
