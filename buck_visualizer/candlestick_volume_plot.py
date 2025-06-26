from __future__ import annotations

import argparse
import asyncio
import plotly.graph_objects as go
from plotly.subplots import make_subplots
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


DESCRIPTION = """
Interactive candlestick chart with volume. Candles illustrate open, high,
low, and close prices, while the bar chart indicates trading volume.
Zoom and hover for details. Large volume on big moves can validate trend
strength.
"""


def plot(df: pd.DataFrame, symbol: str) -> None:
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        vertical_spacing=0.02, row_heights=[0.7, 0.3])

    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df['Open'],
        high=df['High'],
        low=df['Low'],
        close=df['Close'],
        name='Price'
    ), row=1, col=1)

    fig.add_trace(go.Bar(x=df.index, y=df['Volume'], name='Volume'), row=2, col=1)

    fig.update_layout(
        title=f'{symbol} Candlestick with Volume',
        xaxis_title='Date',
        yaxis_title='Price',
        yaxis2_title='Volume',
        hovermode='x unified'
    )

    fig.show()
    print(DESCRIPTION)


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
