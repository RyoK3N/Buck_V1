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
    delta = df['Close'].diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    roll_up = up.rolling(14).mean()
    roll_down = down.rolling(14).mean()
    rs = roll_up / roll_down
    df['RSI'] = 100 - (100 / (1 + rs))
    return df


DESCRIPTION = """
Relative Strength Index identifies overbought (above 70) and oversold (below 30)
zones. Values crossing those lines can signal potential reversals. Hover for
exact readings.
"""


def plot(df: pd.DataFrame, symbol: str) -> None:
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.02)
    fig.add_trace(go.Scatter(x=df.index, y=df['Close'], name='Close'), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], name='RSI'), row=2, col=1)
    fig.add_hline(y=70, line_dash='dash', line_color='red', row=2, col=1)
    fig.add_hline(y=30, line_dash='dash', line_color='green', row=2, col=1)

    fig.update_layout(
        title=f'{symbol} Price and RSI',
        xaxis_title='Date',
        yaxis_title='Price',
        yaxis2_title='RSI',
        hovermode='x unified'
    )

    fig.show()
    print(DESCRIPTION)


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
