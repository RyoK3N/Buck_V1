from __future__ import annotations

import argparse
import asyncio
import plotly.graph_objects as go
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


DESCRIPTION = """
Interactive price chart with 20- and 50-period moving averages. The moving
averages help reveal trend direction. Crossovers may hint at shifts in
momentum. Hover to inspect exact prices.
"""


def plot(df: pd.DataFrame, symbol: str) -> None:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df["Close"], mode="lines", name="Close"))
    fig.add_trace(go.Scatter(x=df.index, y=df["MA20"], mode="lines", name="MA20"))
    fig.add_trace(go.Scatter(x=df.index, y=df["MA50"], mode="lines", name="MA50"))
    fig.update_layout(
        title=f"{symbol} Price with Moving Averages",
        xaxis_title="Date",
        yaxis_title="Price",
        hovermode="x unified",
    )
    fig.show()
    print(DESCRIPTION)


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
