"""
UI.backend.visualizer
──────────────────────
Chart-building logic for the Buck Visualizer.

Provides 14 interactive Plotly chart types covering price action,
trend, momentum, volatility, and volume analysis.  Every chart
includes range selectors, spike-lines / crosshair, hover detail,
and annotation helpers where applicable.

Sources & references used for indicator calculations:
  - https://www.quantifiedstrategies.com/trading-indicators/
  - https://www.quantvps.com/blog/best-trend-indicators-guide
  - https://technical-analysis-library-in-python.readthedocs.io/
  - https://plotly.com/python/candlestick-charts/
"""

from __future__ import annotations

import json
import math
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ─── Shared layout helpers ────────────────────────────────────────────────────

_PALETTE = {
    "price":  "#1f77b4",
    "ma20":   "#ff7f0e",
    "ma50":   "#2ca02c",
    "ma200":  "#d62728",
    "upper":  "rgba(150,150,150,0.35)",
    "lower":  "rgba(150,150,150,0.35)",
    "fill":   "rgba(173,216,230,0.20)",
    "bull":   "#26a69a",
    "bear":   "#ef5350",
    "vol":    "rgba(100,100,200,0.45)",
    "sig":    "#ff9800",
    "span_a": "rgba(76,175,80,0.12)",
    "span_b": "rgba(244,67,54,0.12)",
}

_RANGE_BUTTONS = [
    dict(count=7,  label="1W",  step="day",   stepmode="backward"),
    dict(count=1,  label="1M",  step="month", stepmode="backward"),
    dict(count=3,  label="3M",  step="month", stepmode="backward"),
    dict(count=6,  label="6M",  step="month", stepmode="backward"),
    dict(step="all", label="All"),
]


def _base_layout(title: str, **extra: Any) -> dict:
    """Return a shared layout dict with crosshair, range buttons, theme."""
    layout: dict = dict(
        title=dict(text=title, x=0.5, font=dict(size=16)),
        template="plotly_white",
        hovermode="x unified",
        hoverlabel=dict(bgcolor="white", font_size=12),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=55, r=20, t=60, b=40),
        xaxis=dict(
            rangeslider=dict(visible=False),
            rangeselector=dict(buttons=_RANGE_BUTTONS, font=dict(size=10)),
            showspikes=True, spikemode="across", spikesnap="cursor",
            spikecolor="#999", spikethickness=0.8, spikedash="dot",
        ),
        yaxis=dict(
            showspikes=True, spikemode="across", spikesnap="cursor",
            spikecolor="#999", spikethickness=0.8, spikedash="dot",
        ),
    )
    layout.update(extra)
    return layout


def _subplot_layout(title: str, rows: int, row_heights: List[float],
                    yaxis_titles: List[str], **extra: Any) -> dict:
    """Layout additions for subplot-based charts."""
    layout = _base_layout(title, **extra)
    for i, yt in enumerate(yaxis_titles, start=1):
        key = "yaxis" if i == 1 else f"yaxis{i}"
        ax = layout.get(key, {})
        ax["title_text"] = yt
        ax["showspikes"] = True
        ax["spikemode"] = "across"
        ax["spikesnap"] = "cursor"
        ax["spikecolor"] = "#999"
        ax["spikethickness"] = 0.8
        ax["spikedash"] = "dot"
        layout[key] = ax
    return layout


# ─── Catalogue ────────────────────────────────────────────────────────────────

CHART_CATALOGUE = [
    # ── Price action ──────────────────────────────────────────────────────
    {
        "id": "candlestick",
        "name": "Candlestick + Volume",
        "description": (
            "Interactive candlestick chart with coloured volume bars. "
            "Range selector, crosshair, and hover detail for every bar. "
            "Large volume on big moves validates trend strength."
        ),
    },
    {
        "id": "price_ma",
        "name": "Price + Moving Averages (20 / 50 / 200)",
        "description": (
            "Close price with SMA-20, SMA-50, and SMA-200 overlays. "
            "Golden cross (50 crossing above 200) and death cross annotations "
            "are highlighted automatically."
        ),
    },
    # ── Trend ─────────────────────────────────────────────────────────────
    {
        "id": "bollinger",
        "name": "Bollinger Bands",
        "description": (
            "20-period moving average with upper/lower bands at ±2 standard "
            "deviations. Bandwidth indicator in lower panel shows expansion "
            "and contraction of volatility."
        ),
    },
    {
        "id": "ichimoku",
        "name": "Ichimoku Cloud",
        "description": (
            "Full Ichimoku Kinkō Hyō: Tenkan-sen (9), Kijun-sen (26), "
            "Senkou Span A & B (cloud shaded green/red), Chikou Span. "
            "Cloud acts as dynamic support/resistance."
        ),
    },
    {
        "id": "adx",
        "name": "ADX — Average Directional Index",
        "description": (
            "ADX (14) measures trend strength. +DI above −DI suggests "
            "bullish trend; vice versa bearish. ADX above 25 indicates "
            "a strong trend."
        ),
    },
    # ── Momentum ──────────────────────────────────────────────────────────
    {
        "id": "rsi",
        "name": "RSI — Relative Strength Index",
        "description": (
            "RSI (14) with overbought (70) and oversold (30) zones shaded. "
            "Divergences between price and RSI can precede reversals."
        ),
    },
    {
        "id": "macd",
        "name": "MACD",
        "description": (
            "MACD (12, 26, 9) with signal line and coloured histogram. "
            "Bullish/bearish crossover points are annotated."
        ),
    },
    {
        "id": "stochastic",
        "name": "Stochastic Oscillator",
        "description": (
            "Stochastic %K (14, 3) and %D (3). Readings above 80 are "
            "overbought; below 20 oversold. Crossovers within those zones "
            "generate entry signals."
        ),
    },
    # ── Volume ────────────────────────────────────────────────────────────
    {
        "id": "obv",
        "name": "OBV — On-Balance Volume",
        "description": (
            "Cumulative OBV line with its 20-period EMA. Rising OBV with "
            "rising price confirms trend; divergence warns of reversal."
        ),
    },
    {
        "id": "vwap",
        "name": "VWAP",
        "description": (
            "Volume-Weighted Average Price overlaid on candlestick chart. "
            "Price above VWAP signals bullish intraday bias and vice versa."
        ),
    },
    # ── Volatility ────────────────────────────────────────────────────────
    {
        "id": "volatility",
        "name": "Rolling Volatility",
        "description": (
            "Annualised rolling volatility over 10-day and 30-day windows. "
            "Divergence between the two can signal regime change."
        ),
    },
    {
        "id": "atr",
        "name": "ATR — Average True Range",
        "description": (
            "ATR (14) measures absolute volatility. Spike in ATR alongside "
            "a breakout validates the move."
        ),
    },
    # ── Fibonacci / Statistical ───────────────────────────────────────────
    {
        "id": "fibonacci",
        "name": "Fibonacci Retracement",
        "description": (
            "Auto-detects swing high/low in the window and draws 0 %, "
            "23.6 %, 38.2 %, 50 %, 61.8 %, and 100 % retracement levels "
            "on a candlestick chart."
        ),
    },
    {
        "id": "returns",
        "name": "Returns Distribution",
        "description": (
            "Histogram of daily returns with a KDE overlay and ±1 / ±2 σ "
            "bands. Skewness and kurtosis stats annotated."
        ),
    },
]

CHART_DESCRIPTIONS: Dict[str, str] = {c["id"]: c["description"] for c in CHART_CATALOGUE}


# ─── Data fetching ────────────────────────────────────────────────────────────

async def fetch_df(
    symbol: str,
    start_date: str,
    end_date: str,
    interval: str = "1d",
    indian_api_key: str = "",
) -> pd.DataFrame:
    """Fetch stock data and return a sorted DataFrame (no disk I/O)."""
    from agent_scripts.data_providers import DataProviderFactory

    provider = DataProviderFactory.create_composite_provider(
        indian_api_key=indian_api_key,
        yahoo_timeout=30,
        indian_timeout=30,
    )
    stock_data = await provider.get_stock_data(symbol, start_date, end_date, interval)
    if stock_data is None:
        raise ValueError(f"No data returned for {symbol}")

    df: pd.DataFrame = stock_data["data"].copy()
    df = df.sort_index()
    df.index = df.index.astype(str)
    return df


# ─── Utility ──────────────────────────────────────────────────────────────────

def _fig_dict(fig: go.Figure) -> Dict[str, Any]:
    return json.loads(json.dumps(fig.to_dict(), default=str))


def _colour_volume(df: pd.DataFrame) -> list:
    """Green for up-close bars, red for down-close bars."""
    colors = []
    for i in range(len(df)):
        if i == 0:
            colors.append(_PALETTE["bull"])
        elif df["Close"].iloc[i] >= df["Close"].iloc[i - 1]:
            colors.append(_PALETTE["bull"])
        else:
            colors.append(_PALETTE["bear"])
    return colors


# ═══════════════════════════════════════════════════════════════════════════════
#  CHART BUILDERS
# ═══════════════════════════════════════════════════════════════════════════════

# 1. Candlestick + Volume ──────────────────────────────────────────────────────

def _candlestick(df: pd.DataFrame, symbol: str) -> go.Figure:
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        vertical_spacing=0.03, row_heights=[0.72, 0.28])

    fig.add_trace(go.Candlestick(
        x=df.index, open=df["Open"], high=df["High"],
        low=df["Low"], close=df["Close"], name="OHLC",
        increasing_line_color=_PALETTE["bull"],
        decreasing_line_color=_PALETTE["bear"],
    ), row=1, col=1)

    fig.add_trace(go.Bar(
        x=df.index, y=df["Volume"], name="Volume",
        marker_color=_colour_volume(df), opacity=0.7,
        showlegend=False,
    ), row=2, col=1)

    layout = _subplot_layout(f"{symbol} — Candlestick + Volume",
                             rows=2, row_heights=[0.72, 0.28],
                             yaxis_titles=["Price", "Volume"])
    fig.update_layout(**layout)
    return fig


# 2. Price + MA (20 / 50 / 200) with crossover annotations ────────────────────

def _price_ma(df: pd.DataFrame, symbol: str) -> go.Figure:
    d = df.copy()
    d["MA20"]  = d["Close"].rolling(20).mean()
    d["MA50"]  = d["Close"].rolling(50).mean()
    d["MA200"] = d["Close"].rolling(200).mean()

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=d.index, y=d["Close"], name="Close",
                             line=dict(color=_PALETTE["price"], width=1.4)))
    fig.add_trace(go.Scatter(x=d.index, y=d["MA20"],  name="SMA 20",
                             line=dict(color=_PALETTE["ma20"], width=1, dash="dot")))
    fig.add_trace(go.Scatter(x=d.index, y=d["MA50"],  name="SMA 50",
                             line=dict(color=_PALETTE["ma50"], width=1.1)))
    fig.add_trace(go.Scatter(x=d.index, y=d["MA200"], name="SMA 200",
                             line=dict(color=_PALETTE["ma200"], width=1.3)))

    # Annotate golden / death crosses (50 vs 200)
    if d["MA50"].notna().sum() > 1 and d["MA200"].notna().sum() > 1:
        diff = d["MA50"] - d["MA200"]
        cross = diff * diff.shift(1)
        for idx in d.index[cross < 0]:
            is_golden = float(d.loc[idx, "MA50"]) > float(d.loc[idx, "MA200"])
            fig.add_annotation(
                x=idx, y=float(d.loc[idx, "Close"]),
                text="Golden Cross" if is_golden else "Death Cross",
                showarrow=True, arrowhead=2, ax=0, ay=-35,
                font=dict(size=10, color=_PALETTE["bull"] if is_golden else _PALETTE["bear"]),
                bordercolor=_PALETTE["bull"] if is_golden else _PALETTE["bear"],
                borderwidth=1, borderpad=3, bgcolor="white", opacity=0.9,
            )

    fig.update_layout(**_base_layout(f"{symbol} — Price + Moving Averages",
                                     yaxis_title="Price"))
    return fig


# 3. Bollinger Bands + Bandwidth ───────────────────────────────────────────────

def _bollinger(df: pd.DataFrame, symbol: str) -> go.Figure:
    d = df.copy()
    ma = d["Close"].rolling(20).mean()
    std = d["Close"].rolling(20).std()
    d["MA20"]  = ma
    d["Upper"] = ma + 2 * std
    d["Lower"] = ma - 2 * std
    d["BW"] = ((d["Upper"] - d["Lower"]) / d["MA20"]) * 100  # bandwidth %

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        vertical_spacing=0.04, row_heights=[0.72, 0.28])

    fig.add_trace(go.Scatter(x=d.index, y=d["Upper"], name="Upper",
                             line=dict(color=_PALETTE["upper"], width=0.8),
                             showlegend=False), row=1, col=1)
    fig.add_trace(go.Scatter(x=d.index, y=d["Lower"], name="Lower",
                             line=dict(color=_PALETTE["lower"], width=0.8),
                             fill="tonexty", fillcolor=_PALETTE["fill"],
                             showlegend=False), row=1, col=1)
    fig.add_trace(go.Scatter(x=d.index, y=d["Close"], name="Close",
                             line=dict(color=_PALETTE["price"], width=1.4)), row=1, col=1)
    fig.add_trace(go.Scatter(x=d.index, y=d["MA20"], name="SMA 20",
                             line=dict(color=_PALETTE["ma20"], width=1, dash="dot")), row=1, col=1)
    fig.add_trace(go.Scatter(x=d.index, y=d["BW"], name="Bandwidth %",
                             line=dict(color="#7e57c2", width=1.2)), row=2, col=1)

    layout = _subplot_layout(f"{symbol} — Bollinger Bands",
                             rows=2, row_heights=[0.72, 0.28],
                             yaxis_titles=["Price", "Bandwidth %"])
    fig.update_layout(**layout)
    return fig


# 4. Ichimoku Cloud ────────────────────────────────────────────────────────────

def _ichimoku(df: pd.DataFrame, symbol: str) -> go.Figure:
    d = df.copy()
    h9  = d["High"].rolling(9).max();  l9  = d["Low"].rolling(9).min()
    h26 = d["High"].rolling(26).max(); l26 = d["Low"].rolling(26).min()
    h52 = d["High"].rolling(52).max(); l52 = d["Low"].rolling(52).min()

    d["Tenkan"]  = (h9 + l9) / 2
    d["Kijun"]   = (h26 + l26) / 2
    d["SpanA"]   = ((d["Tenkan"] + d["Kijun"]) / 2).shift(26)
    d["SpanB"]   = ((h52 + l52) / 2).shift(26)
    d["Chikou"]  = d["Close"].shift(-26)

    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=d.index, open=d["Open"], high=d["High"],
        low=d["Low"], close=d["Close"], name="OHLC",
        increasing_line_color=_PALETTE["bull"],
        decreasing_line_color=_PALETTE["bear"],
    ))
    fig.add_trace(go.Scatter(x=d.index, y=d["Tenkan"], name="Tenkan-sen (9)",
                             line=dict(color="#1e88e5", width=1)))
    fig.add_trace(go.Scatter(x=d.index, y=d["Kijun"],  name="Kijun-sen (26)",
                             line=dict(color="#d81b60", width=1)))
    fig.add_trace(go.Scatter(x=d.index, y=d["SpanA"],  name="Span A",
                             line=dict(color="rgba(76,175,80,0.5)", width=0)))
    fig.add_trace(go.Scatter(x=d.index, y=d["SpanB"],  name="Span B",
                             line=dict(color="rgba(244,67,54,0.5)", width=0),
                             fill="tonexty", fillcolor="rgba(76,175,80,0.08)"))
    fig.add_trace(go.Scatter(x=d.index, y=d["Chikou"], name="Chikou Span",
                             line=dict(color="#9c27b0", width=0.9, dash="dot")))

    fig.update_layout(**_base_layout(f"{symbol} — Ichimoku Cloud",
                                     yaxis_title="Price"))
    return fig


# 5. ADX — Average Directional Index ──────────────────────────────────────────

def _adx(df: pd.DataFrame, symbol: str) -> go.Figure:
    d = df.copy()
    period = 14

    plus_dm  = d["High"].diff().clip(lower=0)
    minus_dm = (-d["Low"].diff()).clip(lower=0)
    # When both are positive keep only the larger
    plus_dm[plus_dm < minus_dm] = 0
    minus_dm[minus_dm < plus_dm] = 0

    tr = pd.concat([
        d["High"] - d["Low"],
        (d["High"] - d["Close"].shift()).abs(),
        (d["Low"]  - d["Close"].shift()).abs(),
    ], axis=1).max(axis=1)

    atr   = tr.rolling(period).mean()
    plus_di  = 100 * (plus_dm.rolling(period).mean()  / atr)
    minus_di = 100 * (minus_dm.rolling(period).mean() / atr)
    dx = (plus_di - minus_di).abs() / (plus_di + minus_di) * 100
    d["ADX"]  = dx.rolling(period).mean()
    d["+DI"]  = plus_di
    d["-DI"]  = minus_di

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        vertical_spacing=0.04, row_heights=[0.55, 0.45])

    fig.add_trace(go.Scatter(x=d.index, y=d["Close"], name="Close",
                             line=dict(color=_PALETTE["price"], width=1.4)), row=1, col=1)
    fig.add_trace(go.Scatter(x=d.index, y=d["ADX"], name="ADX",
                             line=dict(color="#424242", width=1.6)), row=2, col=1)
    fig.add_trace(go.Scatter(x=d.index, y=d["+DI"], name="+DI",
                             line=dict(color=_PALETTE["bull"], width=1)), row=2, col=1)
    fig.add_trace(go.Scatter(x=d.index, y=d["-DI"], name="−DI",
                             line=dict(color=_PALETTE["bear"], width=1)), row=2, col=1)
    fig.add_hline(y=25, line_dash="dash", line_color="#aaa",
                  annotation_text="Trend threshold (25)", row=2, col=1)

    layout = _subplot_layout(f"{symbol} — ADX",
                             rows=2, row_heights=[0.55, 0.45],
                             yaxis_titles=["Price", "ADX / DI"])
    fig.update_layout(**layout)
    return fig


# 6. RSI ───────────────────────────────────────────────────────────────────────

def _rsi(df: pd.DataFrame, symbol: str) -> go.Figure:
    d = df.copy()
    delta = d["Close"].diff()
    up   = delta.clip(lower=0).rolling(14).mean()
    down = (-delta.clip(upper=0)).rolling(14).mean()
    d["RSI"] = 100 - (100 / (1 + up / down))

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        vertical_spacing=0.04, row_heights=[0.58, 0.42])

    fig.add_trace(go.Scatter(x=d.index, y=d["Close"], name="Close",
                             line=dict(color=_PALETTE["price"], width=1.4)), row=1, col=1)
    # Overbought / oversold shading
    fig.add_hrect(y0=70, y1=100, fillcolor="rgba(244,67,54,0.07)",
                  line_width=0, row=2, col=1)
    fig.add_hrect(y0=0, y1=30, fillcolor="rgba(76,175,80,0.07)",
                  line_width=0, row=2, col=1)
    fig.add_trace(go.Scatter(x=d.index, y=d["RSI"], name="RSI (14)",
                             line=dict(color="#7e57c2", width=1.4)), row=2, col=1)
    fig.add_hline(y=70, line_dash="dash", line_color=_PALETTE["bear"],
                  line_width=0.8, row=2, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color=_PALETTE["bull"],
                  line_width=0.8, row=2, col=1)
    fig.add_hline(y=50, line_dash="dot", line_color="#bbb",
                  line_width=0.6, row=2, col=1)

    layout = _subplot_layout(f"{symbol} — RSI",
                             rows=2, row_heights=[0.58, 0.42],
                             yaxis_titles=["Price", "RSI"])
    fig.update_layout(**layout)
    fig.update_yaxes(range=[0, 100], row=2, col=1)
    return fig


# 7. MACD with coloured histogram ─────────────────────────────────────────────

def _macd(df: pd.DataFrame, symbol: str) -> go.Figure:
    d = df.copy()
    exp12 = d["Close"].ewm(span=12, adjust=False).mean()
    exp26 = d["Close"].ewm(span=26, adjust=False).mean()
    d["MACD"]   = exp12 - exp26
    d["Signal"] = d["MACD"].ewm(span=9, adjust=False).mean()
    d["Hist"]   = d["MACD"] - d["Signal"]

    hist_colors = [_PALETTE["bull"] if v >= 0 else _PALETTE["bear"]
                   for v in d["Hist"].fillna(0)]

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        vertical_spacing=0.04, row_heights=[0.55, 0.45])

    fig.add_trace(go.Scatter(x=d.index, y=d["Close"], name="Close",
                             line=dict(color=_PALETTE["price"], width=1.4)), row=1, col=1)
    fig.add_trace(go.Bar(x=d.index, y=d["Hist"], name="Histogram",
                         marker_color=hist_colors, opacity=0.6), row=2, col=1)
    fig.add_trace(go.Scatter(x=d.index, y=d["MACD"], name="MACD",
                             line=dict(color="#1e88e5", width=1.2)), row=2, col=1)
    fig.add_trace(go.Scatter(x=d.index, y=d["Signal"], name="Signal",
                             line=dict(color=_PALETTE["sig"], width=1.1, dash="dot")), row=2, col=1)
    fig.add_hline(y=0, line_dash="dash", line_color="#bbb", line_width=0.6,
                  row=2, col=1)

    # Annotate crossovers
    if d["MACD"].notna().sum() > 1:
        diff = d["MACD"] - d["Signal"]
        cross = diff * diff.shift(1)
        for idx in d.index[cross < 0]:
            bullish = float(d.loc[idx, "MACD"]) > float(d.loc[idx, "Signal"])
            fig.add_annotation(
                x=idx, y=float(d.loc[idx, "MACD"]),
                text="▲" if bullish else "▼",
                showarrow=False,
                font=dict(size=14, color=_PALETTE["bull"] if bullish else _PALETTE["bear"]),
                yref="y2", row=2, col=1,
            )

    layout = _subplot_layout(f"{symbol} — MACD (12, 26, 9)",
                             rows=2, row_heights=[0.55, 0.45],
                             yaxis_titles=["Price", "MACD"])
    fig.update_layout(**layout)
    return fig


# 8. Stochastic Oscillator ─────────────────────────────────────────────────────

def _stochastic(df: pd.DataFrame, symbol: str) -> go.Figure:
    d = df.copy()
    period_k, smooth_k, period_d = 14, 3, 3

    low_min  = d["Low"].rolling(period_k).min()
    high_max = d["High"].rolling(period_k).max()
    d["%K"]  = ((d["Close"] - low_min) / (high_max - low_min) * 100
                ).rolling(smooth_k).mean()
    d["%D"]  = d["%K"].rolling(period_d).mean()

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        vertical_spacing=0.04, row_heights=[0.58, 0.42])

    fig.add_trace(go.Scatter(x=d.index, y=d["Close"], name="Close",
                             line=dict(color=_PALETTE["price"], width=1.4)), row=1, col=1)
    fig.add_hrect(y0=80, y1=100, fillcolor="rgba(244,67,54,0.07)",
                  line_width=0, row=2, col=1)
    fig.add_hrect(y0=0, y1=20, fillcolor="rgba(76,175,80,0.07)",
                  line_width=0, row=2, col=1)
    fig.add_trace(go.Scatter(x=d.index, y=d["%K"], name="%K",
                             line=dict(color="#1e88e5", width=1.2)), row=2, col=1)
    fig.add_trace(go.Scatter(x=d.index, y=d["%D"], name="%D",
                             line=dict(color=_PALETTE["sig"], width=1.1, dash="dot")), row=2, col=1)
    fig.add_hline(y=80, line_dash="dash", line_color=_PALETTE["bear"],
                  line_width=0.8, row=2, col=1)
    fig.add_hline(y=20, line_dash="dash", line_color=_PALETTE["bull"],
                  line_width=0.8, row=2, col=1)

    layout = _subplot_layout(f"{symbol} — Stochastic (%K {period_k},{smooth_k}  %D {period_d})",
                             rows=2, row_heights=[0.58, 0.42],
                             yaxis_titles=["Price", "Stochastic"])
    fig.update_layout(**layout)
    fig.update_yaxes(range=[0, 100], row=2, col=1)
    return fig


# 9. OBV ───────────────────────────────────────────────────────────────────────

def _obv(df: pd.DataFrame, symbol: str) -> go.Figure:
    d = df.copy()
    sign = d["Close"].diff().apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))
    d["OBV"]     = (sign * d["Volume"]).cumsum()
    d["OBV_EMA"] = d["OBV"].ewm(span=20, adjust=False).mean()

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        vertical_spacing=0.04, row_heights=[0.55, 0.45])

    fig.add_trace(go.Scatter(x=d.index, y=d["Close"], name="Close",
                             line=dict(color=_PALETTE["price"], width=1.4)), row=1, col=1)
    fig.add_trace(go.Scatter(x=d.index, y=d["OBV"], name="OBV",
                             line=dict(color="#5c6bc0", width=1.2)), row=2, col=1)
    fig.add_trace(go.Scatter(x=d.index, y=d["OBV_EMA"], name="OBV EMA(20)",
                             line=dict(color=_PALETTE["sig"], width=1, dash="dot")), row=2, col=1)

    layout = _subplot_layout(f"{symbol} — On-Balance Volume",
                             rows=2, row_heights=[0.55, 0.45],
                             yaxis_titles=["Price", "OBV"])
    fig.update_layout(**layout)
    return fig


# 10. VWAP ─────────────────────────────────────────────────────────────────────

def _vwap(df: pd.DataFrame, symbol: str) -> go.Figure:
    d = df.copy()
    typical = (d["High"] + d["Low"] + d["Close"]) / 3
    d["VWAP"] = (typical * d["Volume"]).cumsum() / d["Volume"].cumsum()

    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=d.index, open=d["Open"], high=d["High"],
        low=d["Low"], close=d["Close"], name="OHLC",
        increasing_line_color=_PALETTE["bull"],
        decreasing_line_color=_PALETTE["bear"],
    ))
    fig.add_trace(go.Scatter(x=d.index, y=d["VWAP"], name="VWAP",
                             line=dict(color="#ff6f00", width=1.8, dash="dashdot")))

    fig.update_layout(**_base_layout(f"{symbol} — VWAP", yaxis_title="Price"))
    return fig


# 11. Rolling Volatility ──────────────────────────────────────────────────────

def _volatility(df: pd.DataFrame, symbol: str) -> go.Figure:
    d = df.copy()
    ret = d["Close"].pct_change()
    d["Vol10"] = ret.rolling(10).std() * (252 ** 0.5)
    d["Vol30"] = ret.rolling(30).std() * (252 ** 0.5)

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        vertical_spacing=0.04, row_heights=[0.55, 0.45])

    fig.add_trace(go.Scatter(x=d.index, y=d["Close"], name="Close",
                             line=dict(color=_PALETTE["price"], width=1.4)), row=1, col=1)
    fig.add_trace(go.Scatter(x=d.index, y=d["Vol10"], name="10-Day Vol",
                             line=dict(color="#e91e63", width=1.1)), row=2, col=1)
    fig.add_trace(go.Scatter(x=d.index, y=d["Vol30"], name="30-Day Vol",
                             line=dict(color="#3f51b5", width=1.3)), row=2, col=1)

    layout = _subplot_layout(f"{symbol} — Rolling Volatility (annualised)",
                             rows=2, row_heights=[0.55, 0.45],
                             yaxis_titles=["Price", "Volatility"])
    fig.update_layout(**layout)
    return fig


# 12. ATR ──────────────────────────────────────────────────────────────────────

def _atr(df: pd.DataFrame, symbol: str) -> go.Figure:
    d = df.copy()
    tr = pd.concat([
        d["High"] - d["Low"],
        (d["High"] - d["Close"].shift()).abs(),
        (d["Low"]  - d["Close"].shift()).abs(),
    ], axis=1).max(axis=1)
    d["ATR"] = tr.rolling(14).mean()

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        vertical_spacing=0.04, row_heights=[0.55, 0.45])

    fig.add_trace(go.Candlestick(
        x=d.index, open=d["Open"], high=d["High"],
        low=d["Low"], close=d["Close"], name="OHLC",
        increasing_line_color=_PALETTE["bull"],
        decreasing_line_color=_PALETTE["bear"],
    ), row=1, col=1)
    fig.add_trace(go.Scatter(x=d.index, y=d["ATR"], name="ATR (14)",
                             line=dict(color="#e65100", width=1.4),
                             fill="tozeroy",
                             fillcolor="rgba(230,81,0,0.08)"), row=2, col=1)

    layout = _subplot_layout(f"{symbol} — ATR",
                             rows=2, row_heights=[0.55, 0.45],
                             yaxis_titles=["Price", "ATR"])
    layout["xaxis_rangeslider_visible"] = False
    fig.update_layout(**layout)
    return fig


# 13. Fibonacci Retracement ────────────────────────────────────────────────────

def _fibonacci(df: pd.DataFrame, symbol: str) -> go.Figure:
    d = df.copy()
    swing_high = float(d["High"].max())
    swing_low  = float(d["Low"].min())
    diff = swing_high - swing_low

    levels = {
        "0 % (High)":  swing_high,
        "23.6 %":      swing_high - 0.236 * diff,
        "38.2 %":      swing_high - 0.382 * diff,
        "50 %":        swing_high - 0.500 * diff,
        "61.8 %":      swing_high - 0.618 * diff,
        "100 % (Low)": swing_low,
    }
    fib_colors = ["#d32f2f", "#f57c00", "#fbc02d", "#388e3c", "#1976d2", "#7b1fa2"]

    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=d.index, open=d["Open"], high=d["High"],
        low=d["Low"], close=d["Close"], name="OHLC",
        increasing_line_color=_PALETTE["bull"],
        decreasing_line_color=_PALETTE["bear"],
    ))

    for (label, level), color in zip(levels.items(), fib_colors):
        fig.add_hline(
            y=level, line_dash="dash", line_color=color, line_width=1,
            annotation_text=f"{label}  {level:.2f}",
            annotation_position="top left",
            annotation_font=dict(size=10, color=color),
        )

    fig.update_layout(**_base_layout(f"{symbol} — Fibonacci Retracement",
                                     yaxis_title="Price"))
    return fig


# 14. Returns Distribution ─────────────────────────────────────────────────────

def _returns(df: pd.DataFrame, symbol: str) -> go.Figure:
    d = df.copy()
    rets = d["Close"].pct_change().dropna()
    mu = float(rets.mean())
    sigma = float(rets.std())
    skew_val = float(rets.skew())
    kurt_val = float(rets.kurtosis())

    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=rets, nbinsx=60, name="Returns",
        marker_color="rgba(63,81,181,0.55)",
        histnorm="probability density",
    ))

    # KDE overlay via numpy
    xs = np.linspace(float(rets.min()), float(rets.max()), 200)
    kde = (1 / (sigma * math.sqrt(2 * math.pi))) * np.exp(-0.5 * ((xs - mu) / sigma) ** 2)
    fig.add_trace(go.Scatter(x=xs, y=kde, name="Normal fit",
                             line=dict(color="#e91e63", width=1.5)))

    # σ bands
    for n, col in [(1, "rgba(255,152,0,0.4)"), (2, "rgba(244,67,54,0.3)")]:
        fig.add_vrect(x0=mu - n * sigma, x1=mu + n * sigma,
                      fillcolor=col, line_width=0,
                      annotation_text=f"±{n}σ", annotation_position="top left",
                      annotation_font_size=10)

    fig.update_layout(**_base_layout(
        f"{symbol} — Returns Distribution",
        xaxis_title="Daily Return", yaxis_title="Density",
        annotations=[dict(
            text=f"μ={mu:.4f}  σ={sigma:.4f}  skew={skew_val:.2f}  kurt={kurt_val:.2f}",
            xref="paper", yref="paper", x=0.98, y=0.95, showarrow=False,
            font=dict(size=11, family="monospace"), bgcolor="rgba(255,255,255,0.85)",
            bordercolor="#ccc", borderwidth=1, borderpad=4,
        )],
    ))
    return fig


# ─── Registry ─────────────────────────────────────────────────────────────────

_BUILDERS = {
    "candlestick": _candlestick,
    "price_ma":    _price_ma,
    "bollinger":   _bollinger,
    "ichimoku":    _ichimoku,
    "adx":         _adx,
    "rsi":         _rsi,
    "macd":        _macd,
    "stochastic":  _stochastic,
    "obv":         _obv,
    "vwap":        _vwap,
    "volatility":  _volatility,
    "atr":         _atr,
    "fibonacci":   _fibonacci,
    "returns":     _returns,
}


def build_chart(chart_type: str, df: pd.DataFrame, symbol: str) -> Dict[str, Any]:
    """Build the requested chart and return a JSON-serialisable Plotly dict."""
    builder = _BUILDERS.get(chart_type)
    if builder is None:
        raise ValueError(
            f"Unknown chart type '{chart_type}'. "
            f"Valid types: {list(_BUILDERS)}"
        )
    fig = builder(df, symbol)
    return _fig_dict(fig)
