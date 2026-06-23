"""
accuracy.poller
───────────────
Pull latest OHLC for the symbols we have open predictions on, and upsert
them into the `actuals` table. Used by the intraday scheduler job.
"""

from __future__ import annotations

from typing import Iterable, List, Optional

import yfinance as yf

from . import repository
from .market_hours import trading_date_str


def poll_symbols(
    symbols: Iterable[str],
    exchange: str = "NSE",
    is_final: bool = False,
) -> List[dict]:
    """Fetch latest bar per symbol and upsert into `actuals`. Returns rows touched.

    For intraday calls (`is_final=False`) we pull a 1-minute window for today
    and use the last bar as the live snapshot. For EOD calls we pull the
    daily bar so `close` is authoritative.
    """
    symbols = sorted({s for s in symbols if s})
    if not symbols:
        return []

    today = trading_date_str(exchange=exchange)
    rows: list[dict] = []

    if is_final:
        # Pull last two daily bars so we have prev_close too.
        df = _safe_download(symbols, period="5d", interval="1d")
        for s in symbols:
            sub = _slice(df, s)
            if sub is None or sub.empty:
                continue
            try:
                last = sub.iloc[-1]
                prev = sub.iloc[-2] if len(sub) >= 2 else None
                repository.upsert_actual(
                    symbol=s,
                    date=last.name.strftime("%Y-%m-%d") if hasattr(last.name, "strftime") else today,
                    open_=float(last.get("Open")) if "Open" in sub.columns else None,
                    high=float(last.get("High")) if "High" in sub.columns else None,
                    low=float(last.get("Low")) if "Low" in sub.columns else None,
                    close=float(last.get("Close")) if "Close" in sub.columns else None,
                    prev_close=float(prev.get("Close")) if prev is not None and "Close" in sub.columns else None,
                    is_final=True,
                )
                rows.append({"symbol": s, "date": last.name.strftime("%Y-%m-%d") if hasattr(last.name, "strftime") else today})
            except Exception:
                continue
        return rows

    # Intraday: 1-minute bars for today.
    df = _safe_download(symbols, period="1d", interval="1m")
    # Also pull yesterday's close for prev_close context.
    daily = _safe_download(symbols, period="5d", interval="1d")
    for s in symbols:
        sub = _slice(df, s)
        if sub is None or sub.empty:
            continue
        try:
            last = sub.iloc[-1]
            open_ = float(sub.iloc[0].get("Open")) if "Open" in sub.columns else None
            high = float(sub["High"].max()) if "High" in sub.columns else None
            low = float(sub["Low"].min()) if "Low" in sub.columns else None
            close = float(last.get("Close")) if "Close" in sub.columns else None
            prev_close = None
            dsub = _slice(daily, s)
            if dsub is not None and not dsub.empty and len(dsub) >= 2:
                prev_close = float(dsub.iloc[-2].get("Close"))
            repository.upsert_actual(
                symbol=s,
                date=today,
                open_=open_,
                high=high,
                low=low,
                close=close,
                prev_close=prev_close,
                is_final=False,
            )
            rows.append({"symbol": s, "date": today})
        except Exception:
            continue
    return rows


def _safe_download(symbols: List[str], period: str, interval: str):
    try:
        return yf.download(
            tickers=" ".join(symbols),
            period=period,
            interval=interval,
            group_by="ticker",
            threads=False,
            progress=False,
            auto_adjust=False,
        )
    except Exception:
        return None


def _slice(df, symbol: str):
    """yfinance's grouped result is a MultiIndex; single-symbol downloads aren't."""
    if df is None or df.empty:
        return None
    try:
        if hasattr(df, "columns") and getattr(df.columns, "nlevels", 1) > 1:
            if symbol in df.columns.get_level_values(0):
                return df[symbol]
            return None
        return df
    except Exception:
        return None
