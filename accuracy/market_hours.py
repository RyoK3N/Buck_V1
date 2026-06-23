"""
accuracy.market_hours
─────────────────────
Trading-window helpers per exchange.

Default exchange is NSE (matches Buck's Indian-stock focus). The poller and
scheduler consult `is_market_open()` to decide whether to fetch new prices.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, time, timedelta
from typing import Dict, Optional
from zoneinfo import ZoneInfo


@dataclass(frozen=True)
class MarketWindow:
    tz: str
    open: time   # noqa: A003  (open is fine as an attr name here)
    close: time
    weekdays: tuple[int, ...] = (0, 1, 2, 3, 4)  # Mon-Fri


EXCHANGES: Dict[str, MarketWindow] = {
    "NSE": MarketWindow(tz="Asia/Kolkata", open=time(9, 15), close=time(15, 30)),
    "BSE": MarketWindow(tz="Asia/Kolkata", open=time(9, 15), close=time(15, 30)),
    "NYSE": MarketWindow(tz="America/New_York", open=time(9, 30), close=time(16, 0)),
    "NASDAQ": MarketWindow(tz="America/New_York", open=time(9, 30), close=time(16, 0)),
}


def get_window(exchange: str = "NSE") -> MarketWindow:
    return EXCHANGES.get(exchange.upper(), EXCHANGES["NSE"])


def now_in_market_tz(exchange: str = "NSE") -> datetime:
    return datetime.now(tz=ZoneInfo(get_window(exchange).tz))


def is_market_open(now: Optional[datetime] = None, exchange: str = "NSE") -> bool:
    win = get_window(exchange)
    n = (now or datetime.now(tz=ZoneInfo(win.tz))).astimezone(ZoneInfo(win.tz))
    if n.weekday() not in win.weekdays:
        return False
    return win.open <= n.time() <= win.close


def trading_date_str(now: Optional[datetime] = None, exchange: str = "NSE") -> str:
    """YYYY-MM-DD in the exchange's timezone."""
    win = get_window(exchange)
    n = (now or datetime.now(tz=ZoneInfo(win.tz))).astimezone(ZoneInfo(win.tz))
    return n.strftime("%Y-%m-%d")


def time_until_next_open(now: Optional[datetime] = None, exchange: str = "NSE") -> timedelta:
    win = get_window(exchange)
    n = (now or datetime.now(tz=ZoneInfo(win.tz))).astimezone(ZoneInfo(win.tz))
    candidate = n.replace(hour=win.open.hour, minute=win.open.minute, second=0, microsecond=0)
    if candidate <= n:
        candidate += timedelta(days=1)
    while candidate.weekday() not in win.weekdays:
        candidate += timedelta(days=1)
    return candidate - n
