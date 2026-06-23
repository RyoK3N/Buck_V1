"""
accuracy.scheduler
──────────────────
APScheduler jobs that drive the real-time accuracy variable.

- Intraday job: every N minutes during market hours, poll latest prices for
  symbols with open predictions, then reconcile evaluations and broadcast.
- EOD job: once per trading day shortly after the market closes, pull final
  daily bars and re-run reconciliation with `is_final=True`.
"""

from __future__ import annotations

import logging
from typing import Optional

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger

from . import broadcaster, evaluator, poller, repository
from .market_hours import get_window, is_market_open

_logger = logging.getLogger("accuracy.scheduler")


async def _intraday_tick(exchange: str) -> None:
    if not is_market_open(exchange=exchange):
        return
    symbols = repository.distinct_symbols_with_open_predictions()
    if not symbols:
        return
    _logger.info("intraday tick: polling %d symbols", len(symbols))
    poller.poll_symbols(symbols, exchange=exchange, is_final=False)
    written = evaluator.reconcile_open_predictions(is_intraday=True, exchange=exchange)
    for event in written:
        await broadcaster.publish(event)


async def _eod_tick(exchange: str) -> None:
    symbols = repository.distinct_symbols_with_open_predictions()
    if not symbols:
        return
    _logger.info("EOD tick: finalising %d symbols", len(symbols))
    poller.poll_symbols(symbols, exchange=exchange, is_final=True)
    written = evaluator.reconcile_open_predictions(is_intraday=False, exchange=exchange)
    for event in written:
        await broadcaster.publish(event)


def start(
    poll_interval_minutes: int = 5,
    exchange: str = "NSE",
) -> AsyncIOScheduler:
    """Construct, configure, and start the AsyncIOScheduler. Returns it for shutdown."""
    win = get_window(exchange)
    scheduler = AsyncIOScheduler(timezone=win.tz)

    scheduler.add_job(
        _intraday_tick,
        trigger=IntervalTrigger(minutes=max(1, int(poll_interval_minutes))),
        kwargs={"exchange": exchange},
        id="accuracy_intraday",
        replace_existing=True,
        max_instances=1,
        coalesce=True,
    )
    scheduler.add_job(
        _eod_tick,
        trigger=CronTrigger(
            hour=win.close.hour,
            minute=(win.close.minute + 30) % 60,
            day_of_week="mon-fri",
            timezone=win.tz,
        ),
        kwargs={"exchange": exchange},
        id="accuracy_eod",
        replace_existing=True,
        max_instances=1,
        coalesce=True,
    )
    scheduler.start()
    _logger.info(
        "accuracy scheduler started: intraday every %dm (%s); EOD %02d:%02d %s",
        poll_interval_minutes, exchange,
        win.close.hour, (win.close.minute + 30) % 60, win.tz,
    )
    return scheduler


def shutdown(scheduler: Optional[AsyncIOScheduler]) -> None:
    if scheduler is None:
        return
    try:
        scheduler.shutdown(wait=False)
    except Exception:
        pass
