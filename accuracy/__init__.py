"""
accuracy
────────
Persistence + real-time evaluation of Buck forecasts vs market actuals.

Modules:
  db          — sqlite schema + connection helpers
  repository  — write / read predictions, actuals, evaluations
  evaluator   — compute MAE / RMSE / directional accuracy
  poller      — pull live prices from yfinance during market hours
  scheduler   — APScheduler jobs (intraday + EOD)
  broadcaster — fan-out live accuracy updates to WebSocket subscribers
  market_hours — exchange trading-window helpers
"""

from .db import init_db, get_conn  # noqa: F401
from .repository import (  # noqa: F401
    record_prediction,
    list_predictions,
    summary_by_model,
    timeseries,
)
