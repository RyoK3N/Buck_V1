"""
accuracy.evaluator
──────────────────
Compute prediction-vs-actual metrics and reconcile open predictions.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

from . import repository
from .market_hours import trading_date_str


def compute_metrics(
    predicted: Dict[str, Any],
    actual: Dict[str, Any],
) -> Dict[str, Any]:
    """Compute MAE / RMSE / directional_correct / error_pct.

    `predicted` keys: predicted_open/high/low/close (or open/high/low/close).
    `actual` keys: open/high/low/close, prev_close.
    """
    def _p(k: str) -> Optional[float]:
        v = predicted.get(f"predicted_{k}")
        if v is None:
            v = predicted.get(k)
        try:
            return float(v) if v is not None else None
        except (TypeError, ValueError):
            return None

    def _a(k: str) -> Optional[float]:
        v = actual.get(k)
        try:
            return float(v) if v is not None else None
        except (TypeError, ValueError):
            return None

    pairs = []
    for k in ("open", "high", "low", "close"):
        p, a = _p(k), _a(k)
        if p is not None and a is not None:
            pairs.append((p, a))

    mae: Optional[float] = None
    rmse: Optional[float] = None
    if pairs:
        diffs = [p - a for p, a in pairs]
        mae = sum(abs(d) for d in diffs) / len(diffs)
        rmse = math.sqrt(sum(d * d for d in diffs) / len(diffs))

    # Directional correctness (close-vs-prev_close direction)
    directional_correct: Optional[int] = None
    p_close = _p("close")
    a_close = _a("close")
    prev_close = _a("prev_close")
    if p_close is not None and a_close is not None and prev_close is not None:
        actual_up = a_close > prev_close
        predicted_up = p_close > prev_close
        directional_correct = 1 if actual_up == predicted_up else 0

    # Close-error percentage relative to actual close
    error_pct: Optional[float] = None
    if p_close is not None and a_close not in (None, 0):
        error_pct = (p_close - a_close) / a_close * 100.0

    return {
        "mae": mae,
        "rmse": rmse,
        "directional_correct": directional_correct,
        "error_pct": error_pct,
    }


def reconcile_open_predictions(
    today: Optional[str] = None,
    is_intraday: bool = True,
    exchange: str = "NSE",
) -> List[Dict[str, Any]]:
    """For every open prediction whose target_date <= today, look up the actual
    for that symbol+date and write an evaluation. Returns the list of evaluations
    written this run."""
    today = today or trading_date_str(exchange=exchange)
    written: list[dict[str, Any]] = []

    for pred in repository.list_open_predictions_for_date(today):
        actual = repository.get_actual(pred["symbol"], pred["target_date"])
        if not actual or actual.get("close") is None:
            continue
        metrics = compute_metrics(pred, actual)
        repository.record_evaluation(
            prediction_id=pred["id"],
            actual=actual,
            metrics=metrics,
            is_intraday=is_intraday,
        )
        # Only finalize ('evaluated') when the actual is closed/final.
        if not is_intraday and actual.get("is_final"):
            repository.mark_prediction_status(pred["id"], "evaluated")
        written.append({
            "prediction_id": pred["id"],
            "symbol": pred["symbol"],
            "model": pred["model"],
            "target_date": pred["target_date"],
            **metrics,
        })
    return written
