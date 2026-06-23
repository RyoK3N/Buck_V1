"""
accuracy.repository
───────────────────
Read / write helpers for the accuracy database.

These functions are intentionally plain — the orchestrating modules
(buck.py, the scheduler, the API routes) compose them.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from .db import get_conn


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# ── Predictions ──────────────────────────────────────────────────────────────

def record_prediction(
    symbol: str,
    model: str,
    forecast: Dict[str, Any],
    request_metadata: Optional[Dict[str, Any]] = None,
) -> Optional[int]:
    """Persist a forecast. Returns the new prediction id, or None on failure.

    `forecast` should have keys: date, open, high, low, close, confidence, reasoning.
    `request_metadata` may include selected_tools, tool-call trace, prompt, etc.
    """
    try:
        with get_conn() as conn:
            cur = conn.execute(
                """
                INSERT INTO predictions (
                    symbol, model, target_date,
                    predicted_open, predicted_high, predicted_low, predicted_close,
                    confidence, reasoning,
                    created_at, request_metadata_json, status
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'open')
                """,
                (
                    symbol,
                    model,
                    str(forecast.get("date")),
                    _safe_float(forecast.get("open")),
                    _safe_float(forecast.get("high")),
                    _safe_float(forecast.get("low")),
                    _safe_float(forecast.get("close")),
                    _safe_float(forecast.get("confidence")),
                    str(forecast.get("reasoning", ""))[:4000],
                    _utcnow_iso(),
                    json.dumps(request_metadata or {}, default=str),
                ),
            )
            return cur.lastrowid
    except Exception:
        # Never let telemetry break the forecast pipeline.
        return None


def list_predictions(
    symbol: Optional[str] = None,
    model: Optional[str] = None,
    status: Optional[str] = None,
    limit: int = 100,
) -> List[Dict[str, Any]]:
    """Return recent predictions, optionally filtered. Joined with their evaluation."""
    clauses: list[str] = []
    params: list[Any] = []
    if symbol:
        clauses.append("p.symbol = ?")
        params.append(symbol)
    if model:
        clauses.append("p.model = ?")
        params.append(model)
    if status:
        clauses.append("p.status = ?")
        params.append(status)
    where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
    sql = f"""
        SELECT p.*,
               e.actual_open, e.actual_high, e.actual_low, e.actual_close,
               e.mae, e.rmse, e.directional_correct, e.error_pct,
               e.is_intraday, e.evaluated_at
        FROM predictions p
        LEFT JOIN evaluations e ON e.prediction_id = p.id
        {where}
        ORDER BY p.created_at DESC
        LIMIT ?
    """
    params.append(int(limit))
    with get_conn() as conn:
        return [_row_to_dict(r) for r in conn.execute(sql, params).fetchall()]


def get_prediction(prediction_id: int) -> Optional[Dict[str, Any]]:
    with get_conn() as conn:
        r = conn.execute(
            "SELECT * FROM predictions WHERE id = ?", (prediction_id,)
        ).fetchone()
        return _row_to_dict(r) if r else None


def list_open_predictions_for_date(target_date: str) -> List[Dict[str, Any]]:
    """Predictions whose target_date <= given date and still open. Used by evaluator."""
    with get_conn() as conn:
        rows = conn.execute(
            """
            SELECT * FROM predictions
            WHERE status = 'open' AND target_date <= ?
            ORDER BY target_date ASC
            """,
            (target_date,),
        ).fetchall()
        return [_row_to_dict(r) for r in rows]


def distinct_symbols_with_open_predictions() -> List[str]:
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT DISTINCT symbol FROM predictions WHERE status = 'open'"
        ).fetchall()
        return [r["symbol"] for r in rows]


def mark_prediction_status(prediction_id: int, status: str) -> None:
    with get_conn() as conn:
        conn.execute(
            "UPDATE predictions SET status = ? WHERE id = ?",
            (status, prediction_id),
        )


# ── Actuals ──────────────────────────────────────────────────────────────────

def upsert_actual(
    symbol: str,
    date: str,
    open_: Optional[float] = None,
    high: Optional[float] = None,
    low: Optional[float] = None,
    close: Optional[float] = None,
    prev_close: Optional[float] = None,
    is_final: bool = False,
) -> None:
    """Upsert a daily bar. Partial intraday updates allowed (is_final=0)."""
    with get_conn() as conn:
        conn.execute(
            """
            INSERT INTO actuals (symbol, date, open, high, low, close, prev_close, fetched_at, is_final)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(symbol, date) DO UPDATE SET
                open = COALESCE(excluded.open, actuals.open),
                high = COALESCE(excluded.high, actuals.high),
                low = COALESCE(excluded.low, actuals.low),
                close = COALESCE(excluded.close, actuals.close),
                prev_close = COALESCE(excluded.prev_close, actuals.prev_close),
                fetched_at = excluded.fetched_at,
                is_final = MAX(actuals.is_final, excluded.is_final)
            """,
            (
                symbol,
                date,
                _safe_float(open_),
                _safe_float(high),
                _safe_float(low),
                _safe_float(close),
                _safe_float(prev_close),
                _utcnow_iso(),
                1 if is_final else 0,
            ),
        )


def get_actual(symbol: str, date: str) -> Optional[Dict[str, Any]]:
    with get_conn() as conn:
        r = conn.execute(
            "SELECT * FROM actuals WHERE symbol = ? AND date = ?",
            (symbol, date),
        ).fetchone()
        return _row_to_dict(r) if r else None


# ── Evaluations ──────────────────────────────────────────────────────────────

def record_evaluation(
    prediction_id: int,
    actual: Dict[str, Any],
    metrics: Dict[str, Any],
    is_intraday: bool = True,
) -> None:
    with get_conn() as conn:
        conn.execute(
            """
            INSERT INTO evaluations (
                prediction_id, evaluated_at,
                actual_open, actual_high, actual_low, actual_close,
                mae, rmse, directional_correct, error_pct, is_intraday
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(prediction_id) DO UPDATE SET
                evaluated_at = excluded.evaluated_at,
                actual_open = excluded.actual_open,
                actual_high = excluded.actual_high,
                actual_low = excluded.actual_low,
                actual_close = excluded.actual_close,
                mae = excluded.mae,
                rmse = excluded.rmse,
                directional_correct = excluded.directional_correct,
                error_pct = excluded.error_pct,
                is_intraday = excluded.is_intraday
            """,
            (
                prediction_id,
                _utcnow_iso(),
                _safe_float(actual.get("open")),
                _safe_float(actual.get("high")),
                _safe_float(actual.get("low")),
                _safe_float(actual.get("close")),
                _safe_float(metrics.get("mae")),
                _safe_float(metrics.get("rmse")),
                int(metrics["directional_correct"]) if metrics.get("directional_correct") is not None else None,
                _safe_float(metrics.get("error_pct")),
                1 if is_intraday else 0,
            ),
        )


# ── Rollups / summaries ──────────────────────────────────────────────────────

def summary_by_model(
    model: Optional[str] = None,
    symbol: Optional[str] = None,
    window_days: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Rolling per-model accuracy snapshot computed on the fly from evaluations."""
    clauses = ["1 = 1"]
    params: list[Any] = []
    if model:
        clauses.append("p.model = ?")
        params.append(model)
    if symbol:
        clauses.append("p.symbol = ?")
        params.append(symbol)
    if window_days:
        clauses.append("p.created_at >= datetime('now', ?)")
        params.append(f"-{int(window_days)} days")

    sql = f"""
        SELECT p.model,
               COUNT(e.prediction_id) AS n,
               AVG(e.mae) AS mae,
               AVG(e.rmse) AS rmse,
               AVG(CASE WHEN e.directional_correct = 1 THEN 1.0 ELSE 0.0 END) AS directional_accuracy,
               AVG(ABS(e.error_pct)) AS avg_error_pct
        FROM predictions p
        JOIN evaluations e ON e.prediction_id = p.id
        WHERE {' AND '.join(clauses)}
        GROUP BY p.model
        ORDER BY n DESC
    """
    with get_conn() as conn:
        return [_row_to_dict(r) for r in conn.execute(sql, params).fetchall()]


def timeseries(
    model: Optional[str] = None,
    symbol: Optional[str] = None,
    window_days: int = 30,
) -> List[Dict[str, Any]]:
    """Per-day rolling MAE / directional-accuracy series."""
    clauses = ["1 = 1"]
    params: list[Any] = []
    if model:
        clauses.append("p.model = ?")
        params.append(model)
    if symbol:
        clauses.append("p.symbol = ?")
        params.append(symbol)
    clauses.append("p.created_at >= datetime('now', ?)")
    params.append(f"-{int(window_days)} days")

    sql = f"""
        SELECT substr(e.evaluated_at, 1, 10) AS date,
               p.model,
               AVG(e.mae) AS mae,
               AVG(CASE WHEN e.directional_correct = 1 THEN 1.0 ELSE 0.0 END) AS directional_accuracy,
               COUNT(*) AS n
        FROM predictions p
        JOIN evaluations e ON e.prediction_id = p.id
        WHERE {' AND '.join(clauses)}
        GROUP BY date, p.model
        ORDER BY date ASC
    """
    with get_conn() as conn:
        return [_row_to_dict(r) for r in conn.execute(sql, params).fetchall()]


def tool_contribution(model: str = "claude", window_days: int = 30) -> List[Dict[str, Any]]:
    """For Claude predictions, count tool usage across correct vs incorrect outcomes.

    Reads `request_metadata_json` for a `tools_used` list and aggregates.
    """
    rows = []
    with get_conn() as conn:
        for r in conn.execute(
            """
            SELECT p.id, p.request_metadata_json, e.directional_correct
            FROM predictions p
            JOIN evaluations e ON e.prediction_id = p.id
            WHERE p.model = ? AND p.created_at >= datetime('now', ?)
            """,
            (model, f"-{int(window_days)} days"),
        ).fetchall():
            rows.append((r["request_metadata_json"], r["directional_correct"]))

    agg: dict[str, dict[str, int]] = {}
    for meta_json, correct in rows:
        try:
            meta = json.loads(meta_json or "{}")
        except Exception:
            continue
        tools = meta.get("tools_used") or []
        for t in tools:
            name = t if isinstance(t, str) else t.get("name", "")
            if not name:
                continue
            bucket = agg.setdefault(name, {"correct": 0, "incorrect": 0})
            bucket["correct" if correct == 1 else "incorrect"] += 1

    return [
        {"tool": k, "correct": v["correct"], "incorrect": v["incorrect"]}
        for k, v in sorted(agg.items(), key=lambda kv: -(kv[1]["correct"] + kv[1]["incorrect"]))
    ]


# ── Internals ────────────────────────────────────────────────────────────────

def _row_to_dict(row) -> Dict[str, Any]:
    return {k: row[k] for k in row.keys()}


def _safe_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
