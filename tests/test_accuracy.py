"""
Tests for the accuracy/ subsystem: DB schema, repository writes/reads,
evaluator math, and market_hours helpers.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, time, timezone

import pytest

from accuracy import db as accuracy_db
from accuracy import broadcaster, evaluator, market_hours, repository


@pytest.fixture
def fresh_db(tmp_path, monkeypatch):
    """Initialise a temporary sqlite DB for each test and reset broadcaster state."""
    db_path = tmp_path / "accuracy.db"
    accuracy_db.init_db(db_path)
    broadcaster.LIVE_STATE.clear()
    yield db_path
    # global _DB_PATH leaks across tests; reset so the next fixture's init_db wins.
    accuracy_db._DB_PATH = None


# ── DB / repository ──────────────────────────────────────────────────────────

def test_init_db_is_idempotent(fresh_db):
    accuracy_db.init_db(fresh_db)  # call twice; should not raise


def test_record_prediction_persists_required_fields(fresh_db):
    pid = repository.record_prediction(
        symbol="TEST.NS",
        model="openai",
        forecast={
            "date": "2026-06-22",
            "open": 100.5, "high": 105.0, "low": 99.0, "close": 104.2,
            "confidence": 0.72, "reasoning": "unit test",
        },
        request_metadata={"tools_used": ["rsi"], "selected_tools": None},
    )
    assert pid is not None
    rows = repository.list_predictions(limit=10)
    assert len(rows) == 1
    r = rows[0]
    assert r["symbol"] == "TEST.NS"
    assert r["model"] == "openai"
    assert r["predicted_close"] == 104.2
    assert r["confidence"] == 0.72
    assert r["status"] == "open"


def test_distinct_symbols_with_open_predictions(fresh_db):
    for sym in ("A.NS", "B.NS", "A.NS"):
        repository.record_prediction(
            symbol=sym, model="openai",
            forecast={"date": "2026-06-22", "open": 1, "high": 2, "low": 0.5, "close": 1.5, "confidence": 0.5},
        )
    syms = repository.distinct_symbols_with_open_predictions()
    assert sorted(syms) == ["A.NS", "B.NS"]


def test_upsert_actual_merges_intraday_partials(fresh_db):
    repository.upsert_actual(symbol="X.NS", date="2026-06-21", open_=100.0, high=101.0, is_final=False)
    repository.upsert_actual(symbol="X.NS", date="2026-06-21", low=99.5, close=100.3, is_final=False)
    row = repository.get_actual("X.NS", "2026-06-21")
    assert row["open"] == 100.0
    assert row["high"] == 101.0
    assert row["low"] == 99.5
    assert row["close"] == 100.3
    assert row["is_final"] == 0  # still intraday


def test_upsert_actual_finalisation_sticks(fresh_db):
    repository.upsert_actual(symbol="X.NS", date="2026-06-21", close=100.3, is_final=True)
    repository.upsert_actual(symbol="X.NS", date="2026-06-21", close=100.4, is_final=False)
    row = repository.get_actual("X.NS", "2026-06-21")
    # final flag is sticky (MAX of old + new)
    assert row["is_final"] == 1


# ── Evaluator math ───────────────────────────────────────────────────────────

def test_compute_metrics_directional_correct():
    m = evaluator.compute_metrics(
        predicted={"predicted_open": 100, "predicted_high": 105, "predicted_low": 99, "predicted_close": 104},
        actual={"open": 101, "high": 104, "low": 98, "close": 103, "prev_close": 100},
    )
    # predicted close 104 > prev 100 (up); actual close 103 > prev (up) → correct
    assert m["directional_correct"] == 1
    assert m["mae"] is not None and m["mae"] > 0
    assert m["rmse"] is not None and m["rmse"] >= m["mae"]
    assert abs(m["error_pct"] - ((104 - 103) / 103 * 100)) < 1e-9


def test_compute_metrics_directional_wrong():
    m = evaluator.compute_metrics(
        predicted={"predicted_close": 95},
        actual={"close": 102, "prev_close": 100},
    )
    # predicted down, actual up → wrong
    assert m["directional_correct"] == 0


def test_compute_metrics_handles_missing_fields():
    m = evaluator.compute_metrics(predicted={}, actual={"close": 100, "prev_close": 99})
    assert m["mae"] is None
    assert m["rmse"] is None
    assert m["directional_correct"] is None


def test_reconcile_open_predictions_writes_evaluations(fresh_db):
    pid = repository.record_prediction(
        symbol="Y.NS", model="openai",
        forecast={"date": "2026-06-21", "open": 100, "high": 105, "low": 99, "close": 104, "confidence": 0.5},
    )
    repository.upsert_actual(symbol="Y.NS", date="2026-06-21", open_=99, high=103, low=98, close=102, prev_close=100, is_final=True)

    written = evaluator.reconcile_open_predictions(today="2026-06-21", is_intraday=False)
    assert len(written) == 1
    assert written[0]["prediction_id"] == pid
    assert written[0]["directional_correct"] in (0, 1)

    # Status flipped to 'evaluated' for finalised reconciliation
    row = repository.get_prediction(pid)
    assert row["status"] == "evaluated"


# ── Broadcaster ──────────────────────────────────────────────────────────────

def test_broadcaster_updates_live_state(fresh_db):
    async def run():
        await broadcaster.publish({
            "prediction_id": 1, "model": "claude", "symbol": "Z.NS",
            "error_pct": 2.0, "directional_correct": 1,
        })
        await broadcaster.publish({
            "prediction_id": 2, "model": "claude", "symbol": "Z.NS",
            "error_pct": -1.0, "directional_correct": 0,
        })
    asyncio.run(run())

    snap = {(e["model"], e["symbol"]): e for e in broadcaster.snapshot()}
    bucket = snap[("claude", "Z.NS")]
    assert bucket["n"] == 2
    assert bucket["mae_pct"] == 1.5  # mean(|2|, |-1|) = 1.5
    assert bucket["directional_accuracy_pct"] == 50.0


# ── Market hours ─────────────────────────────────────────────────────────────

def test_nse_window_default():
    win = market_hours.get_window("NSE")
    assert win.open == time(9, 15)
    assert win.close == time(15, 30)


def test_is_market_open_weekend_returns_false():
    # 2026-06-21 is a Sunday
    sunday_noon = datetime(2026, 6, 21, 12, 0, tzinfo=timezone.utc)
    assert market_hours.is_market_open(now=sunday_noon, exchange="NSE") is False


def test_trading_date_str_format():
    s = market_hours.trading_date_str(exchange="NSE")
    # YYYY-MM-DD
    assert len(s) == 10 and s[4] == "-" and s[7] == "-"
