"""
accuracy.db
───────────
SQLite schema + connection management for the accuracy tracking subsystem.

Uses stdlib sqlite3 (no SQLAlchemy) to avoid a new heavy dependency.
"""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Optional

_DB_PATH: Optional[Path] = None


_SCHEMA = """
CREATE TABLE IF NOT EXISTS predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol TEXT NOT NULL,
    model TEXT NOT NULL,
    target_date TEXT NOT NULL,
    predicted_open REAL,
    predicted_high REAL,
    predicted_low REAL,
    predicted_close REAL,
    confidence REAL,
    reasoning TEXT,
    created_at TEXT NOT NULL,
    request_metadata_json TEXT,
    status TEXT NOT NULL DEFAULT 'open'
);

CREATE INDEX IF NOT EXISTS idx_pred_status_target ON predictions(status, target_date);
CREATE INDEX IF NOT EXISTS idx_pred_model_symbol_target ON predictions(model, symbol, target_date);
CREATE INDEX IF NOT EXISTS idx_pred_created ON predictions(created_at);

CREATE TABLE IF NOT EXISTS actuals (
    symbol TEXT NOT NULL,
    date TEXT NOT NULL,
    open REAL,
    high REAL,
    low REAL,
    close REAL,
    prev_close REAL,
    fetched_at TEXT NOT NULL,
    is_final INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (symbol, date)
);

CREATE TABLE IF NOT EXISTS evaluations (
    prediction_id INTEGER PRIMARY KEY REFERENCES predictions(id) ON DELETE CASCADE,
    evaluated_at TEXT NOT NULL,
    actual_open REAL,
    actual_high REAL,
    actual_low REAL,
    actual_close REAL,
    mae REAL,
    rmse REAL,
    directional_correct INTEGER,
    error_pct REAL,
    is_intraday INTEGER NOT NULL DEFAULT 1
);

CREATE INDEX IF NOT EXISTS idx_eval_intraday ON evaluations(is_intraday, evaluated_at);

CREATE TABLE IF NOT EXISTS model_rollup (
    date TEXT NOT NULL,
    model TEXT NOT NULL,
    symbol TEXT,
    window TEXT NOT NULL,
    mae REAL,
    rmse REAL,
    directional_accuracy REAL,
    n INTEGER,
    PRIMARY KEY (date, model, symbol, window)
);
"""


def init_db(db_path: str | Path) -> Path:
    """Create the accuracy database (idempotent). Returns the resolved path."""
    global _DB_PATH
    path = Path(db_path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(path) as conn:
        conn.executescript(_SCHEMA)
        conn.commit()
    _DB_PATH = path
    return path


def get_db_path() -> Path:
    if _DB_PATH is None:
        raise RuntimeError(
            "accuracy.db not initialized. Call init_db(path) first "
            "(usually done in UI/backend/main.py lifespan)."
        )
    return _DB_PATH


@contextmanager
def get_conn() -> Iterator[sqlite3.Connection]:
    """Yield a sqlite3 connection with row factory + foreign keys enabled."""
    conn = sqlite3.connect(get_db_path(), detect_types=sqlite3.PARSE_DECLTYPES)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()
