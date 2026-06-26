"""
realtime.state
──────────────
In-memory state for live intraday sessions — the read surface exposed to Claude
via the `rt_session_status` / `rt_session_history` MCP tools.

Modelled on the `accuracy/broadcaster.py` in-process pub/sub style: a single
process holds the running sessions, keyed by symbol. The CLI loop writes; the
MCP read tools (and the REST layer) read. Step summaries are pushed through the
headroom pipeline before storage so what Claude ultimately reads is compact.
"""

from __future__ import annotations

import threading
from collections import deque
from datetime import datetime, timezone
from typing import Any, Deque, Dict, List, Optional

_LOCK = threading.Lock()
_SESSIONS: Dict[str, "LiveSessionState"] = {}
_ACTIVE: Optional[str] = None  # most-recently registered symbol


class LiveSessionState:
    """Mutable state for one symbol's live intraday session."""

    def __init__(self, symbol: str, model_id: str, *, capital: float, exchange: str, replay: bool, max_steps: int = 1000) -> None:
        self.symbol = symbol
        self.model_id = model_id
        self.exchange = exchange
        self.replay = replay
        self.started_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
        self.capital = capital
        # live metrics
        self.last_action: Optional[float] = None
        self.last_signal: Optional[str] = None
        self.last_price: Optional[float] = None
        self.equity: float = capital
        self.intraday_pnl: float = 0.0
        self.intraday_pnl_pct: float = 0.0
        self.n_steps: int = 0
        self.n_updates: int = 0
        self.last_update_stats: Dict[str, float] = {}
        self.market_open: bool = False
        self.status: str = "starting"  # starting | running | waiting_market | stopped
        self.steps: Deque[Dict[str, Any]] = deque(maxlen=max_steps)
        self.updated_at = self.started_at

    def record_step(self, step: Dict[str, Any]) -> None:
        self.steps.append(step)
        self.n_steps += 1
        self.last_action = step.get("target_position")
        self.last_signal = step.get("signal")
        self.last_price = step.get("price")
        if "equity" in step:
            self.equity = step["equity"]
            self.intraday_pnl = round(self.equity - self.capital, 2)
            self.intraday_pnl_pct = round((self.equity / self.capital - 1.0) * 100.0, 4)
        self.updated_at = datetime.now(timezone.utc).isoformat(timespec="seconds")

    def record_update(self, stats: Dict[str, float]) -> None:
        self.n_updates += 1
        self.last_update_stats = {k: round(float(v), 5) for k, v in stats.items()}

    def status_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "model_id": self.model_id,
            "exchange": self.exchange,
            "replay": self.replay,
            "status": self.status,
            "market_open": self.market_open,
            "started_at": self.started_at,
            "updated_at": self.updated_at,
            "capital": self.capital,
            "equity": round(self.equity, 2),
            "intraday_pnl": round(self.intraday_pnl, 2),
            "intraday_pnl_pct": self.intraday_pnl_pct,
            "last_action": self.last_action,
            "last_signal": self.last_signal,
            "last_price": self.last_price,
            "n_steps": self.n_steps,
            "n_updates": self.n_updates,
            "last_update_stats": self.last_update_stats,
        }


def register_session(state: LiveSessionState) -> None:
    global _ACTIVE
    with _LOCK:
        _SESSIONS[state.symbol] = state
        _ACTIVE = state.symbol


def _resolve(symbol: Optional[str]) -> Optional[LiveSessionState]:
    with _LOCK:
        if symbol:
            return _SESSIONS.get(symbol)
        if _ACTIVE:
            return _SESSIONS.get(_ACTIVE)
        return None


def get_status(symbol: Optional[str] = None) -> Dict[str, Any]:
    state = _resolve(symbol)
    if state is None:
        return {"active": False, "reason": "no live session — start one with `python -m realtime.cli`"}
    return {"active": True, **state.status_dict()}


def get_history(symbol: Optional[str] = None, limit: int = 50) -> List[Dict[str, Any]]:
    state = _resolve(symbol)
    if state is None:
        return []
    steps = list(state.steps)
    return steps[-limit:]
