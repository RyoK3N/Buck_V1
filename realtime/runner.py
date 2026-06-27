"""
realtime.runner
───────────────
Manage realtime intraday simulations as background threads so the web UI can
**start and stop** them without the terminal (the CLI `python -m realtime.cli`
runs the same loop in the foreground).

A single process-wide `MANAGER` tracks one running simulator per symbol. Each
session runs in a daemon thread; `stop()` sets the simulator's cooperative stop
flag and the loop finalises at the next bar/poll boundary. Live session state
(action / equity / PnL / step history) is still read through `realtime.state`
(and the `rt_session_*` MCP tools), so the read surface is unchanged.
"""

from __future__ import annotations

import threading
from typing import Any, Dict, List, Optional

from .sim import IntradaySimulator


class _Running:
    __slots__ = ("sim", "thread", "error")

    def __init__(self, sim: IntradaySimulator, thread: threading.Thread) -> None:
        self.sim = sim
        self.thread = thread
        self.error: Optional[str] = None


class SessionManager:
    """Process-wide registry of running realtime simulations, keyed by symbol."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._running: Dict[str, _Running] = {}

    # ── lifecycle ────────────────────────────────────────────────────────────
    def start(
        self,
        *,
        symbol: str,
        model_id: str,
        interval: str = "1m",
        replay: bool = False,
        replay_start: Optional[str] = None,
        replay_end: Optional[str] = None,
        capital: float = 100_000.0,
        max_steps: int = 2000,
        exchange: Optional[str] = None,
        poll_seconds: Optional[float] = None,
        online_update_every: Optional[int] = None,
        speed: float = 1.0,
        indian_api_key: str = "",
    ) -> Dict[str, Any]:
        if replay and (not replay_start or not replay_end):
            raise ValueError("replay mode requires replay_start and replay_end")

        with self._lock:
            self._reap_locked()
            if symbol in self._running and self._running[symbol].thread.is_alive():
                raise RuntimeError(f"a session for {symbol!r} is already running; stop it first")

            # Resolve defaults from settings, tolerating a missing config.
            # Online updates default OFF for replay (deterministic backtest parity);
            # live keeps the configured adaptive cadence.
            live_default_updates = 4
            try:
                from agent_scripts.config import SETTINGS
                exchange = exchange or getattr(SETTINGS, "market_exchange", "NSE")
                poll_seconds = poll_seconds if poll_seconds is not None else getattr(SETTINGS, "rt_poll_seconds", 30.0)
                live_default_updates = getattr(SETTINGS, "rt_online_update_every", 4)
            except Exception:  # noqa: BLE001
                exchange = exchange or "NSE"
                poll_seconds = poll_seconds if poll_seconds is not None else 30.0
            if online_update_every is None:
                online_update_every = 0 if replay else live_default_updates

            import os
            sim = IntradaySimulator(
                symbol=symbol,
                model_id=model_id,
                interval=interval,
                exchange=exchange,
                poll_seconds=poll_seconds,
                capital=capital,
                replay=replay,
                replay_start=replay_start,
                replay_end=replay_end,
                online_update_every=online_update_every,
                api_key=indian_api_key or os.environ.get("INDIAN_API_KEY", ""),
                max_steps=max_steps,
                speed=speed,
            )

            running = _Running(sim, thread=None)  # type: ignore[arg-type]

            def _run() -> None:
                try:
                    sim.run()
                except Exception as exc:  # noqa: BLE001
                    running.error = str(exc)
                    sim.state.status = "error"

            thread = threading.Thread(target=_run, name=f"rt-sim-{symbol}", daemon=True)
            running.thread = thread
            self._running[symbol] = running
            thread.start()

        return self.status(symbol)

    def stop(self, symbol: str, *, join_timeout: float = 5.0) -> Dict[str, Any]:
        with self._lock:
            running = self._running.get(symbol)
        if running is None:
            raise KeyError(f"no session for {symbol!r}")
        running.sim.request_stop()
        running.thread.join(timeout=join_timeout)
        return self.status(symbol)

    # ── introspection ────────────────────────────────────────────────────────
    def status(self, symbol: str) -> Dict[str, Any]:
        from .state import get_status
        st = get_status(symbol)
        with self._lock:
            running = self._running.get(symbol)
            st["running"] = bool(running and running.thread.is_alive())
            if running and running.error:
                st["error"] = running.error
        return st

    def list_sessions(self) -> List[Dict[str, Any]]:
        with self._lock:
            self._reap_locked()
            symbols = list(self._running.keys())
        return [self.status(s) for s in symbols]

    # ── internals ────────────────────────────────────────────────────────────
    def _reap_locked(self) -> None:
        """Drop finished sessions whose state is terminal (called under lock)."""
        for sym, running in list(self._running.items()):
            if not running.thread.is_alive() and running.sim.state.status in {"stopped", "error"}:
                # Keep the last finished session readable via realtime.state, but
                # free the manager slot so the symbol can be started again.
                self._running.pop(sym, None)


# Process-wide singleton used by the REST layer.
MANAGER = SessionManager()
