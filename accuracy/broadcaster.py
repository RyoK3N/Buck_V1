"""
accuracy.broadcaster
────────────────────
In-process pub/sub for live accuracy updates.

`LIVE_STATE` is a `(model, symbol) -> snapshot` dict refreshed by the evaluator
after every reconciliation. WebSocket subscribers receive delta events through
asyncio queues registered via `subscribe()`.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Any, Dict, List, Tuple

# (model, symbol) — symbol="*" for cross-symbol model rollups.
LIVE_STATE: Dict[Tuple[str, str], Dict[str, Any]] = {}

_subscribers: list[asyncio.Queue] = []
_lock = asyncio.Lock()


def _iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


async def publish(event: Dict[str, Any]) -> None:
    """Fan an evaluation event out to all subscribers; update LIVE_STATE."""
    key = (event.get("model", "unknown"), event.get("symbol") or "*")
    bucket = LIVE_STATE.setdefault(key, {
        "model": key[0],
        "symbol": event.get("symbol"),
        "n": 0,
        "mae_pct_sum": 0.0,
        "directional_hits": 0,
        "mae_pct": None,
        "directional_accuracy_pct": None,
        "updated_at": None,
    })
    bucket["n"] += 1
    if event.get("error_pct") is not None:
        bucket["mae_pct_sum"] += abs(float(event["error_pct"]))
        bucket["mae_pct"] = bucket["mae_pct_sum"] / bucket["n"]
    if event.get("directional_correct") is not None:
        bucket["directional_hits"] += int(event["directional_correct"])
        bucket["directional_accuracy_pct"] = bucket["directional_hits"] / bucket["n"] * 100.0
    bucket["updated_at"] = _iso()

    payload = {
        "type": "evaluation",
        "model": key[0],
        "symbol": event.get("symbol"),
        "prediction_id": event.get("prediction_id"),
        "error_pct": event.get("error_pct"),
        "directional_correct": event.get("directional_correct"),
        "rolling": {
            "mae_pct": bucket["mae_pct"],
            "directional_accuracy_pct": bucket["directional_accuracy_pct"],
            "n": bucket["n"],
        },
        "ts": bucket["updated_at"],
    }
    async with _lock:
        dead: list[asyncio.Queue] = []
        for q in _subscribers:
            try:
                q.put_nowait(payload)
            except asyncio.QueueFull:
                dead.append(q)
        for q in dead:
            try:
                _subscribers.remove(q)
            except ValueError:
                pass


def publish_sync(event: Dict[str, Any]) -> None:
    """Publish from non-async contexts. Schedules onto the running loop if any."""
    try:
        loop = asyncio.get_running_loop()
        loop.create_task(publish(event))
    except RuntimeError:
        # No loop — fall back to synchronous LIVE_STATE update only.
        asyncio.run(publish(event))


async def subscribe() -> asyncio.Queue:
    q: asyncio.Queue = asyncio.Queue(maxsize=256)
    async with _lock:
        _subscribers.append(q)
    return q


async def unsubscribe(q: asyncio.Queue) -> None:
    async with _lock:
        try:
            _subscribers.remove(q)
        except ValueError:
            pass


def snapshot() -> List[Dict[str, Any]]:
    return [
        {
            "model": v["model"],
            "symbol": v["symbol"],
            "mae_pct": v["mae_pct"],
            "directional_accuracy_pct": v["directional_accuracy_pct"],
            "n": v["n"],
            "updated_at": v["updated_at"],
        }
        for v in LIVE_STATE.values()
    ]
