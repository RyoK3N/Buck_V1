"""
tools.rl.sessions
─────────────────
Lightweight persistence for RL training sessions.

`/rl/train` already returns rich per-episode metrics (reward, return, sharpe,
drawdown, losses) plus an equity curve, but historically nothing was saved — so
once the HTTP response was consumed, the training run was gone. This module
writes one JSON file per run under `tools/rl/sessions/` so the run can be
revisited later for d3 observability (see `UI/backend/d3_viz.py`) and exposed to
Claude through the `list_training_sessions` / `visualize_training` MCP tools.

Files are named `<model_id>__<UTC-timestamp>.json`; the stem is the session_id.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

SESSIONS_DIR = Path(__file__).resolve().parent / "sessions"


def _safe(name: str) -> str:
    return "".join(c if (c.isalnum() or c in "-._") else "_" for c in str(name))


def save_session(result: Dict[str, Any], *, request: Any = None) -> Optional[str]:
    """Persist a training result dict. Returns the session_id (stem), or None.

    Best-effort: never raises into the training path — a failed save just logs.
    """
    try:
        SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        model_id = result.get("model_id", "model")
        session_id = f"{_safe(model_id)}__{ts}"

        record: Dict[str, Any] = {
            "session_id": session_id,
            "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "model_id": model_id,
            "symbol": result.get("symbol"),
            "algorithm": result.get("algorithm"),
            "episodes": result.get("episodes"),
            "total_steps": result.get("total_steps"),
            "episode_rewards": result.get("episode_rewards", []),
            "equity_curve": result.get("equity_curve", []),
            "final_summary": result.get("final_summary", {}),
        }
        if request is not None:
            record["interval"] = getattr(request, "interval", None)
            record["start_date"] = getattr(request, "start_date", None)
            record["end_date"] = getattr(request, "end_date", None)
            record["hyperparams"] = {
                "hidden_dim": getattr(request, "hidden_dim", None),
                "learning_rate": getattr(request, "learning_rate", None),
                "initial_capital": getattr(request, "initial_capital", None),
            }

        path = SESSIONS_DIR / f"{session_id}.json"
        path.write_text(json.dumps(record, indent=2, default=str))
        return session_id
    except Exception:  # noqa: BLE001
        try:
            from agent_scripts.config import LOGGER

            LOGGER.warning("Failed to persist training session for %s", result.get("model_id"))
        except Exception:
            pass
        return None


def list_sessions(
    model_id: Optional[str] = None,
    symbol: Optional[str] = None,
    limit: int = 50,
) -> List[Dict[str, Any]]:
    """List saved sessions (newest first) as lightweight summaries."""
    if not SESSIONS_DIR.exists():
        return []
    out: List[Dict[str, Any]] = []
    for path in sorted(SESSIONS_DIR.glob("*.json"), reverse=True):
        try:
            rec = json.loads(path.read_text())
        except Exception:
            continue
        if model_id and rec.get("model_id") != model_id:
            continue
        if symbol and rec.get("symbol") != symbol:
            continue
        eps = rec.get("episode_rewards") or []
        last = eps[-1] if eps else {}
        out.append({
            "session_id": rec.get("session_id", path.stem),
            "model_id": rec.get("model_id"),
            "symbol": rec.get("symbol"),
            "algorithm": rec.get("algorithm"),
            "interval": rec.get("interval"),
            "episodes": rec.get("episodes") or len(eps),
            "created_at": rec.get("created_at"),
            "final_return_pct": last.get("return_pct"),
            "final_sharpe": last.get("sharpe"),
        })
        if len(out) >= limit:
            break
    return out


def load_session(session_id: str) -> Optional[Dict[str, Any]]:
    """Load a full session record by id (file stem)."""
    path = SESSIONS_DIR / f"{_safe(session_id)}.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except Exception:
        return None
