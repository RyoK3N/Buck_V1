"""
realtime
────────
Real-time intraday simulation for Buck.

A CLI-driven loop (`python -m realtime.cli`) that, during market hours (or in
`--replay`), polls live data, has a trained RL model **predict before each
event**, observes the realised outcome, runs **online PPO updates** to maximise
intraday profit, and streams its state through the headroom context-engineering
pipeline so Claude can read it via the `rt_session_status` / `rt_session_history`
MCP tools.

The online updates train an *in-session adaptive copy* of the model and save to
a `<model_id>_live` checkpoint — the base model is never overwritten.
"""

from __future__ import annotations

from .state import LiveSessionState, get_status, get_history, register_session

__all__ = ["LiveSessionState", "get_status", "get_history", "register_session"]
