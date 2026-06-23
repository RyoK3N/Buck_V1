"""
tools.rl.ensemble
─────────────────
Multi-timeframe ensemble inference.

Given a list of model specs `[{"model_id", "interval", "weight"}, ...]`,
loads each model, fetches data at its native interval, runs the last-bar
inference, and returns a weighted combination of their signals along with
the per-model breakdown.

Conventions:
  * For PPO-continuous models, the signal is a target position fraction in
    [0, 1]. We use that directly.
  * For discrete DQN/A2C/PPO-discrete models, the signal is mapped from
    action {0=HOLD, 1=BUY, 2=SELL} to a target fraction:
        HOLD = current_position (i.e., no change), pragmatically encoded as 0.5
        BUY  = 1.0
        SELL = 0.0
    This intentionally penalizes HOLD slightly in the ensemble — when the
    discrete model is unsure (HOLD), it contributes a neutral 0.5 so PPO
    models can swing the vote.

The returned `ensemble_action` is the weighted average of per-model target
fractions, suitable for direct use by the live trader or as the seed for a
Claude-side confluence check.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from .dqn_agent import load_agent, extract_state
from .features import STATE_DIM
from .ppo_continuous import PPOContinuousAgent
from .rl_tool import fetch_historical_data, fetch_live_data
from .env import TradingEnvironment


_DISCRETE_TO_FRACTION = {0: 0.5, 1: 1.0, 2: 0.0}  # HOLD / BUY / SELL


@dataclass
class ModelSignal:
    model_id: str
    algorithm: str
    interval: str
    weight: float
    raw_action: float      # in the model's native action space (0/1/2 or 0..1)
    target_fraction: float # mapped to [0, 1]
    note: Optional[str] = None
    error: Optional[str] = None


def _load_any_agent(model_id: str) -> Any:
    """Try PPO-continuous first, fall back to the legacy DQN loader."""
    agent = PPOContinuousAgent()
    if agent.load(model_id):
        return agent
    return load_agent(model_id)


def _signal_from_dqn_like(agent, df: pd.DataFrame) -> int:
    """Last-bar discrete action from DQN/A2C/PPO-discrete."""
    if df is None or df.empty:
        raise ValueError("no data")
    last_idx = len(df) - 1
    # Try to use a current cash_ratio of 1.0 (no position) — this is the
    # "fresh look" inference setting. Matches how the FastAPI route uses it.
    state = extract_state(df, last_idx, position=0, cash_ratio=1.0)
    return int(agent.act(state, eval_mode=True))


def _signal_from_ppo_continuous(agent: PPOContinuousAgent, df: pd.DataFrame) -> float:
    """Last-bar continuous action from PPO-continuous. Warms the LSTM window
    by replaying the env over the available bars in eval mode."""
    env = TradingEnvironment(df)
    state = env.reset()
    agent.reset_window()
    action = 0.5
    while True:
        action, _ = agent.act(state, eval_mode=True)
        if env.idx >= env.n - 1:
            break
        state, _, done, _ = env.step(action)
        if done:
            break
    return float(action)


def ensemble_predict(
    symbol: str,
    models: List[Dict[str, Any]],
    start_date: str,
    end_date: str,
    fallback_interval: str = "1d",
) -> Dict[str, Any]:
    """Run each model on data at its native interval and return weighted aggregate.

    Args:
        symbol: Ticker (use exchange suffix where appropriate, e.g. 'BHEL.NS').
        models: list of dicts: {"model_id": str, "interval": str (optional),
                                "weight": float (optional, default 1.0)}
        start_date, end_date: YYYY-MM-DD window for each model's data fetch.
        fallback_interval: used when a model spec doesn't specify one.

    Returns:
        dict with keys: ensemble_action (float in [0,1]), per_model (list of
        ModelSignal as dicts), n_used, weights_sum, note.
    """
    if not models:
        return {"ensemble_action": 0.5, "per_model": [], "n_used": 0,
                "weights_sum": 0.0, "note": "no models provided"}

    per_model: List[ModelSignal] = []
    weighted_sum = 0.0
    weight_total = 0.0

    for spec in models:
        model_id = spec.get("model_id")
        interval = spec.get("interval", fallback_interval)
        weight = float(spec.get("weight", 1.0))
        if not model_id:
            continue
        agent = _load_any_agent(model_id)
        if agent is None:
            per_model.append(ModelSignal(model_id=model_id, algorithm="?", interval=interval,
                                          weight=weight, raw_action=0.0, target_fraction=0.5,
                                          error="model not found"))
            continue
        try:
            df = fetch_historical_data(symbol, start_date, end_date, interval)
            if df is None or df.empty:
                raise ValueError(f"no data for {symbol} @ {interval}")
            algo = getattr(agent, "algorithm", "dqn")
            if algo == "ppo_continuous":
                target = _signal_from_ppo_continuous(agent, df)
                sig = ModelSignal(model_id=model_id, algorithm=algo, interval=interval,
                                  weight=weight, raw_action=target, target_fraction=target)
            else:
                a = _signal_from_dqn_like(agent, df)
                target = _DISCRETE_TO_FRACTION.get(int(a), 0.5)
                sig = ModelSignal(model_id=model_id, algorithm=algo, interval=interval,
                                  weight=weight, raw_action=float(a), target_fraction=target)
            per_model.append(sig)
            weighted_sum += sig.target_fraction * weight
            weight_total += weight
        except Exception as exc:  # noqa: BLE001
            per_model.append(ModelSignal(model_id=model_id, algorithm=getattr(agent, "algorithm", "?"),
                                          interval=interval, weight=weight, raw_action=0.0,
                                          target_fraction=0.5, error=str(exc)))

    if weight_total <= 1e-9:
        return {"ensemble_action": 0.5,
                "per_model": [s.__dict__ for s in per_model],
                "n_used": 0, "weights_sum": 0.0,
                "note": "no model contributed (all errored or zero-weight)"}

    ensemble_action = float(weighted_sum / weight_total)
    note = None
    if any(s.error for s in per_model):
        note = "some models errored — see per_model[].error"

    return {
        "symbol": symbol,
        "ensemble_action": round(ensemble_action, 4),
        "ensemble_signal": _classify_signal(ensemble_action),
        "n_used": sum(1 for s in per_model if s.error is None),
        "n_total": len(per_model),
        "weights_sum": round(weight_total, 4),
        "per_model": [s.__dict__ for s in per_model],
        "note": note,
    }


def _classify_signal(target_fraction: float) -> str:
    """Human-readable bucket for the ensemble action."""
    if target_fraction >= 0.66:
        return "BUY"
    if target_fraction <= 0.33:
        return "SELL"
    return "HOLD"


if __name__ == "__main__":
    import os
    os.environ.setdefault("INDIAN_API_KEY", "")
    result = ensemble_predict(
        symbol="BHEL.NS",
        models=[
            {"model_id": "dqn_model_best", "interval": "1d", "weight": 1.0},
            {"model_id": "dqn_model", "interval": "1d", "weight": 0.5},
        ],
        start_date="2025-12-23",
        end_date="2026-06-20",
    )
    import json
    print(json.dumps(result, indent=2, default=str))
