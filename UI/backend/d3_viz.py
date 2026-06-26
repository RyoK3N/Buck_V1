"""
UI.backend.d3_viz
─────────────────
Framework-agnostic **d3** chart specs for RL training-session observability.

Unlike `UI/backend/visualizer.py` (which emits Plotly figure dicts for market
charts), this module emits compact `d3-buck/1` specs — pure data + encoding —
that the frontend renders with d3 (`UI/frontend/src/components/D3Chart.tsx`).
The same specs are returned to Claude through the `visualize_training` MCP tool
(compressed by the headroom layer), giving the model structured numbers to
reason over instead of an opaque image.

Spec shape:
    {
      "spec_version": "d3-buck/1",
      "chart": "<chart_id>",
      "mark": "line" | "area" | "multiline" | "bar" | "heatmap",
      "data": [ {...}, ... ],
      "encoding": {"x": {...}, "y": {...}, "series": [...]},
      "meta": {...}
    }
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

SPEC_VERSION = "d3-buck/1"


D3_CHART_CATALOGUE: List[Dict[str, str]] = [
    {"id": "reward_curve", "label": "Reward Curve",
     "description": "Per-episode total reward with a rolling mean — shows learning progress / convergence."},
    {"id": "equity_curve", "label": "Equity Curve",
     "description": "Portfolio value over the run — the agent's simulated PnL trajectory."},
    {"id": "loss_curves", "label": "Loss Curves",
     "description": "Policy loss, value loss and entropy per episode (PPO) — diagnose under/over-fitting."},
    {"id": "return_distribution", "label": "Return Distribution",
     "description": "Histogram of per-episode return % — spread and skew of outcomes."},
    {"id": "drawdown_curve", "label": "Drawdown Curve",
     "description": "Max drawdown % per episode — risk profile across training."},
    {"id": "action_heatmap", "label": "Action Heatmap",
     "description": "Target-position / action intensity over time — when the agent is long vs flat."},
]

D3_CHART_DESCRIPTIONS: Dict[str, str] = {c["id"]: c["description"] for c in D3_CHART_CATALOGUE}


# ── helpers ──────────────────────────────────────────────────────────────────

def _rolling_mean(values: List[float], window: int = 10) -> List[Optional[float]]:
    out: List[Optional[float]] = []
    acc: List[float] = []
    for v in values:
        acc.append(v)
        if len(acc) > window:
            acc.pop(0)
        out.append(round(sum(acc) / len(acc), 4))
    return out


def _episodes(session: Dict[str, Any]) -> List[Dict[str, Any]]:
    return session.get("episode_rewards") or []


def _empty(chart: str, reason: str) -> Dict[str, Any]:
    return {
        "spec_version": SPEC_VERSION,
        "chart": chart,
        "mark": "line",
        "data": [],
        "encoding": {},
        "meta": {"empty": True, "reason": reason},
    }


def _base_meta(session: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "session_id": session.get("session_id"),
        "model_id": session.get("model_id"),
        "symbol": session.get("symbol"),
        "algorithm": session.get("algorithm"),
    }


# ── chart builders ───────────────────────────────────────────────────────────

def _reward_curve(session: Dict[str, Any]) -> Dict[str, Any]:
    eps = _episodes(session)
    if not eps:
        return _empty("reward_curve", "no episode_rewards")
    rewards = [float(e.get("total_reward", 0.0)) for e in eps]
    roll = _rolling_mean(rewards, window=max(2, len(rewards) // 10 or 2))
    data = [
        {"episode": int(e.get("episode", i + 1)), "total_reward": rewards[i], "rolling_mean": roll[i]}
        for i, e in enumerate(eps)
    ]
    return {
        "spec_version": SPEC_VERSION,
        "chart": "reward_curve",
        "mark": "multiline",
        "data": data,
        "encoding": {
            "x": {"field": "episode", "type": "quantitative", "title": "Episode"},
            "y": {"field": "total_reward", "type": "quantitative", "title": "Total reward"},
            "series": [
                {"field": "total_reward", "label": "Reward", "color": "#6366f1"},
                {"field": "rolling_mean", "label": "Rolling mean", "color": "#f59e0b"},
            ],
        },
        "meta": _base_meta(session),
    }


def _equity_curve(session: Dict[str, Any]) -> Dict[str, Any]:
    curve = session.get("equity_curve") or []
    if not curve:
        return _empty("equity_curve", "no equity_curve")
    data = []
    for i, pt in enumerate(curve):
        if isinstance(pt, dict):
            pv = pt.get("portfolio_value", pt.get("equity"))
            ts = pt.get("timestamp", i)
        else:
            pv, ts = pt, i
        if pv is None:
            continue
        data.append({"t": ts, "step": i, "portfolio_value": round(float(pv), 2)})
    return {
        "spec_version": SPEC_VERSION,
        "chart": "equity_curve",
        "mark": "area",
        "data": data,
        "encoding": {
            "x": {"field": "step", "type": "quantitative", "title": "Step"},
            "y": {"field": "portfolio_value", "type": "quantitative", "title": "Portfolio value"},
            "series": [{"field": "portfolio_value", "label": "Equity", "color": "#10b981"}],
        },
        "meta": _base_meta(session),
    }


def _loss_curves(session: Dict[str, Any]) -> Dict[str, Any]:
    eps = _episodes(session)
    has_loss = any("policy_loss" in e or "value_loss" in e or "entropy" in e for e in eps)
    if not eps or not has_loss:
        return _empty("loss_curves", "no loss metrics (PPO-only)")
    data = [
        {
            "episode": int(e.get("episode", i + 1)),
            "policy_loss": float(e.get("policy_loss", 0.0)),
            "value_loss": float(e.get("value_loss", 0.0)),
            "entropy": float(e.get("entropy", 0.0)),
        }
        for i, e in enumerate(eps)
    ]
    return {
        "spec_version": SPEC_VERSION,
        "chart": "loss_curves",
        "mark": "multiline",
        "data": data,
        "encoding": {
            "x": {"field": "episode", "type": "quantitative", "title": "Episode"},
            "y": {"field": "policy_loss", "type": "quantitative", "title": "Value"},
            "series": [
                {"field": "policy_loss", "label": "Policy loss", "color": "#ef4444"},
                {"field": "value_loss", "label": "Value loss", "color": "#3b82f6"},
                {"field": "entropy", "label": "Entropy", "color": "#a855f7"},
            ],
        },
        "meta": _base_meta(session),
    }


def _return_distribution(session: Dict[str, Any]) -> Dict[str, Any]:
    eps = _episodes(session)
    rets = [float(e.get("return_pct", 0.0)) for e in eps if e.get("return_pct") is not None]
    if not rets:
        return _empty("return_distribution", "no return_pct")
    lo, hi = min(rets), max(rets)
    n_bins = min(20, max(5, len(rets) // 3 or 5))
    width = (hi - lo) / n_bins if hi > lo else 1.0
    bins = [0] * n_bins
    for r in rets:
        idx = min(n_bins - 1, int((r - lo) / width)) if width else 0
        bins[idx] += 1
    data = [
        {"bin_start": round(lo + i * width, 4), "bin_end": round(lo + (i + 1) * width, 4), "count": bins[i]}
        for i in range(n_bins)
    ]
    return {
        "spec_version": SPEC_VERSION,
        "chart": "return_distribution",
        "mark": "bar",
        "data": data,
        "encoding": {
            "x": {"field": "bin_start", "type": "quantitative", "title": "Return %"},
            "y": {"field": "count", "type": "quantitative", "title": "Episodes"},
            "series": [{"field": "count", "label": "Episodes", "color": "#0ea5e9"}],
        },
        "meta": {**_base_meta(session), "min": round(lo, 4), "max": round(hi, 4), "mean": round(sum(rets) / len(rets), 4)},
    }


def _drawdown_curve(session: Dict[str, Any]) -> Dict[str, Any]:
    eps = _episodes(session)
    if any(e.get("max_drawdown_pct") is not None for e in eps):
        data = [
            {"episode": int(e.get("episode", i + 1)), "max_drawdown_pct": float(e.get("max_drawdown_pct", 0.0))}
            for i, e in enumerate(eps)
        ]
        return {
            "spec_version": SPEC_VERSION,
            "chart": "drawdown_curve",
            "mark": "area",
            "data": data,
            "encoding": {
                "x": {"field": "episode", "type": "quantitative", "title": "Episode"},
                "y": {"field": "max_drawdown_pct", "type": "quantitative", "title": "Max drawdown %"},
                "series": [{"field": "max_drawdown_pct", "label": "Drawdown %", "color": "#dc2626"}],
            },
            "meta": _base_meta(session),
        }
    # Derive a running drawdown from the equity curve if no per-episode metric.
    curve = session.get("equity_curve") or []
    pvs = [float(p.get("portfolio_value")) for p in curve if isinstance(p, dict) and p.get("portfolio_value") is not None]
    if not pvs:
        return _empty("drawdown_curve", "no drawdown data")
    peak = pvs[0]
    data = []
    for i, pv in enumerate(pvs):
        peak = max(peak, pv)
        dd = (pv - peak) / peak * 100 if peak else 0.0
        data.append({"step": i, "drawdown_pct": round(dd, 4)})
    return {
        "spec_version": SPEC_VERSION,
        "chart": "drawdown_curve",
        "mark": "area",
        "data": data,
        "encoding": {
            "x": {"field": "step", "type": "quantitative", "title": "Step"},
            "y": {"field": "drawdown_pct", "type": "quantitative", "title": "Drawdown %"},
            "series": [{"field": "drawdown_pct", "label": "Drawdown %", "color": "#dc2626"}],
        },
        "meta": _base_meta(session),
    }


def _action_heatmap(session: Dict[str, Any]) -> Dict[str, Any]:
    """Action / target-position intensity over time.

    Works off a per-step `steps`/`signals` list (live RT sessions and rl_predict
    output) when present; otherwise falls back to colouring episodes by return.
    """
    steps = session.get("steps") or session.get("signals") or []
    if steps:
        data = []
        for i, s in enumerate(steps):
            pos = s.get("target_position", s.get("realized_position", s.get("position")))
            data.append({
                "step": int(s.get("step", i)),
                "value": round(float(pos), 4) if pos is not None else 0.0,
                "price": s.get("price"),
            })
        return {
            "spec_version": SPEC_VERSION,
            "chart": "action_heatmap",
            "mark": "heatmap",
            "data": data,
            "encoding": {
                "x": {"field": "step", "type": "quantitative", "title": "Step"},
                "intensity": {"field": "value", "type": "quantitative", "title": "Target position",
                              "scheme": "viridis", "domain": [0, 1]},
            },
            "meta": _base_meta(session),
        }
    eps = _episodes(session)
    if not eps:
        return _empty("action_heatmap", "no per-step actions or episodes")
    data = [
        {"step": int(e.get("episode", i + 1)), "value": float(e.get("return_pct", 0.0))}
        for i, e in enumerate(eps)
    ]
    return {
        "spec_version": SPEC_VERSION,
        "chart": "action_heatmap",
        "mark": "heatmap",
        "data": data,
        "encoding": {
            "x": {"field": "step", "type": "quantitative", "title": "Episode"},
            "intensity": {"field": "value", "type": "quantitative", "title": "Return %", "scheme": "rdylgn"},
        },
        "meta": {**_base_meta(session), "fallback": "episode_return"},
    }


_BUILDERS = {
    "reward_curve": _reward_curve,
    "equity_curve": _equity_curve,
    "loss_curves": _loss_curves,
    "return_distribution": _return_distribution,
    "drawdown_curve": _drawdown_curve,
    "action_heatmap": _action_heatmap,
}


def build_d3_spec(chart: str, session: Dict[str, Any]) -> Dict[str, Any]:
    """Build a d3-buck spec for `chart` from a training/realtime session record."""
    builder = _BUILDERS.get(chart)
    if builder is None:
        return _empty(chart, f"unknown chart {chart!r}; valid: {sorted(_BUILDERS)}")
    return builder(session)
