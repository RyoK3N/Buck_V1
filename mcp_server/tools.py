"""
mcp_server.tools
────────────────
Implementations for every tool declared in `registry.py`.

Each tool function:
  * accepts JSON-serializable arguments (matching the input_schema)
  * returns a JSON-serializable dict
  * is async — long-running ones offload to executors as needed
  * pulls API keys from `os.environ` if not provided in args, so MCP calls
    (which have no per-request header) Just Work as long as the server has
    `.env` loaded.
"""

from __future__ import annotations

import asyncio
import os
import time
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable, Dict, List, Optional

from .registry import BUCK_TOOLS_BY_NAME

# ── Per-tool last-call telemetry (for /mcp/status) ──────────────────────────
LAST_CALL: Dict[str, Dict[str, Any]] = {}


def _record_call(name: str, ok: bool, latency_ms: float, error: Optional[str] = None) -> None:
    LAST_CALL[name] = {
        "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "ok": ok,
        "latency_ms": round(latency_ms, 1),
        "error": error,
    }


# ── Tool implementations ─────────────────────────────────────────────────────

async def single_analyze(
    symbol: str,
    start_date: str,
    end_date: str,
    interval: str = "1h",
    selected_tools: Optional[List[str]] = None,
    openai_api_key: Optional[str] = None,
    indian_api_key: Optional[str] = None,
    model: Optional[str] = None,
    base_url: Optional[str] = None,
) -> Dict[str, Any]:
    from agent_scripts.buck import BuckFactory
    buck = BuckFactory.create_production_agent(
        openai_api_key=openai_api_key or os.environ.get("OPENAI_API_KEY", ""),
        indian_api_key=indian_api_key or os.environ.get("INDIAN_API_KEY", ""),
        model=model or "gpt-4o",
        base_url=base_url,
        selected_tools=selected_tools,
    )
    return await buck.analyze_and_predict(
        symbol=symbol,
        start_date=start_date,
        end_date=end_date,
        interval=interval,
        save_results=False,
    )


async def batch_analyze(
    symbols: List[str],
    start_date: str,
    end_date: str,
    interval: str = "1h",
    selected_tools: Optional[List[str]] = None,
    max_concurrent: int = 3,
    openai_api_key: Optional[str] = None,
    indian_api_key: Optional[str] = None,
    model: Optional[str] = None,
    base_url: Optional[str] = None,
) -> Dict[str, Any]:
    from agent_scripts.buck import BuckFactory
    buck = BuckFactory.create_production_agent(
        openai_api_key=openai_api_key or os.environ.get("OPENAI_API_KEY", ""),
        indian_api_key=indian_api_key or os.environ.get("INDIAN_API_KEY", ""),
        model=model or "gpt-4o",
        base_url=base_url,
        selected_tools=selected_tools,
    )
    return await buck.batch_analyze(
        symbols=symbols,
        start_date=start_date,
        end_date=end_date,
        interval=interval,
        max_concurrent=max_concurrent,
    )


async def list_tools_registry() -> Dict[str, Any]:
    from agent_scripts.tools import ToolFactory
    return ToolFactory.get_registry()


async def list_available_intervals() -> Dict[str, Any]:
    return {
        "intervals": [
            "1m", "2m", "5m", "15m", "30m", "60m", "90m",
            "1h", "1d", "5d", "1wk", "1mo", "3mo",
        ],
    }


async def list_chart_types() -> Dict[str, Any]:
    from UI.backend.visualizer import CHART_CATALOGUE
    return {"chart_types": CHART_CATALOGUE}


# ── RL ──────────────────────────────────────────────────────────────────────

async def rl_train(
    symbol: str,
    start_date: str,
    end_date: str,
    model_id: str,
    interval: str = "1d",
    algorithm: str = "dqn",
    episodes: int = 50,
    hidden_dim: int = 128,
    learning_rate: float = 0.001,
    initial_capital: float = 100000.0,
    indian_api_key: Optional[str] = None,
) -> Dict[str, Any]:
    # Reuse the same orchestration code as the REST route by calling it.
    from UI.backend.models import RLTrainRequest
    from UI.backend.routes import rl_train as rl_train_route
    req = RLTrainRequest(
        symbol=symbol, start_date=start_date, end_date=end_date, interval=interval,
        algorithm=algorithm, model_id=model_id, episodes=episodes, hidden_dim=hidden_dim,
        learning_rate=learning_rate, initial_capital=initial_capital,
        indian_api_key=indian_api_key or os.environ.get("INDIAN_API_KEY", ""),
    )
    return await rl_train_route(req)


async def rl_predict(
    symbol: str,
    start_date: str,
    end_date: str,
    model_id: str,
    interval: str = "1d",
    initial_capital: float = 100000.0,
    indian_api_key: Optional[str] = None,
) -> Dict[str, Any]:
    from UI.backend.models import RLPredictRequest
    from UI.backend.routes import rl_predict as rl_predict_route
    req = RLPredictRequest(
        symbol=symbol, start_date=start_date, end_date=end_date, interval=interval,
        model_id=model_id, initial_capital=initial_capital,
        indian_api_key=indian_api_key or os.environ.get("INDIAN_API_KEY", ""),
    )
    return await rl_predict_route(req)


async def rl_simulate(
    symbol: str,
    model_id: str,
    interval: str = "1m",
    initial_capital: float = 100000.0,
    indian_api_key: Optional[str] = None,
) -> Dict[str, Any]:
    from UI.backend.models import RLSimulateRequest
    from UI.backend.routes import rl_simulate as rl_simulate_route
    req = RLSimulateRequest(
        symbol=symbol, model_id=model_id, interval=interval,
        initial_capital=initial_capital,
        indian_api_key=indian_api_key or os.environ.get("INDIAN_API_KEY", ""),
    )
    return await rl_simulate_route(req)


async def list_rl_models() -> Dict[str, Any]:
    from tools.rl.dqn_agent import list_models
    return {"models": list_models()}


async def rl_ensemble_predict(
    symbol: str,
    start_date: str,
    end_date: str,
    models: List[Dict[str, Any]],
    fallback_interval: str = "1d",
    indian_api_key: Optional[str] = None,
) -> Dict[str, Any]:
    if indian_api_key:
        os.environ["INDIAN_API_KEY"] = indian_api_key
    from tools.rl.ensemble import ensemble_predict
    return ensemble_predict(
        symbol=symbol,
        models=models,
        start_date=start_date,
        end_date=end_date,
        fallback_interval=fallback_interval,
    )


# ── Visualization ───────────────────────────────────────────────────────────

async def visualize(
    symbol: str,
    start_date: str,
    end_date: str,
    chart_type: str,
    interval: str = "1d",
    indian_api_key: Optional[str] = None,
) -> Dict[str, Any]:
    from UI.backend.visualizer import fetch_df, build_chart, CHART_DESCRIPTIONS
    df = await fetch_df(
        symbol=symbol, start_date=start_date, end_date=end_date, interval=interval,
        indian_api_key=indian_api_key or os.environ.get("INDIAN_API_KEY", ""),
    )
    return {
        "chart": build_chart(chart_type, df, symbol),
        "chart_type": chart_type,
        "symbol": symbol,
        "description": CHART_DESCRIPTIONS.get(chart_type, ""),
    }


async def visualize_accuracy(
    model: Optional[str] = None,
    symbol: Optional[str] = None,
    window_days: int = 30,
    metric: str = "mae",
) -> Dict[str, Any]:
    """d3 spec of rolling accuracy (MAE or directional accuracy) per model over time."""
    from accuracy.repository import timeseries
    from UI.backend.d3_viz import build_accuracy_spec
    rows = timeseries(model=model, symbol=symbol, window_days=window_days)
    return {
        "metric": metric,
        "window_days": window_days,
        "spec": build_accuracy_spec(rows, metric=metric),
    }


async def visualize_predictions(
    symbol: str,
    lookback_days: int = 30,
    model: Optional[str] = None,
) -> Dict[str, Any]:
    """d3 spec of predicted-vs-actual close for a symbol over a lookback window."""
    from UI.backend.d3_viz import build_predictions_spec
    data = await compare_predictions_vs_actual(symbol=symbol, lookback_days=lookback_days, model=model)
    return {
        "symbol": symbol,
        "lookback_days": lookback_days,
        "spec": build_predictions_spec(symbol, data.get("series", [])),
    }


async def visualize_compare(
    symbols: List[str],
    start_date: str,
    end_date: str,
    interval: str = "1d",
    normalize: bool = True,
    indian_api_key: Optional[str] = None,
) -> Dict[str, Any]:
    """d3 spec overlaying multiple symbols' price series (rebased to 100 by default)."""
    from UI.backend.visualizer import fetch_df
    from UI.backend.d3_viz import build_compare_spec
    key = indian_api_key or os.environ.get("INDIAN_API_KEY", "")
    series: List[Dict[str, Any]] = []
    errors: Dict[str, str] = {}
    for sym in symbols:
        try:
            df = await fetch_df(symbol=sym, start_date=start_date, end_date=end_date,
                                interval=interval, indian_api_key=key)
            points = [
                {"date": str(idx), "value": float(row["Close"])}
                for idx, row in df.iterrows()
                if row.get("Close") is not None
            ]
            series.append({"symbol": sym, "points": points})
        except Exception as exc:  # noqa: BLE001
            errors[sym] = str(exc)
    return {
        "symbols": symbols,
        "normalized": normalize,
        "errors": errors,
        "spec": build_compare_spec(series, normalized=normalize),
    }


async def visualize_session(
    symbol: Optional[str] = None,
    chart: str = "equity_curve",
) -> Dict[str, Any]:
    """d3 spec for a live realtime session (equity_curve / action_heatmap / drawdown_curve)."""
    from realtime.state import get_status, get_history
    from UI.backend.d3_viz import build_d3_spec
    status = get_status(symbol)
    steps = get_history(symbol, limit=500)
    session = {
        "session_id": status.get("symbol"),
        "model_id": status.get("model_id"),
        "symbol": status.get("symbol"),
        "algorithm": "ppo_continuous_live",
        "equity_curve": [{"portfolio_value": s["equity"]} for s in steps if "equity" in s],
        "steps": steps,
    }
    return {
        "symbol": status.get("symbol"),
        "chart": chart,
        "active": status.get("active", False),
        "status": status,
        "spec": build_d3_spec(chart, session),
    }


# ── Accuracy ────────────────────────────────────────────────────────────────

async def get_prediction_accuracy(
    symbol: Optional[str] = None,
    model: Optional[str] = None,
    window_days: Optional[int] = None,
) -> Dict[str, Any]:
    from accuracy.repository import summary_by_model
    return {"window_days": window_days, "summaries": summary_by_model(model=model, symbol=symbol, window_days=window_days)}


async def list_recent_predictions(
    symbol: Optional[str] = None,
    model: Optional[str] = None,
    status: Optional[str] = None,
    limit: int = 20,
) -> Dict[str, Any]:
    from accuracy.repository import list_predictions
    return {"predictions": list_predictions(symbol=symbol, model=model, status=status, limit=limit)}


async def compare_predictions_vs_actual(
    symbol: str,
    lookback_days: int = 30,
    model: Optional[str] = None,
) -> Dict[str, Any]:
    from accuracy.repository import list_predictions
    rows = list_predictions(symbol=symbol, model=model, limit=200)
    series = []
    for r in rows:
        if r.get("actual_close") is None:
            continue
        series.append({
            "target_date": r["target_date"],
            "model": r["model"],
            "predicted_close": r.get("predicted_close"),
            "actual_close": r.get("actual_close"),
            "error_pct": r.get("error_pct"),
            "directional_correct": r.get("directional_correct"),
        })
    return {"symbol": symbol, "lookback_days": lookback_days, "series": series}


# ── Context engineering (headroom) ───────────────────────────────────────────

async def headroom_stats() -> Dict[str, Any]:
    """Token + cost accounting for the headroom compression layer."""
    from mcp_server.context_engineering import USAGE
    from mcp_server.context_engineering.compressor import headroom_available
    from mcp_server.context_engineering.middleware import cache_stats
    return {
        "headroom_available": headroom_available(),
        "usage": USAGE.snapshot(),
        "cache": cache_stats(),
    }


async def headroom_reset() -> Dict[str, Any]:
    """Reset the cost/token tracker and clear the compression cache."""
    from mcp_server.context_engineering import USAGE
    from mcp_server.context_engineering.middleware import clear_cache
    USAGE.reset()
    clear_cache()
    return {"status": "reset", "usage": USAGE.snapshot()}


# ── Training-session visualization (d3) ──────────────────────────────────────

async def list_training_sessions(
    model_id: Optional[str] = None,
    symbol: Optional[str] = None,
    limit: int = 50,
) -> Dict[str, Any]:
    from tools.rl.sessions import list_sessions
    return {"sessions": list_sessions(model_id=model_id, symbol=symbol, limit=limit)}


async def list_d3_chart_types() -> Dict[str, Any]:
    from UI.backend.d3_viz import D3_CHART_CATALOGUE
    return {"chart_types": D3_CHART_CATALOGUE}


async def visualize_training(
    session_id: str,
    chart: str = "reward_curve",
) -> Dict[str, Any]:
    from tools.rl.sessions import load_session
    from UI.backend.d3_viz import build_d3_spec, D3_CHART_DESCRIPTIONS
    session = load_session(session_id)
    if session is None:
        return {"error": f"training session {session_id!r} not found"}
    return {
        "session_id": session_id,
        "chart": chart,
        "description": D3_CHART_DESCRIPTIONS.get(chart, ""),
        "spec": build_d3_spec(chart, session),
    }


# ── Real-time intraday session (read-only) ───────────────────────────────────

async def rt_session_status(symbol: Optional[str] = None) -> Dict[str, Any]:
    from realtime.state import get_status
    return get_status(symbol)


async def rt_session_history(symbol: Optional[str] = None, limit: int = 50) -> Dict[str, Any]:
    from realtime.state import get_history
    return {"steps": get_history(symbol, limit=limit)}


# ── Dispatcher ──────────────────────────────────────────────────────────────

_IMPLS: Dict[str, Callable[..., Awaitable[Dict[str, Any]]]] = {
    "single_analyze": single_analyze,
    "batch_analyze": batch_analyze,
    "list_tools_registry": list_tools_registry,
    "list_available_intervals": list_available_intervals,
    "list_chart_types": list_chart_types,
    "rl_train": rl_train,
    "rl_predict": rl_predict,
    "rl_simulate": rl_simulate,
    "list_rl_models": list_rl_models,
    "rl_ensemble_predict": rl_ensemble_predict,
    "visualize": visualize,
    "visualize_accuracy": visualize_accuracy,
    "visualize_predictions": visualize_predictions,
    "visualize_compare": visualize_compare,
    "visualize_session": visualize_session,
    "get_prediction_accuracy": get_prediction_accuracy,
    "list_recent_predictions": list_recent_predictions,
    "compare_predictions_vs_actual": compare_predictions_vs_actual,
    # context engineering
    "headroom_stats": headroom_stats,
    "headroom_reset": headroom_reset,
    # d3 training visualization
    "list_training_sessions": list_training_sessions,
    "list_d3_chart_types": list_d3_chart_types,
    "visualize_training": visualize_training,
    # real-time intraday session (read-only)
    "rt_session_status": rt_session_status,
    "rt_session_history": rt_session_history,
}


# Wrapped impls (headroom compression + battle-tested-patterns middleware).
# Built lazily so importing `tools` never hard-depends on the CE layer.
_WRAPPED: Dict[str, Callable[..., Awaitable[Dict[str, Any]]]] = {}


def get_wrapped(name: str) -> Callable[..., Awaitable[Dict[str, Any]]]:
    """Return the context-engineered wrapper for a tool, building it on first use.

    Returns the structured ``{_headroom, data}`` envelope — used by the REST layer,
    internal callers and tests.
    """
    fn = _WRAPPED.get(name)
    if fn is None:
        from mcp_server.context_engineering import wrap_tool
        fn = wrap_tool(name, _IMPLS[name])
        _WRAPPED[name] = fn
    return fn


# MCP-facing wrappers return a compact string (single content block, no
# structuredContent duplication) so headroom's compression actually reaches Claude.
_MCP_WRAPPED: Dict[str, Callable[..., Awaitable[str]]] = {}


def get_mcp_wrapped(name: str) -> Callable[..., Awaitable[str]]:
    """Return the MCP-facing (string-returning) wrapper for a tool."""
    fn = _MCP_WRAPPED.get(name)
    if fn is None:
        from mcp_server.context_engineering import wrap_tool_for_mcp
        fn = wrap_tool_for_mcp(name, _IMPLS[name])
        _MCP_WRAPPED[name] = fn
    return fn


def list_tool_names() -> List[str]:
    return list(_IMPLS.keys())


async def dispatch_async(name: str, args: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Look up and invoke a tool by name (through the headroom wrapper layer).

    Returns the compression envelope ``{"_headroom": {...}, "data": ...}``.
    """
    if name not in _IMPLS:
        raise KeyError(f"unknown tool: {name!r}. Known: {sorted(_IMPLS)}")
    if name not in BUCK_TOOLS_BY_NAME:
        # Registry/impl mismatch — surface clearly.
        raise RuntimeError(f"tool {name!r} has an implementation but no registry entry")

    start = time.perf_counter()
    try:
        result = await get_wrapped(name)(**(args or {}))
        _record_call(name, ok=True, latency_ms=(time.perf_counter() - start) * 1000)
        return result
    except Exception as exc:  # noqa: BLE001
        _record_call(name, ok=False, latency_ms=(time.perf_counter() - start) * 1000, error=str(exc))
        raise


async def dispatch_raw_async(name: str, args: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Invoke a tool *without* the compression wrapper (raw dict result).

    Used by internal callers (e.g. REST routes) that want the original payload.
    """
    if name not in _IMPLS:
        raise KeyError(f"unknown tool: {name!r}. Known: {sorted(_IMPLS)}")
    return await _IMPLS[name](**(args or {}))


def dispatch(name: str, args: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Synchronous wrapper for callers without a running loop (rare)."""
    return asyncio.run(dispatch_async(name, args))
