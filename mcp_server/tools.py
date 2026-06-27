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
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
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


# ── Buck web-app client ──────────────────────────────────────────────────────
# Realtime tools talk to the RUNNING FastAPI web app (not in-process state) so a
# session Claude drives is the same one shown in the UI — they're separate
# processes. The base URL comes from BUCK_API_URL (default http://localhost:8000).

def _api_base() -> str:
    base = os.environ.get("BUCK_API_URL")
    if not base:
        try:
            from agent_scripts.config import SETTINGS
            base = getattr(SETTINGS, "buck_api_url", "http://localhost:8000")
        except Exception:
            base = "http://localhost:8000"
    return base.rstrip("/")


def _ui_base() -> str:
    base = os.environ.get("BUCK_UI_URL")
    if not base:
        try:
            from agent_scripts.config import SETTINGS
            base = getattr(SETTINGS, "buck_ui_url", "http://localhost:5173")
        except Exception:
            base = "http://localhost:5173"
    return base.rstrip("/")


class BuckAppUnavailable(RuntimeError):
    """Raised when the Buck web app can't be reached — surfaced as a clear hint."""


async def _api_request(method: str, path: str, *, params: Optional[Dict[str, Any]] = None,
                       json_body: Optional[Dict[str, Any]] = None, timeout: float = 15.0) -> Any:
    """Call the running Buck web app. Raises BuckAppUnavailable if it's not up.

    Uses a short (2s) CONNECT timeout so an absent app fails fast, with the longer
    `timeout` applying only to reading the response of a connected request.
    """
    import requests

    url = f"{_api_base()}{path}"

    def _do() -> Any:
        resp = requests.request(method, url, params=params, json=json_body, timeout=(2.0, timeout))
        if resp.status_code >= 400:
            try:
                detail = resp.json().get("detail", resp.text)
            except Exception:
                detail = resp.text
            raise RuntimeError(f"{resp.status_code}: {detail}")
        return resp.json()

    try:
        return await asyncio.to_thread(_do)
    except requests.exceptions.RequestException as exc:
        raise BuckAppUnavailable(
            f"Could not reach the Buck web app at {_api_base()}. Start it with "
            f"`python main.py` or the start_buck_app tool (set BUCK_API_URL if it runs "
            f"elsewhere). [{type(exc).__name__}]"
        ) from exc


# ── Web-app lifecycle (auto-start) ────────────────────────────────────────────
# The MCP server runs from the repo root, so it can launch the web app itself
# instead of forcing the user to run `python main.py` in a terminal first.

_REPO_ROOT = Path(__file__).resolve().parents[1]
_APP_PROC = None          # the subprocess we launched (if any)
_APP_LOCK = threading.Lock()


def _autostart_enabled() -> bool:
    val = os.environ.get("BUCK_AUTOSTART")
    if val is not None:
        return val.strip().lower() not in ("0", "false", "no", "off")
    try:
        from agent_scripts.config import SETTINGS
        return bool(getattr(SETTINGS, "buck_autostart", True))
    except Exception:
        return True


def _app_healthy(timeout: float = 1.5) -> bool:
    import requests
    try:
        r = requests.get(f"{_api_base()}/health", timeout=(1.5, timeout))
        return r.status_code == 200
    except Exception:
        return False


def _launch_app(full_ui: bool) -> None:
    """Spawn `python main.py` from the repo root, fully detached from the MCP
    server's stdio (critical: this server speaks MCP over stdout/stdin)."""
    global _APP_PROC
    import subprocess
    cmd = [sys.executable, "main.py", "--no-reload"]
    if not full_ui:
        cmd.append("--backend-only")
    _APP_PROC = subprocess.Popen(
        cmd,
        cwd=str(_REPO_ROOT),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,  # own process group → survives, won't catch our signals
    )


def ensure_app_running(full_ui: bool = False, wait_seconds: float = 25.0) -> Dict[str, Any]:
    """Ensure the Buck web app is reachable, auto-starting it if needed.

    Returns {"healthy": bool, "started": bool, "api": url, "ui": url}. Raises
    BuckAppUnavailable if it can't be brought up (and autostart is on)."""
    import time as _time

    if _app_healthy():
        return {"healthy": True, "started": False, "api": _api_base(), "ui": _ui_base()}

    if not _autostart_enabled():
        raise BuckAppUnavailable(
            f"The Buck web app isn't running at {_api_base()} and autostart is disabled "
            f"(BUCK_AUTOSTART=0). Start it with `python main.py`."
        )

    with _APP_LOCK:
        # Re-check inside the lock — another concurrent call may have started it.
        if not _app_healthy():
            global _APP_PROC
            already = _APP_PROC is not None and _APP_PROC.poll() is None
            if not already:
                _launch_app(full_ui=full_ui)
        deadline = _time.time() + wait_seconds
        while _time.time() < deadline:
            if _app_healthy():
                return {"healthy": True, "started": True, "api": _api_base(), "ui": _ui_base()}
            _time.sleep(1.0)

    raise BuckAppUnavailable(
        f"Started the Buck web app but it didn't become healthy at {_api_base()} within "
        f"{wait_seconds:.0f}s. Check `python main.py` output / dependencies."
    )


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
    """d3 spec for a live realtime session (equity_curve / action_heatmap / drawdown_curve).

    Reads the session from the running web app so it matches what the UI shows;
    falls back to in-process state if the app isn't reachable.
    """
    from UI.backend.d3_viz import build_d3_spec
    try:
        status = await rt_session_status(symbol)
        steps = (await rt_session_history(symbol, limit=500)).get("steps", [])
    except BuckAppUnavailable:
        from realtime.state import get_status, get_history
        status, steps = get_status(symbol), get_history(symbol, limit=500)
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
    """Live status of a realtime session running in the Buck web app."""
    params = {"symbol": symbol} if symbol else None
    try:
        return await _api_request("GET", "/rt/status", params=params)
    except BuckAppUnavailable:
        # Fallback: read in-process state (e.g. when mounted in the API process).
        from realtime.state import get_status
        return get_status(symbol)


async def rt_session_history(symbol: Optional[str] = None, limit: int = 50) -> Dict[str, Any]:
    """Recent per-step records from a realtime session in the Buck web app."""
    params: Dict[str, Any] = {"limit": limit}
    if symbol:
        params["symbol"] = symbol
    try:
        return await _api_request("GET", "/rt/history", params=params)
    except BuckAppUnavailable:
        from realtime.state import get_history
        return {"steps": get_history(symbol, limit=limit)}


async def rt_start_session(
    symbol: str,
    model_id: str,
    replay: bool = True,
    replay_start: Optional[str] = None,
    replay_end: Optional[str] = None,
    interval: str = "1d",
    capital: float = 100000.0,
    max_steps: int = 2000,
    speed: float = 60.0,
    open_ui: bool = True,
) -> Dict[str, Any]:
    """Start a realtime simulation in the Buck web app so it streams live in the UI.

    Defaults to replay mode (works any time). Requires the web app to be running
    and a trained `ppo_continuous` model. `speed` is the replay fast-forward factor
    (per-bar delay = bar_period/speed; 1=real-time, higher=faster, >=1000=instant).
    When `open_ui` is true, also opens the browser to the Realtime tab.
    If the web app isn't running it is auto-started (unless BUCK_AUTOSTART=0).
    """
    # Make sure the web app is up first (auto-start it if needed). Start the full
    # stack when we're going to open the UI so there's a frontend to watch.
    await asyncio.to_thread(ensure_app_running, open_ui)

    body = {
        "symbol": symbol,
        "model_id": model_id,
        "interval": interval,
        "replay": replay,
        "replay_start": replay_start,
        "replay_end": replay_end,
        "capital": capital,
        "max_steps": max_steps,
        "speed": speed,
        "indian_api_key": os.environ.get("INDIAN_API_KEY", ""),
    }
    status = await _api_request("POST", "/rt/start", json_body=body)
    result: Dict[str, Any] = {"started": True, "status": status}
    if open_ui:
        result["ui"] = await open_buck_ui(tab="realtime", symbol=symbol)
    return result


async def rt_stop_session(symbol: str) -> Dict[str, Any]:
    """Stop a running realtime session in the Buck web app."""
    status = await _api_request("POST", "/rt/stop", json_body={"symbol": symbol})
    return {"stopped": True, "status": status}


async def open_buck_ui(
    tab: str = "realtime",
    symbol: Optional[str] = None,
    autostart: bool = False,
) -> Dict[str, Any]:
    """Open the Buck web UI in the user's browser, deep-linked to a tab/symbol.

    Use this to let the user *watch* what Buck is doing (e.g. a replay session)
    rather than only reading results in chat.
    """
    from urllib.parse import urlencode
    import webbrowser

    # Make sure the full stack (incl. frontend) is up so the page actually loads.
    app: Dict[str, Any] = {}
    try:
        app = await asyncio.to_thread(ensure_app_running, True)
    except BuckAppUnavailable as exc:
        app = {"healthy": False, "error": str(exc)}

    params: Dict[str, Any] = {"tab": tab}
    if symbol:
        params["symbol"] = symbol
    if autostart:
        params["autostart"] = "1"
    url = f"{_ui_base()}/?{urlencode(params)}"
    opened = False
    try:
        opened = await asyncio.to_thread(webbrowser.open, url)
    except Exception:  # noqa: BLE001
        opened = False
    return {
        "url": url,
        "opened": bool(opened),
        "app": app,
        "note": "Open this URL in your browser to watch." if not opened else "Opened in your browser.",
    }


async def start_buck_app(frontend: bool = True, wait_seconds: float = 30.0) -> Dict[str, Any]:
    """Ensure the Buck web app (`python main.py`) is running, launching it from the
    repo if needed. Most realtime tools auto-start it, but call this to bring it up
    explicitly (e.g. before a batch of sessions). `frontend=True` also starts the UI."""
    try:
        return await asyncio.to_thread(ensure_app_running, frontend, wait_seconds)
    except BuckAppUnavailable as exc:
        return {"healthy": False, "started": False, "error": str(exc)}


async def buck_app_status() -> Dict[str, Any]:
    """Report whether the Buck web app is reachable, and the API/UI URLs."""
    healthy = await asyncio.to_thread(_app_healthy)
    return {"healthy": healthy, "api": _api_base(), "ui": _ui_base(),
            "autostart": _autostart_enabled()}


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
    # real-time intraday session (read + run-control via the web app)
    "rt_session_status": rt_session_status,
    "rt_session_history": rt_session_history,
    "rt_start_session": rt_start_session,
    "rt_stop_session": rt_stop_session,
    "open_buck_ui": open_buck_ui,
    "start_buck_app": start_buck_app,
    "buck_app_status": buck_app_status,
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
