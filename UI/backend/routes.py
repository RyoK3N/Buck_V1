"""
UI.backend.routes
──────────────────
FastAPI route handlers for all Buck endpoints.
"""

from __future__ import annotations
import asyncio
import logging
import os
import traceback
from typing import Any, Dict

from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect

from ._env_scope import temporary_env
from .models import (
    AnalyzeRequest,
    BatchRequest,
    VisualizeRequest,
    HealthResponse,
    IntervalsResponse,
    ToolsResponse,
    ConfigResponse,
    ChartTypesResponse,
    ChartTypeInfo,
    VisualizeResponse,
    ToolInfo,
    ToolCategory,
    ToolsRegistryResponse,
    RLTrainRequest,
    RLPredictRequest,
    RLSimulateRequest,
    RLEnsembleRequest,
    RLModelsResponse,
    PredictionRow,
    PredictionsResponse,
    ModelSummary,
    AccuracySummaryResponse,
    AccuracyPoint,
    AccuracyTimeseriesResponse,
    ToolContributionRow,
    ToolContributionResponse,
    LiveAccuracyEntry,
    LiveAccuracyResponse,
    EvaluateNowRequest,
    EvaluateNowResponse,
    MCPToolInfo,
    MCPToolsResponse,
    MCPStatusResponse,
    MCPInvokeRequest,
    MCPInvokeResponse,
    ClaudePredictRequest,
    ClaudeChatRequest,
    ClaudeChatResponse,
    ToolUseTrace,
    RTStartRequest,
    RTStopRequest,
    RTStatusResponse,
    RTHistoryResponse,
    RTSessionsResponse,
)  # noqa: F401

logger = logging.getLogger(__name__)
router = APIRouter()

VALID_INTERVALS = ["1m", "2m", "5m", "15m", "30m", "60m", "90m", "1h",
                   "1d", "5d", "1wk", "1mo", "3mo"]


def _internal_error(exc: Exception, context: str) -> HTTPException:
    """Log the full exception server-side; return a client-safe 500.

    Returning str(exc) directly to callers leaks stack traces, file paths,
    and library internals. Callers only need to know something failed and
    where to look (server logs).
    """
    logger.error("%s failed: %s\n%s", context, exc, traceback.format_exc())
    return HTTPException(status_code=500, detail=f"{context} failed — see server logs for details.")


def _resolve_openai_key(requested: str | None) -> str:
    """Use the request's key if given, else the server's own .env key.

    Keeps the browser from ever having to round-trip a server-configured
    secret just to fill out a form field.
    """
    from agent_scripts.config import SETTINGS
    key = requested or SETTINGS.openai_api_key
    if not key or key == "__placeholder__":
        raise HTTPException(
            status_code=400,
            detail="No OPENAI_API_KEY configured on the server and none was provided in the request.",
        )
    return key


def _make_buck(req: AnalyzeRequest | BatchRequest):
    """Instantiate Buck with per-request API keys."""
    from agent_scripts.buck import BuckFactory
    return BuckFactory.create_production_agent(
        openai_api_key=_resolve_openai_key(req.openai_api_key),
        indian_api_key=req.indian_api_key or "",
        model=req.model or "gpt-4o",
        base_url=req.base_url or None,
        selected_tools=req.selected_tools,
    )


# ── Health ────────────────────────────────────────────────────────────────────

@router.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse()


# ── Config ────────────────────────────────────────────────────────────────────

@router.get("/config", response_model=ConfigResponse)
async def get_config() -> ConfigResponse:
    """Return server config loaded from .env (used by the UI to pre-fill forms).

    Never echoes secret values back to the client — only whether each key is
    configured. The UI only needs this to decide whether to show "using
    server key" vs. prompting the user for one.
    """
    from agent_scripts.config import SETTINGS
    return ConfigResponse(
        openai_api_key_configured=SETTINGS.openai_api_key not in ("", "__placeholder__"),
        openai_base_url=SETTINGS.openai_base_url,
        chat_model=SETTINGS.chat_model,
        indian_api_key_configured=bool(SETTINGS.indian_api_key),
    )


# ── Meta ──────────────────────────────────────────────────────────────────────

@router.get("/intervals", response_model=IntervalsResponse)
async def intervals() -> IntervalsResponse:
    return IntervalsResponse(intervals=VALID_INTERVALS)


@router.get("/tools", response_model=ToolsResponse)
async def tools() -> ToolsResponse:
    from agent_scripts.tools import ToolFactory
    return ToolsResponse(tools=ToolFactory.get_available_tools())


@router.get("/tools-registry", response_model=ToolsRegistryResponse)
async def tools_registry() -> ToolsRegistryResponse:
    """Return all dynamically discovered tool categories."""
    from agent_scripts.tools import ToolFactory

    raw = ToolFactory.get_registry()  # {"categories": [...]}
    categories = [
        ToolCategory(
            id=cat["id"],
            name=cat["name"],
            description=cat.get("description", ""),
            tools=[ToolInfo(**t) for t in cat["tools"]],
        )
        for cat in raw["categories"]
    ]
    return ToolsRegistryResponse(categories=categories)


# ── Analysis ──────────────────────────────────────────────────────────────────

@router.post("/analyze")
async def analyze(req: AnalyzeRequest) -> Dict[str, Any]:
    # Keys are passed explicitly to BuckFactory below — no need to touch
    # process-wide os.environ for this path.
    logger.info("POST /analyze  symbol=%s  selected_tools=%s", req.symbol, req.selected_tools)
    try:
        buck = _make_buck(req)
        result = await buck.analyze_and_predict(
            symbol=req.symbol,
            start_date=req.start_date,
            end_date=req.end_date,
            interval=req.interval,
            save_results=False,
        )
        return result
    except HTTPException:
        raise
    except Exception as exc:
        raise _internal_error(exc, "Analysis")


@router.post("/batch")
async def batch(req: BatchRequest) -> Dict[str, Any]:
    try:
        buck = _make_buck(req)
        result = await buck.batch_analyze(
            symbols=req.symbols,
            start_date=req.start_date,
            end_date=req.end_date,
            interval=req.interval,
            max_concurrent=req.max_concurrent or 3,
        )
        return result
    except HTTPException:
        raise
    except Exception as exc:
        raise _internal_error(exc, "Batch analysis")


# ── Visualizer ────────────────────────────────────────────────────────────────

@router.get("/chart-types", response_model=ChartTypesResponse)
async def chart_types() -> ChartTypesResponse:
    from .visualizer import CHART_CATALOGUE
    return ChartTypesResponse(
        chart_types=[ChartTypeInfo(**c) for c in CHART_CATALOGUE]
    )


@router.post("/visualize", response_model=VisualizeResponse)
async def visualize(req: VisualizeRequest) -> VisualizeResponse:
    from .visualizer import fetch_df, build_chart, CHART_DESCRIPTIONS

    try:
        df = await fetch_df(
            symbol=req.symbol,
            start_date=req.start_date,
            end_date=req.end_date,
            interval=req.interval,
            indian_api_key=req.indian_api_key or "",
        )
        chart_dict = build_chart(req.chart_type, df, req.symbol)
        return VisualizeResponse(
            chart=chart_dict,
            chart_type=req.chart_type,
            symbol=req.symbol,
            description=CHART_DESCRIPTIONS.get(req.chart_type, ""),
        )
    except Exception as exc:
        raise _internal_error(exc, "Chart generation")


# ── d3 training-session observability ────────────────────────────────────────

@router.get("/viz/d3-chart-types")
async def viz_d3_chart_types() -> Dict[str, Any]:
    from .d3_viz import D3_CHART_CATALOGUE
    return {"chart_types": D3_CHART_CATALOGUE}


@router.get("/viz/training-sessions")
async def viz_training_sessions(
    model_id: str | None = None,
    symbol: str | None = None,
    limit: int = 50,
) -> Dict[str, Any]:
    from tools.rl.sessions import list_sessions
    return {"sessions": list_sessions(model_id=model_id, symbol=symbol, limit=limit)}


@router.get("/viz/training/{session_id}/{chart}")
async def viz_training_chart(session_id: str, chart: str) -> Dict[str, Any]:
    from tools.rl.sessions import load_session
    from .d3_viz import build_d3_spec, D3_CHART_DESCRIPTIONS
    session = load_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail=f"training session {session_id!r} not found")
    return {
        "session_id": session_id,
        "chart": chart,
        "description": D3_CHART_DESCRIPTIONS.get(chart, ""),
        "spec": build_d3_spec(chart, session),
    }


# ── RL / Reinforcement Learning ──────────────────────────────────────────────

async def _rl_train_ppo_continuous(req) -> Dict[str, Any]:
    """Continuous-action PPO training loop using TradingEnvironment.

    Each "episode" walks the env once end-to-end, collecting one or more
    rollouts of `rollout_steps` and running PPO updates between them.
    """
    try:
        from tools.rl.rl_tool import fetch_historical_data
        from tools.rl.env import TradingEnvironment
        from tools.rl.ppo_continuous import PPOContinuousAgent

        df = fetch_historical_data(req.symbol, req.start_date, req.end_date, req.interval)
        if df is None or df.empty:
            raise HTTPException(status_code=400, detail=f"No data for {req.symbol} in given range")
        if len(df) < 60:
            raise HTTPException(status_code=400,
                                detail=f"PPO-continuous needs at least 60 bars; got {len(df)}")

        agent = PPOContinuousAgent(
            hidden_dim=req.hidden_dim,
            lr=req.learning_rate,
            rollout_steps=min(256, max(32, len(df) // 4)),
        )
        best_return = -float("inf")
        episode_rewards: List[Dict[str, Any]] = []
        last_summary: Dict[str, Any] = {}

        for ep in range(req.episodes):
            env = TradingEnvironment(df, initial_capital=req.initial_capital)
            env.reset()
            agent.reset_window()
            total_reward = 0.0
            update_stats: Dict[str, float] = {}
            n_updates = 0
            while True:
                rollout = agent.collect_rollout(env)
                if rollout["n_steps"] == 0:
                    break
                stats = agent.update(rollout)
                total_reward += float(rollout["rewards"].sum())
                update_stats = stats
                n_updates += 1
                if rollout["episode_done"]:
                    break
            env.close_position()
            summary = env.summary()
            last_summary = summary
            episode_rewards.append({
                "episode": ep + 1,
                "total_reward": round(total_reward, 4),
                "portfolio_value": summary["final_portfolio_value"],
                "return_pct": summary["total_return_pct"],
                "sharpe": summary["annualized_sharpe"],
                "max_drawdown_pct": summary["max_drawdown_pct"],
                "trades": summary["total_trades"],
                "n_updates": n_updates,
                "policy_loss": round(update_stats.get("policy_loss", 0.0), 4),
                "value_loss": round(update_stats.get("value_loss", 0.0), 4),
                "entropy": round(update_stats.get("entropy", 0.0), 4),
            })
            if summary["total_return_pct"] > best_return:
                best_return = summary["total_return_pct"]
                agent.save(f"{req.model_id}_best")
            logger.info("PPO-cont ep %d/%d: ret=%.2f%% sharpe=%.2f dd=%.2f%% trades=%d",
                        ep + 1, req.episodes, summary["total_return_pct"],
                        summary["annualized_sharpe"], summary["max_drawdown_pct"],
                        summary["total_trades"])

        agent.save(req.model_id)
        result = {
            "model_id": req.model_id,
            "algorithm": "ppo_continuous",
            "symbol": req.symbol,
            "episodes": req.episodes,
            "total_steps": len(df),
            "episode_rewards": episode_rewards,
            "equity_curve": env.equity_curve[-200:] if getattr(env, "equity_curve", None) else [],
            "final_summary": last_summary,
            "best_return_pct": round(best_return, 4),
            "status": "trained",
        }
        from tools.rl.sessions import save_session
        result["session_id"] = save_session(result, request=req)
        return result
    except HTTPException:
        raise
    except Exception as exc:
        raise _internal_error(exc, "PPO-continuous training")


async def _rl_predict_ppo_continuous(req) -> Dict[str, Any]:
    """Continuous-action PPO inference: run env in eval mode, record signals."""
    try:
        from tools.rl.rl_tool import fetch_historical_data
        from tools.rl.env import TradingEnvironment
        from tools.rl.ppo_continuous import PPOContinuousAgent

        df = fetch_historical_data(req.symbol, req.start_date, req.end_date, req.interval)
        if df is None or df.empty:
            raise HTTPException(status_code=400, detail=f"No data for {req.symbol}")

        agent = PPOContinuousAgent()
        if not agent.load(req.model_id):
            raise HTTPException(status_code=404,
                                detail=f"Model {req.model_id} not found or not a ppo_continuous checkpoint")

        env = TradingEnvironment(df, initial_capital=req.initial_capital)
        state = env.reset()
        agent.reset_window()
        signals: List[Dict[str, Any]] = []
        while True:
            action, info = agent.act(state, eval_mode=True)
            step_idx = env.idx
            price = float(env.close[step_idx])
            if env.idx >= env.n - 1:
                signals.append({
                    "step": step_idx,
                    "target_position": round(action, 4),
                    "signal": "BUY" if action >= 0.66 else ("SELL" if action <= 0.33 else "HOLD"),
                    "mu": round(info["mu"], 4),
                    "log_std": round(info["log_std"], 4),
                    "value": round(info["value"], 4),
                    "price": round(price, 2),
                })
                break
            state, reward, done, step_info = env.step(action)
            signals.append({
                "step": step_idx,
                "target_position": round(action, 4),
                "realized_position": round(step_info.realized_position, 4),
                "signal": "BUY" if action >= 0.66 else ("SELL" if action <= 0.33 else "HOLD"),
                "mu": round(info["mu"], 4),
                "log_std": round(info["log_std"], 4),
                "value": round(info["value"], 4),
                "price": round(price, 2),
                "portfolio_value": round(step_info.portfolio_value, 2),
                "reward": round(reward, 4),
                "forced_exit": step_info.forced_exit,
            })
            if done:
                break
        env.close_position()
        summary = env.summary()
        return {
            "model_id": req.model_id,
            "algorithm": "ppo_continuous",
            "symbol": req.symbol,
            "total_signals": len(signals),
            "signals": signals[-150:],
            "equity_curve": env.equity_curve[-200:],
            "summary": summary,
            "status": "predicted",
        }
    except HTTPException:
        raise
    except Exception as exc:
        raise _internal_error(exc, "PPO-continuous prediction")


@router.post("/rl/train")
async def rl_train(req: RLTrainRequest) -> Dict[str, Any]:
    # tools.rl.rl_tool.fetch_historical_data reads INDIAN_API_KEY from
    # os.environ rather than taking it as a parameter — scope the mutation to
    # this request instead of leaving it set process-wide.
    async with temporary_env(INDIAN_API_KEY=req.indian_api_key):
        return await _rl_train_impl(req)


async def _rl_train_impl(req: RLTrainRequest) -> Dict[str, Any]:
    logger.info("POST /rl/train  symbol=%s  model_id=%s  algorithm=%s  episodes=%d",
                req.symbol, req.model_id, req.algorithm, req.episodes)
    if req.algorithm == "ppo_continuous":
        return await _rl_train_ppo_continuous(req)
    try:
        from tools.rl.dqn_agent import create_agent, extract_state
        from tools.rl.rl_tool import fetch_historical_data
        from tools.rl.wallet import Wallet
        df = fetch_historical_data(req.symbol, req.start_date, req.end_date, req.interval)
        if df is None or df.empty:
            raise HTTPException(status_code=400, detail=f"No data for {req.symbol} in given range")
        required_cols = ['Close', 'Open', 'High', 'Low', 'Volume']
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            raise HTTPException(status_code=400, detail=f"Missing columns: {missing}. Got: {list(df.columns)}")
        close = df['Close'].values
        total_steps = len(close)
        if total_steps < 10:
            raise HTTPException(status_code=400, detail=f"Need at least 10 data points, got {total_steps}")
        agent = create_agent(req.algorithm, input_dim=12, hidden_dim=req.hidden_dim, output_dim=3, lr=req.learning_rate)
        best_reward = -float('inf')
        episode_rewards = []
        for ep in range(req.episodes):
            wallet = Wallet(initial_capital=req.initial_capital)
            total_reward = 0.0
            prev_pv = wallet.get_portfolio_value(float(close[0]))
            for step in range(1, total_steps):
                position = 1 if wallet.holdings > 1e-6 else 0
                cash_ratio = wallet.cash / (prev_pv + 1e-10)
                state = extract_state(df, step, position, cash_ratio)
                action = agent.act(state)
                price = float(close[step])
                if action == 1 and position == 0:
                    wallet.buy(price)
                elif action == 2 and position == 1:
                    wallet.sell(price)
                current_pv = wallet.get_portfolio_value(price)
                reward = current_pv - prev_pv
                next_pos = 1 if wallet.holdings > 1e-6 else 0
                next_cr = wallet.cash / (current_pv + 1e-10)
                next_state = extract_state(df, min(step + 1, total_steps - 1), next_pos, next_cr)
                done = (step == total_steps - 1)
                prev_pv = current_pv
                if hasattr(agent, 'remember'):
                    agent.remember(state, action, reward, next_state, done)
                if agent.algorithm == 'a2c' and hasattr(agent, 'train_step'):
                    agent.train_step(reward, next_state, done)
                if agent.algorithm == 'dqn' and hasattr(agent, 'replay'):
                    agent.replay()
                total_reward += reward
            if agent.algorithm == 'dqn' and hasattr(agent, 'end_episode'):
                agent.end_episode()
            if agent.algorithm == 'ppo' and hasattr(agent, 'end_episode'):
                agent.end_episode()
            if agent.algorithm == 'a2c' and hasattr(agent, 'end_episode'):
                agent.end_episode()
            final_pv = wallet.get_portfolio_value(float(close[-1]))
            pnl_pct = ((final_pv - req.initial_capital) / req.initial_capital) * 100
            episode_rewards.append({
                'episode': ep + 1,
                'total_reward': round(total_reward, 4),
                'portfolio_value': round(final_pv, 2),
                'return_pct': round(pnl_pct, 2),
                'trades': len(wallet.trades),
            })
            if total_reward > best_reward:
                best_reward = total_reward
                agent.save(f"{req.model_id}_best")
            logger.info("Episode %d/%d: reward=%.2f return=%.2f%% algorithm=%s",
                        ep + 1, req.episodes, total_reward, pnl_pct, req.algorithm)
        agent.save(req.model_id)
        wallet.close_position(float(close[-1]))
        summary = wallet.get_summary(float(close[-1]))
        result = {
            'model_id': req.model_id,
            'algorithm': req.algorithm,
            'symbol': req.symbol,
            'episodes': req.episodes,
            'total_steps': total_steps,
            'episode_rewards': episode_rewards,
            'final_summary': summary,
            'equity_curve': wallet.equity_curve[-200:] if wallet.equity_curve else [],
            'best_reward': round(best_reward, 4),
            'status': 'trained',
        }
        from tools.rl.sessions import save_session
        result["session_id"] = save_session(result, request=req)
        return result
    except HTTPException:
        raise
    except Exception as exc:
        raise _internal_error(exc, "RL training")


@router.post("/rl/predict")
async def rl_predict(req: RLPredictRequest) -> Dict[str, Any]:
    async with temporary_env(INDIAN_API_KEY=req.indian_api_key):
        return await _rl_predict_impl(req)


async def _rl_predict_impl(req: RLPredictRequest) -> Dict[str, Any]:
    logger.info("POST /rl/predict  symbol=%s  model_id=%s", req.symbol, req.model_id)
    # Auto-detect PPO-continuous checkpoints and use the continuous code path.
    # NOTE: only swallow errors from the *detection* itself; once we've decided
    # it's a PPO-continuous checkpoint, surface any predict-time error rather
    # than silently falling through to the legacy DQN path (which would mask
    # the real failure as a state-shape mismatch).
    _is_ppo_continuous = False
    try:
        from tools.rl.ppo_continuous import WEIGHTS_DIR as _PPO_WEIGHTS_DIR
        import torch as _torch
        _ckpt_path = _PPO_WEIGHTS_DIR / f"{req.model_id}.pt"
        if _ckpt_path.exists():
            _ckpt = _torch.load(_ckpt_path, map_location="cpu", weights_only=True)
            _is_ppo_continuous = _ckpt.get("algorithm") == "ppo_continuous"
    except Exception as exc:  # noqa: BLE001
        logger.warning("PPO-continuous checkpoint detect failed for %s: %s", req.model_id, exc)
    if _is_ppo_continuous:
        return await _rl_predict_ppo_continuous(req)
    try:
        from tools.rl.dqn_agent import load_agent, extract_state
        from tools.rl.rl_tool import fetch_historical_data
        from tools.rl.wallet import Wallet
        df = fetch_historical_data(req.symbol, req.start_date, req.end_date, req.interval)
        if df is None or df.empty:
            raise HTTPException(status_code=400, detail=f"No data for {req.symbol}")
        agent = load_agent(req.model_id)
        if agent is None:
            raise HTTPException(status_code=404, detail=f"Model {req.model_id} not found")
        wallet = Wallet(initial_capital=req.initial_capital)
        close = df['Close'].values
        signals = []
        for step in range(len(close)):
            position = 1 if wallet.holdings > 1e-6 else 0
            cash_ratio = wallet.cash / (wallet.get_portfolio_value(close[step]) + 1e-10)
            state = extract_state(df, step, position, cash_ratio)
            action = agent.act(state, eval_mode=True)
            price = close[step]
            if action == 1 and position == 0:
                wallet.buy(price)
            elif action == 2 and position == 1:
                wallet.sell(price)
            signals.append({
                'step': step,
                'action': ['HOLD', 'BUY', 'SELL'][action],
                'price': round(float(price), 2),
                'portfolio_value': round(float(wallet.get_portfolio_value(price)), 2),
            })
        wallet.close_position(close[-1])
        summary = wallet.get_summary(close[-1])
        return {
            'model_id': req.model_id,
            'symbol': req.symbol,
            'total_signals': len(signals),
            'signals': signals[-100:],
            'equity_curve': wallet.equity_curve[-200:] if wallet.equity_curve else [],
            'summary': summary,
            'status': 'predicted',
        }
    except HTTPException:
        raise
    except Exception as exc:
        raise _internal_error(exc, "RL prediction")


@router.get("/rl/models", response_model=RLModelsResponse)
async def rl_models() -> RLModelsResponse:
    from tools.rl.dqn_agent import list_models
    return RLModelsResponse(models=list_models())


@router.post("/rl/ensemble-predict")
async def rl_ensemble_predict(req: RLEnsembleRequest) -> Dict[str, Any]:
    """Multi-timeframe ensemble inference. See tools.rl.ensemble for details."""
    async with temporary_env(INDIAN_API_KEY=req.indian_api_key):
        return await _rl_ensemble_predict_impl(req)


async def _rl_ensemble_predict_impl(req: RLEnsembleRequest) -> Dict[str, Any]:
    logger.info("POST /rl/ensemble-predict  symbol=%s  n_models=%d", req.symbol, len(req.models))
    try:
        from tools.rl.ensemble import ensemble_predict
        result = ensemble_predict(
            symbol=req.symbol,
            models=[m.model_dump() for m in req.models],
            start_date=req.start_date,
            end_date=req.end_date,
            fallback_interval=req.fallback_interval,
        )
        return result
    except Exception as exc:
        raise _internal_error(exc, "RL ensemble prediction")


@router.delete("/rl/models/{model_id}")
async def rl_delete_model(model_id: str) -> Dict[str, Any]:
    from tools.rl.dqn_agent import WEIGHTS_DIR
    path = WEIGHTS_DIR / f"{model_id}.pt"
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Model {model_id} not found")
    path.unlink()
    logger.info("Deleted RL model: %s", model_id)
    return {'status': 'deleted', 'model_id': model_id}


# ── Accuracy tracking ────────────────────────────────────────────────────────

@router.get("/accuracy/predictions", response_model=PredictionsResponse)
async def accuracy_predictions(
    symbol: str | None = None,
    model: str | None = None,
    status: str | None = None,
    limit: int = 100,
) -> PredictionsResponse:
    from accuracy.repository import list_predictions
    rows = list_predictions(symbol=symbol, model=model, status=status, limit=limit)
    return PredictionsResponse(predictions=[PredictionRow(**_coerce_row(r)) for r in rows])


@router.get("/accuracy/summary", response_model=AccuracySummaryResponse)
async def accuracy_summary(
    model: str | None = None,
    symbol: str | None = None,
    window_days: int | None = None,
) -> AccuracySummaryResponse:
    from accuracy.repository import summary_by_model
    rows = summary_by_model(model=model, symbol=symbol, window_days=window_days)
    return AccuracySummaryResponse(
        window_days=window_days,
        summaries=[ModelSummary(**_coerce_row(r)) for r in rows],
    )


@router.get("/accuracy/timeseries", response_model=AccuracyTimeseriesResponse)
async def accuracy_timeseries(
    model: str | None = None,
    symbol: str | None = None,
    window_days: int = 30,
) -> AccuracyTimeseriesResponse:
    from accuracy.repository import timeseries
    rows = timeseries(model=model, symbol=symbol, window_days=window_days)
    return AccuracyTimeseriesResponse(
        points=[AccuracyPoint(**_coerce_row(r)) for r in rows],
    )


@router.get("/accuracy/tool-contribution", response_model=ToolContributionResponse)
async def accuracy_tool_contribution(
    model: str = "claude",
    window_days: int = 30,
) -> ToolContributionResponse:
    from accuracy.repository import tool_contribution
    rows = tool_contribution(model=model, window_days=window_days)
    return ToolContributionResponse(
        model=model,
        rows=[ToolContributionRow(**r) for r in rows],
    )


@router.get("/accuracy/live", response_model=LiveAccuracyResponse)
async def accuracy_live() -> LiveAccuracyResponse:
    from accuracy.broadcaster import snapshot
    return LiveAccuracyResponse(entries=[LiveAccuracyEntry(**e) for e in snapshot()])


@router.post("/accuracy/evaluate-now", response_model=EvaluateNowResponse)
async def accuracy_evaluate_now(req: EvaluateNowRequest) -> EvaluateNowResponse:
    """Force a poll + reconcile right now. Bypasses the schedule."""
    from agent_scripts.config import SETTINGS
    from accuracy import broadcaster, evaluator, poller, repository
    symbols = [req.symbol] if req.symbol else repository.distinct_symbols_with_open_predictions()
    polled = len(poller.poll_symbols(symbols, exchange=SETTINGS.market_exchange, is_final=req.is_final))
    written = evaluator.reconcile_open_predictions(
        is_intraday=not req.is_final, exchange=SETTINGS.market_exchange
    )
    for event in written:
        await broadcaster.publish(event)
    return EvaluateNowResponse(polled=polled, evaluated=len(written))


# ── Claude predictor + chat ─────────────────────────────────────────────────

@router.post("/claude/predict")
async def claude_predict(req: ClaudePredictRequest) -> Dict[str, Any]:
    """Run Buck's analysis pipeline with the Claude research-agent predictor.

    Claude uses Buck's MCP tools (single_analyze, batch_analyze, rl_predict,
    visualize, accuracy introspection) to gather evidence, then emits the
    forecast. The forecast + tool-call trace are persisted to the accuracy DB
    under model='claude'.
    """
    from agent_scripts.buck import Buck, BuckFactory
    from agent_scripts.predictors import PredictorFactory
    from agent_scripts.config import SETTINGS

    anthropic_key = req.anthropic_api_key or SETTINGS.anthropic_api_key
    if not anthropic_key:
        raise HTTPException(status_code=400, detail="ANTHROPIC_API_KEY required (in request body or .env)")
    openai_key = _resolve_openai_key(req.openai_api_key)

    # Claude's tool-calling loop dispatches through mcp_server.tools, which
    # resolves keys from os.environ as a fallback when a tool call doesn't
    # carry them explicitly — scope the mutation to this request only.
    async with temporary_env(
        OPENAI_API_KEY=openai_key,
        INDIAN_API_KEY=req.indian_api_key,
        ANTHROPIC_API_KEY=anthropic_key,
    ):
        return await _claude_predict_impl(req, openai_key, anthropic_key)


async def _claude_predict_impl(req: ClaudePredictRequest, openai_key: str, anthropic_key: str) -> Dict[str, Any]:
    from agent_scripts.buck import Buck, BuckFactory
    from agent_scripts.predictors import PredictorFactory

    try:
        # Build a Buck wired with the Claude predictor (analyzer still runs to
        # produce the seed analysis the predictor receives).
        base = BuckFactory.create_production_agent(
            openai_api_key=openai_key,
            indian_api_key=req.indian_api_key or "",
            model="gpt-4o",
            base_url=req.base_url,
            selected_tools=req.selected_tools,
        )
        claude_predictor = PredictorFactory.create_claude_predictor(
            api_key=anthropic_key,
            model=req.claude_model,
            max_iterations=req.max_iterations,
        )
        buck = Buck(
            config=base.config,
            data_provider=base.data_provider,
            analyzer=base.analyzer,
            predictor=claude_predictor,
        )
        result = await buck.analyze_and_predict(
            symbol=req.symbol,
            start_date=req.start_date,
            end_date=req.end_date,
            interval=req.interval,
            save_results=False,
            model_name="claude",
        )
        # Surface the trace in the API response too (it's also persisted via telemetry).
        result.setdefault("metadata", {})["tool_trace"] = getattr(claude_predictor, "last_trace", [])
        return result
    except Exception as exc:
        raise _internal_error(exc, "Claude prediction")


@router.post("/claude/chat", response_model=ClaudeChatResponse)
async def claude_chat(req: ClaudeChatRequest) -> ClaudeChatResponse:
    """Free-form chat with Claude, who can call Buck's MCP tools."""
    from anthropic import Anthropic
    from agent_scripts.config import SETTINGS
    from agent_scripts.claude_predictor import _SYSTEM_PROMPT as _DEFAULT_SYS
    from mcp_server.registry import BUCK_TOOLS
    from mcp_server.tools import dispatch_async

    anthropic_key = req.anthropic_api_key or SETTINGS.anthropic_api_key
    if not anthropic_key:
        raise HTTPException(status_code=400, detail="ANTHROPIC_API_KEY required")

    model = req.claude_model or SETTINGS.claude_model
    max_iter = req.max_iterations or SETTINGS.claude_max_iterations
    client = Anthropic(api_key=anthropic_key)

    chat_system = (
        "You are Buck's stock-research assistant. Use the available tools to "
        "answer the user's question with concrete data. Cite the tools you "
        "called and the symbols you looked at. Be direct and quantitative.\n\n"
        + _DEFAULT_SYS.split("Workflow")[0]  # reuse the tool catalog blurb
    )
    messages = [{"role": m.role, "content": m.content} for m in req.messages]
    trace: list[dict] = []

    for iteration in range(max_iter):
        try:
            response = await asyncio.to_thread(
                client.messages.create,
                model=model,
                max_tokens=4096,
                system=chat_system,
                tools=BUCK_TOOLS,
                messages=messages,
            )
        except Exception as exc:
            raise _internal_error(exc, "Anthropic call")

        messages.append({"role": "assistant", "content": [b.model_dump() for b in response.content]})

        if response.stop_reason != "tool_use":
            text = "".join(b.text for b in response.content if getattr(b, "type", None) == "text")
            return ClaudeChatResponse(text=text, trace=[ToolUseTrace(**t) for t in trace])

        tool_results = []
        for block in response.content:
            if getattr(block, "type", None) != "tool_use":
                continue
            try:
                result = await dispatch_async(block.name, block.input or {})
                ok, err = True, None
            except Exception as exc:
                result = {"error": str(exc)}
                ok, err = False, str(exc)
            trace.append({"iter": iteration, "name": block.name, "args": block.input or {}, "ok": ok, "error": err})
            import json as _json
            tool_results.append({
                "type": "tool_result",
                "tool_use_id": block.id,
                "content": _json.dumps(result, default=str)[:8000],
                "is_error": not ok,
            })
        messages.append({"role": "user", "content": tool_results})

    return ClaudeChatResponse(
        text="(reached max_iterations without a final answer — try a more focused question)",
        trace=[ToolUseTrace(**t) for t in trace],
    )


# ── MCP server introspection (REST shadow for the UI) ───────────────────────

@router.get("/mcp/tools", response_model=MCPToolsResponse)
async def mcp_tools() -> MCPToolsResponse:
    from mcp_server.registry import BUCK_TOOLS
    from mcp_server.tools import LAST_CALL
    return MCPToolsResponse(
        tools=[
            MCPToolInfo(
                name=t["name"],
                description=t["description"],
                input_schema=t["input_schema"],
                last_call=LAST_CALL.get(t["name"]),
            )
            for t in BUCK_TOOLS
        ]
    )


@router.get("/mcp/status", response_model=MCPStatusResponse)
async def mcp_status() -> MCPStatusResponse:
    from agent_scripts.config import SETTINGS
    from mcp_server.tools import list_tool_names
    return MCPStatusResponse(
        mounted_in_api=SETTINGS.mount_mcp_in_api,
        standalone_runner="python -m mcp_server.runner --transport stdio  # for Claude Desktop",
        mount_path="/mcp" if SETTINGS.mount_mcp_in_api else "",
        transport="sse" if SETTINGS.mount_mcp_in_api else "stdio",
        tool_count=len(list_tool_names()),
    )


@router.get("/mcp/headroom")
async def mcp_headroom() -> Dict[str, Any]:
    """Token + cost accounting for the headroom context-engineering layer."""
    from mcp_server.context_engineering import USAGE
    from mcp_server.context_engineering.compressor import headroom_available
    from mcp_server.context_engineering.middleware import cache_stats
    return {
        "headroom_available": headroom_available(),
        "usage": USAGE.snapshot(),
        "cache": cache_stats(),
    }


@router.post("/mcp/headroom/reset")
async def mcp_headroom_reset() -> Dict[str, Any]:
    from mcp_server.context_engineering import USAGE
    from mcp_server.context_engineering.middleware import clear_cache
    USAGE.reset()
    clear_cache()
    return {"status": "reset", "usage": USAGE.snapshot()}


@router.post("/mcp/invoke", response_model=MCPInvokeResponse)
async def mcp_invoke(req: MCPInvokeRequest) -> MCPInvokeResponse:
    """Invoke an MCP tool from the web UI without going through the MCP protocol.
    Useful for the 'Test' button in the MCP Tools tab."""
    import time as _time
    from mcp_server.tools import dispatch_async
    start = _time.perf_counter()
    try:
        result = await dispatch_async(req.tool, req.args or {})
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(exc))
    return MCPInvokeResponse(
        tool=req.tool,
        result=result,
        latency_ms=round((_time.perf_counter() - start) * 1000, 1),
    )


@router.websocket("/accuracy/ws")
async def accuracy_ws(ws: WebSocket) -> None:
    from accuracy.broadcaster import subscribe, unsubscribe, snapshot
    await ws.accept()
    q = await subscribe()
    try:
        # Initial snapshot so clients can render immediately
        await ws.send_json({"type": "snapshot", "entries": snapshot()})
        while True:
            event = await q.get()
            await ws.send_json(event)
    except WebSocketDisconnect:
        pass
    finally:
        await unsubscribe(q)


def _coerce_row(row: Dict[str, Any]) -> Dict[str, Any]:
    """sqlite3.Row -> dict already done in repository; this drops nulls Pydantic doesn't expect."""
    return {k: v for k, v in row.items()}


@router.post("/rl/simulate")
async def rl_simulate(req: RLSimulateRequest) -> Dict[str, Any]:
    # fetch_live_data takes the key explicitly below — no env mutation needed.
    logger.info("POST /rl/simulate  symbol=%s  model_id=%s", req.symbol, req.model_id)
    try:
        from tools.rl.dqn_agent import load_agent, extract_state
        from tools.rl.rl_tool import fetch_live_data
        from tools.rl.wallet import Wallet
        agent = load_agent(req.model_id)
        if agent is None:
            raise HTTPException(status_code=404, detail=f"Model {req.model_id} not found")
        api_key = req.indian_api_key or os.environ.get("INDIAN_API_KEY", "")
        live = fetch_live_data(req.symbol, api_key)
        if live is None:
            raise HTTPException(status_code=503, detail="Could not fetch live data")
        price = live['price']
        wallet = Wallet(initial_capital=req.initial_capital)
        position = 0
        cash_ratio = 1.0
        dummy_df_data = {
            'Close': [price * 0.98, price * 0.99, price],
            'High': [price * 1.01, price * 1.01, price * 1.005],
            'Low': [price * 0.97, price * 0.98, price * 0.99],
            'Volume': [1000, 1200, 1100],
            'Open': [price * 0.99, price * 0.99, price * 1.0],
        }
        import pandas as pd
        dummy_df = pd.DataFrame(dummy_df_data)
        state = extract_state(dummy_df, len(dummy_df) - 1, position, cash_ratio)
        action = agent.act(state, eval_mode=True)
        action_name = ['HOLD', 'BUY', 'SELL'][action]
        if action == 1:
            wallet.buy(price)
        elif action == 2 and position == 1:
            wallet.sell(price)
        summary = wallet.get_summary(price)
        return {
            'action': action_name,
            'price': round(price, 2),
            'symbol': req.symbol,
            'model_id': req.model_id,
            'wallet': summary,
            'live_data': live,
            'status': 'simulated',
        }
    except HTTPException:
        raise
    except Exception as exc:
        raise _internal_error(exc, "RL simulation")


# ── Realtime intraday session: monitor + run controls ──────────────────────────

@router.get("/rt/status", response_model=RTStatusResponse)
async def rt_status(symbol: str | None = None) -> RTStatusResponse:
    """Current state of a realtime session (action / equity / PnL / step counts)."""
    from realtime.runner import MANAGER
    from realtime.state import get_status
    if symbol:
        return RTStatusResponse(**MANAGER.status(symbol))
    return RTStatusResponse(**get_status(None))


@router.get("/rt/history", response_model=RTHistoryResponse)
async def rt_history(symbol: str | None = None, limit: int = 100) -> RTHistoryResponse:
    """Recent per-step records from a realtime session (for the equity chart + table)."""
    from realtime.state import get_history
    return RTHistoryResponse(symbol=symbol, steps=get_history(symbol, limit=limit))


@router.get("/rt/sessions", response_model=RTSessionsResponse)
async def rt_sessions() -> RTSessionsResponse:
    """List all sessions known to the run manager (running or recently finished)."""
    from realtime.runner import MANAGER
    return RTSessionsResponse(sessions=MANAGER.list_sessions())


@router.get("/rt/chart")
async def rt_chart(symbol: str | None = None, chart: str = "equity_curve") -> Dict[str, Any]:
    """d3-buck spec for a realtime session (equity_curve / action_heatmap / drawdown_curve)."""
    from realtime.state import get_status, get_history
    from .d3_viz import build_d3_spec
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
    return {"chart": chart, "active": status.get("active", False), "spec": build_d3_spec(chart, session)}


@router.post("/rt/start", response_model=RTStatusResponse)
async def rt_start(req: RTStartRequest) -> RTStatusResponse:
    """Start a realtime simulation in a background thread (use replay for off-hours)."""
    from realtime.runner import MANAGER
    # indian_api_key is passed explicitly to MANAGER.start() below — no env
    # mutation needed (the session runs in a background thread, so a
    # request-scoped env var would go out of scope while the session is
    # still running).
    logger.info("POST /rt/start  symbol=%s  model_id=%s  replay=%s", req.symbol, req.model_id, req.replay)
    try:
        status = MANAGER.start(
            symbol=req.symbol,
            model_id=req.model_id,
            interval=req.interval,
            replay=req.replay,
            replay_start=req.replay_start,
            replay_end=req.replay_end,
            capital=req.capital,
            max_steps=req.max_steps,
            speed=req.speed,
            online_update_every=req.online_update_every,
            indian_api_key=req.indian_api_key or os.environ.get("INDIAN_API_KEY", ""),
        )
        return RTStatusResponse(**status)
    except (ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=409, detail=str(exc))
    except Exception as exc:  # noqa: BLE001
        raise _internal_error(exc, "Realtime session start")


@router.post("/rt/stop", response_model=RTStatusResponse)
async def rt_stop(req: RTStopRequest) -> RTStatusResponse:
    """Request a running realtime session to finish at the next safe point."""
    from realtime.runner import MANAGER
    logger.info("POST /rt/stop  symbol=%s", req.symbol)
    try:
        return RTStatusResponse(**MANAGER.stop(req.symbol))
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
