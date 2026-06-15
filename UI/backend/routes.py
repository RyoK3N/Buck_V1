"""
UI.backend.routes
──────────────────
FastAPI route handlers for all Buck endpoints.
"""

from __future__ import annotations
import logging
import os
import traceback
from typing import Any, Dict

from fastapi import APIRouter, HTTPException

logger = logging.getLogger(__name__)

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
    RLModelsResponse,
)  # noqa: F401

router = APIRouter()

VALID_INTERVALS = ["1m", "2m", "5m", "15m", "30m", "60m", "90m", "1h",
                   "1d", "5d", "1wk", "1mo", "3mo"]


def _make_buck(req: AnalyzeRequest | BatchRequest):
    """Instantiate Buck with per-request API keys."""
    from agent_scripts.buck import BuckFactory
    return BuckFactory.create_production_agent(
        openai_api_key=req.openai_api_key,
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
    """Return server config loaded from .env (used by the UI to pre-fill forms)."""
    from agent_scripts.config import SETTINGS
    return ConfigResponse(
        openai_api_key=(
            SETTINGS.openai_api_key
            if SETTINGS.openai_api_key != "__placeholder__"
            else ""
        ),
        openai_base_url=SETTINGS.openai_base_url,
        chat_model=SETTINGS.chat_model,
        indian_api_key=SETTINGS.indian_api_key,
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
    os.environ["OPENAI_API_KEY"] = req.openai_api_key
    if req.indian_api_key:
        os.environ["INDIAN_API_KEY"] = req.indian_api_key

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
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/batch")
async def batch(req: BatchRequest) -> Dict[str, Any]:
    os.environ["OPENAI_API_KEY"] = req.openai_api_key
    if req.indian_api_key:
        os.environ["INDIAN_API_KEY"] = req.indian_api_key

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
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


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
        raise HTTPException(status_code=500, detail=str(exc))


# ── RL / Reinforcement Learning ──────────────────────────────────────────────

@router.post("/rl/train")
async def rl_train(req: RLTrainRequest) -> Dict[str, Any]:
    if req.indian_api_key:
        os.environ["INDIAN_API_KEY"] = req.indian_api_key
    logger.info("POST /rl/train  symbol=%s  model_id=%s  episodes=%d", req.symbol, req.model_id, req.episodes)
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
        return {
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
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("RL train error: %s\n%s", exc, traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/rl/predict")
async def rl_predict(req: RLPredictRequest) -> Dict[str, Any]:
    if req.indian_api_key:
        os.environ["INDIAN_API_KEY"] = req.indian_api_key
    logger.info("POST /rl/predict  symbol=%s  model_id=%s", req.symbol, req.model_id)
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
        logger.error("RL predict error: %s\n%s", exc, traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/rl/models", response_model=RLModelsResponse)
async def rl_models() -> RLModelsResponse:
    from tools.rl.dqn_agent import list_models
    return RLModelsResponse(models=list_models())


@router.delete("/rl/models/{model_id}")
async def rl_delete_model(model_id: str) -> Dict[str, Any]:
    from tools.rl.dqn_agent import WEIGHTS_DIR
    path = WEIGHTS_DIR / f"{model_id}.pt"
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Model {model_id} not found")
    path.unlink()
    logger.info("Deleted RL model: %s", model_id)
    return {'status': 'deleted', 'model_id': model_id}


@router.post("/rl/simulate")
async def rl_simulate(req: RLSimulateRequest) -> Dict[str, Any]:
    if req.indian_api_key:
        os.environ["INDIAN_API_KEY"] = req.indian_api_key
    logger.info("POST /rl/simulate  symbol=%s  model_id=%s", req.symbol, req.model_id)
    try:
        from tools.rl.dqn_agent import load_agent, extract_state
        from tools.rl.rl_tool import fetch_live_data
        from tools.rl.wallet import Wallet
        agent = load_agent(req.model_id)
        if agent is None:
            raise HTTPException(status_code=404, detail=f"Model {req.model_id} not found")
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
        logger.error("RL simulate error: %s\n%s", exc, traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(exc))
