from __future__ import annotations
import json
import os
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yfinance as yf
from langchain_core.tools import tool

from agent_scripts.tools import BaseTool, get_stock_data

from .dqn_agent import create_agent, extract_state
from .wallet import Wallet


def fetch_historical_data(symbol: str, start_date: str, end_date: str, interval: str = "1d") -> Optional[pd.DataFrame]:
    try:
        if not symbol.endswith('.NS'):
            yf_sym = f"{symbol}.NS"
        else:
            yf_sym = symbol
        data = yf.download(yf_sym, start=start_date, end=end_date, interval=interval, progress=False)
        if data is not None and not data.empty:
            if isinstance(data.columns, pd.MultiIndex):
                flat = data.columns.get_level_values(0)
                data.columns = [str(c) for c in flat]
            data = data.dropna()
            if data.empty:
                return None
            return data
    except Exception:
        pass
    try:
        import requests
        symbol_clean = symbol.replace('.NS', '')
        api_key = os.environ.get("INDIAN_API_KEY", "")
        if api_key:
            resp = requests.get(
                f"https://stock.indianapi.in/stock",
                headers={"X-Api-Key": api_key},
                params={"symbol": symbol_clean},
                timeout=30,
            )
            if resp.status_code == 200:
                raw = resp.json()
                rows = raw if isinstance(raw, list) else [raw]
                if rows:
                    df = pd.DataFrame(rows)
                    df.columns = [str(c).capitalize() for c in df.columns]
                    name_map = {'Close': 'Close', 'close': 'Close', 'Open': 'Open', 'open': 'Open',
                                'High': 'High', 'high': 'High', 'Low': 'Low', 'low': 'Low',
                                'Volume': 'Volume', 'volume': 'Volume'}
                    df = df.rename(columns=name_map)
                    required = ['Close', 'Open', 'High', 'Low', 'Volume']
                    if all(c in df.columns for c in required):
                        date_col = None
                        for c in ['Date', 'date', 'timestamp', 'datetime']:
                            if c in df.columns:
                                date_col = c
                                break
                        if date_col:
                            df['Date'] = pd.to_datetime(df[date_col])
                        else:
                            df['Date'] = pd.date_range(start=start_date, periods=len(df), freq='D')
                        df = df.sort_values('Date')
                        df = df[(df['Date'] >= pd.Timestamp(start_date)) & (df['Date'] <= pd.Timestamp(end_date))]
                        df = df.reset_index(drop=True)
                        return df[required + ['Date']]
    except Exception:
        pass
    return None


def fetch_live_data(symbol: str, api_key: str = "") -> Optional[Dict[str, float]]:
    try:
        import requests
        symbol_clean = symbol.replace('.NS', '')
        headers = {"X-Api-Key": api_key} if api_key else {}
        resp = requests.get(
            f"https://stock.indianapi.in/stock",
            headers=headers,
            params={"symbol": symbol_clean},
            timeout=15,
        )
        if resp.status_code == 200:
            data = resp.json()
            if isinstance(data, dict):
                return {
                    'price': float(data.get('price', data.get('close', data.get('last_price', 0)))),
                    'open': float(data.get('open', 0)),
                    'high': float(data.get('high', 0)),
                    'low': float(data.get('low', 0)),
                    'volume': float(data.get('volume', 0)),
                    'timestamp': data.get('timestamp', datetime.now().isoformat()),
                }
    except Exception:
        pass
    try:
        if not symbol.endswith('.NS'):
            yf_sym = f"{symbol}.NS"
        else:
            yf_sym = symbol
        ticker = yf.Ticker(yf_sym)
        hist = ticker.history(period="1d", interval="1m")
        if hist is not None and not hist.empty:
            last = hist.iloc[-1]
            return {
                'price': float(last['Close']),
                'open': float(last['Open']),
                'high': float(last['High']),
                'low': float(last['Low']),
                'volume': float(last['Volume']),
                'timestamp': str(last.name) if hasattr(last, 'name') else datetime.now().isoformat(),
            }
    except Exception:
        pass
    return None


class RLAgentTool(BaseTool):
    def __init__(self):
        super().__init__(
            "rl_agent",
            "Reinforcement Learning agent for trading (DQN/PPO/A2C). Trains on market data and outputs BUY/SELL/HOLD signals."
        )

    def execute(self, data: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        if not self._validate_data(data):
            raise ValueError("Invalid data structure")
        mode = kwargs.get('mode', 'predict')
        algorithm = kwargs.get('algorithm', 'dqn')
        model_id = kwargs.get('model_id', 'default_rl_model')
        episodes = kwargs.get('episodes', 50)
        hidden_dim = kwargs.get('hidden_dim', 128)
        lr = kwargs.get('lr', 1e-3)
        initial_capital = kwargs.get('initial_capital', 100000.0)

        if mode == 'train':
            return self._train(data, algorithm, model_id, episodes, hidden_dim, lr, initial_capital)
        return self._predict(data, model_id, initial_capital)

    def _train(
        self, data: pd.DataFrame, algorithm: str, model_id: str,
        episodes: int, hidden_dim: int, lr: float, initial_capital: float
    ) -> Dict[str, Any]:
        agent = create_agent(algorithm, input_dim=12, hidden_dim=hidden_dim, output_dim=3, lr=lr)
        close = data['Close'].values
        total_steps = len(close)
        best_reward = -float('inf')
        episode_rewards = []

        for ep in range(episodes):
            wallet = Wallet(initial_capital=initial_capital)
            total_reward = 0.0
            prev_pv = wallet.get_portfolio_value(float(close[0]))
            for step in range(1, total_steps):
                position = 1 if wallet.holdings > 1e-6 else 0
                cash_ratio = wallet.cash / (prev_pv + 1e-10)
                state = extract_state(data, step, position, cash_ratio)
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
                next_state = extract_state(data, min(step + 1, total_steps - 1), next_pos, next_cr)
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
            pnl_pct = ((final_pv - initial_capital) / initial_capital) * 100
            episode_rewards.append({
                'episode': ep + 1,
                'total_reward': round(total_reward, 4),
                'portfolio_value': round(final_pv, 2),
                'return_pct': round(pnl_pct, 2),
                'trades': len(wallet.trades),
            })
            if total_reward > best_reward:
                best_reward = total_reward
                agent.save(f"{model_id}_best")

        agent.save(model_id)
        wallet.close_position(float(close[-1]))
        summary = wallet.get_summary(float(close[-1]))

        return {
            'model_id': model_id,
            'algorithm': algorithm,
            'episodes': episodes,
            'total_steps': total_steps,
            'episode_rewards': episode_rewards,
            'final_summary': summary,
            'best_reward': round(best_reward, 4),
            'status': 'trained',
        }

    def _predict(self, data: pd.DataFrame, model_id: str, initial_capital: float) -> Dict[str, Any]:
        from .dqn_agent import load_agent
        agent = load_agent(model_id)
        if agent is None:
            return {'signal': 'NEUTRAL', 'strength': 0.0, 'error': f'Model {model_id} not found. Train first.', 'status': 'no_model'}
        wallet = Wallet(initial_capital=initial_capital)
        close = data['Close'].values
        total_steps = len(close)
        signals = []

        for step in range(total_steps):
            position = 1 if wallet.holdings > 1e-6 else 0
            cash_ratio = wallet.cash / (wallet.get_portfolio_value(float(close[step])) + 1e-10)
            state = extract_state(data, step, position, cash_ratio)
            action = agent.act(state, eval_mode=True)
            price = float(close[step])
            if action == 1 and position == 0:
                wallet.buy(price)
            elif action == 2 and position == 1:
                wallet.sell(price)
            signals.append({
                'step': step,
                'action': ['HOLD', 'BUY', 'SELL'][action],
                'price': round(price, 2),
                'portfolio_value': round(float(wallet.get_portfolio_value(price)), 2),
            })

        wallet.close_position(float(close[-1]))
        summary = wallet.get_summary(float(close[-1]))
        last_action = signals[-1]['action'] if signals else 'HOLD'
        last_pv = signals[-1]['portfolio_value'] if signals else initial_capital
        strength = min(abs(last_pv - initial_capital) / initial_capital * 2, 1.0)

        return {
            'signal': last_action, 'strength': round(strength, 4), 'model_id': model_id,
            'total_signals': len(signals), 'summary': summary, 'signals': signals[-50:], 'status': 'predicted',
        }


@tool
def rl_agent(
    mode: str = "predict",
    algorithm: str = "dqn",
    model_id: str = "default_rl_model",
    episodes: int = 50,
    hidden_dim: int = 128,
    lr: float = 0.001,
    initial_capital: float = 100000.0,
) -> str:
    """Run the Reinforcement Learning trading agent (DQN/PPO/A2C). Modes: 'train' to train a new model, 'predict' to use a trained model. Returns BUY/SELL/HOLD signals with portfolio performance."""
    data = get_stock_data()
    if data is None:
        return json.dumps({"error": "No stock data available"})
    t = RLAgentTool()
    result = t.execute(
        data, mode=mode, algorithm=algorithm, model_id=model_id, episodes=episodes,
        hidden_dim=hidden_dim, lr=lr, initial_capital=initial_capital,
    )
    return json.dumps(result, default=str)


TOOL_CLASS = RLAgentTool
TOOL_FUNC = rl_agent
