# Reinforcement Learning Trading Agent

DQN-based reinforcement learning agent for stock trading.

## Architecture

- **dqn_agent.py** - DQN network (PyTorch), replay buffer, state extraction, agent save/load
- **wallet.py** - Simulation wallet tracking cash, holdings, portfolio value, trades, and metrics (Sharpe, drawdown, return)
- **rl_tool.py** - BaseTool wrapper, data fetching (yfinance → Indian API fallback), training loop, prediction pipeline

## Features

- **Train** on historical data with configurable episodes, hidden dimensions, learning rate
- **Evaluate** trained models on out-of-sample data with full wallet simulation
- **Live Simulate** using Indian API for real-time market data
- **Weights Management** - Save/load/list/delete model weights
- **Benchmarking** - Track Sharpe ratio, max drawdown, total return across training episodes

## Actions

| Action | Description |
|--------|-------------|
| HOLD   | Do nothing |
| BUY    | Invest all available cash |
| SELL   | Liquidate all holdings |

## State Features (12-dim)

1. Price z-score
2. Volume z-score
3. Log return
4. Price momentum
5. Volatility
6. RSI (normalized)
7. High-Low range
8. Bollinger Band position
9. Position flag (holding or not)
10. Cash ratio
11. Instant return
12. Short-term trend

## API Endpoints

- `POST /rl/train` - Train a new model
- `POST /rl/predict` - Run evaluation on historical data
- `POST /rl/simulate` - Single-step live simulation
- `GET /rl/models` - List saved models
- `DELETE /rl/models/{id}` - Delete a model
