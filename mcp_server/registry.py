"""
mcp_server.registry
───────────────────
Single source of truth for the Buck MCP tool surface.

Each entry has the shape Anthropic's `messages.create(tools=[...])` expects:
  {name, description, input_schema}

`mcp_server.server` decorates the matching implementations in `mcp_server.tools`
with FastMCP `@mcp.tool` so external MCP clients see the same tools.
"""

from __future__ import annotations

from typing import Any, Dict, List

# Common parameter shapes reused across multiple tools
_SYMBOL = {"type": "string", "description": "Ticker, e.g. 'BHEL.NS' (NSE) or 'AAPL'"}
_DATE = {"type": "string", "description": "YYYY-MM-DD"}
_INTERVAL = {
    "type": "string",
    "description": "Data interval (1m, 5m, 15m, 30m, 1h, 1d, ...)",
    "default": "1h",
}


BUCK_TOOLS: List[Dict[str, Any]] = [
    # ── Analysis (existing /analyze) ────────────────────────────────────────
    {
        "name": "single_analyze",
        "description": (
            "Run Buck's full analysis pipeline (technical indicators + sentiment + OpenAI forecast) "
            "for one symbol over a date range. This is the primary evidence call — invoke this first "
            "when you need a comprehensive read on a stock. Returns analysis_results, forecast (OHLC + "
            "confidence + reasoning), and metadata."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "symbol": _SYMBOL,
                "start_date": _DATE,
                "end_date": _DATE,
                "interval": _INTERVAL,
                "selected_tools": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional subset of tool IDs (rsi, macd, moving_average, obv, support_resistance, candlestick_patterns, lstm_prediction, ...). Omit for all available tools.",
                },
            },
            "required": ["symbol", "start_date", "end_date"],
        },
    },
    # ── Batch analysis (existing /batch) ────────────────────────────────────
    {
        "name": "batch_analyze",
        "description": (
            "Run analysis concurrently across multiple symbols (e.g. sector peers, a watchlist) "
            "for cross-sectional context. Use this after `single_analyze` if you want to compare a "
            "candidate against peers."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "symbols": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of ticker symbols",
                },
                "start_date": _DATE,
                "end_date": _DATE,
                "interval": _INTERVAL,
                "selected_tools": {
                    "type": "array",
                    "items": {"type": "string"},
                },
                "max_concurrent": {"type": "integer", "default": 3, "minimum": 1, "maximum": 10},
            },
            "required": ["symbols", "start_date", "end_date"],
        },
    },
    # ── Discovery (existing /tools-registry, /intervals, /chart-types) ──────
    {
        "name": "list_tools_registry",
        "description": "List every Buck analysis tool grouped by category (maths, dl, rl, utility, web). Use this first to discover what you can pass in `selected_tools`.",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "list_available_intervals",
        "description": "List supported data intervals for `single_analyze` / `batch_analyze` / `visualize`.",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "list_chart_types",
        "description": "List available chart_type values for `visualize`.",
        "input_schema": {"type": "object", "properties": {}},
    },
    # ── RL Lab (existing /rl/*) ─────────────────────────────────────────────
    {
        "name": "rl_train",
        "description": (
            "Train a fresh RL trading agent on a symbol's historical data. Long-running — call only "
            "when you genuinely need a model that does not yet exist (check `list_rl_models` first). "
            "Prefer `algorithm='ppo_continuous'` for new models: it outputs a continuous target-position "
            "fraction in [0, 1], uses an LSTM encoder over a 30-feature state, and trains with Sharpe-with-"
            "drawdown reward shaping. The legacy discrete options (dqn/ppo/a2c) are kept for backward "
            "compatibility with existing checkpoints like `dqn_model_best`."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "symbol": _SYMBOL,
                "start_date": _DATE,
                "end_date": _DATE,
                "interval": {"type": "string", "default": "1d"},
                "algorithm": {
                    "type": "string",
                    "enum": ["dqn", "a2c", "ppo", "ppo_continuous"],
                    "default": "ppo_continuous",
                },
                "model_id": {"type": "string", "description": "Identifier for the saved model"},
                "episodes": {"type": "integer", "default": 200, "minimum": 1, "maximum": 1000,
                              "description": "200+ is reasonable for ppo_continuous; 50 is too few"},
                "hidden_dim": {"type": "integer", "default": 128},
                "learning_rate": {"type": "number", "default": 3e-4,
                                   "description": "3e-4 is PPO-standard; 1e-3 fits DQN better"},
                "initial_capital": {"type": "number", "default": 100000.0},
            },
            "required": ["symbol", "start_date", "end_date", "model_id"],
        },
    },
    {
        "name": "rl_predict",
        "description": "Run a trained RL model over a date range and inspect its trade signals + equity curve. Use this for backtesting an RL signal before incorporating it into a forecast.",
        "input_schema": {
            "type": "object",
            "properties": {
                "symbol": _SYMBOL,
                "start_date": _DATE,
                "end_date": _DATE,
                "interval": {"type": "string", "default": "1d"},
                "model_id": {"type": "string"},
                "initial_capital": {"type": "number", "default": 100000.0},
            },
            "required": ["symbol", "start_date", "end_date", "model_id"],
        },
    },
    {
        "name": "rl_simulate",
        "description": "Get the live RL action (HOLD / BUY / SELL) for a symbol using the latest market snapshot from a trained model.",
        "input_schema": {
            "type": "object",
            "properties": {
                "symbol": _SYMBOL,
                "model_id": {"type": "string"},
                "interval": {"type": "string", "default": "1m"},
                "initial_capital": {"type": "number", "default": 100000.0},
            },
            "required": ["symbol", "model_id"],
        },
    },
    {
        "name": "list_rl_models",
        "description": "List all trained RL models on disk. Call this before `rl_train` to avoid retraining a model that already exists.",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "rl_ensemble_predict",
        "description": (
            "Multi-timeframe ensemble inference: run several RL models (each at its own native interval) "
            "and return a weighted aggregate. Use this when you want a more robust signal than any single "
            "model gives — e.g. stack a daily-trained model with an hourly one and the symbol-specific "
            "model. Output is a target position fraction in [0, 1] plus a BUY/HOLD/SELL bucket plus per-"
            "model breakdown. Recommended weighting: ∝ each model's out-of-sample Sharpe ratio (call "
            "`get_prediction_accuracy` to get those numbers)."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "symbol": _SYMBOL,
                "start_date": _DATE,
                "end_date": _DATE,
                "models": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "model_id": {"type": "string"},
                            "interval": {"type": "string"},
                            "weight": {"type": "number", "default": 1.0},
                        },
                        "required": ["model_id"],
                    },
                    "minItems": 1,
                    "description": "Each model spec runs at its own interval (1d/1h/15m/...).",
                },
                "fallback_interval": {"type": "string", "default": "1d"},
            },
            "required": ["symbol", "start_date", "end_date", "models"],
        },
    },
    # ── Visualization (existing /visualize) ─────────────────────────────────
    {
        "name": "visualize",
        "description": "Generate a Plotly chart JSON for a symbol. Useful when you want to reason over chart structure (candlestick patterns, Bollinger squeeze, MACD divergence, ...). Returns the Plotly figure dict.",
        "input_schema": {
            "type": "object",
            "properties": {
                "symbol": _SYMBOL,
                "start_date": _DATE,
                "end_date": _DATE,
                "interval": {"type": "string", "default": "1d"},
                "chart_type": {"type": "string", "default": "price_ma"},
            },
            "required": ["symbol", "start_date", "end_date", "chart_type"],
        },
    },
    {
        "name": "visualize_accuracy",
        "description": (
            "Render rolling forecast accuracy over time as a d3 chart spec (d3-buck/1): one line "
            "per model showing daily MAE or directional accuracy. Use this to see whether a model is "
            "improving, drifting, or degrading. Pair with `get_prediction_accuracy` for point stats."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "model": {"type": "string", "description": "Optional model filter (openai, claude, ensemble)"},
                "symbol": {"type": "string", "description": "Optional symbol filter"},
                "window_days": {"type": "integer", "default": 30, "minimum": 1, "maximum": 365},
                "metric": {"type": "string", "enum": ["mae", "directional_accuracy"], "default": "mae"},
            },
        },
    },
    {
        "name": "visualize_predictions",
        "description": (
            "Render predicted-vs-actual close for a symbol over a lookback window as a d3 chart spec "
            "(two lines: predicted and actual). Use this to visually inspect bias and tracking error "
            "for a specific ticker."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "symbol": _SYMBOL,
                "lookback_days": {"type": "integer", "default": 30, "minimum": 1, "maximum": 365},
                "model": {"type": "string", "description": "Optional model filter"},
            },
            "required": ["symbol"],
        },
    },
    {
        "name": "visualize_compare",
        "description": (
            "Overlay multiple symbols' price series on one d3 chart (rebased to 100 by default) for "
            "relative-performance comparison across peers or a watchlist. Returns a d3-buck/1 multiline spec."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "symbols": {"type": "array", "items": {"type": "string"}, "minItems": 1,
                            "description": "Ticker symbols to overlay"},
                "start_date": _DATE,
                "end_date": _DATE,
                "interval": {"type": "string", "default": "1d"},
                "normalize": {"type": "boolean", "default": True,
                              "description": "Rebase each series to 100 at its first point"},
            },
            "required": ["symbols", "start_date", "end_date"],
        },
    },
    {
        "name": "visualize_session",
        "description": (
            "Render the currently running realtime intraday session as a d3 chart spec — equity_curve, "
            "action_heatmap (target position over time), or drawdown_curve. Use this to monitor a live "
            "simulation Claude or the RL agent is driving. Returns {active, status, spec}; when no session "
            "is running the spec is empty."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "symbol": {"type": "string", "description": "Optional symbol; omit for the active session"},
                "chart": {
                    "type": "string",
                    "enum": ["equity_curve", "action_heatmap", "drawdown_curve"],
                    "default": "equity_curve",
                },
            },
        },
    },
    # ── Accuracy self-introspection (new) ───────────────────────────────────
    {
        "name": "get_prediction_accuracy",
        "description": (
            "Look up rolling accuracy for previous Buck forecasts (per-model, optionally per-symbol). "
            "ALWAYS consult this before committing to a forecast — it tells you how well the model has "
            "been doing recently and lets you calibrate confidence accordingly."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "symbol": {"type": "string", "description": "Optional symbol filter"},
                "model": {"type": "string", "description": "Optional model filter (openai, claude, ensemble)"},
                "window_days": {"type": "integer", "description": "Lookback window in days; null = all-time"},
            },
        },
    },
    {
        "name": "list_recent_predictions",
        "description": "List recent predictions (with actuals + evaluation metrics where available). Useful for inspecting where the model has historically been right or wrong.",
        "input_schema": {
            "type": "object",
            "properties": {
                "symbol": {"type": "string"},
                "model": {"type": "string"},
                "status": {"type": "string", "enum": ["open", "evaluated", "expired"]},
                "limit": {"type": "integer", "default": 20, "minimum": 1, "maximum": 200},
            },
        },
    },
    {
        "name": "compare_predictions_vs_actual",
        "description": "Return a time series of predicted-vs-actual close for a symbol over the lookback window. Use this to see whether the model is consistently biased high/low or drifting.",
        "input_schema": {
            "type": "object",
            "properties": {
                "symbol": {"type": "string"},
                "lookback_days": {"type": "integer", "default": 30},
                "model": {"type": "string"},
            },
            "required": ["symbol"],
        },
    },
    # ── Context engineering / headroom (new) ────────────────────────────────
    {
        "name": "headroom_stats",
        "description": (
            "Report the context-engineering layer's token + cost accounting: how many tokens "
            "the headroom compression layer has saved across all tool calls this session, the "
            "estimated USD cost saved, a per-tool breakdown, and cache hit-rate. Use this to "
            "audit how much context budget Buck's tools are consuming."
        ),
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "headroom_reset",
        "description": "Reset the headroom token/cost tracker and clear the compression cache. Use at the start of a fresh measurement.",
        "input_schema": {"type": "object", "properties": {}},
    },
    # ── Training-session observability (d3) ──────────────────────────────────
    {
        "name": "list_training_sessions",
        "description": (
            "List saved RL training sessions (one per `rl_train` run) with their model_id, symbol, "
            "algorithm, interval, episode count and final return. Call this to discover session_ids "
            "to pass to `visualize_training`."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "model_id": {"type": "string", "description": "Optional model_id filter"},
                "symbol": {"type": "string", "description": "Optional symbol filter"},
                "limit": {"type": "integer", "default": 50, "minimum": 1, "maximum": 200},
            },
        },
    },
    {
        "name": "list_d3_chart_types",
        "description": "List the available d3 chart types for `visualize_training` (reward_curve, equity_curve, loss_curves, return_distribution, drawdown_curve, action_heatmap).",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "visualize_training",
        "description": (
            "Render a training session as an interactive d3 chart spec (d3-buck/1) for observability: "
            "reward curve, equity curve, policy/value/entropy loss curves, return distribution, drawdown, "
            "or action heatmap. Returns a framework-agnostic data+encoding spec the Buck frontend renders "
            "with d3. Use this to reason about how an RL model trained — convergence, overfitting, drawdowns."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "session_id": {"type": "string", "description": "From `list_training_sessions`"},
                "chart": {
                    "type": "string",
                    "enum": [
                        "reward_curve", "equity_curve", "loss_curves",
                        "return_distribution", "drawdown_curve", "action_heatmap",
                    ],
                    "default": "reward_curve",
                },
            },
            "required": ["session_id"],
        },
    },
    # ── Real-time intraday session (read-only) ──────────────────────────────
    {
        "name": "rt_session_status",
        "description": (
            "Get the current state of a live intraday simulation session started via the CLI "
            "(`python -m realtime.cli`): latest action / target position, equity, intraday PnL, "
            "number of online learning updates, and market-open status. Read-only — the loop itself "
            "is CLI-driven."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "symbol": {"type": "string", "description": "Optional symbol filter; omit for the active session"},
            },
        },
    },
    {
        "name": "rt_session_history",
        "description": "Recent per-step records from the live intraday session (prediction-before-event, realized reward, online-update stats). Visualizable via `visualize_training` equity/reward charts.",
        "input_schema": {
            "type": "object",
            "properties": {
                "symbol": {"type": "string"},
                "limit": {"type": "integer", "default": 50, "minimum": 1, "maximum": 500},
            },
        },
    },
]


BUCK_TOOLS_BY_NAME: Dict[str, Dict[str, Any]] = {t["name"]: t for t in BUCK_TOOLS}
