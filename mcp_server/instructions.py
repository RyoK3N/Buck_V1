"""
mcp_server.instructions
───────────────────────
Top-level context the MCP server advertises to clients (Claude Desktop, Claude
Code, etc.) via FastMCP's `instructions` field, plus the text of the workflow
`@mcp.prompt`s. Centralised here so the "what Buck is / how to drive it" story is
written once and stays consistent with the tool registry.

Key facts a client MUST know up front:
  • Buck is for **Indian equities (NSE) only** — not US/global tickers.
  • Symbols use the **`.NS` suffix** (e.g. RELIANCE.NS).
  • All clock/market logic is **IST**; NSE trades 09:15–15:30, Mon–Fri.
"""

from __future__ import annotations

SERVER_INSTRUCTIONS = """\
Buck — AI-powered analysis, forecasting and RL trading for **Indian stock markets (NSE)**.

╔══ MARKET SCOPE (read first) ══════════════════════════════════════════════════╗
Buck targets **NSE-listed Indian equities ONLY**. It does NOT support US/global
tickers (AAPL, NVDA, TSLA, …) — price news/sentiment, RL/live data and the
market-hours logic are all India-specific.
  • Symbols MUST use the NSE `.NS` suffix, e.g.
      RELIANCE.NS, TCS.NS, INFY.NS, HDFCBANK.NS, SBIN.NS, ICICIBANK.NS, BHEL.NS
  • All times are **IST (Asia/Kolkata)**. NSE regular session: 09:15–15:30, Mon–Fri.
  • Dates are `YYYY-MM-DD`. Intraday intervals (1m/5m/15m/…) only have a limited
    recent-history window; use `1d` for long ranges.
If a user names a US ticker, tell them Buck is NSE-only and suggest the Indian
equivalent or an NSE symbol.
╚═══════════════════════════════════════════════════════════════════════════════╝

WHAT BUCK CAN DO (by category)
  • Analysis & forecasting — technical indicators (RSI, MACD, moving averages,
    OBV, support/resistance, candlestick patterns), LSTM price-direction model,
    and an LLM-written OHLC forecast with confidence + reasoning.
      tools: single_analyze, batch_analyze
  • Reinforcement-learning trading agents — train, backtest, get a live signal,
    and stack models into an ensemble.
      tools: rl_train, rl_predict, rl_simulate, rl_ensemble_predict, list_rl_models
  • Realtime intraday sessions — a live online-learning loop you can START, WATCH
    and STOP. Sessions run inside the web app and show live in its Realtime tab.
      tools: rt_start_session, rt_stop_session, rt_session_status,
             rt_session_history, visualize_session, open_buck_ui
      (replay mode works any time; needs the web app running — `python main.py`.)
  • Prediction accuracy — how well past forecasts held up; check before trusting one.
      tools: get_prediction_accuracy, list_recent_predictions, compare_predictions_vs_actual
  • Visualization — Plotly market charts + d3 specs for training/accuracy/comparison.
      tools: visualize, visualize_training, visualize_accuracy, visualize_predictions,
             visualize_compare, visualize_session
  • Discovery / introspection — enumerate options before calling other tools.
      tools: list_tools_registry, list_available_intervals, list_chart_types,
             list_d3_chart_types, list_training_sessions, headroom_stats

RECOMMENDED WORKFLOWS
  1) Analyze & forecast one stock:
       get_prediction_accuracy (calibrate trust) → single_analyze → optionally
       visualize for a chart. Add batch_analyze to compare sector peers.
  2) RL lifecycle (a model must exist before predict/simulate):
       list_rl_models → rl_train (algorithm='ppo_continuous', episodes ≥ 200) →
       rl_predict (backtest over a date range) → rl_simulate (latest-snapshot
       signal) and/or rl_ensemble_predict → visualize_training to inspect the run.
  3) Run a REPLAY simulation the user can WATCH in the web UI (needs the web app
     running — `python main.py`):
       (ensure a ppo_continuous model exists via list_rl_models, else rl_train) →
       rt_start_session(symbol, model_id, replay=True, replay_start, replay_end,
       open_ui=True) — this starts the sim in the web app AND opens the browser to
       the Realtime tab → poll rt_session_status / visualize_session to narrate
       progress → rt_stop_session when done. Use open_buck_ui any time to let the
       user watch a tab. The rt_* tools drive the SAME session the UI shows.
  4) Accuracy review:
       get_prediction_accuracy → list_recent_predictions →
       compare_predictions_vs_actual → visualize_accuracy / visualize_predictions.

CONVENTIONS & GOTCHAS
  • Always sanity-check `get_prediction_accuracy` before presenting a forecast as
    reliable, and state the model's recent hit-rate alongside any prediction.
  • Prefer `ppo_continuous` for new RL models (continuous position sizing + LSTM
    encoder). The discrete dqn/a2c/ppo options exist only for legacy checkpoints.
  • API keys are read from the server's `.env` (OpenAI/OpenRouter for forecasts,
    INDIAN_API_KEY for live/RL data & news) — you don't pass them in tool args.
  • Tool results pass through a compression layer; inspect savings with headroom_stats.
"""


# ── Workflow prompts (exposed via prompts/list) ──────────────────────────────

def analyze_stock_prompt(symbol: str, start_date: str, end_date: str, interval: str = "1d") -> str:
    return (
        f"Analyze the NSE stock {symbol} from {start_date} to {end_date} at {interval} interval using Buck.\n\n"
        "Follow Buck's analysis workflow:\n"
        f"1. Call get_prediction_accuracy (symbol={symbol!r}) to see how reliable recent forecasts have been.\n"
        f"2. Call single_analyze (symbol={symbol!r}, start_date={start_date!r}, end_date={end_date!r}, "
        f"interval={interval!r}) for indicators, sentiment and the OHLC forecast.\n"
        "3. Optionally call visualize for a chart, or batch_analyze to compare sector peers.\n"
        "Report the forecast WITH the model's recent accuracy so the confidence is calibrated. "
        "Remember Buck is NSE-only — symbols use the .NS suffix."
    )


def train_and_simulate_prompt(symbol: str, model_id: str, start_date: str, end_date: str) -> str:
    return (
        f"Train and evaluate a reinforcement-learning trading agent for NSE stock {symbol} with Buck.\n\n"
        "Workflow:\n"
        f"1. list_rl_models — check whether {model_id!r} (or a suitable model) already exists; skip training if so.\n"
        f"2. rl_train (symbol={symbol!r}, model_id={model_id!r}, start_date={start_date!r}, "
        f"end_date={end_date!r}, algorithm='ppo_continuous', episodes=200) — this is long-running.\n"
        f"3. rl_predict (same symbol/model, a held-out date range) to backtest the equity curve.\n"
        f"4. rl_simulate (symbol={symbol!r}, model_id={model_id!r}) for the latest-snapshot BUY/HOLD/SELL signal.\n"
        "5. visualize_training on the saved session to inspect reward/equity/loss curves.\n"
        "Summarize return %, drawdown and whether the agent is trustworthy."
    )


def simulate_replay_prompt(
    symbol: str,
    model_id: str = "",
    replay_start: str = "",
    replay_end: str = "",
    interval: str = "1d",
) -> str:
    model_line = (
        f"Use model_id={model_id!r}." if model_id
        else "Pick a trained ppo_continuous model from list_rl_models (train one with rl_train if none exists)."
    )
    dates = (
        f"replay_start={replay_start!r}, replay_end={replay_end!r}"
        if (replay_start and replay_end)
        else "a recent ~30-day window (e.g. last month)"
    )
    return (
        f"Run a REPLAY trading simulation for the NSE stock {symbol} that I can watch live in the Buck web UI.\n\n"
        "Prerequisite: the Buck web app must be running (`python main.py`).\n\n"
        "Steps:\n"
        f"1. {model_line}\n"
        f"2. Call rt_start_session(symbol={symbol!r}, model_id=<model>, replay=true, "
        f"{dates}, interval={interval!r}, open_ui=true). This starts the sim inside the web app and "
        "opens my browser to the Realtime tab so I can watch the equity curve, signals and PnL update.\n"
        "3. Poll rt_session_status every few seconds and narrate progress (signal, equity, PnL, steps); "
        "use visualize_session for the equity/action chart.\n"
        "4. When the run finishes (or I ask), call rt_stop_session and summarize return %, drawdown and behaviour.\n"
        "Buck is NSE-only — keep the symbol in .NS form."
    )


def compare_peers_prompt(symbols: str, start_date: str, end_date: str, interval: str = "1d") -> str:
    return (
        f"Compare these NSE stocks with Buck: {symbols} ({start_date}→{end_date}, {interval}).\n\n"
        "1. batch_analyze across the symbols for cross-sectional indicators + forecasts.\n"
        "2. visualize_compare to overlay their price series (rebased to 100) for relative performance.\n"
        "3. get_prediction_accuracy per symbol to weight which forecasts to trust.\n"
        "Rank the names and explain the relative-strength picture. All symbols must be NSE (.NS)."
    )
