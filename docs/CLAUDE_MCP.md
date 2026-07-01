# Buck MCP Server

Buck ships an MCP server (`mcp_server/`) that exposes its analysis, RL lab,
visualizer, accuracy tracking, and realtime-session operations as tools for
Claude Desktop (or any MCP client). This doc covers the full tool surface
and the available transports. For install steps, see the "Claude Desktop
(MCP)" section of the [README](../README.md#claude-desktop-mcp).

## Transports

```bash
# Claude Desktop — stdio (what install_mcp.sh registers)
python -m mcp_server.runner --transport stdio

# HTTP/SSE — for external MCP clients. Only bind beyond 127.0.0.1 if you
# really need to expose it, and put it behind auth / a reverse proxy —
# the MCP protocol itself has no built-in auth.
python -m mcp_server.runner --transport sse --host 127.0.0.1 --port 8765

# Streamable HTTP (newer MCP transport)
python -m mcp_server.runner --transport streamable-http --host 127.0.0.1 --port 8765
```

Set `MOUNT_MCP_IN_API=true` in `.env` to also mount the SSE app inside the
FastAPI backend at `/mcp-sse` (off by default).

API keys are read from the repo's `.env` (loaded automatically by the
runner) — you don't need to put secrets in the Claude Desktop config.

## Tool surface

All tools live in `mcp_server/tools.py`; the schema each one exposes to
Claude is declared in `mcp_server/registry.py` (`BUCK_TOOLS`). Grouped by
what they do:

**Analysis**
| Tool | What it does |
|------|---------------|
| `single_analyze` | Run Buck's full pipeline (indicators + sentiment + forecast) for one symbol |
| `batch_analyze` | Same, concurrently across multiple symbols |
| `list_tools_registry` | Discover available indicator tools before picking `selected_tools` |
| `list_available_intervals` | Valid `interval` values for analyze/visualize calls |
| `list_chart_types` | Valid `chart_type` values for `visualize` |

**RL lab**
| Tool | What it does |
|------|---------------|
| `rl_train` | Train a DQN/A2C/PPO/PPO-continuous model on a symbol's history |
| `rl_predict` | Run a trained model over a date range; returns signals + equity curve |
| `rl_simulate` | Latest live action from a trained model against the current market snapshot |
| `rl_ensemble_predict` | Stack multiple models (e.g. daily + hourly) into one signal |
| `list_rl_models` | List trained models on disk (check before retraining) |

**Visualizer**
| Tool | What it does |
|------|---------------|
| `visualize` | Plotly chart JSON for a symbol (candlestick, Bollinger, RSI, MACD, …) |
| `visualize_accuracy` / `visualize_predictions` / `visualize_compare` | Accuracy-tracking charts |
| `visualize_session` / `visualize_training` | Realtime-session and RL-training-run charts (d3-buck spec) |
| `list_d3_chart_types` | Valid chart names for `visualize_training` |

**Accuracy tracking**
| Tool | What it does |
|------|---------------|
| `get_prediction_accuracy` | MAE/RMSE/directional-accuracy summary for a model/symbol |
| `list_recent_predictions` | Recent predictions with actuals + evaluation metrics where available |
| `compare_predictions_vs_actual` | Predicted-vs-actual close time series for a symbol |

**Realtime sessions**
| Tool | What it does |
|------|---------------|
| `rt_start_session` / `rt_stop_session` | Start/stop a live or replay intraday session in the running web app |
| `rt_session_status` / `rt_session_history` | Poll a session's current state / per-step history |
| `start_buck_app` / `buck_app_status` | Ensure `python main.py` is running; check its health |
| `open_buck_ui` | Open the web UI in the user's browser, deep-linked to a tab/symbol |

**Context engineering**
| Tool | What it does |
|------|---------------|
| `headroom_stats` | Token/cost accounting for the headroom MCP-output compression layer |
| `headroom_reset` | Reset the tracker and clear the compression cache |

`list_training_sessions` lists past RL training runs (for `visualize_training`).

## Notes

- `rt_start_session` / `open_buck_ui` drive the **running web app**
  (`BUCK_API_URL` / `BUCK_UI_URL`, default `localhost:8000` / `localhost:5173`)
  rather than in-process state, so a session Claude starts shows up live in
  the browser tab. Only `http`/`https` values pointing at a well-formed host
  are honored for these — anything else falls back to the localhost default.
- Tool outputs are passed through the `headroom` compression layer
  (`mcp_server/context_engineering/`) when `HEADROOM_ENABLED=true`, to cut
  token usage on large results (e.g. `rt_session_history`).
- See [SECURITY.md](../SECURITY.md) for the MCP server's trust model — it
  has no built-in authentication; keep it on stdio (Claude Desktop's default)
  or behind a reverse proxy if you use HTTP/SSE.
