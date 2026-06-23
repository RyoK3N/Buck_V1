# Claude MCP + Live Accuracy

This guide covers the Claude integration shipped on top of Buck:

1. **MCP server** — exposes Buck's user-facing operations (Single Analyze, Batch, RL Lab, Visualizer, accuracy introspection) as tools for any MCP client (Claude Desktop, Claude Code, custom).
2. **Claude predictor** — an in-app predictor that uses the Anthropic SDK with native tool use to orchestrate those same operations and emit a forecast.
3. **Real-time accuracy** — every forecast is recorded to SQLite; a background scheduler polls market data, reconciles predictions vs actuals, and streams updates to the new **Claude** tab in the web UI.

## Setup

1. Install new dependencies:
   ```
   pip install -r requirements.txt
   ```
2. Add Anthropic credentials to `.env` (or supply per-request via the UI sidebar):
   ```
   ANTHROPIC_API_KEY=sk-ant-…
   CLAUDE_MODEL=claude-opus-4-5
   ```
3. Optionally tune the accuracy scheduler:
   ```
   ACCURACY_POLL_INTERVAL_MINUTES=5
   ACCURACY_DB_PATH=accuracy/buck_accuracy.db
   MARKET_TZ=Asia/Kolkata
   MARKET_EXCHANGE=NSE
   ```

The `accuracy/` SQLite database is created automatically on first FastAPI startup.

## Running

### Web UI (everything in one process)

```
python main.py
```

The FastAPI lifespan initialises the accuracy DB and (when `ACCURACY_SCHEDULER_ENABLED=true`) starts the APScheduler intraday + EOD jobs. The new **Claude** tab in the frontend gives you:

- **Chat** — open-ended chat with Claude, who calls Buck's MCP tools and shows the trace.
- **Predictions** — every forecast Buck has produced, joined with its actual once available.
- **Accuracy Dashboard** — live rolling MAE / directional-accuracy tiles, per-model time-series chart, and Claude tool-contribution heatmap.
- **MCP Tools** — inspect and test each MCP tool with an auto-generated form.
- **Settings** — Claude key + model + helpful pointers.

### Standalone MCP server (for Claude Desktop)

```
python -m mcp_server.runner --transport stdio
```

Then in Claude Desktop's `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "buck": {
      "command": "python",
      "args": ["-m", "mcp_server.runner", "--transport", "stdio"],
      "cwd": "/absolute/path/to/Buck_V1",
      "env": {
        "OPENAI_API_KEY": "sk-...",
        "INDIAN_API_KEY": ""
      }
    }
  }
}
```

For HTTP/SSE transport (used by browser-based MCP clients):

```
python -m mcp_server.runner --transport sse --port 8765
```

You can also mount the MCP server inside the FastAPI process at `/mcp-sse` by setting `MOUNT_MCP_IN_API=true` in `.env`.

## MCP tool surface

All tools live in `mcp_server/registry.py` (single source of truth) and are implemented in `mcp_server/tools.py`. They mirror Buck's existing user-facing operations 1:1, plus three accuracy-introspection tools.

| Tool | Wraps |
|---|---|
| `single_analyze` | `POST /analyze` |
| `batch_analyze` | `POST /batch` |
| `list_tools_registry`, `list_available_intervals`, `list_chart_types` | `GET /tools-registry`, `GET /intervals`, `GET /chart-types` |
| `rl_train`, `rl_predict`, `rl_simulate`, `list_rl_models` | `POST /rl/*` + `GET /rl/models` |
| `visualize` | `POST /visualize` |
| `get_prediction_accuracy`, `list_recent_predictions`, `compare_predictions_vs_actual` | accuracy DB |

When new algorithms are added under `tools/ml/`, `tools/utility/`, `tools/web/` they flow into `single_analyze`/`batch_analyze` automatically via Buck's `ToolFactory` auto-discovery — no MCP changes required. To add a brand-new top-level capability, append an entry to `BUCK_TOOLS` and a coroutine to `mcp_server/tools.py`.

## How Claude makes predictions

`ClaudePredictor` (`agent_scripts/claude_predictor.py`) drives an iterative tool-use loop:

1. Receives the seed analysis Buck already computed.
2. Sends Anthropic a system prompt framing Claude as a research analyst and exposes `BUCK_TOOLS` as native tools.
3. Claude typically: calls `get_prediction_accuracy` → `single_analyze` → optionally `batch_analyze` for peers or `rl_predict` for a directional check → reasons → emits a JSON forecast wrapped in `` ```json ``` ``.
4. The full tool-call trace is stored in `predictions.request_metadata_json` so the dashboard can attribute correct/incorrect outcomes to specific tools.

## Real-time accuracy

- Every prediction (from any model) is written to `predictions` by the telemetry hook in `agent_scripts/buck.py`.
- `accuracy/scheduler.py` runs two jobs on `AsyncIOScheduler`:
  - **Intraday**: every `ACCURACY_POLL_INTERVAL_MINUTES` during market hours — polls latest prices via yfinance, upserts `actuals`, reconciles open predictions, broadcasts to `WS /accuracy/ws`.
  - **EOD**: shortly after market close — pulls final daily bars, finalises evaluations, marks predictions as `evaluated`.
- The live tiles in the Accuracy Dashboard are driven by the WebSocket; the per-model summary tables read from `/accuracy/summary` and `/accuracy/timeseries`.

You can force a poll + reconcile from the UI ("Evaluate now" button) or via:

```
curl -X POST http://localhost:8000/accuracy/evaluate-now -H 'content-type: application/json' -d '{"is_final": false}'
```

## Cost & safety notes

- `ClaudePredictor` enforces a per-session token budget (`CLAUDE_SESSION_TOKEN_BUDGET`, default 200k). When exceeded it fails loud — no silent fallback to OpenAI.
- Hard cap of `max_tokens=4096` per Anthropic call and `CLAUDE_MAX_ITERATIONS` (default 10) tool-use iterations.
- MCP tools read API keys from `os.environ` if not provided in args, so the MCP server inherits `.env`.

## Troubleshooting

- **"ANTHROPIC_API_KEY required"** — set it in `.env` or paste into the sidebar Anthropic field.
- **Empty Accuracy Dashboard** — no predictions in the DB yet. Run a Single Analyze or `POST /claude/predict` first.
- **Live tile not updating** — outside market hours by design. The "Evaluate now" button bypasses the schedule.
- **MCP server fails on Claude Desktop start** — double-check `cwd` in `claude_desktop_config.json` points to the Buck repo root.
