# Wrapper Pre-Flight Checklist

Before wrapping **any** Buck MCP tool (or adding a new one) we screen it against a
small set of battle-tested patterns, drawn from
[`Totoro-jam/battle-tested-patterns`](https://github.com/Totoro-jam/battle-tested-patterns).
The reusable primitives live in
[`mcp_server/context_engineering/patterns.py`](../mcp_server/context_engineering/patterns.py)
and are composed by the middleware chain in
[`middleware.py`](../mcp_server/context_engineering/middleware.py).

The point of the layer: shrink the tokens Claude pays for (via `headroom`
compression) **without** sacrificing correctness or making the server fragile.

## The checklist

For every tool, answer these before wiring it in:

| # | Question | Pattern | Where applied |
|---|----------|---------|---------------|
| 1 | **Is the call idempotent / read-only?** If yes it can be cached. | LRU Cache + TTL | `READ_ONLY_TOOLS` set in `middleware.py` |
| 2 | **Does it hit an external data provider?** (Indian API / yfinance) If yes, rate-limit it. | Token-Bucket Rate Limiter | `DATA_FETCH_TOOLS` set in `middleware.py` |
| 3 | **Can the dependency go down or hang?** (headroom service, broker API) Guard it so we fail fast. | Circuit Breaker | `compressor.py`, data-fetch path |
| 4 | **Are failures transient?** If yes, retry with backoff before giving up. | Retry + Backoff | `retry_with_backoff` in `middleware.py` |
| 5 | **Is the tool already registered as the single source of truth?** Do not duplicate metadata. | Registry | reuse `registry.BUCK_TOOLS_BY_NAME` |
| 6 | **Can the resilience steps be composed in a fixed order?** | Middleware Chain | `wrap_tool()` in `middleware.py` |
| 7 | **Does usage need to be observed centrally?** (tokens + cost) | Observer | `USAGE` tracker in `cost_tracker.py` |
| 8 | **Is the output safe to compress?** Compression must degrade to passthrough on any error so the tool never returns garbage. | (correctness invariant) | `compress_payload()` try/except + breaker |

## Decision rules baked into the layer

- **Reads are cached, writes/long-running are not.** `single_analyze`, `rl_train`,
  `rl_predict`, `rl_simulate`, `rl_ensemble_predict`, `visualize`, `visualize_training`
  are *not* cached (they're expensive/stateful or parametrised by live data);
  the `list_*`, accuracy, and `headroom_stats` reads are.
- **Only data-fetch tools pass through the rate limiter.** Pure DB/registry reads do not.
- **Compression is best-effort.** If `headroom` is missing, the breaker is open,
  or compression raises, we return the original payload and record zero savings —
  correctness always wins over token savings.
- **One accounting path.** Both the FastMCP/Claude path (`server.py`) and the REST
  `/mcp/invoke` path (`tools.dispatch_async`) route through `wrap_tool`, so the
  `USAGE` tracker reflects every call exactly once.

## Adding a new tool

1. Add the spec to `mcp_server/registry.py` (source of truth).
2. Add the async impl to `mcp_server/tools.py` and register it in `_IMPLS`.
3. Decide cache/rate-limit by adding the name to `READ_ONLY_TOOLS` and/or
   `DATA_FETCH_TOOLS` in `middleware.py` per the checklist above.
4. Nothing else — `wrap_tool` applies the chain automatically at registration time.
