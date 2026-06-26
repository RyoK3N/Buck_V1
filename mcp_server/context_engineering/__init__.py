"""
mcp_server.context_engineering
───────────────────────────────
Context-engineering layer for the Buck MCP surface.

Goal: every MCP tool result Claude sees is run through `headroom` compression so
the conversation uses far fewer tokens, while a single process-wide tracker
accounts for tokens + estimated cost (raw vs compressed).

Before wrapping any tool we screen it against a small set of resilience /
structure patterns lifted from `battle-tested-patterns`
(https://github.com/Totoro-jam/battle-tested-patterns):

  * CircuitBreaker      — fail fast when headroom / a data provider is down
  * TokenBucketRateLimiter — cap outbound data-provider calls
  * retry_with_backoff  — ride out transient fetch / compress errors
  * LRUCacheTTL         — memoise idempotent read-only tool outputs
  * Registry            — reuse the existing single source of truth
  * Middleware Chain    — compose the above around each impl
  * Observer            — the usage tracker subscribes to every wrapped call

See `docs/WRAPPER_CHECKLIST.md` for the full pre-flight checklist.

Public surface:
  * USAGE             — the process-wide cost/token tracker singleton
  * wrap_tool         — decorate a tool impl with the middleware chain
  * compress_payload  — run a JSON payload through headroom
"""

from __future__ import annotations

from .cost_tracker import USAGE, UsageTracker
from .compressor import compress_payload, headroom_available
from .middleware import wrap_tool, READ_ONLY_TOOLS, DATA_FETCH_TOOLS

__all__ = [
    "USAGE",
    "UsageTracker",
    "compress_payload",
    "headroom_available",
    "wrap_tool",
    "READ_ONLY_TOOLS",
    "DATA_FETCH_TOOLS",
]
