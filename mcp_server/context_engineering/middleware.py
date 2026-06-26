"""
mcp_server.context_engineering.middleware
────────────────────────────────────────────
The Middleware Chain that wraps every Buck MCP tool implementation.

`wrap_tool(name, impl)` returns an async callable with the same signature as
`impl` that, in order:

  1. (read-only tools) checks an LRU+TTL cache keyed by (name, args);
  2. (data-fetch tools) passes through a token-bucket rate limiter;
  3. runs the impl inside a per-tool circuit breaker + retry-with-backoff;
  4. compresses the result via headroom;
  5. records tokens + cost into the `USAGE` observer;
  6. returns a backward-compatible envelope.

The envelope is:
    {"_headroom": {"compressed": bool, "tokens_raw": N, "tokens_compressed": M,
                   "tokens_saved": K, "cache": bool, "reason": str},
     "data": <compressed-text-or-original-payload>}

Callers that don't care about compression can simply read ``["data"]``; when
compression is off/unavailable, ``data`` is the original dict unchanged.
"""

from __future__ import annotations

import functools
import inspect
import json
from typing import Any, Awaitable, Callable, Dict, Optional

from .compressor import compress_payload
from .cost_tracker import USAGE
from .patterns import (
    CircuitBreaker,
    LRUCacheTTL,
    TokenBucketRateLimiter,
    retry_with_backoff,
    CircuitOpenError,
)

ToolImpl = Callable[..., Awaitable[Dict[str, Any]]]

# ── Per-tool classification (see docs/WRAPPER_CHECKLIST.md) ──────────────────
# Idempotent, cheap reads → safe to cache.
READ_ONLY_TOOLS = {
    "list_tools_registry",
    "list_available_intervals",
    "list_chart_types",
    "list_rl_models",
    "get_prediction_accuracy",
    "list_recent_predictions",
    "compare_predictions_vs_actual",
    "visualize_accuracy",
    "visualize_predictions",
    "headroom_stats",
    "list_training_sessions",
    "list_d3_chart_types",
    "rt_session_status",
    "rt_session_history",
}

# Tools that hit an external data provider → throttle outbound calls.
DATA_FETCH_TOOLS = {
    "single_analyze",
    "batch_analyze",
    "rl_train",
    "rl_predict",
    "rl_simulate",
    "rl_ensemble_predict",
    "visualize",
    "visualize_compare",
}

# Tools whose results must never be cached (stateful / mutating).
NEVER_CACHE = {"headroom_reset"}

# Shared infrastructure.
_CACHE = LRUCacheTTL(maxsize=256, ttl=300.0)
_RATE_LIMITER = TokenBucketRateLimiter(capacity=12.0, refill_rate=6.0)
_BREAKERS: Dict[str, CircuitBreaker] = {}


def _breaker_for(name: str) -> CircuitBreaker:
    cb = _BREAKERS.get(name)
    if cb is None:
        cb = CircuitBreaker(fail_max=5, reset_timeout=30.0, name=name)
        _BREAKERS[name] = cb
    return cb


def _cache_key(name: str, args: Dict[str, Any]) -> str:
    try:
        return name + "|" + json.dumps(args, sort_keys=True, default=str)
    except Exception:
        return name + "|" + repr(sorted(args.items()))


def _envelope(data: Any, stats: Dict[str, Any], *, cache: bool) -> Dict[str, Any]:
    tokens_saved = max(0, stats.get("tokens_raw", 0) - stats.get("tokens_compressed", 0))
    return {
        "_headroom": {
            "compressed": bool(stats.get("compressed", False)),
            "tokens_raw": stats.get("tokens_raw", 0),
            "tokens_compressed": stats.get("tokens_compressed", 0),
            "tokens_saved": tokens_saved,
            "cache": cache,
            "reason": stats.get("reason", ""),
        },
        "data": data,
    }


def wrap_tool(name: str, impl: ToolImpl) -> ToolImpl:
    """Wrap a raw tool impl with the resilience + compression middleware chain."""

    cacheable = name in READ_ONLY_TOOLS and name not in NEVER_CACHE
    rate_limited = name in DATA_FETCH_TOOLS

    @functools.wraps(impl)
    async def _wrapped(**kwargs: Any) -> Dict[str, Any]:
        args = dict(kwargs)

        # 1) cache lookup (read-only tools only)
        key: Optional[str] = None
        if cacheable:
            key = _cache_key(name, args)
            found, cached = _CACHE.get(key)
            if found:
                stats = cached["_headroom"]
                USAGE.record(
                    name,
                    stats.get("tokens_raw", 0),
                    stats.get("tokens_compressed", 0),
                    compressed=stats.get("compressed", False),
                    cache_hit=True,
                )
                # Re-stamp the cache flag without mutating the stored object.
                return _envelope(cached["data"], stats, cache=True)

        # 2) rate limit outbound data calls
        if rate_limited:
            await _RATE_LIMITER.acquire_async(1.0, timeout=15.0)

        # 3) run impl inside circuit breaker + retry-with-backoff
        breaker = _breaker_for(name)

        async def _call() -> Dict[str, Any]:
            return await breaker.call_async(impl, **args)

        result = await retry_with_backoff(
            _call,
            attempts=3,
            base_delay=0.3,
            max_delay=3.0,
            no_retry_on=(CircuitOpenError,),
        )

        # 4) compress + 5) record
        payload, stats = compress_payload(result)
        USAGE.record(
            name,
            stats.get("tokens_raw", 0),
            stats.get("tokens_compressed", 0),
            compressed=stats.get("compressed", False),
        )

        envelope = _envelope(payload, stats, cache=False)
        if cacheable and key is not None:
            _CACHE.set(key, envelope)
        return envelope

    # CRITICAL: expose the original impl's signature so FastMCP builds the real
    # input schema (symbol, start_date, …) instead of a single opaque `kwargs`
    # field. The wrapper keeps its **kwargs body, so FastMCP can still call it
    # with the validated arguments.
    #
    # The impls use `from __future__ import annotations`, so their annotations are
    # PEP 563 *strings*. We resolve them to real type objects in the impl's own
    # module namespace (eval_str=True) and pin both __signature__ and
    # __annotations__ — otherwise FastMCP would try to eval those strings in this
    # middleware module's globals (which lack `List`, etc.) and fail to build the
    # pydantic argument model.
    try:
        sig = inspect.signature(impl, eval_str=True)
    except (ValueError, TypeError, NameError):
        sig = inspect.signature(impl)
    _wrapped.__signature__ = sig  # type: ignore[attr-defined]
    resolved = {
        p_name: p.annotation
        for p_name, p in sig.parameters.items()
        if p.annotation is not inspect.Parameter.empty
    }
    if sig.return_annotation is not inspect.Signature.empty:
        resolved["return"] = sig.return_annotation
    _wrapped.__annotations__ = resolved
    return _wrapped


def cache_stats() -> Dict[str, Any]:
    return _CACHE.stats()


def clear_cache() -> None:
    _CACHE.clear()
