"""
Tests for the context-engineering layer: the battle-tested-patterns primitives,
the cost/token tracker, and the wrap_tool middleware (compression envelope +
caching + accounting). Network-free.
"""

from __future__ import annotations

import asyncio
import time

import pytest

from mcp_server.context_engineering.patterns import (
    CircuitBreaker,
    CircuitOpenError,
    LRUCacheTTL,
    TokenBucketRateLimiter,
    retry_with_backoff,
)
from mcp_server.context_engineering.cost_tracker import UsageTracker
from mcp_server.context_engineering.middleware import wrap_tool


# ── Circuit breaker ──────────────────────────────────────────────────────────
def test_circuit_breaker_opens_after_failures():
    cb = CircuitBreaker(fail_max=3, reset_timeout=60.0, name="t")

    def boom():
        raise ValueError("nope")

    for _ in range(3):
        with pytest.raises(ValueError):
            cb.call(boom)
    assert cb.state == CircuitBreaker.OPEN
    with pytest.raises(CircuitOpenError):
        cb.call(boom)


def test_circuit_breaker_half_open_recovers():
    cb = CircuitBreaker(fail_max=1, reset_timeout=0.05, name="t")
    with pytest.raises(ValueError):
        cb.call(lambda: (_ for _ in ()).throw(ValueError()))
    assert cb.state == CircuitBreaker.OPEN
    time.sleep(0.06)
    assert cb.state == CircuitBreaker.HALF_OPEN
    assert cb.call(lambda: 42) == 42
    assert cb.state == CircuitBreaker.CLOSED


# ── Rate limiter ─────────────────────────────────────────────────────────────
def test_token_bucket_limits_then_refills():
    rl = TokenBucketRateLimiter(capacity=2, refill_rate=100.0)
    assert rl.try_acquire()
    assert rl.try_acquire()
    assert not rl.try_acquire()  # bucket empty
    time.sleep(0.05)
    assert rl.try_acquire()  # refilled


# ── Retry with backoff ───────────────────────────────────────────────────────
def test_retry_succeeds_after_transient_failures():
    calls = {"n": 0}

    async def flaky():
        calls["n"] += 1
        if calls["n"] < 3:
            raise RuntimeError("transient")
        return "ok"

    out = asyncio.run(retry_with_backoff(flaky, attempts=5, base_delay=0.001))
    assert out == "ok"
    assert calls["n"] == 3


def test_retry_does_not_retry_excluded():
    async def boom():
        raise CircuitOpenError("open")

    with pytest.raises(CircuitOpenError):
        asyncio.run(retry_with_backoff(boom, attempts=5, base_delay=0.001, no_retry_on=(CircuitOpenError,)))


# ── LRU cache TTL ────────────────────────────────────────────────────────────
def test_lru_cache_hit_miss_and_ttl():
    c = LRUCacheTTL(maxsize=2, ttl=0.05)
    c.set("a", 1)
    found, val = c.get("a")
    assert found and val == 1
    time.sleep(0.06)
    found, _ = c.get("a")
    assert not found  # expired


def test_lru_cache_evicts_oldest():
    c = LRUCacheTTL(maxsize=2, ttl=100)
    c.set("a", 1)
    c.set("b", 2)
    c.get("a")  # touch a so b is the LRU
    c.set("c", 3)  # evicts b
    assert c.get("b")[0] is False
    assert c.get("a")[0] is True
    assert c.get("c")[0] is True


# ── Cost tracker ─────────────────────────────────────────────────────────────
def test_usage_tracker_accumulates_and_costs():
    t = UsageTracker(price_per_mtok=10.0)
    t.record("toolA", tokens_raw=1000, tokens_compressed=200)
    t.record("toolA", tokens_raw=500, tokens_compressed=100)
    snap = t.snapshot()
    assert snap["calls"] == 2
    assert snap["tokens_raw"] == 1500
    assert snap["tokens_saved"] == 1200
    # 1200 tokens @ $10/M = $0.012
    assert snap["est_cost_saved_usd"] == pytest.approx(0.012, rel=1e-3)
    assert snap["per_tool"]["toolA"]["calls"] == 2


# ── Middleware (wrap_tool) ───────────────────────────────────────────────────
def test_wrap_tool_envelope_and_caching():
    calls = {"n": 0}

    async def impl(**kwargs):
        calls["n"] += 1
        return {"value": kwargs.get("x", 0) * 2}

    # list_rl_models is in READ_ONLY_TOOLS, but use a real read-only name to hit cache.
    wrapped = wrap_tool("list_rl_models", impl)

    env1 = asyncio.run(wrapped(x=3))
    assert "_headroom" in env1 and "data" in env1
    assert env1["data"]["value"] == 6
    # second identical call should be served from cache (impl not invoked again)
    env2 = asyncio.run(wrapped(x=3))
    assert env2["_headroom"]["cache"] is True
    assert calls["n"] == 1
