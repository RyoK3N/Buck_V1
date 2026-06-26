"""
mcp_server.context_engineering.patterns
─────────────────────────────────────────
Reusable resilience / structure primitives, modelled on the reference
implementations in `battle-tested-patterns`
(https://github.com/Totoro-jam/battle-tested-patterns).

These are deliberately dependency-free and individually unit-testable so the
wrapper layer can compose them (Middleware Chain) without dragging in any
framework. Each primitive maps to a named pattern in that repo:

  * CircuitBreaker          → Systems / Circuit Breaker
  * TokenBucketRateLimiter  → Systems / Rate Limiter
  * retry_with_backoff      → Systems / Retry Backoff
  * LRUCacheTTL             → Data Structures / LRU Cache (+ TTL eviction)
"""

from __future__ import annotations

import asyncio
import random
import threading
import time
from collections import OrderedDict
from typing import Any, Awaitable, Callable, Dict, Hashable, Optional, Tuple, TypeVar

T = TypeVar("T")


# ─────────────────────────────────────────────────────────────────────────────
# Circuit Breaker
# ─────────────────────────────────────────────────────────────────────────────
class CircuitOpenError(RuntimeError):
    """Raised when a call is attempted while the breaker is open."""


class CircuitBreaker:
    """Three-state circuit breaker (CLOSED → OPEN → HALF_OPEN).

    Opens after `fail_max` consecutive failures; stays open for
    `reset_timeout` seconds, then allows a single trial call (HALF_OPEN). A
    success closes it; a failure re-opens it. Thread-safe.
    """

    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"

    def __init__(self, fail_max: int = 5, reset_timeout: float = 30.0, name: str = "cb") -> None:
        self.fail_max = fail_max
        self.reset_timeout = reset_timeout
        self.name = name
        self._state = self.CLOSED
        self._fail_count = 0
        self._opened_at = 0.0
        self._lock = threading.Lock()

    @property
    def state(self) -> str:
        with self._lock:
            return self._eval_state()

    def _eval_state(self) -> str:
        if self._state == self.OPEN and (time.monotonic() - self._opened_at) >= self.reset_timeout:
            self._state = self.HALF_OPEN
        return self._state

    def _on_success(self) -> None:
        with self._lock:
            self._fail_count = 0
            self._state = self.CLOSED

    def _on_failure(self) -> None:
        with self._lock:
            self._fail_count += 1
            if self._fail_count >= self.fail_max or self._state == self.HALF_OPEN:
                self._state = self.OPEN
                self._opened_at = time.monotonic()

    def _guard(self) -> None:
        with self._lock:
            if self._eval_state() == self.OPEN:
                raise CircuitOpenError(f"circuit {self.name!r} is open")

    async def call_async(self, fn: Callable[..., Awaitable[T]], *args: Any, **kwargs: Any) -> T:
        self._guard()
        try:
            result = await fn(*args, **kwargs)
        except Exception:
            self._on_failure()
            raise
        self._on_success()
        return result

    def call(self, fn: Callable[..., T], *args: Any, **kwargs: Any) -> T:
        self._guard()
        try:
            result = fn(*args, **kwargs)
        except Exception:
            self._on_failure()
            raise
        self._on_success()
        return result


# ─────────────────────────────────────────────────────────────────────────────
# Token-bucket rate limiter
# ─────────────────────────────────────────────────────────────────────────────
class TokenBucketRateLimiter:
    """Classic token bucket. `capacity` tokens, refilled at `refill_rate`/sec.

    `try_acquire()` is non-blocking; `acquire_async()` waits for a token.
    """

    def __init__(self, capacity: float = 10.0, refill_rate: float = 5.0) -> None:
        self.capacity = float(capacity)
        self.refill_rate = float(refill_rate)
        self._tokens = float(capacity)
        self._last = time.monotonic()
        self._lock = threading.Lock()

    def _refill(self) -> None:
        now = time.monotonic()
        elapsed = now - self._last
        self._last = now
        self._tokens = min(self.capacity, self._tokens + elapsed * self.refill_rate)

    def try_acquire(self, tokens: float = 1.0) -> bool:
        with self._lock:
            self._refill()
            if self._tokens >= tokens:
                self._tokens -= tokens
                return True
            return False

    async def acquire_async(self, tokens: float = 1.0, timeout: float = 10.0) -> bool:
        deadline = time.monotonic() + timeout
        while True:
            if self.try_acquire(tokens):
                return True
            if time.monotonic() >= deadline:
                return False
            # Sleep roughly until one token is available.
            await asyncio.sleep(max(0.01, tokens / max(self.refill_rate, 1e-6)))


# ─────────────────────────────────────────────────────────────────────────────
# Retry with exponential backoff + jitter
# ─────────────────────────────────────────────────────────────────────────────
async def retry_with_backoff(
    fn: Callable[..., Awaitable[T]],
    *args: Any,
    attempts: int = 3,
    base_delay: float = 0.25,
    max_delay: float = 4.0,
    jitter: float = 0.1,
    retry_on: Tuple[type, ...] = (Exception,),
    no_retry_on: Tuple[type, ...] = (),
    **kwargs: Any,
) -> T:
    """Call `fn` up to `attempts` times with exponential backoff + jitter.

    `no_retry_on` exceptions (e.g. CircuitOpenError) are raised immediately.
    """
    last_exc: Optional[BaseException] = None
    for i in range(attempts):
        try:
            return await fn(*args, **kwargs)
        except no_retry_on:
            raise
        except retry_on as exc:  # noqa: BLE001
            last_exc = exc
            if i == attempts - 1:
                break
            delay = min(max_delay, base_delay * (2 ** i))
            delay += random.uniform(0, jitter)
            await asyncio.sleep(delay)
    assert last_exc is not None
    raise last_exc


# ─────────────────────────────────────────────────────────────────────────────
# LRU cache with TTL
# ─────────────────────────────────────────────────────────────────────────────
class LRUCacheTTL:
    """Bounded LRU cache with per-entry TTL.

    Used to memoise idempotent, read-only tool outputs (lists, accuracy reads,
    visualisations) so repeated Claude calls within a session are free. Stores
    the already-compressed payload so cache hits also skip recompression.
    """

    def __init__(self, maxsize: int = 128, ttl: float = 300.0) -> None:
        self.maxsize = maxsize
        self.ttl = ttl
        self._store: "OrderedDict[Hashable, Tuple[float, Any]]" = OrderedDict()
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0

    def get(self, key: Hashable) -> Tuple[bool, Any]:
        """Return (found, value). `found` is False on miss or expiry."""
        with self._lock:
            item = self._store.get(key)
            if item is None:
                self.misses += 1
                return False, None
            expires_at, value = item
            if time.monotonic() >= expires_at:
                del self._store[key]
                self.misses += 1
                return False, None
            self._store.move_to_end(key)
            self.hits += 1
            return True, value

    def set(self, key: Hashable, value: Any) -> None:
        with self._lock:
            if key in self._store:
                self._store.move_to_end(key)
            self._store[key] = (time.monotonic() + self.ttl, value)
            while len(self._store) > self.maxsize:
                self._store.popitem(last=False)

    def clear(self) -> None:
        with self._lock:
            self._store.clear()

    def stats(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "size": len(self._store),
                "maxsize": self.maxsize,
                "ttl": self.ttl,
                "hits": self.hits,
                "misses": self.misses,
            }
