"""
UI.backend.ratelimit
─────────────────────
Per-client sliding-window rate limiting for the expensive endpoints.

Every LLM-calling or model-training endpoint (/analyze, /batch, /claude/*,
/rl/*, /mcp/invoke) turns one HTTP request into real spend: an OpenAI or
Anthropic API call, or minutes of LSTM/RL training CPU. Without backpressure
a single client can run the bill or the box into the ground. This applies a
simple in-process sliding window per client IP to just those paths — cheap
read-only endpoints (/health, /config, /accuracy/*, /rt/status, ...) are
never throttled.

In-process state only: if you run multiple workers, each worker enforces the
limit independently (good enough for the localhost trust model; put a real
gateway limiter in front for anything multi-instance).
"""

from __future__ import annotations

import threading
import time
from collections import defaultdict, deque
from typing import Deque, Dict

from fastapi import Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

# Path prefixes that trigger LLM spend or heavy training work.
_EXPENSIVE_PREFIXES = (
    "/analyze",
    "/batch",
    "/claude/",
    "/rl/",
    "/mcp/invoke",
    "/accuracy/evaluate-now",
)

_WINDOW_SECONDS = 60.0


class RateLimitMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, limit_per_minute: int):
        super().__init__(app)
        self._limit = limit_per_minute
        self._lock = threading.Lock()
        self._hits: Dict[str, Deque[float]] = defaultdict(deque)

    async def dispatch(self, request: Request, call_next):
        path = request.url.path
        if request.method == "OPTIONS" or not path.startswith(_EXPENSIVE_PREFIXES):
            return await call_next(request)

        client = request.client.host if request.client else "unknown"
        now = time.monotonic()
        with self._lock:
            window = self._hits[client]
            while window and now - window[0] > _WINDOW_SECONDS:
                window.popleft()
            if len(window) >= self._limit:
                retry_after = max(1, int(_WINDOW_SECONDS - (now - window[0])) + 1)
                return JSONResponse(
                    status_code=429,
                    content={
                        "detail": (
                            f"Rate limit exceeded: {self._limit} requests/minute to "
                            "analysis/training endpoints. Retry later or raise "
                            "BUCK_RATE_LIMIT_PER_MINUTE in .env."
                        )
                    },
                    headers={"Retry-After": str(retry_after)},
                )
            window.append(now)

        return await call_next(request)


def maybe_add_rate_limit(app, limit_per_minute: int) -> None:
    """Attach the limiter unless disabled (limit <= 0)."""
    if limit_per_minute > 0:
        app.add_middleware(RateLimitMiddleware, limit_per_minute=limit_per_minute)
