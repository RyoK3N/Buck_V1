"""
UI.backend._env_scope
───────────────────────
A handful of downstream modules (mcp_server.tools' fallback key resolution,
tools.rl.rl_tool's data fetchers) read API keys straight from
``os.environ`` instead of accepting them as call arguments. Until those are
refactored to take explicit parameters, request handlers that need to pass a
per-request key through to them have to set process-wide environment
variables.

Setting ``os.environ[...]`` directly from a request handler and leaving it
set is unsafe: the value leaks into every other concurrent/subsequent
request on the same process until something overwrites it again. This
module scopes that mutation to the lifetime of a single request and
serializes access so concurrent requests can't interleave with each other's
credentials.
"""

from __future__ import annotations

import asyncio
import os
from contextlib import asynccontextmanager
from typing import AsyncIterator, Optional

# Serializes all scoped-env mutations so two concurrent requests can never
# observe (or stomp on) each other's temporarily-set keys.
_LOCK = asyncio.Lock()

_UNSET = object()


@asynccontextmanager
async def temporary_env(**values: Optional[str]) -> AsyncIterator[None]:
    """Set the given environment variables for the duration of the `async with`
    block, then restore whatever was there before (or unset it if it wasn't
    set). Falsy values are skipped (nothing to set, nothing to restore).

    Usage:
        async with temporary_env(INDIAN_API_KEY=req.indian_api_key):
            ...do work that reads os.environ["INDIAN_API_KEY"]...
    """
    to_set = {k: v for k, v in values.items() if v}
    if not to_set:
        yield
        return

    async with _LOCK:
        previous = {k: os.environ.get(k, _UNSET) for k in to_set}
        try:
            os.environ.update(to_set)
            yield
        finally:
            for k, prev in previous.items():
                if prev is _UNSET:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = prev
