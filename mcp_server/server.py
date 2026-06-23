"""
mcp_server.server
─────────────────
FastMCP server wiring. The `mcp` instance can be:
  * run over stdio (for Claude Desktop) via `python -m mcp_server.runner`
  * mounted into the FastAPI app via `asgi_app()` (gated by MOUNT_MCP_IN_API)
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from mcp.server.fastmcp import FastMCP

from .registry import BUCK_TOOLS_BY_NAME
from .tools import _IMPLS  # noqa: F401  (imported for binding side-effect below)
from . import tools as _tools_module


mcp = FastMCP("buck")


def _register_all() -> None:
    """Register every implementation in `tools._IMPLS` against the FastMCP instance,
    using metadata from the registry. Doing this dynamically keeps the registry
    as the single source of truth."""
    for name, impl in _tools_module._IMPLS.items():
        meta = BUCK_TOOLS_BY_NAME.get(name)
        if meta is None:
            continue
        mcp.tool(name=name, description=meta.get("description", ""))(impl)


_register_all()


def asgi_app(transport: str = "sse"):
    """Return an ASGI app for mounting into FastAPI.

    Defaults to SSE; FastMCP also supports `streamable_http`.
    """
    if transport == "sse":
        return mcp.sse_app()
    if transport in ("http", "streamable_http"):
        return mcp.streamable_http_app()
    raise ValueError(f"unknown transport: {transport!r}")
