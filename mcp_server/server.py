"""
mcp_server.server
─────────────────
FastMCP server wiring. The `mcp` instance can be:
  * run over stdio (for Claude Desktop) via `python -m mcp_server.runner`
  * mounted into the FastAPI app via `asgi_app()` (gated by MOUNT_MCP_IN_API)
"""

from __future__ import annotations

from mcp.server.fastmcp import FastMCP

from .registry import BUCK_TOOLS_BY_NAME
from . import tools as _tools_module
from .instructions import (
    SERVER_INSTRUCTIONS,
    analyze_stock_prompt,
    train_and_simulate_prompt,
    compare_peers_prompt,
)


# `instructions` is surfaced to MCP clients as top-level context (what Buck is,
# that it's NSE/Indian-only, symbol/date conventions, and the recommended
# workflows) so the model drives Buck correctly instead of guessing.
mcp = FastMCP("buck", instructions=SERVER_INSTRUCTIONS)


def _register_prompts() -> None:
    """Expose Buck's standard workflows as MCP prompts (prompts/list)."""
    mcp.prompt(
        name="analyze_stock",
        description="Buck's analysis + forecast workflow for one NSE stock (.NS).",
    )(analyze_stock_prompt)
    mcp.prompt(
        name="train_and_simulate",
        description="Buck's RL lifecycle for an NSE stock: train → backtest → live signal → visualize.",
    )(train_and_simulate_prompt)
    mcp.prompt(
        name="compare_peers",
        description="Compare several NSE stocks with batch analysis + a rebased overlay chart.",
    )(compare_peers_prompt)


def _register_all() -> None:
    """Register every implementation in `tools._IMPLS` against the FastMCP instance,
    using metadata from the registry. Doing this dynamically keeps the registry
    as the single source of truth.

    Each impl is wrapped by the context-engineering middleware (headroom
    compression + battle-tested-patterns resilience) so every tool result Claude
    sees is compressed and accounted for in the `USAGE` tracker. We register the
    MCP-facing (string-returning) wrapper so the compressed payload reaches Claude
    as a single content block — not duplicated into `structuredContent`.
    """
    for name in _tools_module._IMPLS:
        meta = BUCK_TOOLS_BY_NAME.get(name)
        if meta is None:
            continue
        wrapped = _tools_module.get_mcp_wrapped(name)
        # structured_output=False: the wrapper already returns the compact
        # (headroom-compressed) payload as text; without this FastMCP would also
        # emit it as `structuredContent`, sending the payload to Claude twice.
        mcp.tool(
            name=name,
            description=meta.get("description", ""),
            structured_output=False,
        )(wrapped)


_register_all()
_register_prompts()


def asgi_app(transport: str = "sse"):
    """Return an ASGI app for mounting into FastAPI.

    Defaults to SSE; FastMCP also supports `streamable_http`.
    """
    if transport == "sse":
        return mcp.sse_app()
    if transport in ("http", "streamable_http"):
        return mcp.streamable_http_app()
    raise ValueError(f"unknown transport: {transport!r}")
