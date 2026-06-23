"""
mcp_server
──────────
MCP server that exposes Buck's user-facing operations to Claude (and any
other MCP client). The same tool registry powers both this server and the
in-app `ClaudePredictor` so the two surfaces stay in lock-step.
"""

from .registry import BUCK_TOOLS  # noqa: F401
from .tools import dispatch, dispatch_async  # noqa: F401
