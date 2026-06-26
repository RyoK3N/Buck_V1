"""
mcp_server.runner
─────────────────
Stand-alone entry point.

Examples:
    # Claude Desktop (stdio)
    python -m mcp_server.runner --transport stdio

    # HTTP/SSE for external clients (bind 0.0.0.0 only when you really need
    # to expose it beyond localhost, and put it behind auth/a reverse proxy)
    python -m mcp_server.runner --transport sse --host 127.0.0.1 --port 8765
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Make the repo root importable when invoked as a script.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from dotenv import load_dotenv
load_dotenv(_REPO_ROOT / ".env")


def _init_accuracy_db() -> None:
    """Initialize the accuracy SQLite DB (same setup as the FastAPI lifespan)."""
    from agent_scripts.config import SETTINGS
    from accuracy.db import init_db
    db_path = SETTINGS.accuracy_db_path
    if not Path(db_path).is_absolute():
        db_path = str(_REPO_ROOT / db_path)
    init_db(db_path)


def main() -> None:
    p = argparse.ArgumentParser(description="Buck MCP server")
    p.add_argument("--transport", choices=["stdio", "sse", "streamable-http"], default="stdio")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8765)
    args = p.parse_args()

    _init_accuracy_db()

    from mcp_server.server import mcp

    if args.transport == "stdio":
        mcp.run(transport="stdio")
    elif args.transport == "sse":
        mcp.settings.host = args.host
        mcp.settings.port = args.port
        mcp.run(transport="sse")
    else:
        mcp.settings.host = args.host
        mcp.settings.port = args.port
        mcp.run(transport="streamable-http")


if __name__ == "__main__":
    main()
