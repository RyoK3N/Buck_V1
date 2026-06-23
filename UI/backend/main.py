"""
UI.backend.main
────────────────
FastAPI application entry point.

Run from the repo root:
    uvicorn UI.backend.main:app --reload --port 8000
"""

from __future__ import annotations
import os
import sys
from pathlib import Path
from contextlib import asynccontextmanager

# ── Path setup ────────────────────────────────────────────────────────────────
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# ── Load .env FIRST so real keys end up in os.environ ─────────────────────────
# agent_scripts.config also calls load_dotenv() but by then the values are
# already set, so this call here takes precedence (no-override behaviour).
from dotenv import load_dotenv
load_dotenv(_REPO_ROOT / ".env")

# Only set a placeholder if the key is still absent after .env load.
# This allows the server to start without a key; every real request
# supplies keys from the request body.
if not os.environ.get("OPENAI_API_KEY"):
    os.environ["OPENAI_API_KEY"] = "__placeholder__"

# ── App ───────────────────────────────────────────────────────────────────────
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .routes import router


@asynccontextmanager
async def lifespan(_app: FastAPI):
    # Accuracy DB — always initialized so /analyze can record predictions
    from agent_scripts.config import SETTINGS, LOGGER
    from accuracy.db import init_db
    from accuracy import scheduler as accuracy_scheduler

    db_path = SETTINGS.accuracy_db_path
    if not Path(db_path).is_absolute():
        db_path = str(_REPO_ROOT / db_path)
    init_db(db_path)
    LOGGER.info("Accuracy DB initialized at %s", db_path)

    scheduler = None
    if SETTINGS.accuracy_scheduler_enabled:
        try:
            scheduler = accuracy_scheduler.start(
                poll_interval_minutes=SETTINGS.accuracy_poll_interval_minutes,
                exchange=SETTINGS.market_exchange,
            )
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("Accuracy scheduler failed to start: %s", exc)

    try:
        yield
    finally:
        accuracy_scheduler.shutdown(scheduler)


app = FastAPI(
    title="Buck API",
    description="AI-powered stock analysis and prediction",
    version="1.4.1",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"^http://localhost:\d+$",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)

# Optional: mount the FastMCP SSE app at /mcp for browser-based MCP clients.
# Disabled by default; flip MOUNT_MCP_IN_API=true in .env to enable.
from agent_scripts.config import SETTINGS as _SETTINGS  # noqa: E402
if _SETTINGS.mount_mcp_in_api:
    try:
        from mcp_server.server import asgi_app as _mcp_asgi
        app.mount("/mcp-sse", _mcp_asgi(transport="sse"))
    except Exception as _exc:  # noqa: BLE001
        import logging
        logging.getLogger(__name__).warning("MCP mount skipped: %s", _exc)
