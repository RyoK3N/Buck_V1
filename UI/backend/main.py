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
    yield


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
