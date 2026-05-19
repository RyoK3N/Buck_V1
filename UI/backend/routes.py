"""
UI.backend.routes
──────────────────
FastAPI route handlers for all Buck endpoints.
"""

from __future__ import annotations
import logging
import os
from typing import Any, Dict

from fastapi import APIRouter, HTTPException

logger = logging.getLogger(__name__)

from .models import (
    AnalyzeRequest,
    BatchRequest,
    VisualizeRequest,
    HealthResponse,
    IntervalsResponse,
    ToolsResponse,
    ConfigResponse,
    ChartTypesResponse,
    ChartTypeInfo,
    VisualizeResponse,
    ToolInfo,
    ToolCategory,
    ToolsRegistryResponse,
)  # noqa: F401

router = APIRouter()

VALID_INTERVALS = ["1m", "2m", "5m", "15m", "30m", "60m", "90m", "1h",
                   "1d", "5d", "1wk", "1mo", "3mo"]


def _make_buck(req: AnalyzeRequest | BatchRequest):
    """Instantiate Buck with per-request API keys."""
    from agent_scripts.buck import BuckFactory
    return BuckFactory.create_production_agent(
        openai_api_key=req.openai_api_key,
        indian_api_key=req.indian_api_key or "",
        model=req.model or "gpt-4o",
        base_url=req.base_url or None,
        selected_tools=req.selected_tools,
    )


# ── Health ────────────────────────────────────────────────────────────────────

@router.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse()


# ── Config ────────────────────────────────────────────────────────────────────

@router.get("/config", response_model=ConfigResponse)
async def get_config() -> ConfigResponse:
    """Return server config loaded from .env (used by the UI to pre-fill forms)."""
    from agent_scripts.config import SETTINGS
    return ConfigResponse(
        openai_api_key=(
            SETTINGS.openai_api_key
            if SETTINGS.openai_api_key != "__placeholder__"
            else ""
        ),
        openai_base_url=SETTINGS.openai_base_url,
        chat_model=SETTINGS.chat_model,
        indian_api_key=SETTINGS.indian_api_key,
    )


# ── Meta ──────────────────────────────────────────────────────────────────────

@router.get("/intervals", response_model=IntervalsResponse)
async def intervals() -> IntervalsResponse:
    return IntervalsResponse(intervals=VALID_INTERVALS)


@router.get("/tools", response_model=ToolsResponse)
async def tools() -> ToolsResponse:
    from agent_scripts.tools import ToolFactory
    return ToolsResponse(tools=ToolFactory.get_available_tools())


@router.get("/tools-registry", response_model=ToolsRegistryResponse)
async def tools_registry() -> ToolsRegistryResponse:
    """Return all dynamically discovered tool categories."""
    from agent_scripts.tools import ToolFactory

    raw = ToolFactory.get_registry()  # {"categories": [...]}
    categories = [
        ToolCategory(
            id=cat["id"],
            name=cat["name"],
            description=cat.get("description", ""),
            tools=[ToolInfo(**t) for t in cat["tools"]],
        )
        for cat in raw["categories"]
    ]
    return ToolsRegistryResponse(categories=categories)


# ── Analysis ──────────────────────────────────────────────────────────────────

@router.post("/analyze")
async def analyze(req: AnalyzeRequest) -> Dict[str, Any]:
    os.environ["OPENAI_API_KEY"] = req.openai_api_key
    if req.indian_api_key:
        os.environ["INDIAN_API_KEY"] = req.indian_api_key

    logger.info("POST /analyze  symbol=%s  selected_tools=%s", req.symbol, req.selected_tools)
    try:
        buck = _make_buck(req)
        result = await buck.analyze_and_predict(
            symbol=req.symbol,
            start_date=req.start_date,
            end_date=req.end_date,
            interval=req.interval,
            save_results=False,
        )
        return result
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/batch")
async def batch(req: BatchRequest) -> Dict[str, Any]:
    os.environ["OPENAI_API_KEY"] = req.openai_api_key
    if req.indian_api_key:
        os.environ["INDIAN_API_KEY"] = req.indian_api_key

    try:
        buck = _make_buck(req)
        result = await buck.batch_analyze(
            symbols=req.symbols,
            start_date=req.start_date,
            end_date=req.end_date,
            interval=req.interval,
            max_concurrent=req.max_concurrent or 3,
        )
        return result
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


# ── Visualizer ────────────────────────────────────────────────────────────────

@router.get("/chart-types", response_model=ChartTypesResponse)
async def chart_types() -> ChartTypesResponse:
    from .visualizer import CHART_CATALOGUE
    return ChartTypesResponse(
        chart_types=[ChartTypeInfo(**c) for c in CHART_CATALOGUE]
    )


@router.post("/visualize", response_model=VisualizeResponse)
async def visualize(req: VisualizeRequest) -> VisualizeResponse:
    from .visualizer import fetch_df, build_chart, CHART_DESCRIPTIONS

    try:
        df = await fetch_df(
            symbol=req.symbol,
            start_date=req.start_date,
            end_date=req.end_date,
            interval=req.interval,
            indian_api_key=req.indian_api_key or "",
        )
        chart_dict = build_chart(req.chart_type, df, req.symbol)
        return VisualizeResponse(
            chart=chart_dict,
            chart_type=req.chart_type,
            symbol=req.symbol,
            description=CHART_DESCRIPTIONS.get(req.chart_type, ""),
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
