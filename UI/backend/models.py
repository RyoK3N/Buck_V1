"""
UI.backend.models
─────────────────
Pydantic request/response schemas for the Buck API.
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


# ── Requests ──────────────────────────────────────────────────────────────────

class AnalyzeRequest(BaseModel):
    symbol: str
    start_date: str = Field(..., description="YYYY-MM-DD")
    end_date: str = Field(..., description="YYYY-MM-DD")
    interval: str = "1h"
    openai_api_key: str
    indian_api_key: Optional[str] = ""
    model: Optional[str] = "gpt-4o"
    base_url: Optional[str] = None
    selected_tools: Optional[List[str]] = None


class BatchRequest(BaseModel):
    symbols: List[str]
    start_date: str = Field(..., description="YYYY-MM-DD")
    end_date: str = Field(..., description="YYYY-MM-DD")
    interval: str = "1h"
    openai_api_key: str
    indian_api_key: Optional[str] = ""
    model: Optional[str] = "gpt-4o"
    base_url: Optional[str] = None
    max_concurrent: Optional[int] = 3
    selected_tools: Optional[List[str]] = None


class VisualizeRequest(BaseModel):
    symbol: str
    start_date: str = Field(..., description="YYYY-MM-DD")
    end_date: str = Field(..., description="YYYY-MM-DD")
    interval: str = "1d"
    chart_type: str = "price_ma"
    indian_api_key: Optional[str] = ""


# ── Responses ─────────────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status: str = "ok"
    version: str = "1.4.1"


class IntervalsResponse(BaseModel):
    intervals: List[str]


class ToolsResponse(BaseModel):
    tools: List[str]


class ConfigResponse(BaseModel):
    """Current server configuration loaded from .env"""
    openai_api_key: str
    openai_base_url: Optional[str]
    chat_model: str
    indian_api_key: str


class ChartTypeInfo(BaseModel):
    id: str
    name: str
    description: str


class ChartTypesResponse(BaseModel):
    chart_types: List[ChartTypeInfo]


class VisualizeResponse(BaseModel):
    chart: Dict[str, Any]   # Plotly figure dict (data + layout)
    chart_type: str
    symbol: str
    description: str


# ── Tools Registry ────────────────────────────────────────────────────────────

class ToolInfo(BaseModel):
    id: str
    name: str
    description: str = ""


class ToolCategory(BaseModel):
    id: str
    name: str
    description: str = ""
    tools: List[ToolInfo] = []


class ToolsRegistryResponse(BaseModel):
    categories: List[ToolCategory]


# Analyze/batch responses are arbitrary dicts from Buck — return as-is
AnalyzeResponse = Dict[str, Any]
BatchResponse = Dict[str, Any]


# ── RL / Reinforcement Learning ──────────────────────────────────────────────

class RLTrainRequest(BaseModel):
    symbol: str
    start_date: str = Field(..., description="YYYY-MM-DD")
    end_date: str = Field(..., description="YYYY-MM-DD")
    interval: str = "1d"
    algorithm: str = "dqn"
    model_id: str = "default_rl_model"
    episodes: int = 50
    hidden_dim: int = 128
    learning_rate: float = 0.001
    initial_capital: float = 100000.0
    indian_api_key: Optional[str] = ""


class RLPredictRequest(BaseModel):
    symbol: str
    start_date: str = Field(..., description="YYYY-MM-DD")
    end_date: str = Field(..., description="YYYY-MM-DD")
    interval: str = "1d"
    model_id: str = "default_rl_model"
    initial_capital: float = 100000.0
    indian_api_key: Optional[str] = ""


class RLSimulateRequest(BaseModel):
    model_id: str
    symbol: str
    interval: str = "1m"
    initial_capital: float = 100000.0
    indian_api_key: Optional[str] = ""


class RLModelsResponse(BaseModel):
    models: List[Dict[str, Any]]
