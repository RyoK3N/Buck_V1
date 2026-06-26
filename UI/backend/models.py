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
    algorithm: str = Field("ppo_continuous", description="dqn | ppo | a2c | ppo_continuous")
    model_id: str = "default_rl_model"
    episodes: int = 200          # bumped from 50 — 50 is too few for convergence
    hidden_dim: int = 128
    learning_rate: float = 3e-4  # PPO-standard; was 1e-3 (DQN default)
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


class RLEnsembleModel(BaseModel):
    """One entry in an ensemble spec."""
    model_id: str
    interval: Optional[str] = None
    weight: float = 1.0


class RLEnsembleRequest(BaseModel):
    """Multi-timeframe ensemble inference."""
    symbol: str
    start_date: str = Field(..., description="YYYY-MM-DD")
    end_date: str = Field(..., description="YYYY-MM-DD")
    models: List[RLEnsembleModel] = Field(..., description="At least one model")
    fallback_interval: str = "1d"
    indian_api_key: Optional[str] = ""


class RLModelsResponse(BaseModel):
    models: List[Dict[str, Any]]


# ── Accuracy tracking ────────────────────────────────────────────────────────

class PredictionRow(BaseModel):
    """A row from the predictions table joined with its evaluation (if any)."""
    id: int
    symbol: str
    model: str
    target_date: str
    predicted_open: Optional[float] = None
    predicted_high: Optional[float] = None
    predicted_low: Optional[float] = None
    predicted_close: Optional[float] = None
    confidence: Optional[float] = None
    reasoning: Optional[str] = None
    created_at: str
    status: str
    # Evaluation fields (may be null when prediction is unevaluated)
    actual_open: Optional[float] = None
    actual_high: Optional[float] = None
    actual_low: Optional[float] = None
    actual_close: Optional[float] = None
    mae: Optional[float] = None
    rmse: Optional[float] = None
    directional_correct: Optional[int] = None
    error_pct: Optional[float] = None
    is_intraday: Optional[int] = None
    evaluated_at: Optional[str] = None


class PredictionsResponse(BaseModel):
    predictions: List[PredictionRow]


class ModelSummary(BaseModel):
    model: str
    n: int
    mae: Optional[float] = None
    rmse: Optional[float] = None
    directional_accuracy: Optional[float] = None
    avg_error_pct: Optional[float] = None


class AccuracySummaryResponse(BaseModel):
    window_days: Optional[int] = None
    summaries: List[ModelSummary]


class AccuracyPoint(BaseModel):
    date: str
    model: str
    mae: Optional[float] = None
    directional_accuracy: Optional[float] = None
    n: int


class AccuracyTimeseriesResponse(BaseModel):
    points: List[AccuracyPoint]


class ToolContributionRow(BaseModel):
    tool: str
    correct: int
    incorrect: int


class ToolContributionResponse(BaseModel):
    model: str
    rows: List[ToolContributionRow]


class LiveAccuracyEntry(BaseModel):
    model: str
    symbol: Optional[str] = None
    mae_pct: Optional[float] = None
    directional_accuracy_pct: Optional[float] = None
    n: int = 0
    updated_at: Optional[str] = None


class LiveAccuracyResponse(BaseModel):
    entries: List[LiveAccuracyEntry]


class EvaluateNowRequest(BaseModel):
    """Force an immediate poll + reconcile (useful for tests / on-demand refresh)."""
    is_final: bool = False
    symbol: Optional[str] = None  # if set, only this symbol; else all open


class EvaluateNowResponse(BaseModel):
    polled: int
    evaluated: int


# ── MCP server introspection ────────────────────────────────────────────────

class MCPToolInfo(BaseModel):
    name: str
    description: str
    input_schema: Dict[str, Any]
    last_call: Optional[Dict[str, Any]] = None


class MCPToolsResponse(BaseModel):
    tools: List[MCPToolInfo]


class MCPStatusResponse(BaseModel):
    mounted_in_api: bool
    standalone_runner: str
    mount_path: str
    transport: str
    tool_count: int


class MCPInvokeRequest(BaseModel):
    tool: str
    args: Optional[Dict[str, Any]] = None


class MCPInvokeResponse(BaseModel):
    tool: str
    result: Any
    latency_ms: float


# ── Claude predictor / chat ──────────────────────────────────────────────────

class ClaudePredictRequest(BaseModel):
    symbol: str
    start_date: str = Field(..., description="YYYY-MM-DD")
    end_date: str = Field(..., description="YYYY-MM-DD")
    interval: str = "1h"
    anthropic_api_key: Optional[str] = None
    claude_model: Optional[str] = None
    openai_api_key: str
    indian_api_key: Optional[str] = ""
    base_url: Optional[str] = None
    selected_tools: Optional[List[str]] = None
    max_iterations: Optional[int] = None


class ChatMessage(BaseModel):
    role: str  # 'user' | 'assistant'
    content: str


class ClaudeChatRequest(BaseModel):
    messages: List[ChatMessage]
    anthropic_api_key: Optional[str] = None
    claude_model: Optional[str] = None
    max_iterations: Optional[int] = None
    stream: bool = False


class ToolUseTrace(BaseModel):
    iter: int
    name: str
    args: Dict[str, Any] = {}
    ok: bool
    error: Optional[str] = None


class ClaudeChatResponse(BaseModel):
    text: str
    trace: List[ToolUseTrace] = []


# ── Realtime intraday session (monitor + run controls) ─────────────────────────

class RTStartRequest(BaseModel):
    symbol: str
    model_id: str
    interval: str = "1m"
    replay: bool = False
    replay_start: Optional[str] = None
    replay_end: Optional[str] = None
    capital: float = 100000.0
    max_steps: int = 2000
    indian_api_key: Optional[str] = None


class RTStopRequest(BaseModel):
    symbol: str


class RTStatusResponse(BaseModel):
    """Live session status; mirrors realtime.state.status_dict + manager flags.
    Kept permissive (extra fields allowed) so backend changes don't break the UI."""
    active: bool = False
    running: bool = False
    symbol: Optional[str] = None
    model_id: Optional[str] = None
    status: Optional[str] = None
    market_open: Optional[bool] = None
    replay: Optional[bool] = None
    started_at: Optional[str] = None
    updated_at: Optional[str] = None
    capital: Optional[float] = None
    equity: Optional[float] = None
    intraday_pnl: Optional[float] = None
    intraday_pnl_pct: Optional[float] = None
    last_action: Optional[float] = None
    last_signal: Optional[str] = None
    last_price: Optional[float] = None
    n_steps: Optional[int] = None
    n_updates: Optional[int] = None
    error: Optional[str] = None
    reason: Optional[str] = None

    model_config = {"extra": "allow"}


class RTHistoryResponse(BaseModel):
    symbol: Optional[str] = None
    steps: List[Dict[str, Any]] = []


class RTSessionsResponse(BaseModel):
    sessions: List[Dict[str, Any]] = []
