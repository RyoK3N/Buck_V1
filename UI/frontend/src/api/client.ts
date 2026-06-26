import axios from 'axios'
import type {
  AnalyzeResponse,
  BatchResponse,
  ServerConfig,
  ChartTypeInfo,
  VisualizeRequest,
  VisualizeResponse,
  ToolsRegistry,
  RLTrainRequest,
  RLTrainResponse,
  RLPredictRequest,
  RLPredictResponse,
  RLSimulateRequest,
  RLSimulateResponse,
  RLModelsResponse,
  ChatMessage,
  ClaudeChatResponse,
  ClaudePredictResponse,
  PredictionRow,
  ModelSummary,
  AccuracyPoint,
  LiveAccuracyEntry,
  ToolContributionRow,
  MCPToolInfo,
  MCPStatus,
} from '../types'

const BASE_URL = import.meta.env.VITE_API_URL ?? 'http://localhost:8000'

const http = axios.create({ baseURL: BASE_URL })

export interface AnalyzeRequest {
  symbol: string
  start_date: string
  end_date: string
  interval: string
  openai_api_key: string
  indian_api_key?: string
  model?: string
  base_url?: string
  selected_tools?: string[]
}

export interface BatchRequest {
  symbols: string[]
  start_date: string
  end_date: string
  interval: string
  openai_api_key: string
  indian_api_key?: string
  model?: string
  base_url?: string
  max_concurrent?: number
  selected_tools?: string[]
}

// ── Config ────────────────────────────────────────────────────────────────────

export async function getServerConfig(): Promise<ServerConfig> {
  const { data } = await http.get<ServerConfig>('/config')
  return data
}

// ── Analysis ──────────────────────────────────────────────────────────────────

export async function analyzeStock(req: AnalyzeRequest): Promise<AnalyzeResponse> {
  const { data } = await http.post<AnalyzeResponse>('/analyze', req)
  return data
}

export async function batchAnalyze(req: BatchRequest): Promise<BatchResponse> {
  const { data } = await http.post<BatchResponse>('/batch', req)
  return data
}

// ── Meta ──────────────────────────────────────────────────────────────────────

export async function getHealth(): Promise<{ status: string; version: string }> {
  const { data } = await http.get('/health')
  return data
}

export async function getIntervals(): Promise<string[]> {
  const { data } = await http.get<{ intervals: string[] }>('/intervals')
  return data.intervals
}

export async function getTools(): Promise<string[]> {
  const { data } = await http.get<{ tools: string[] }>('/tools')
  return data.tools
}

export async function getToolsRegistry(): Promise<ToolsRegistry> {
  const { data } = await http.get<ToolsRegistry>('/tools-registry')
  return data
}

// ── Visualizer ────────────────────────────────────────────────────────────────

export async function getChartTypes(): Promise<ChartTypeInfo[]> {
  const { data } = await http.get<{ chart_types: ChartTypeInfo[] }>('/chart-types')
  return data.chart_types
}

export async function visualize(req: VisualizeRequest): Promise<VisualizeResponse> {
  const { data } = await http.post<VisualizeResponse>('/visualize', req)
  return data
}

// ── RL / Reinforcement Learning ──────────────────────────────────────────────

export async function rlTrain(req: RLTrainRequest): Promise<RLTrainResponse> {
  const { data } = await http.post<RLTrainResponse>('/rl/train', req)
  return data
}

export async function rlPredict(req: RLPredictRequest): Promise<RLPredictResponse> {
  const { data } = await http.post<RLPredictResponse>('/rl/predict', req)
  return data
}

export async function rlSimulate(req: RLSimulateRequest): Promise<RLSimulateResponse> {
  const { data } = await http.post<RLSimulateResponse>('/rl/simulate', req)
  return data
}

export async function rlGetModels(): Promise<RLModelsResponse> {
  const { data } = await http.get<RLModelsResponse>('/rl/models')
  return data
}

export async function rlDeleteModel(modelId: string): Promise<{ status: string; model_id: string }> {
  const { data } = await http.delete(`/rl/models/${modelId}`)
  return data
}

// ── Claude predictor + chat ──────────────────────────────────────────────────

export interface ClaudePredictRequestBody {
  symbol: string
  start_date: string
  end_date: string
  interval: string
  openai_api_key: string
  indian_api_key?: string
  anthropic_api_key?: string
  claude_model?: string
  base_url?: string
  selected_tools?: string[]
  max_iterations?: number
}

export async function claudePredict(req: ClaudePredictRequestBody): Promise<ClaudePredictResponse> {
  const { data } = await http.post<ClaudePredictResponse>('/claude/predict', req)
  return data
}

export interface ClaudeChatRequestBody {
  messages: ChatMessage[]
  anthropic_api_key?: string
  claude_model?: string
  max_iterations?: number
  stream?: boolean
}

export async function claudeChat(req: ClaudeChatRequestBody): Promise<ClaudeChatResponse> {
  const { data } = await http.post<ClaudeChatResponse>('/claude/chat', req)
  return data
}

// ── Accuracy tracking ────────────────────────────────────────────────────────

export async function getPredictions(params: { symbol?: string; model?: string; status?: string; limit?: number } = {}): Promise<PredictionRow[]> {
  const { data } = await http.get<{ predictions: PredictionRow[] }>('/accuracy/predictions', { params })
  return data.predictions
}

export async function getAccuracySummary(params: { model?: string; symbol?: string; window_days?: number } = {}): Promise<ModelSummary[]> {
  const { data } = await http.get<{ summaries: ModelSummary[] }>('/accuracy/summary', { params })
  return data.summaries
}

export async function getAccuracyTimeseries(params: { model?: string; symbol?: string; window_days?: number } = {}): Promise<AccuracyPoint[]> {
  const { data } = await http.get<{ points: AccuracyPoint[] }>('/accuracy/timeseries', { params })
  return data.points
}

export async function getLiveAccuracy(): Promise<LiveAccuracyEntry[]> {
  const { data } = await http.get<{ entries: LiveAccuracyEntry[] }>('/accuracy/live')
  return data.entries
}

export async function getToolContribution(params: { model?: string; window_days?: number } = {}): Promise<ToolContributionRow[]> {
  const { data } = await http.get<{ model: string; rows: ToolContributionRow[] }>('/accuracy/tool-contribution', { params })
  return data.rows
}

export async function evaluateNow(body: { is_final?: boolean; symbol?: string } = {}): Promise<{ polled: number; evaluated: number }> {
  const { data } = await http.post<{ polled: number; evaluated: number }>('/accuracy/evaluate-now', body)
  return data
}

export function openAccuracyWebSocket(): WebSocket {
  const wsUrl = BASE_URL.replace(/^http/, 'ws') + '/accuracy/ws'
  return new WebSocket(wsUrl)
}

// ── d3 training-session observability ─────────────────────────────────────────

export interface D3ChartTypeInfo {
  id: string
  label: string
  description: string
}

export interface TrainingSessionSummary {
  session_id: string
  model_id: string
  symbol: string | null
  algorithm: string | null
  interval: string | null
  episodes: number | null
  created_at: string | null
  final_return_pct: number | null
  final_sharpe: number | null
}

// A framework-agnostic d3-buck/1 spec produced by UI/backend/d3_viz.py.
export interface D3Spec {
  spec_version: string
  chart: string
  mark: 'line' | 'area' | 'multiline' | 'bar' | 'heatmap'
  data: Record<string, unknown>[]
  encoding: Record<string, unknown>
  meta: Record<string, unknown>
}

export async function getD3ChartTypes(): Promise<D3ChartTypeInfo[]> {
  const { data } = await http.get<{ chart_types: D3ChartTypeInfo[] }>('/viz/d3-chart-types')
  return data.chart_types
}

export async function getTrainingSessions(
  params: { model_id?: string; symbol?: string; limit?: number } = {},
): Promise<TrainingSessionSummary[]> {
  const { data } = await http.get<{ sessions: TrainingSessionSummary[] }>('/viz/training-sessions', { params })
  return data.sessions
}

export async function getTrainingChart(sessionId: string, chart: string): Promise<{ spec: D3Spec; description: string }> {
  const { data } = await http.get<{ spec: D3Spec; description: string }>(`/viz/training/${sessionId}/${chart}`)
  return data
}

// ── Headroom (context engineering) ────────────────────────────────────────────

export interface HeadroomUsage {
  calls: number
  tokens_raw: number
  tokens_compressed: number
  tokens_saved: number
  reduction_pct: number
  est_cost_raw_usd: number
  est_cost_compressed_usd: number
  est_cost_saved_usd: number
  per_tool: Record<string, Record<string, number>>
}

export async function getHeadroomStats(): Promise<{ headroom_available: boolean; usage: HeadroomUsage; cache: Record<string, number> }> {
  const { data } = await http.get('/mcp/headroom')
  return data
}

// ── MCP introspection ────────────────────────────────────────────────────────

export async function getMCPTools(): Promise<MCPToolInfo[]> {
  const { data } = await http.get<{ tools: MCPToolInfo[] }>('/mcp/tools')
  return data.tools
}

export async function getMCPStatus(): Promise<MCPStatus> {
  const { data } = await http.get<MCPStatus>('/mcp/status')
  return data
}

export async function invokeMCPTool(tool: string, args: Record<string, unknown> = {}): Promise<{ tool: string; result: unknown; latency_ms: number }> {
  const { data } = await http.post('/mcp/invoke', { tool, args })
  return data
}

// ── Realtime intraday session (monitor + run controls) ────────────────────────

export interface RTStatus {
  active: boolean
  running: boolean
  symbol?: string | null
  model_id?: string | null
  status?: string | null
  market_open?: boolean | null
  replay?: boolean | null
  started_at?: string | null
  updated_at?: string | null
  capital?: number | null
  equity?: number | null
  intraday_pnl?: number | null
  intraday_pnl_pct?: number | null
  last_action?: number | null
  last_signal?: string | null
  last_price?: number | null
  n_steps?: number | null
  n_updates?: number | null
  error?: string | null
  reason?: string | null
}

export interface RTStep {
  ts?: string
  price?: number
  target_position?: number
  signal?: string
  realized_return?: number
  bar_return?: number
  equity?: number
  [k: string]: unknown
}

export interface RTStartBody {
  symbol: string
  model_id: string
  interval?: string
  replay?: boolean
  replay_start?: string
  replay_end?: string
  capital?: number
  max_steps?: number
  indian_api_key?: string
}

export async function getRealtimeStatus(symbol?: string): Promise<RTStatus> {
  const { data } = await http.get<RTStatus>('/rt/status', { params: symbol ? { symbol } : {} })
  return data
}

export async function getRealtimeHistory(symbol?: string, limit = 100): Promise<RTStep[]> {
  const { data } = await http.get<{ symbol: string | null; steps: RTStep[] }>('/rt/history', {
    params: { ...(symbol ? { symbol } : {}), limit },
  })
  return data.steps
}

export async function getRealtimeSessions(): Promise<RTStatus[]> {
  const { data } = await http.get<{ sessions: RTStatus[] }>('/rt/sessions')
  return data.sessions
}

export async function getRealtimeChart(symbol: string | undefined, chart: string): Promise<{ spec: D3Spec; active: boolean; chart: string }> {
  const { data } = await http.get<{ spec: D3Spec; active: boolean; chart: string }>('/rt/chart', {
    params: { ...(symbol ? { symbol } : {}), chart },
  })
  return data
}

export async function startRealtime(body: RTStartBody): Promise<RTStatus> {
  const { data } = await http.post<RTStatus>('/rt/start', body)
  return data
}

export async function stopRealtime(symbol: string): Promise<RTStatus> {
  const { data } = await http.post<RTStatus>('/rt/stop', { symbol })
  return data
}
