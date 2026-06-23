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
