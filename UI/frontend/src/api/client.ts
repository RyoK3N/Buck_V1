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
