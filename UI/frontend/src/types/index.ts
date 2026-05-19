export interface Forecast {
  date: string
  open: number
  high: number
  low: number
  close: number
  confidence: number
  reasoning: string
}

export interface AnalysisResult {
  symbol: string
  analysis_type: string
  data: Record<string, unknown>
  timestamp: string
  confidence: number
}

export interface DataInfo {
  start_date: string
  end_date: string
  interval: string
  data_points: number
  news_available: boolean
}

export interface Metadata {
  agent_version: string
  model_used: string
  analysis_confidence: number
  prediction_confidence: number
}

export interface AnalyzeResponse {
  symbol: string
  timestamp: string
  data_info: DataInfo
  analysis_results: AnalysisResult[]
  forecast: Forecast
  metadata: Metadata
  error?: string
  status?: string
}

export interface BatchSummary {
  successful: number
  failed: number
  avg_confidence: number
}

export interface BatchInfo {
  symbols: string[]
  start_date: string
  end_date: string
  interval: string
  timestamp: string
  total_symbols: number
}

export interface BatchResponse {
  batch_info: BatchInfo
  results: Record<string, AnalyzeResponse>
  summary: BatchSummary
}

export interface Config {
  openai_api_key: string
  indian_api_key: string
  model: string
  base_url: string
}

export interface ServerConfig {
  openai_api_key: string
  openai_base_url: string | null
  chat_model: string
  indian_api_key: string
}

export interface ChartTypeInfo {
  id: string
  name: string
  description: string
}

export interface VisualizeRequest {
  symbol: string
  start_date: string
  end_date: string
  interval: string
  chart_type: string
  indian_api_key?: string
}

export interface PlotlyFigure {
  data: unknown[]
  layout: Record<string, unknown>
}

export interface VisualizeResponse {
  chart: PlotlyFigure
  chart_type: string
  symbol: string
  description: string
}

// ── Tools Registry ───────────────────────────────────────────────────────────

export interface ToolInfo {
  id: string
  name: string
  description: string
}

export interface ToolCategory {
  id: string
  name: string
  description: string
  tools: ToolInfo[]
}

export interface ToolsRegistry {
  categories: ToolCategory[]
}
