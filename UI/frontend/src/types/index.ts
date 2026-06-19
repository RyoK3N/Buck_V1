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

// ── RL / Reinforcement Learning ─────────────────────────────────────────────

export interface RLTrainRequest {
  symbol: string
  start_date: string
  end_date: string
  interval: string
  algorithm: string
  model_id: string
  episodes: number
  hidden_dim: number
  learning_rate: number
  initial_capital: number
  indian_api_key?: string
}

export interface RLEpisodeReward {
  episode: number
  total_reward: number
  portfolio_value: number
  return_pct: number
  epsilon: number
  trades: number
}

export interface RLWalletSummary {
  initial_capital: number
  cash: number
  holdings: number
  current_price: number
  portfolio_value: number
  total_return_pct: number
  total_trades: number
  win_rate_pct?: number
  sharpe_ratio: number
  max_drawdown_pct: number
  transaction_cost: number
}

export interface RLTrainResponse {
  model_id: string
  symbol: string
  algorithm?: string
  episodes: number
  total_steps: number
  episode_rewards: RLEpisodeReward[]
  final_summary: RLWalletSummary
  equity_curve: Array<{ timestamp: string; cash: number; holdings: number; price: number | null; portfolio_value: number }>
  best_reward: number
  status: string
}

export interface RLPredictRequest {
  symbol: string
  start_date: string
  end_date: string
  interval: string
  model_id: string
  initial_capital: number
  indian_api_key?: string
}

export interface RLPredictSignal {
  step: number
  action: string
  price: number
  portfolio_value: number
}

export interface RLPredictResponse {
  model_id: string
  symbol: string
  total_signals: number
  signals: RLPredictSignal[]
  equity_curve: Array<{ timestamp: string; cash: number; holdings: number; price: number | null; portfolio_value: number }>
  summary: RLWalletSummary
  status: string
}

export interface RLSimulateRequest {
  model_id: string
  symbol: string
  interval: string
  initial_capital: number
  indian_api_key?: string
}

export interface RLSimulateResponse {
  action: string
  price: number
  symbol: string
  model_id: string
  wallet: RLWalletSummary
  live_data: Record<string, unknown>
  status: string
}

export interface RLModelInfo {
  id: string
  path: string
  created: string
  input_dim: number
  hidden_dim: number
  train_steps: number
  epsilon: number
  error?: string
}

export interface RLModelsResponse {
  models: RLModelInfo[]
}
