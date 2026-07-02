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
  anthropic_api_key: string
  claude_model: string
}

export interface ServerConfig {
  // The server never sends key values back — only whether one is configured
  // server-side (from .env), so the UI can leave the field blank and let
  // requests fall back to the server's own key instead of round-tripping it.
  openai_api_key_configured: boolean
  openai_base_url: string | null
  chat_model: string
  indian_api_key_configured: boolean
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

// ── Claude / Chat ────────────────────────────────────────────────────────────

export interface ChatMessage {
  role: 'user' | 'assistant'
  content: string
}

export interface ToolUseTrace {
  iter: number
  name: string
  args: Record<string, unknown>
  ok: boolean
  error?: string | null
}

export interface ClaudeChatResponse {
  text: string
  trace: ToolUseTrace[]
}

export interface ClaudePredictResponse extends AnalyzeResponse {
  metadata: Metadata & { tool_trace?: ToolUseTrace[] }
}

// ── Accuracy tracking ────────────────────────────────────────────────────────

export interface PredictionRow {
  id: number
  symbol: string
  model: string
  target_date: string
  predicted_open: number | null
  predicted_high: number | null
  predicted_low: number | null
  predicted_close: number | null
  confidence: number | null
  reasoning: string | null
  created_at: string
  status: string
  actual_open: number | null
  actual_high: number | null
  actual_low: number | null
  actual_close: number | null
  mae: number | null
  rmse: number | null
  directional_correct: number | null
  error_pct: number | null
  is_intraday: number | null
  evaluated_at: string | null
}

export interface ModelSummary {
  model: string
  n: number
  mae: number | null
  rmse: number | null
  directional_accuracy: number | null
  avg_error_pct: number | null
}

export interface AccuracyPoint {
  date: string
  model: string
  mae: number | null
  directional_accuracy: number | null
  n: number
}

export interface LiveAccuracyEntry {
  model: string
  symbol: string | null
  mae_pct: number | null
  directional_accuracy_pct: number | null
  n: number
  updated_at: string | null
}

export interface ToolContributionRow {
  tool: string
  correct: number
  incorrect: number
}

// ── MCP introspection ────────────────────────────────────────────────────────

export interface MCPToolInfo {
  name: string
  description: string
  input_schema: Record<string, unknown>
  last_call?: { ts: string; ok: boolean; latency_ms: number; error?: string | null } | null
}

export interface MCPStatus {
  mounted_in_api: boolean
  standalone_runner: string
  mount_path: string
  transport: string
  tool_count: number
}
