import { useCallback, useState } from 'react'
import { analyzeStock, batchAnalyze } from './api/client'
import type { AnalyzeResponse, BatchResponse, Config } from './types'
import Header from './components/Header'
import ConfigPanel from './components/ConfigPanel'
import AnalysisForm from './components/AnalysisForm'
import BatchForm from './components/BatchForm'
import ResultsPanel from './components/ResultsPanel'
import VisualizerPanel from './components/VisualizerPanel'
import ToolsConfigPanel from './components/ToolsConfigPanel'
import LoadingSpinner from './components/LoadingSpinner'
import RLPanel from './components/RLPanel'

type Tab = 'single' | 'batch' | 'visualizer' | 'rl'

export default function App() {
  const [config, setConfig] = useState<Config>({
    openai_api_key: '',
    indian_api_key: '',
    model: 'gpt-4o',
    base_url: '',
  })
  const [tab, setTab] = useState<Tab>('single')
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [result, setResult] = useState<AnalyzeResponse | BatchResponse | null>(null)
  const [selectedTools, setSelectedTools] = useState<string[]>([])

  const handleConfigChange = useCallback((cfg: Config) => setConfig(cfg), [])
  const handleToolsChange = useCallback((tools: string[]) => setSelectedTools(tools), [])

  async function handleSingleSubmit(payload: {
    symbol: string
    start_date: string
    end_date: string
    interval: string
  }) {
    setLoading(true)
    setError(null)
    setResult(null)
    try {
      const res = await analyzeStock({
        ...payload,
        openai_api_key: config.openai_api_key,
        indian_api_key: config.indian_api_key || undefined,
        model: config.model || undefined,
        base_url: config.base_url || undefined,
        selected_tools: selectedTools.length > 0 ? selectedTools : undefined,
      })
      setResult(res)
    } catch (e: unknown) {
      const msg = e instanceof Error ? e.message : 'Request failed'
      // Try to extract server detail from axios error
      const detail = (e as { response?: { data?: { detail?: string } } })?.response?.data?.detail
      setError(detail ?? msg)
    } finally {
      setLoading(false)
    }
  }

  async function handleBatchSubmit(payload: {
    symbols: string[]
    start_date: string
    end_date: string
    interval: string
    max_concurrent: number
  }) {
    setLoading(true)
    setError(null)
    setResult(null)
    try {
      const res = await batchAnalyze({
        ...payload,
        openai_api_key: config.openai_api_key,
        indian_api_key: config.indian_api_key || undefined,
        model: config.model || undefined,
        base_url: config.base_url || undefined,
        selected_tools: selectedTools.length > 0 ? selectedTools : undefined,
      })
      setResult(res)
    } catch (e: unknown) {
      const msg = e instanceof Error ? e.message : 'Request failed'
      const detail = (e as { response?: { data?: { detail?: string } } })?.response?.data?.detail
      setError(detail ?? msg)
    } finally {
      setLoading(false)
    }
  }

  function switchTab(t: Tab) {
    setTab(t)
    if (t !== 'visualizer' && t !== 'rl') {
      setResult(null)
      setError(null)
    }
  }

  const TAB_LABELS: Record<Tab, string> = {
    single: 'Single',
    batch: 'Batch',
    visualizer: 'Visualizer',
    rl: 'RL Lab',
  }

  return (
    <div className="min-h-screen bg-gray-50">
      <Header />

      <div className="mx-auto flex max-w-screen-xl gap-6 p-6">
        {/* Sidebar */}
        <div className="w-64 flex-shrink-0">
          <ConfigPanel onChange={handleConfigChange} />
        </div>

        {/* Main */}
        <div className="min-w-0 flex-1 space-y-6">
          {/* Tab switcher */}
          <div className="flex gap-1 rounded-lg border border-gray-200 bg-white p-1 shadow-sm w-fit">
            {(['single', 'batch', 'visualizer', 'rl'] as Tab[]).map(t => (
              <button
                key={t}
                onClick={() => switchTab(t)}
                className={`rounded px-4 py-1.5 text-sm font-medium transition-colors ${
                  tab === t
                    ? 'bg-blue-600 text-white'
                    : 'text-gray-600 hover:text-gray-900'
                }`}
              >
                {TAB_LABELS[t]}
              </button>
            ))}
          </div>

          {/* RL Lab tab */}
          {tab === 'rl' && (
            <RLPanel config={config} />
          )}

          {/* Visualizer tab */}
          {tab === 'visualizer' && (
            <VisualizerPanel config={config} />
          )}

          {/* Analysis tabs */}
          {tab !== 'visualizer' && tab !== 'rl' && (
            <>
              <div className="rounded-lg border border-gray-200 bg-white p-4 shadow-sm space-y-4">
                {tab === 'single' ? (
                  <AnalysisForm config={config} onSubmit={handleSingleSubmit} loading={loading} />
                ) : (
                  <BatchForm config={config} onSubmit={handleBatchSubmit} loading={loading} />
                )}
                <ToolsConfigPanel onChange={handleToolsChange} />
              </div>

              {error && (
                <div className="rounded-lg border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-700">
                  <strong>Error:</strong> {error}
                </div>
              )}

              {loading && <LoadingSpinner />}
              {!loading && result && (
                <ResultsPanel result={result} mode={tab as 'single' | 'batch'} />
              )}
            </>
          )}
        </div>
      </div>
    </div>
  )
}
