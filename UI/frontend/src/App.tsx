import { useCallback, useEffect, useState } from 'react'
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
import ClaudePanel from './components/claude/ClaudePanel'
import TrainingObservability from './components/TrainingObservability'
import RealtimePanel from './components/RealtimePanel'

type Tab = 'single' | 'batch' | 'visualizer' | 'rl' | 'realtime' | 'training' | 'claude'
const TABS: Tab[] = ['single', 'batch', 'visualizer', 'rl', 'realtime', 'training', 'claude']

export interface RealtimeDeepLink {
  symbol?: string
  model?: string
  interval?: string
  start?: string
  end?: string
  autostart?: boolean
}

/** Parse deep-link query params (set by the open_buck_ui MCP tool / shared links). */
function readDeepLink(): { tab?: Tab; realtime: RealtimeDeepLink } {
  const q = new URLSearchParams(window.location.search)
  const t = q.get('tab') as Tab | null
  return {
    tab: t && TABS.includes(t) ? t : undefined,
    realtime: {
      symbol: q.get('symbol') ?? undefined,
      model: q.get('model') ?? undefined,
      interval: q.get('interval') ?? undefined,
      start: q.get('start') ?? undefined,
      end: q.get('end') ?? undefined,
      autostart: q.get('autostart') === '1' || q.get('autostart') === 'true',
    },
  }
}

export default function App() {
  const [config, setConfig] = useState<Config>({
    openai_api_key: '',
    indian_api_key: '',
    model: 'gpt-4o',
    base_url: '',
    anthropic_api_key: '',
    claude_model: 'claude-opus-4-5',
  })
  // Deep-link (e.g. opened by the open_buck_ui MCP tool) — read once on mount.
  const [deepLink] = useState(readDeepLink)
  const [tab, setTab] = useState<Tab>(deepLink.tab ?? 'single')
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [result, setResult] = useState<AnalyzeResponse | BatchResponse | null>(null)
  const [selectedTools, setSelectedTools] = useState<string[]>([])

  // If a deep link targeted a tab, reflect it in the document title briefly.
  useEffect(() => {
    if (deepLink.tab === 'realtime' && deepLink.realtime.symbol) {
      document.title = `Buck · Realtime · ${deepLink.realtime.symbol}`
    }
  }, [deepLink])

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

  const NON_ANALYSIS: Tab[] = ['visualizer', 'rl', 'realtime', 'training', 'claude']

  function switchTab(t: Tab) {
    setTab(t)
    if (!NON_ANALYSIS.includes(t)) {
      setResult(null)
      setError(null)
    }
  }

  const TAB_LABELS: Record<Tab, string> = {
    single: 'Single',
    batch: 'Batch',
    visualizer: 'Visualizer',
    rl: 'RL Lab',
    realtime: 'Realtime',
    training: 'Training',
    claude: 'Claude',
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
            {(['single', 'batch', 'visualizer', 'rl', 'realtime', 'training', 'claude'] as Tab[]).map(t => (
              <button
                key={t}
                onClick={() => switchTab(t)}
                className={`rounded px-4 py-1.5 text-sm font-medium transition-colors ${
                  tab === t
                    ? t === 'claude' ? 'bg-purple-600 text-white' : 'bg-blue-600 text-white'
                    : 'text-gray-600 hover:text-gray-900'
                }`}
              >
                {TAB_LABELS[t]}
              </button>
            ))}
          </div>

          {/* Claude tab (isolated) */}
          {tab === 'claude' && (
            <ClaudePanel config={config} />
          )}

          {/* RL Lab tab */}
          {tab === 'rl' && (
            <RLPanel config={config} />
          )}

          {/* Realtime monitoring tab */}
          {tab === 'realtime' && (
            <div className="rounded-lg border border-gray-200 bg-white p-4 shadow-sm">
              <RealtimePanel config={config} initial={deepLink.tab === 'realtime' ? deepLink.realtime : undefined} />
            </div>
          )}

          {/* Visualizer tab */}
          {tab === 'visualizer' && (
            <VisualizerPanel config={config} />
          )}

          {/* Training observability tab (d3) */}
          {tab === 'training' && (
            <div className="rounded-lg border border-gray-200 bg-white p-4 shadow-sm">
              <TrainingObservability />
            </div>
          )}

          {/* Analysis tabs */}
          {!NON_ANALYSIS.includes(tab) && (
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
