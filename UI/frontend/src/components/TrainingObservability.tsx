import { useEffect, useState } from 'react'
import {
  getTrainingSessions,
  getD3ChartTypes,
  getTrainingChart,
  getHeadroomStats,
  type TrainingSessionSummary,
  type D3ChartTypeInfo,
  type D3Spec,
} from '../api/client'
import D3Chart from './D3Chart'

/**
 * Training observability tab: pick a saved RL training session, then explore it
 * through interactive d3 charts (reward / equity / loss / return dist / drawdown
 * / action heatmap). Also surfaces the headroom compression savings so you can
 * see how much Claude-facing token budget the MCP layer is reclaiming.
 */
export default function TrainingObservability() {
  const [sessions, setSessions] = useState<TrainingSessionSummary[]>([])
  const [chartTypes, setChartTypes] = useState<D3ChartTypeInfo[]>([])
  const [selected, setSelected] = useState<string>('')
  const [chart, setChart] = useState<string>('reward_curve')
  const [spec, setSpec] = useState<D3Spec | null>(null)
  const [description, setDescription] = useState<string>('')
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string>('')
  const [headroom, setHeadroom] = useState<Awaited<ReturnType<typeof getHeadroomStats>> | null>(null)

  function refresh() {
    getTrainingSessions({ limit: 100 })
      .then((s) => {
        setSessions(s)
        if (s.length && !selected) setSelected(s[0].session_id)
      })
      .catch((e) => setError(String(e)))
    getD3ChartTypes().then(setChartTypes).catch(() => {})
    getHeadroomStats().then(setHeadroom).catch(() => {})
  }

  useEffect(refresh, [])

  useEffect(() => {
    if (!selected || !chart) return
    setLoading(true)
    setError('')
    getTrainingChart(selected, chart)
      .then((r) => {
        setSpec(r.spec)
        setDescription(r.description)
      })
      .catch((e) => setError(String(e?.response?.data?.detail ?? e)))
      .finally(() => setLoading(false))
    getHeadroomStats().then(setHeadroom).catch(() => {})
  }, [selected, chart])

  const usage = headroom?.usage

  return (
    <div className="flex flex-col gap-4">
      <div className="flex items-center justify-between">
        <h2 className="text-lg font-semibold text-gray-800">Training Observability</h2>
        <button
          onClick={refresh}
          className="rounded bg-gray-100 px-3 py-1.5 text-sm font-medium text-gray-700 hover:bg-gray-200"
        >
          ↻ Refresh
        </button>
      </div>

      {/* Headroom savings banner */}
      {usage && (
        <div className="rounded border border-emerald-200 bg-emerald-50 px-3 py-2 text-xs text-emerald-800">
          <span className="font-semibold">Context engineering (headroom):</span>{' '}
          {headroom?.headroom_available ? 'active' : 'passthrough (lib not installed)'} ·{' '}
          {usage.calls} calls · saved {usage.tokens_saved.toLocaleString()} tokens (
          {usage.reduction_pct}%) · ≈ ${usage.est_cost_saved_usd.toFixed(4)} saved
        </div>
      )}

      <div className="grid grid-cols-2 gap-4">
        <label className="flex flex-col gap-1">
          <span className="text-xs font-medium text-gray-700">Training session</span>
          <select
            className="rounded border border-gray-300 px-2 py-1.5 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
            value={selected}
            onChange={(e) => setSelected(e.target.value)}
          >
            {sessions.length === 0 && <option value="">No sessions — train an RL model first</option>}
            {sessions.map((s) => (
              <option key={s.session_id} value={s.session_id}>
                {s.model_id} · {s.symbol ?? '?'} · {s.algorithm ?? '?'} · {s.episodes ?? '?'} ep
                {s.final_return_pct != null ? ` · ${s.final_return_pct}%` : ''}
              </option>
            ))}
          </select>
        </label>

        <label className="flex flex-col gap-1">
          <span className="text-xs font-medium text-gray-700">Chart</span>
          <select
            className="rounded border border-gray-300 px-2 py-1.5 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
            value={chart}
            onChange={(e) => setChart(e.target.value)}
          >
            {chartTypes.map((c) => (
              <option key={c.id} value={c.id}>
                {c.label}
              </option>
            ))}
          </select>
        </label>
      </div>

      {description && <p className="text-xs text-gray-500">{description}</p>}
      {error && <p className="text-sm text-red-600">{error}</p>}

      <div className="rounded border border-gray-200 bg-white p-2">
        {loading ? (
          <div className="flex h-64 items-center justify-center text-sm text-gray-400">Loading…</div>
        ) : (
          <D3Chart spec={spec} height={340} />
        )}
      </div>
    </div>
  )
}
