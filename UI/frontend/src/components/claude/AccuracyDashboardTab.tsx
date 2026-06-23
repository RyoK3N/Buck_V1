import { useEffect, useRef, useState } from 'react'
import {
  getAccuracySummary,
  getAccuracyTimeseries,
  getLiveAccuracy,
  getToolContribution,
  evaluateNow,
  openAccuracyWebSocket,
} from '../../api/client'
import type {
  AccuracyPoint,
  LiveAccuracyEntry,
  ModelSummary,
  ToolContributionRow,
} from '../../types'

declare const Plotly: { newPlot: (el: HTMLElement, data: unknown[], layout: Record<string, unknown>, config?: Record<string, unknown>) => void }

export default function AccuracyDashboardTab() {
  const [live, setLive] = useState<LiveAccuracyEntry[]>([])
  const [summaries, setSummaries] = useState<ModelSummary[]>([])
  const [series, setSeries] = useState<AccuracyPoint[]>([])
  const [tools, setTools] = useState<ToolContributionRow[]>([])
  const [window, setWindow] = useState<number>(30)
  const [wsConnected, setWsConnected] = useState(false)
  const [evaluating, setEvaluating] = useState(false)
  const chartRef = useRef<HTMLDivElement | null>(null)
  const heatmapRef = useRef<HTMLDivElement | null>(null)

  async function reload() {
    const [s, ts, lv, tc] = await Promise.all([
      getAccuracySummary({ window_days: window }),
      getAccuracyTimeseries({ window_days: window }),
      getLiveAccuracy(),
      getToolContribution({ model: 'claude', window_days: window }),
    ])
    setSummaries(s)
    setSeries(ts)
    setLive(lv)
    setTools(tc)
  }

  useEffect(() => {
    void reload()
  }, [window])

  // WebSocket subscription for real-time updates
  useEffect(() => {
    let ws: WebSocket | null = null
    try {
      ws = openAccuracyWebSocket()
      ws.onopen = () => setWsConnected(true)
      ws.onclose = () => setWsConnected(false)
      ws.onmessage = ev => {
        try {
          const msg = JSON.parse(ev.data)
          if (msg.type === 'snapshot') setLive(msg.entries)
          if (msg.type === 'evaluation') void reload()  // simple refresh on each tick
        } catch {/* ignore */}
      }
    } catch {/* WS unsupported */}
    return () => { ws?.close() }
  }, [])

  // Plotly line chart of per-model accuracy
  useEffect(() => {
    if (!chartRef.current || typeof Plotly === 'undefined') return
    const byModel: Record<string, AccuracyPoint[]> = {}
    for (const p of series) {
      (byModel[p.model] ??= []).push(p)
    }
    const traces = Object.entries(byModel).map(([model, pts]) => ({
      type: 'scatter',
      mode: 'lines+markers',
      name: model,
      x: pts.map(p => p.date),
      y: pts.map(p => p.mae),
    }))
    Plotly.newPlot(chartRef.current, traces, {
      margin: { t: 30, l: 50, r: 20, b: 40 },
      yaxis: { title: 'MAE (price units)' },
      xaxis: { title: 'date' },
      title: 'Per-model rolling MAE',
      legend: { orientation: 'h', y: -0.25 },
    }, { displayModeBar: false, responsive: true })
  }, [series])

  // Plotly heatmap of tool contribution
  useEffect(() => {
    if (!heatmapRef.current || typeof Plotly === 'undefined') return
    if (tools.length === 0) {
      heatmapRef.current.innerHTML = '<div class="text-sm text-gray-500 p-4">No Claude tool-trace data yet.</div>'
      return
    }
    const z = [tools.map(t => t.correct), tools.map(t => t.incorrect)]
    Plotly.newPlot(heatmapRef.current, [{
      type: 'heatmap',
      z,
      x: tools.map(t => t.tool),
      y: ['Correct', 'Incorrect'],
      colorscale: 'Viridis',
      hoverongaps: false,
    }], {
      margin: { t: 30, l: 80, r: 20, b: 80 },
      title: 'Claude tool usage by outcome',
      xaxis: { tickangle: -30 },
    }, { displayModeBar: false, responsive: true })
  }, [tools])

  async function handleEvaluateNow() {
    setEvaluating(true)
    try {
      await evaluateNow({ is_final: false })
      await reload()
    } finally {
      setEvaluating(false)
    }
  }

  return (
    <div className="space-y-4">
      <div className="flex items-center gap-3">
        <label className="text-xs text-gray-600">
          Window
          <select
            className="ml-2 rounded border border-gray-300 px-2 py-1 text-sm"
            value={window}
            onChange={e => setWindow(Number(e.target.value))}
          >
            <option value={7}>7 days</option>
            <option value={30}>30 days</option>
            <option value={90}>90 days</option>
            <option value={365}>1 year</option>
          </select>
        </label>
        <button
          onClick={() => void handleEvaluateNow()}
          disabled={evaluating}
          className="rounded border border-gray-300 px-3 py-1 text-sm hover:bg-gray-50 disabled:opacity-50"
        >
          {evaluating ? 'Polling…' : 'Evaluate now'}
        </button>
        <span className={`text-xs ${wsConnected ? 'text-green-700' : 'text-gray-400'}`}>
          {wsConnected ? '● live' : '○ disconnected'}
        </span>
      </div>

      {/* Live tiles */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
        {live.length === 0 && (
          <div className="md:col-span-3 rounded-lg border border-gray-200 bg-white p-4 text-sm text-gray-500">
            No live accuracy yet. Predictions accumulate here as the intraday poller reconciles them.
          </div>
        )}
        {live.map((e, i) => (
          <div key={i} className="rounded-lg border border-gray-200 bg-white p-4 shadow-sm">
            <div className="text-xs uppercase tracking-wide text-gray-500">{e.model} {e.symbol && e.symbol !== '*' ? `· ${e.symbol}` : ''}</div>
            <div className="mt-2 grid grid-cols-2 gap-2 text-sm">
              <div>
                <div className="text-gray-500 text-xs">Rolling MAE</div>
                <div className="text-lg font-semibold">{e.mae_pct == null ? '—' : `${e.mae_pct.toFixed(2)}%`}</div>
              </div>
              <div>
                <div className="text-gray-500 text-xs">Directional acc</div>
                <div className="text-lg font-semibold">{e.directional_accuracy_pct == null ? '—' : `${e.directional_accuracy_pct.toFixed(0)}%`}</div>
              </div>
              <div>
                <div className="text-gray-500 text-xs">N</div>
                <div className="text-lg">{e.n}</div>
              </div>
              <div>
                <div className="text-gray-500 text-xs">Updated</div>
                <div className="text-xs text-gray-600">{e.updated_at?.slice(11, 19) ?? '—'}</div>
              </div>
            </div>
          </div>
        ))}
      </div>

      {/* Per-model summaries table */}
      <div className="rounded-lg border border-gray-200 bg-white p-3 shadow-sm">
        <h3 className="mb-2 text-sm font-semibold text-gray-700">Per-model summary (window: {window}d)</h3>
        <table className="min-w-full text-sm">
          <thead className="bg-gray-50 text-xs uppercase text-gray-500">
            <tr>
              <th className="px-3 py-2 text-left">Model</th>
              <th className="px-3 py-2 text-right">N</th>
              <th className="px-3 py-2 text-right">MAE</th>
              <th className="px-3 py-2 text-right">RMSE</th>
              <th className="px-3 py-2 text-right">Directional acc</th>
              <th className="px-3 py-2 text-right">Avg |err %|</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-gray-100">
            {summaries.map(s => (
              <tr key={s.model}>
                <td className="px-3 py-2 font-medium">{s.model}</td>
                <td className="px-3 py-2 text-right">{s.n}</td>
                <td className="px-3 py-2 text-right">{s.mae?.toFixed(3) ?? '—'}</td>
                <td className="px-3 py-2 text-right">{s.rmse?.toFixed(3) ?? '—'}</td>
                <td className="px-3 py-2 text-right">{s.directional_accuracy == null ? '—' : `${(s.directional_accuracy * 100).toFixed(0)}%`}</td>
                <td className="px-3 py-2 text-right">{s.avg_error_pct?.toFixed(2) ?? '—'}%</td>
              </tr>
            ))}
            {summaries.length === 0 && (
              <tr><td colSpan={6} className="px-3 py-4 text-center text-gray-500">No evaluated predictions yet.</td></tr>
            )}
          </tbody>
        </table>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        <div className="rounded-lg border border-gray-200 bg-white p-3 shadow-sm">
          <div ref={chartRef} style={{ width: '100%', height: 320 }} />
        </div>
        <div className="rounded-lg border border-gray-200 bg-white p-3 shadow-sm">
          <div ref={heatmapRef} style={{ width: '100%', height: 320 }} />
        </div>
      </div>
    </div>
  )
}
