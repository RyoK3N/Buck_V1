import { useEffect, useState } from 'react'
import { getPredictions } from '../../api/client'
import type { PredictionRow } from '../../types'

export default function PredictionsTab() {
  const [rows, setRows] = useState<PredictionRow[]>([])
  const [model, setModel] = useState<string>('')
  const [symbol, setSymbol] = useState<string>('')
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [expanded, setExpanded] = useState<number | null>(null)

  async function refresh() {
    setLoading(true)
    setError(null)
    try {
      const rs = await getPredictions({
        model: model || undefined,
        symbol: symbol || undefined,
        limit: 100,
      })
      setRows(rs)
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : 'failed')
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => { void refresh() }, [model, symbol])

  return (
    <div className="space-y-3">
      <div className="rounded-lg border border-gray-200 bg-white p-3 shadow-sm flex flex-wrap items-end gap-3">
        <label className="flex flex-col text-xs">
          <span className="text-gray-600 mb-1">Model</span>
          <select
            className="rounded border border-gray-300 px-2 py-1 text-sm"
            value={model}
            onChange={e => setModel(e.target.value)}
          >
            <option value="">All models</option>
            <option value="claude">claude</option>
            <option value="openai">openai (gpt-4o etc.)</option>
            <option value="gpt-4o">gpt-4o</option>
          </select>
        </label>
        <label className="flex flex-col text-xs">
          <span className="text-gray-600 mb-1">Symbol</span>
          <input
            className="rounded border border-gray-300 px-2 py-1 text-sm"
            placeholder="e.g. RELIANCE.NS"
            value={symbol}
            onChange={e => setSymbol(e.target.value.toUpperCase())}
          />
        </label>
        <button
          onClick={() => void refresh()}
          className="rounded border border-gray-300 px-3 py-1.5 text-sm hover:bg-gray-50"
        >
          Refresh
        </button>
        {loading && <span className="text-xs text-gray-500">loading…</span>}
        {error && <span className="text-xs text-red-600">{error}</span>}
        <span className="ml-auto text-xs text-gray-500">{rows.length} rows</span>
      </div>

      <div className="rounded-lg border border-gray-200 bg-white shadow-sm overflow-x-auto">
        <table className="min-w-full text-sm">
          <thead className="bg-gray-50 text-xs uppercase text-gray-500">
            <tr>
              <th className="px-3 py-2 text-left">ID</th>
              <th className="px-3 py-2 text-left">Target</th>
              <th className="px-3 py-2 text-left">Symbol</th>
              <th className="px-3 py-2 text-left">Model</th>
              <th className="px-3 py-2 text-right">Pred Close</th>
              <th className="px-3 py-2 text-right">Actual</th>
              <th className="px-3 py-2 text-right">Err %</th>
              <th className="px-3 py-2 text-right">Conf</th>
              <th className="px-3 py-2 text-left">Status</th>
              <th className="px-3 py-2"></th>
            </tr>
          </thead>
          <tbody className="divide-y divide-gray-100">
            {rows.map(r => (
              <>
                <tr key={r.id} className="hover:bg-gray-50">
                  <td className="px-3 py-2 text-gray-500">{r.id}</td>
                  <td className="px-3 py-2">{r.target_date}</td>
                  <td className="px-3 py-2 font-medium">{r.symbol}</td>
                  <td className="px-3 py-2 text-gray-700">{r.model}</td>
                  <td className="px-3 py-2 text-right">{r.predicted_close?.toFixed(2) ?? '—'}</td>
                  <td className="px-3 py-2 text-right">{r.actual_close?.toFixed(2) ?? '—'}</td>
                  <td className={`px-3 py-2 text-right ${
                    r.error_pct == null ? '' : Math.abs(r.error_pct) < 1 ? 'text-green-700' : Math.abs(r.error_pct) < 3 ? 'text-amber-700' : 'text-red-700'
                  }`}>
                    {r.error_pct == null ? '—' : `${r.error_pct.toFixed(2)}%`}
                  </td>
                  <td className="px-3 py-2 text-right">
                    {r.confidence == null ? '—' : `${(r.confidence * 100).toFixed(0)}%`}
                  </td>
                  <td className="px-3 py-2">
                    <span className={`rounded px-2 py-0.5 text-xs ${
                      r.status === 'evaluated' ? 'bg-green-100 text-green-700' :
                      r.status === 'expired' ? 'bg-gray-100 text-gray-600' :
                      'bg-amber-100 text-amber-700'
                    }`}>{r.status}</span>
                  </td>
                  <td className="px-3 py-2 text-right">
                    <button
                      className="text-xs text-blue-600 hover:underline"
                      onClick={() => setExpanded(expanded === r.id ? null : r.id)}
                    >
                      {expanded === r.id ? 'Hide' : 'Reasoning'}
                    </button>
                  </td>
                </tr>
                {expanded === r.id && (
                  <tr key={`${r.id}-exp`} className="bg-gray-50">
                    <td colSpan={10} className="px-3 py-3 text-xs text-gray-700 whitespace-pre-wrap">
                      {r.reasoning || '(no reasoning recorded)'}
                    </td>
                  </tr>
                )}
              </>
            ))}
            {rows.length === 0 && !loading && (
              <tr>
                <td colSpan={10} className="px-3 py-6 text-center text-sm text-gray-500">
                  No predictions yet. Run /analyze or /claude/predict to populate.
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </div>
    </div>
  )
}
