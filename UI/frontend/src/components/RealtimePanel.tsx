import { useCallback, useEffect, useRef, useState } from 'react'
import {
  startRealtime,
  stopRealtime,
  getRealtimeStatus,
  getRealtimeHistory,
  getRealtimeChart,
  rlGetModels,
  type RTStatus,
  type RTStep,
  type D3Spec,
} from '../api/client'
import type { Config, RLModelInfo } from '../types'
import D3Chart from './D3Chart'

export interface RealtimeInitial {
  symbol?: string
  model?: string
  interval?: string
  start?: string
  end?: string
  autostart?: boolean
}

/**
 * Realtime tab: start/stop an intraday simulation and monitor the work live —
 * action / equity / PnL tiles, an equity (or action / drawdown) d3 chart, and a
 * scrolling step history. Polls /rt/status + /rt/history while a session runs.
 *
 * Replay mode streams recent historical bars through the same loop, so it works
 * any time (off market hours) — the recommended default for trying it out.
 */
const POLL_MS = 4000
const REPLAY_POLL_MS = 1500 // poll faster while watching an accelerated replay
const SPEED_OPTIONS = [
  { v: 1, label: 'Real-time (1×)' },
  { v: 5, label: '5×' },
  { v: 30, label: '30×' },
  { v: 60, label: '60×' },
  { v: 300, label: '300×' },
  { v: 1000, label: 'Instant' },
]
const RT_CHARTS = [
  { id: 'equity_curve', label: 'Equity curve' },
  { id: 'action_heatmap', label: 'Action heatmap' },
  { id: 'drawdown_curve', label: 'Drawdown' },
]

function todayISO(offsetDays = 0): string {
  const d = new Date()
  d.setDate(d.getDate() + offsetDays)
  return d.toISOString().slice(0, 10)
}

function Tile({ label, value, tone = 'default' }: { label: string; value: string; tone?: 'default' | 'pos' | 'neg' | 'accent' }) {
  const toneCls =
    tone === 'pos' ? 'text-emerald-600' : tone === 'neg' ? 'text-red-600' : tone === 'accent' ? 'text-blue-600' : 'text-gray-900'
  return (
    <div className="rounded-lg border border-gray-200 bg-white px-3 py-2 shadow-sm">
      <div className="text-[11px] font-medium uppercase tracking-wide text-gray-500">{label}</div>
      <div className={`mt-0.5 text-lg font-semibold tabular-nums ${toneCls}`}>{value}</div>
    </div>
  )
}

function signalTone(sig?: string | null): 'pos' | 'neg' | 'default' {
  if (sig === 'BUY') return 'pos'
  if (sig === 'SELL') return 'neg'
  return 'default'
}

export default function RealtimePanel({ config, initial }: { config: Config; initial?: RealtimeInitial }) {
  const [models, setModels] = useState<RLModelInfo[]>([])
  const [symbol, setSymbol] = useState((initial?.symbol ?? 'BHEL.NS').toUpperCase())
  const [modelId, setModelId] = useState(initial?.model ?? '')
  const [interval, setIntervalVal] = useState(initial?.interval ?? '1d')
  const [replay, setReplay] = useState(true)
  const [replayStart, setReplayStart] = useState(initial?.start ?? todayISO(-30))
  const [replayEnd, setReplayEnd] = useState(initial?.end ?? todayISO())
  const [capital, setCapital] = useState(100000)
  const [speed, setSpeed] = useState(60)
  const autostartedRef = useRef(false)

  const [status, setStatus] = useState<RTStatus | null>(null)
  const [steps, setSteps] = useState<RTStep[]>([])
  const [chart, setChart] = useState('equity_curve')
  const [spec, setSpec] = useState<D3Spec | null>(null)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')

  const pollRef = useRef<number | null>(null)

  useEffect(() => {
    rlGetModels()
      .then((r) => {
        setModels(r.models)
        if (r.models.length && !modelId) setModelId(r.models[0].id)
      })
      .catch(() => {})
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  const refresh = useCallback(async () => {
    try {
      const [st, hist, ch] = await Promise.all([
        getRealtimeStatus(symbol || undefined),
        getRealtimeHistory(symbol || undefined, 200),
        getRealtimeChart(symbol || undefined, chart),
      ])
      setStatus(st)
      setSteps(hist)
      setSpec(ch.spec)
    } catch (e) {
      const detail = (e as { response?: { data?: { detail?: string } } })?.response?.data?.detail
      setError(detail ?? String(e))
    }
  }, [symbol, chart])

  // Poll while a session is running.
  useEffect(() => {
    refresh()
    const running = status?.running
    if (running) {
      // Poll faster during an accelerated replay so the stream is watchable.
      const pollMs = status?.replay ? REPLAY_POLL_MS : POLL_MS
      pollRef.current = window.setInterval(refresh, pollMs)
      return () => {
        if (pollRef.current) window.clearInterval(pollRef.current)
      }
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [refresh, status?.running, status?.replay])

  async function handleStart() {
    setBusy(true)
    setError('')
    try {
      if (replay && (!replayStart || !replayEnd)) throw new Error('Replay needs a start and end date')
      const st = await startRealtime({
        symbol,
        model_id: modelId,
        interval,
        replay,
        replay_start: replay ? replayStart : undefined,
        replay_end: replay ? replayEnd : undefined,
        capital,
        speed: replay ? speed : undefined,
        indian_api_key: config.indian_api_key || undefined,
      })
      setStatus(st)
      setTimeout(refresh, 600)
    } catch (e) {
      const detail = (e as { response?: { data?: { detail?: string } } })?.response?.data?.detail
      setError(detail ?? String(e))
    } finally {
      setBusy(false)
    }
  }

  async function handleStop() {
    setBusy(true)
    setError('')
    try {
      const st = await stopRealtime(symbol)
      setStatus(st)
    } catch (e) {
      const detail = (e as { response?: { data?: { detail?: string } } })?.response?.data?.detail
      setError(detail ?? String(e))
    } finally {
      setBusy(false)
    }
  }

  const running = !!status?.running

  // Auto-start once when opened via a deep link with ?autostart=1 (e.g. from the
  // open_buck_ui MCP tool), as soon as a model is available.
  useEffect(() => {
    if (!initial?.autostart || autostartedRef.current) return
    if (!modelId || running) return
    autostartedRef.current = true
    handleStart()
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [initial?.autostart, modelId, running])

  const pnlPct = status?.intraday_pnl_pct ?? null
  const recent = [...steps].reverse().slice(0, 40)

  return (
    <div className="flex flex-col gap-4">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <div>
          <h2 className="text-lg font-semibold text-gray-800">Realtime Simulation</h2>
          <p className="text-xs text-gray-500">
            Start an intraday RL session and monitor it live. Replay mode works any time.
          </p>
        </div>
        <div className="flex items-center gap-2">
          <span
            className={`inline-flex items-center gap-1.5 rounded-full px-2.5 py-1 text-xs font-medium ${
              running ? 'bg-emerald-100 text-emerald-700' : 'bg-gray-100 text-gray-600'
            }`}
          >
            <span className={`h-2 w-2 rounded-full ${running ? 'animate-pulse bg-emerald-500' : 'bg-gray-400'}`} />
            {running ? 'Running' : status?.status ?? 'Idle'}
          </span>
          <button
            onClick={refresh}
            className="rounded bg-gray-100 px-3 py-1.5 text-sm font-medium text-gray-700 hover:bg-gray-200"
          >
            ↻ Refresh
          </button>
        </div>
      </div>

      {/* Run controls */}
      <div className="rounded-lg border border-gray-200 bg-gray-50 p-3">
        <div className="grid grid-cols-2 gap-3 sm:grid-cols-3 lg:grid-cols-6">
          <label className="flex flex-col gap-1">
            <span className="text-xs font-medium text-gray-700">Symbol</span>
            <input
              className="rounded border border-gray-300 px-2 py-1.5 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
              value={symbol}
              onChange={(e) => setSymbol(e.target.value.toUpperCase())}
              disabled={running}
            />
          </label>
          <label className="flex flex-col gap-1">
            <span className="text-xs font-medium text-gray-700">Model</span>
            <select
              className="rounded border border-gray-300 px-2 py-1.5 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
              value={modelId}
              onChange={(e) => setModelId(e.target.value)}
              disabled={running}
            >
              {models.length === 0 && <option value="">No models — train one first</option>}
              {models.map((m) => (
                <option key={m.id} value={m.id}>
                  {m.id}
                </option>
              ))}
            </select>
          </label>
          <label className="flex flex-col gap-1">
            <span className="text-xs font-medium text-gray-700">Interval</span>
            <select
              className="rounded border border-gray-300 px-2 py-1.5 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
              value={interval}
              onChange={(e) => setIntervalVal(e.target.value)}
              disabled={running}
            >
              {['1m', '5m', '15m', '30m', '1h', '1d'].map((iv) => (
                <option key={iv} value={iv}>
                  {iv}
                </option>
              ))}
            </select>
          </label>
          <label className="flex flex-col gap-1">
            <span className="text-xs font-medium text-gray-700">Capital</span>
            <input
              type="number"
              className="rounded border border-gray-300 px-2 py-1.5 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
              value={capital}
              onChange={(e) => setCapital(Number(e.target.value))}
              disabled={running}
            />
          </label>
          <label className="flex items-center gap-2 pt-5">
            <input type="checkbox" checked={replay} onChange={(e) => setReplay(e.target.checked)} disabled={running} />
            <span className="text-xs font-medium text-gray-700">Replay</span>
          </label>
          <div className="flex items-end gap-2 pt-1">
            {!running ? (
              <button
                onClick={handleStart}
                disabled={busy || !modelId || !symbol}
                className="w-full rounded bg-blue-600 px-3 py-1.5 text-sm font-medium text-white hover:bg-blue-700 disabled:opacity-50"
              >
                ▶ Start
              </button>
            ) : (
              <button
                onClick={handleStop}
                disabled={busy}
                className="w-full rounded bg-red-600 px-3 py-1.5 text-sm font-medium text-white hover:bg-red-700 disabled:opacity-50"
              >
                ■ Stop
              </button>
            )}
          </div>
        </div>

        {replay && (
          <div className="mt-3 grid grid-cols-2 gap-3 sm:grid-cols-3 sm:max-w-2xl">
            <label className="flex flex-col gap-1">
              <span className="text-xs font-medium text-gray-700">Replay start</span>
              <input
                type="date"
                className="rounded border border-gray-300 px-2 py-1.5 text-sm"
                value={replayStart}
                onChange={(e) => setReplayStart(e.target.value)}
                disabled={running}
              />
            </label>
            <label className="flex flex-col gap-1">
              <span className="text-xs font-medium text-gray-700">Replay end</span>
              <input
                type="date"
                className="rounded border border-gray-300 px-2 py-1.5 text-sm"
                value={replayEnd}
                onChange={(e) => setReplayEnd(e.target.value)}
                disabled={running}
              />
            </label>
            <label className="flex flex-col gap-1">
              <span className="text-xs font-medium text-gray-700">Fast-forward</span>
              <select
                className="rounded border border-gray-300 px-2 py-1.5 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
                value={speed}
                onChange={(e) => setSpeed(Number(e.target.value))}
                disabled={running}
                title="Replay speed: per-bar delay = bar period / speed"
              >
                {SPEED_OPTIONS.map((o) => (
                  <option key={o.v} value={o.v}>
                    {o.label}
                  </option>
                ))}
              </select>
            </label>
          </div>
        )}
      </div>

      {error && <p className="rounded border border-red-200 bg-red-50 px-3 py-2 text-sm text-red-700">{error}</p>}

      {/* Live tiles */}
      <div className="grid grid-cols-2 gap-3 sm:grid-cols-3 lg:grid-cols-6">
        <Tile label="Signal" value={status?.last_signal ?? '—'} tone={signalTone(status?.last_signal)} />
        <Tile
          label="Target pos"
          value={status?.last_action != null ? status.last_action.toFixed(2) : '—'}
          tone="accent"
        />
        <Tile label="Price" value={status?.last_price != null ? status.last_price.toFixed(2) : '—'} />
        <Tile
          label="Equity"
          value={status?.equity != null ? status.equity.toLocaleString(undefined, { maximumFractionDigits: 0 }) : '—'}
        />
        <Tile
          label="PnL"
          value={pnlPct != null ? `${pnlPct >= 0 ? '+' : ''}${pnlPct.toFixed(2)}%` : '—'}
          tone={pnlPct == null ? 'default' : pnlPct >= 0 ? 'pos' : 'neg'}
        />
        <Tile label="Steps / updates" value={`${status?.n_steps ?? 0} / ${status?.n_updates ?? 0}`} />
      </div>

      {/* Chart + history: responsive two-column on large screens, stacked on small */}
      <div className="grid grid-cols-1 gap-4 lg:grid-cols-3">
        <div className="lg:col-span-2 rounded-lg border border-gray-200 bg-white p-3 shadow-sm">
          <div className="mb-2 flex items-center justify-between">
            <h3 className="text-sm font-semibold text-gray-700">Live chart</h3>
            <select
              className="rounded border border-gray-300 px-2 py-1 text-xs focus:outline-none focus:ring-2 focus:ring-blue-500"
              value={chart}
              onChange={(e) => setChart(e.target.value)}
            >
              {RT_CHARTS.map((c) => (
                <option key={c.id} value={c.id}>
                  {c.label}
                </option>
              ))}
            </select>
          </div>
          <D3Chart spec={spec} height={320} />
        </div>

        <div className="rounded-lg border border-gray-200 bg-white p-3 shadow-sm">
          <h3 className="mb-2 text-sm font-semibold text-gray-700">Step history</h3>
          <div className="max-h-80 overflow-auto">
            <table className="w-full text-left text-xs">
              <thead className="sticky top-0 bg-white text-gray-500">
                <tr>
                  <th className="py-1 pr-2 font-medium">Time</th>
                  <th className="py-1 pr-2 font-medium">Px</th>
                  <th className="py-1 pr-2 font-medium">Sig</th>
                  <th className="py-1 pr-2 font-medium text-right">Ret</th>
                  <th className="py-1 font-medium text-right">Equity</th>
                </tr>
              </thead>
              <tbody className="tabular-nums">
                {recent.length === 0 && (
                  <tr>
                    <td colSpan={5} className="py-6 text-center text-gray-400">
                      No steps yet
                    </td>
                  </tr>
                )}
                {recent.map((s, i) => {
                  const ret = typeof s.realized_return === 'number' ? s.realized_return * 100 : null
                  return (
                    <tr key={i} className="border-t border-gray-100">
                      <td className="py-1 pr-2 text-gray-600">{String(s.ts ?? '').slice(-8) || '—'}</td>
                      <td className="py-1 pr-2">{typeof s.price === 'number' ? s.price.toFixed(2) : '—'}</td>
                      <td
                        className={`py-1 pr-2 font-medium ${
                          s.signal === 'BUY' ? 'text-emerald-600' : s.signal === 'SELL' ? 'text-red-600' : 'text-gray-500'
                        }`}
                      >
                        {s.signal ?? '—'}
                      </td>
                      <td className={`py-1 pr-2 text-right ${ret == null ? '' : ret >= 0 ? 'text-emerald-600' : 'text-red-600'}`}>
                        {ret == null ? '—' : `${ret >= 0 ? '+' : ''}${ret.toFixed(3)}%`}
                      </td>
                      <td className="py-1 text-right">
                        {typeof s.equity === 'number' ? s.equity.toLocaleString(undefined, { maximumFractionDigits: 0 }) : '—'}
                      </td>
                    </tr>
                  )
                })}
              </tbody>
            </table>
          </div>
        </div>
      </div>
    </div>
  )
}
