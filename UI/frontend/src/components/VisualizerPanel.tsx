import { useEffect, useRef, useState } from 'react'
import { getChartTypes, getIntervals, visualize } from '../api/client'
import type { ChartTypeInfo, Config, PlotlyFigure } from '../types'
import LoadingSpinner from './LoadingSpinner'

declare const Plotly: {
  react: (el: HTMLElement, data: unknown[], layout: unknown, config?: unknown) => void
  relayout: (el: HTMLElement, update: unknown) => void
}

// Group chart types by category for the selector
const GROUPS: Record<string, string[]> = {
  'Price Action':  ['candlestick', 'price_ma', 'fibonacci', 'vwap'],
  'Trend':         ['bollinger', 'ichimoku', 'adx'],
  'Momentum':      ['rsi', 'macd', 'stochastic'],
  'Volume':        ['obv'],
  'Volatility':    ['volatility', 'atr'],
  'Statistical':   ['returns'],
}

interface Props {
  config: Config
}

export default function VisualizerPanel({ config }: Props) {
  const [chartTypes, setChartTypes] = useState<ChartTypeInfo[]>([])
  const [intervals, setIntervals] = useState<string[]>(['1d'])

  const [symbol, setSymbol] = useState('')
  const [startDate, setStartDate] = useState('')
  const [endDate, setEndDate] = useState('')
  const [interval, setInterval] = useState('1d')
  const [chartType, setChartType] = useState('candlestick')

  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [description, setDescription] = useState('')

  const plotDivRef = useRef<HTMLDivElement>(null)
  const [hasChart, setHasChart] = useState(false)

  useEffect(() => {
    getChartTypes().then(ct => {
      setChartTypes(ct)
      if (ct.length > 0) setChartType(ct[0].id)
    }).catch(() => {})
    getIntervals().then(setIntervals).catch(() => {})
  }, [])

  function renderPlotly(fig: PlotlyFigure) {
    if (!plotDivRef.current) return
    const el = plotDivRef.current
    Plotly.react(
      el,
      fig.data as unknown[],
      {
        ...fig.layout,
        autosize: true,
        height: 600,
        paper_bgcolor: 'white',
        plot_bgcolor: 'white',
      },
      {
        responsive: true,
        displaylogo: false,
        modeBarButtonsToAdd: ['drawline', 'drawopenpath', 'eraseshape'],
        toImageButtonOptions: {
          format: 'png',
          filename: `buck_${chartType}_${symbol}`,
          height: 900,
          width: 1400,
          scale: 2,
        },
      },
    )
    setHasChart(true)
  }

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault()
    setLoading(true)
    setError(null)

    try {
      const res = await visualize({
        symbol,
        start_date: startDate,
        end_date: endDate,
        interval,
        chart_type: chartType,
        indian_api_key: config.indian_api_key || undefined,
      })
      setDescription(res.description)
      requestAnimationFrame(() => renderPlotly(res.chart))
    } catch (err: unknown) {
      const detail = (err as { response?: { data?: { detail?: string } } })
        ?.response?.data?.detail
      setError(detail ?? (err instanceof Error ? err.message : 'Chart generation failed'))
    } finally {
      setLoading(false)
    }
  }

  const selectedTypeInfo = chartTypes.find(c => c.id === chartType)
  const disabled = loading || !symbol || !startDate || !endDate
  const chartIndex = new Map(chartTypes.map(c => [c.id, c]))

  return (
    <div className="flex flex-col gap-5">
      {/* ── Form ─────────────────────────────────────────────────── */}
      <div className="rounded-lg border border-gray-200 bg-white p-4 shadow-sm">
        <form onSubmit={handleSubmit} className="flex flex-col gap-4">
          {/* Row 1: Symbol + dates + interval */}
          <div className="grid grid-cols-4 gap-3">
            <label className="flex flex-col gap-1">
              <span className="text-xs font-medium text-gray-700">Symbol *</span>
              <input
                className="rounded border border-gray-300 px-2 py-1.5 text-sm uppercase focus:outline-none focus:ring-2 focus:ring-blue-500"
                placeholder="RELIANCE.NS"
                value={symbol}
                onChange={e => setSymbol(e.target.value.toUpperCase())}
              />
            </label>
            <label className="flex flex-col gap-1">
              <span className="text-xs font-medium text-gray-700">Start *</span>
              <input type="date"
                className="rounded border border-gray-300 px-2 py-1.5 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
                value={startDate} onChange={e => setStartDate(e.target.value)} />
            </label>
            <label className="flex flex-col gap-1">
              <span className="text-xs font-medium text-gray-700">End *</span>
              <input type="date"
                className="rounded border border-gray-300 px-2 py-1.5 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
                value={endDate} onChange={e => setEndDate(e.target.value)} />
            </label>
            <label className="flex flex-col gap-1">
              <span className="text-xs font-medium text-gray-700">Interval</span>
              <select
                className="rounded border border-gray-300 px-2 py-1.5 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
                value={interval} onChange={e => setInterval(e.target.value)}>
                {intervals.map(iv => <option key={iv} value={iv}>{iv}</option>)}
              </select>
            </label>
          </div>

          {/* Row 2: Chart type grouped selector */}
          <div>
            <span className="mb-1.5 block text-xs font-medium text-gray-700">Chart Type</span>
            <div className="flex flex-wrap gap-x-5 gap-y-2">
              {Object.entries(GROUPS).map(([group, ids]) => {
                const available = ids.filter(id => chartIndex.has(id))
                if (available.length === 0) return null
                return (
                  <div key={group} className="flex items-center gap-1.5">
                    <span className="text-[10px] font-semibold uppercase tracking-wider text-gray-400">
                      {group}
                    </span>
                    {available.map(id => {
                      const info = chartIndex.get(id)
                      const active = chartType === id
                      return (
                        <button
                          key={id}
                          type="button"
                          onClick={() => setChartType(id)}
                          title={info?.description}
                          className={`rounded-full px-2.5 py-0.5 text-xs font-medium transition-colors ${
                            active
                              ? 'bg-blue-600 text-white shadow-sm'
                              : 'bg-gray-100 text-gray-600 hover:bg-gray-200'
                          }`}
                        >
                          {info?.name?.replace(/ — .+/, '') ?? id}
                        </button>
                      )
                    })}
                  </div>
                )
              })}
            </div>
          </div>

          {/* Description + submit */}
          {selectedTypeInfo && (
            <p className="rounded bg-gray-50 px-3 py-2 text-xs leading-relaxed text-gray-600">
              {selectedTypeInfo.description}
            </p>
          )}

          <button type="submit" disabled={disabled}
            className="rounded bg-blue-600 px-4 py-2 text-sm font-semibold text-white hover:bg-blue-700 disabled:cursor-not-allowed disabled:opacity-50">
            {loading ? 'Generating chart…' : 'Generate Chart'}
          </button>
        </form>
      </div>

      {/* ── Error ────────────────────────────────────────────────── */}
      {error && (
        <div className="rounded-lg border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-700">
          <strong>Error:</strong> {error}
        </div>
      )}

      {/* ── Loading ──────────────────────────────────────────────── */}
      {loading && <LoadingSpinner />}

      {/* ── Chart ────────────────────────────────────────────────── */}
      <div className={`rounded-lg border border-gray-200 bg-white shadow-sm transition-all ${
        hasChart ? 'block' : 'hidden'
      }`}>
        <div ref={plotDivRef} style={{ width: '100%', minHeight: '600px' }} />
        {description && (
          <p className="border-t border-gray-100 px-4 py-2.5 text-xs leading-relaxed text-gray-500">
            {description}
          </p>
        )}
      </div>
    </div>
  )
}
