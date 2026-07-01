import { useEffect, useState } from 'react'
import { getIntervals } from '../api/client'

interface Props {
  onSubmit: (payload: {
    symbols: string[]
    start_date: string
    end_date: string
    interval: string
    max_concurrent: number
  }) => void
  loading: boolean
  /** True if either the user typed a key or the server already has one configured (.env). */
  openaiKeyAvailable: boolean
}

export default function BatchForm({ onSubmit, loading, openaiKeyAvailable }: Props) {
  const [rawSymbols, setRawSymbols] = useState('')
  const [startDate, setStartDate] = useState('')
  const [endDate, setEndDate] = useState('')
  const [interval, setInterval] = useState('1d')
  const [maxConcurrent, setMaxConcurrent] = useState(3)
  const [intervals, setIntervals] = useState<string[]>(['1h', '1d'])

  useEffect(() => {
    getIntervals().then(setIntervals).catch(() => {})
  }, [])

  function handleSubmit(e: React.FormEvent) {
    e.preventDefault()
    const symbols = rawSymbols
      .split(/[\n,]+/)
      .map(s => s.trim().toUpperCase())
      .filter(Boolean)
    onSubmit({ symbols, start_date: startDate, end_date: endDate, interval, max_concurrent: maxConcurrent })
  }

  const symbols = rawSymbols.split(/[\n,]+/).map(s => s.trim()).filter(Boolean)
  const disabled = loading || !openaiKeyAvailable || symbols.length === 0 || !startDate || !endDate

  return (
    <form onSubmit={handleSubmit} className="flex flex-col gap-4">
      <label className="flex flex-col gap-1">
        <span className="text-xs font-medium text-gray-700">
          Symbols * <span className="font-normal text-gray-400">(comma or newline separated)</span>
        </span>
        <textarea
          className="rounded border border-gray-300 px-2 py-1.5 text-sm uppercase focus:outline-none focus:ring-2 focus:ring-blue-500"
          rows={3}
          placeholder={"RELIANCE.NS, TCS.NS\nINFY.NS"}
          value={rawSymbols}
          onChange={e => setRawSymbols(e.target.value)}
        />
        {symbols.length > 0 && (
          <span className="text-xs text-gray-400">{symbols.length} symbol{symbols.length !== 1 ? 's' : ''}</span>
        )}
      </label>

      <div className="grid grid-cols-2 gap-4">
        <label className="flex flex-col gap-1">
          <span className="text-xs font-medium text-gray-700">Start Date *</span>
          <input
            type="date"
            className="rounded border border-gray-300 px-2 py-1.5 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
            value={startDate}
            onChange={e => setStartDate(e.target.value)}
          />
        </label>

        <label className="flex flex-col gap-1">
          <span className="text-xs font-medium text-gray-700">End Date *</span>
          <input
            type="date"
            className="rounded border border-gray-300 px-2 py-1.5 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
            value={endDate}
            onChange={e => setEndDate(e.target.value)}
          />
        </label>

        <label className="flex flex-col gap-1">
          <span className="text-xs font-medium text-gray-700">Interval</span>
          <select
            className="rounded border border-gray-300 px-2 py-1.5 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
            value={interval}
            onChange={e => setInterval(e.target.value)}
          >
            {intervals.map(iv => (
              <option key={iv} value={iv}>{iv}</option>
            ))}
          </select>
        </label>

        <label className="flex flex-col gap-1">
          <span className="text-xs font-medium text-gray-700">Max Concurrent</span>
          <input
            type="number"
            min={1}
            max={10}
            className="rounded border border-gray-300 px-2 py-1.5 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
            value={maxConcurrent}
            onChange={e => setMaxConcurrent(Number(e.target.value))}
          />
        </label>
      </div>

      {!openaiKeyAvailable && (
        <p className="text-xs text-red-500">Set your OpenAI API key in the Config panel.</p>
      )}

      <button
        type="submit"
        disabled={disabled}
        className="rounded bg-blue-600 px-4 py-2 text-sm font-semibold text-white hover:bg-blue-700 disabled:cursor-not-allowed disabled:opacity-50"
      >
        {loading ? 'Analyzing batch...' : `Analyze ${symbols.length > 0 ? symbols.length + ' symbols' : ''}`}
      </button>
    </form>
  )
}
