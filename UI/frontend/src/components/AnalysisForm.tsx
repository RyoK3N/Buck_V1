import { useEffect, useState } from 'react'
import { getIntervals } from '../api/client'
import type { Config } from '../types'

interface Props {
  config: Config
  onSubmit: (payload: {
    symbol: string
    start_date: string
    end_date: string
    interval: string
  }) => void
  loading: boolean
}

export default function AnalysisForm({ config, onSubmit, loading }: Props) {
  const [symbol, setSymbol] = useState('')
  const [startDate, setStartDate] = useState('')
  const [endDate, setEndDate] = useState('')
  const [interval, setInterval] = useState('1d')
  const [intervals, setIntervals] = useState<string[]>(['1h', '1d'])

  useEffect(() => {
    getIntervals().then(setIntervals).catch(() => {})
  }, [])

  function handleSubmit(e: React.FormEvent) {
    e.preventDefault()
    onSubmit({ symbol, start_date: startDate, end_date: endDate, interval })
  }

  const disabled = loading || !config.openai_api_key || !symbol || !startDate || !endDate

  return (
    <form onSubmit={handleSubmit} className="flex flex-col gap-4">
      <div className="grid grid-cols-2 gap-4">
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
      </div>

      {!config.openai_api_key && (
        <p className="text-xs text-red-500">Set your OpenAI API key in the Config panel.</p>
      )}

      <button
        type="submit"
        disabled={disabled}
        className="rounded bg-blue-600 px-4 py-2 text-sm font-semibold text-white hover:bg-blue-700 disabled:cursor-not-allowed disabled:opacity-50"
      >
        {loading ? 'Analyzing...' : 'Analyze'}
      </button>
    </form>
  )
}
