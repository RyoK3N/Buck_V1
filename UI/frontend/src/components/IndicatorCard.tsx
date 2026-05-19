import type { AnalysisResult } from '../types'
import SignalBadge from './SignalBadge'

interface Props {
  result: AnalysisResult
}

const DISPLAY_KEYS: Record<string, string[]> = {
  moving_average: ['short_ma', 'long_ma', 'type'],
  rsi:            ['rsi', 'window', 'condition'],
  macd:           ['macd', 'signal', 'histogram'],
  obv:            ['obv', 'obv_trend', 'obv_ma'],
  candlestick_patterns: ['bullish_score', 'bearish_score', 'pattern_count'],
  support_resistance:   ['nearest_support', 'nearest_resistance', 'current_price'],
  lstm_prediction:      ['predicted_direction', 'predicted_probability', 'accuracy', 'training_samples'],
}

function formatValue(v: unknown): string {
  if (v === null || v === undefined) return '—'
  if (typeof v === 'number') return Number.isInteger(v) ? String(v) : v.toFixed(4)
  return String(v)
}

export default function IndicatorCard({ result }: Props) {
  const keys = DISPLAY_KEYS[result.analysis_type] ?? Object.keys(result.data).slice(0, 4)
  const signal = result.data['signal'] as string | undefined

  return (
    <div className="rounded-lg border border-gray-200 bg-white p-4 shadow-sm">
      <div className="mb-2 flex items-center justify-between">
        <h4 className="text-sm font-semibold capitalize text-gray-800">
          {result.analysis_type.replace(/_/g, ' ')}
        </h4>
        {signal && <SignalBadge signal={signal} />}
      </div>

      <div className="space-y-1">
        {keys.map(k => (
          <div key={k} className="flex justify-between text-xs">
            <span className="text-gray-500 capitalize">{k.replace(/_/g, ' ')}</span>
            <span className="font-mono text-gray-800">{formatValue(result.data[k])}</span>
          </div>
        ))}
      </div>

      {typeof result.data['note'] === 'string' && (
        <p className="mt-2 text-xs italic text-amber-600">{result.data['note']}</p>
      )}
    </div>
  )
}
