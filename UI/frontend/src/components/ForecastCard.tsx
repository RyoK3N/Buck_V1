import type { Forecast } from '../types'

interface Props {
  forecast: Forecast
}

function PriceRow({ label, value }: { label: string; value: number }) {
  return (
    <div className="flex justify-between">
      <span className="text-sm text-gray-500">{label}</span>
      <span className="text-sm font-mono font-medium text-gray-900">
        {typeof value === 'number' ? value.toFixed(2) : '—'}
      </span>
    </div>
  )
}

export default function ForecastCard({ forecast }: Props) {
  const pct = Math.round((forecast.confidence ?? 0) * 100)

  return (
    <div className="rounded-lg border border-gray-200 bg-white p-4 shadow-sm">
      <div className="mb-3 flex items-center justify-between">
        <h3 className="font-semibold text-gray-800">Forecast — {forecast.date}</h3>
        <span className="text-sm font-medium text-blue-600">{pct}% confidence</span>
      </div>

      {/* Confidence bar */}
      <div className="mb-4 h-2 w-full overflow-hidden rounded-full bg-gray-100">
        <div
          className="h-2 rounded-full bg-blue-500 transition-all"
          style={{ width: `${pct}%` }}
        />
      </div>

      <div className="mb-4 grid grid-cols-2 gap-x-8 gap-y-1">
        <PriceRow label="Open" value={forecast.open} />
        <PriceRow label="High" value={forecast.high} />
        <PriceRow label="Low"  value={forecast.low} />
        <PriceRow label="Close" value={forecast.close} />
      </div>

      {forecast.reasoning && (
        <p className="text-xs leading-relaxed text-gray-600">{forecast.reasoning}</p>
      )}
    </div>
  )
}
