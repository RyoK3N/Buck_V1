interface Props {
  signal: string
}

const COLORS: Record<string, string> = {
  BUY:  'bg-green-100 text-green-800',
  SELL: 'bg-red-100 text-red-800',
  HOLD: 'bg-yellow-100 text-yellow-800',
}

export default function SignalBadge({ signal }: Props) {
  const upper = signal?.toUpperCase() ?? 'HOLD'
  const cls = COLORS[upper] ?? 'bg-gray-100 text-gray-800'
  return (
    <span className={`inline-block rounded-full px-3 py-0.5 text-sm font-semibold ${cls}`}>
      {upper}
    </span>
  )
}
