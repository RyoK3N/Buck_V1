import type { AnalysisResult } from '../types'

interface Props {
  result: AnalysisResult
}

function ScoreBar({ label, value }: { label: string; value: number }) {
  const pct = Math.max(0, Math.min(100, Math.round(value * 100)))
  const color = value >= 0.6 ? 'bg-green-500' : value >= 0.4 ? 'bg-yellow-400' : 'bg-red-400'

  return (
    <div>
      <div className="mb-0.5 flex justify-between text-xs">
        <span className="text-gray-600">{label}</span>
        <span className="font-medium text-gray-800">{pct}%</span>
      </div>
      <div className="h-1.5 w-full overflow-hidden rounded-full bg-gray-100">
        <div className={`h-1.5 rounded-full ${color}`} style={{ width: `${pct}%` }} />
      </div>
    </div>
  )
}

export default function SentimentSection({ result }: Props) {
  const d = result.data as Record<string, unknown>

  const overall = typeof d['overall_sentiment'] === 'number' ? d['overall_sentiment'] as number : null
  const newsCount = typeof d['news_count'] === 'number' ? d['news_count'] as number : null
  const freshness = typeof d['freshness_score'] === 'number' ? d['freshness_score'] as number : null
  const topics = typeof d['topic_distribution'] === 'object' && d['topic_distribution']
    ? d['topic_distribution'] as Record<string, number>
    : null

  return (
    <div className="rounded-lg border border-gray-200 bg-white p-4 shadow-sm">
      <h3 className="mb-3 font-semibold text-gray-800">Sentiment Analysis</h3>

      <div className="space-y-2">
        {overall !== null && <ScoreBar label="Overall Sentiment" value={overall} />}
        {freshness !== null && <ScoreBar label="News Freshness" value={freshness} />}
      </div>

      {newsCount !== null && (
        <p className="mt-2 text-xs text-gray-500">{newsCount} articles analyzed</p>
      )}

      {topics && Object.keys(topics).length > 0 && (
        <div className="mt-3">
          <p className="mb-1 text-xs font-medium text-gray-600">Topics</p>
          <div className="flex flex-wrap gap-1">
            {Object.entries(topics).map(([topic, weight]) => (
              <span
                key={topic}
                className="rounded bg-gray-100 px-2 py-0.5 text-xs text-gray-700"
                title={`weight: ${(weight as number).toFixed(3)}`}
              >
                {topic}
              </span>
            ))}
          </div>
        </div>
      )}
    </div>
  )
}
