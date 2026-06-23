import type { AnalysisResult, AnalyzeResponse, BatchResponse } from '../types'
import ForecastCard from './ForecastCard'
import IndicatorCard from './IndicatorCard'
import SentimentSection from './SentimentSection'

// ── Helpers ──────────────────────────────────────────────────────────────────

/**
 * Unwrap the composite analysis structure returned by the backend.
 *
 * The backend returns analysis_results as:
 *   [{analysis_type: "composite_analysis", data: {analysis_results: {technical_analysis: ..., sentiment_analysis: ...}}}]
 *
 * We extract individual tool results from the technical_analysis and the
 * sentiment result from the sentiment_analysis.
 */
function extractResults(analysisResults: AnalysisResult[] | undefined) {
  const toolCards: AnalysisResult[] = []
  let sentiment: AnalysisResult | undefined
  let toolsUsed: string[] = []

  if (!analysisResults) return { toolCards, sentiment, toolsUsed }

  for (const ar of analysisResults) {
    if (ar.analysis_type === 'composite_analysis') {
      // Unwrap composite → nested analysis_results
      const nested = (ar.data as Record<string, unknown>)?.analysis_results as
        | Record<string, AnalysisResult>
        | undefined

      if (nested) {
        // Technical analysis → per-tool cards
        const tech = nested['technical_analysis']
        if (tech) {
          const techData = tech.data as Record<string, unknown>
          toolsUsed = (techData?.tools_used as string[]) ?? []
          const toolsResults = techData?.tools_results as
            | Record<string, Record<string, unknown>>
            | undefined

          if (toolsResults) {
            for (const [toolName, toolResult] of Object.entries(toolsResults)) {
              toolCards.push({
                symbol: ar.symbol,
                analysis_type: toolName,
                data: toolResult,
                timestamp: ar.timestamp,
                confidence: tech.confidence ?? 0,
              })
            }
          }
        }

        // Sentiment
        const sent = nested['sentiment_analysis']
        if (sent) {
          sentiment = sent
        }
      }
    } else if (ar.analysis_type === 'sentiment_analysis') {
      sentiment = ar
    } else if (ar.analysis_type === 'technical_analysis') {
      // Direct technical analysis (non-composite path)
      const techData = ar.data as Record<string, unknown>
      toolsUsed = (techData?.tools_used as string[]) ?? []
      const toolsResults = techData?.tools_results as
        | Record<string, Record<string, unknown>>
        | undefined

      if (toolsResults) {
        for (const [toolName, toolResult] of Object.entries(toolsResults)) {
          toolCards.push({
            symbol: ar.symbol,
            analysis_type: toolName,
            data: toolResult,
            timestamp: ar.timestamp,
            confidence: ar.confidence ?? 0,
          })
        }
      }
    } else {
      toolCards.push(ar)
    }
  }

  return { toolCards, sentiment, toolsUsed }
}

// ── Single result ─────────────────────────────────────────────────────────────

function SingleResult({ result }: { result: AnalyzeResponse }) {
  if (result.error) {
    return (
      <div className="rounded-lg border border-red-200 bg-red-50 p-4 text-sm text-red-700">
        <strong>{result.symbol}</strong> — {result.error}
      </div>
    )
  }

  const { toolCards, sentiment, toolsUsed } = extractResults(result.analysis_results)

  return (
    <div className="space-y-4">
      <div className="flex items-center gap-2 text-sm text-gray-500">
        <span className="font-semibold text-gray-800">{result.symbol}</span>
        <span>·</span>
        <span>{result.data_info?.interval}</span>
        <span>·</span>
        <span>{result.data_info?.data_points} data points</span>
        <span>·</span>
        <span>{new Date(result.timestamp).toLocaleString()}</span>
      </div>

      {toolsUsed.length > 0 && (
        <div className="flex flex-wrap items-center gap-1.5 text-xs text-gray-500">
          <span className="font-medium text-gray-600">Tools used:</span>
          {toolsUsed.map(t => (
            <span
              key={t}
              className="rounded bg-gray-100 px-1.5 py-0.5 text-[11px] font-mono text-gray-600"
            >
              {t}
            </span>
          ))}
        </div>
      )}

      {result.forecast && <ForecastCard forecast={result.forecast} />}

      {toolCards.length > 0 && (
        <div>
          <h3 className="mb-2 text-sm font-semibold text-gray-600">Technical Indicators</h3>
          <div className="grid grid-cols-1 gap-3 sm:grid-cols-2 xl:grid-cols-3">
            {toolCards.map(r => (
              <IndicatorCard key={r.analysis_type} result={r} />
            ))}
          </div>
        </div>
      )}

      {sentiment && <SentimentSection result={sentiment} />}
    </div>
  )
}

// ── Props ─────────────────────────────────────────────────────────────────────

interface Props {
  result: AnalyzeResponse | BatchResponse | null
  mode: 'single' | 'batch'
}

export default function ResultsPanel({ result, mode }: Props) {
  if (!result) return null

  if (mode === 'single') {
    return <SingleResult result={result as AnalyzeResponse} />
  }

  // Batch mode
  const batch = result as BatchResponse
  const { summary, results } = batch

  return (
    <div className="space-y-6">
      <div className="flex gap-6 rounded-lg border border-gray-200 bg-white p-4 text-sm shadow-sm">
        <div>
          <span className="text-gray-500">Successful </span>
          <span className="font-semibold text-green-700">{summary.successful}</span>
        </div>
        <div>
          <span className="text-gray-500">Failed </span>
          <span className="font-semibold text-red-700">{summary.failed}</span>
        </div>
        <div>
          <span className="text-gray-500">Avg Confidence </span>
          <span className="font-semibold text-blue-700">
            {Math.round((summary.avg_confidence ?? 0) * 100)}%
          </span>
        </div>
      </div>

      {Object.entries(results).map(([sym, res]) => (
        <div key={sym} className="rounded-lg border border-gray-100 p-4 shadow-sm">
          <SingleResult result={res} />
        </div>
      ))}
    </div>
  )
}
