import { useRef, useState } from 'react'
import { claudeChat } from '../../api/client'
import type { ChatMessage, Config, ToolUseTrace } from '../../types'
import TraceViewer from './TraceViewer'

interface Props {
  config: Config
}

interface AssistantTurn {
  text: string
  trace: ToolUseTrace[]
}

export default function ChatTab({ config }: Props) {
  const [messages, setMessages] = useState<ChatMessage[]>([])
  const [traces, setTraces] = useState<Record<number, AssistantTurn>>({})
  const [input, setInput] = useState('')
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const listRef = useRef<HTMLDivElement>(null)

  async function send() {
    const text = input.trim()
    if (!text || loading) return
    if (!config.anthropic_api_key) {
      setError('Anthropic API key required — set it in the sidebar.')
      return
    }
    setError(null)
    const next: ChatMessage[] = [...messages, { role: 'user', content: text }]
    setMessages(next)
    setInput('')
    setLoading(true)
    try {
      const res = await claudeChat({
        messages: next,
        anthropic_api_key: config.anthropic_api_key,
        claude_model: config.claude_model || undefined,
      })
      const finalMessages: ChatMessage[] = [...next, { role: 'assistant', content: res.text }]
      setMessages(finalMessages)
      setTraces(t => ({ ...t, [finalMessages.length - 1]: { text: res.text, trace: res.trace } }))
      setTimeout(() => listRef.current?.scrollTo({ top: listRef.current.scrollHeight, behavior: 'smooth' }), 50)
    } catch (e: unknown) {
      const msg = e instanceof Error ? e.message : 'Chat failed'
      const detail = (e as { response?: { data?: { detail?: string } } })?.response?.data?.detail
      setError(detail ?? msg)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="rounded-lg border border-gray-200 bg-white shadow-sm flex flex-col" style={{ height: '70vh' }}>
      <div ref={listRef} className="flex-1 overflow-y-auto p-4 space-y-4">
        {messages.length === 0 && (
          <div className="text-sm text-gray-500">
            Ask Claude anything about the markets Buck supports. Claude will call Buck's tools
            (single_analyze, batch_analyze, rl_predict, visualize, get_prediction_accuracy …) to gather
            evidence before answering. Try: <em>"Forecast tomorrow's close for INFY.NS and compare to TCS.NS"</em>.
          </div>
        )}
        {messages.map((m, i) => (
          <div key={i} className={m.role === 'user' ? 'flex justify-end' : 'flex flex-col items-start gap-2'}>
            <div
              className={`max-w-[80%] whitespace-pre-wrap rounded-lg px-3 py-2 text-sm ${
                m.role === 'user'
                  ? 'bg-blue-600 text-white'
                  : 'bg-gray-100 text-gray-900'
              }`}
            >
              {m.content}
            </div>
            {m.role === 'assistant' && traces[i] && traces[i].trace.length > 0 && (
              <div className="w-full max-w-[80%]">
                <TraceViewer trace={traces[i].trace} />
              </div>
            )}
          </div>
        ))}
        {loading && (
          <div className="text-xs text-gray-500 italic">Claude is thinking and calling tools…</div>
        )}
      </div>
      {error && (
        <div className="border-t border-red-200 bg-red-50 px-4 py-2 text-sm text-red-700">{error}</div>
      )}
      <div className="border-t border-gray-200 p-3 flex gap-2">
        <textarea
          className="flex-1 rounded border border-gray-300 px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-purple-500"
          rows={2}
          placeholder="Ask Claude about a stock, peer comparison, or accuracy of past forecasts…"
          value={input}
          onChange={e => setInput(e.target.value)}
          onKeyDown={e => {
            if (e.key === 'Enter' && (e.metaKey || e.ctrlKey)) {
              e.preventDefault()
              void send()
            }
          }}
          disabled={loading}
        />
        <button
          onClick={() => void send()}
          disabled={loading || !input.trim()}
          className="rounded bg-purple-600 px-4 py-2 text-sm font-medium text-white disabled:opacity-50 hover:bg-purple-700"
        >
          Send
        </button>
      </div>
    </div>
  )
}
