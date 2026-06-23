import { useEffect, useState } from 'react'
import { getMCPStatus, getMCPTools, invokeMCPTool } from '../../api/client'
import type { MCPStatus, MCPToolInfo } from '../../types'

export default function MCPToolsTab() {
  const [status, setStatus] = useState<MCPStatus | null>(null)
  const [tools, setTools] = useState<MCPToolInfo[]>([])
  const [selected, setSelected] = useState<string | null>(null)
  const [argsText, setArgsText] = useState<string>('{}')
  const [result, setResult] = useState<unknown>(null)
  const [err, setErr] = useState<string | null>(null)
  const [running, setRunning] = useState(false)

  useEffect(() => {
    void Promise.all([getMCPStatus(), getMCPTools()]).then(([s, t]) => {
      setStatus(s)
      setTools(t)
    })
  }, [])

  async function refreshTools() {
    setTools(await getMCPTools())
  }

  async function runTool() {
    if (!selected) return
    setRunning(true)
    setErr(null)
    setResult(null)
    try {
      const parsed = argsText.trim() ? JSON.parse(argsText) : {}
      const r = await invokeMCPTool(selected, parsed)
      setResult(r)
      await refreshTools()
    } catch (e: unknown) {
      setErr(e instanceof Error ? e.message : 'failed')
    } finally {
      setRunning(false)
    }
  }

  const selectedTool = tools.find(t => t.name === selected)

  return (
    <div className="space-y-4">
      <div className="rounded-lg border border-gray-200 bg-white p-3 shadow-sm text-sm">
        {status ? (
          <div className="flex flex-wrap gap-4">
            <span><span className="text-gray-500">Mounted in API:</span> <code>{String(status.mounted_in_api)}</code></span>
            <span><span className="text-gray-500">Mount path:</span> <code>{status.mount_path || '(none)'}</code></span>
            <span><span className="text-gray-500">Transport:</span> <code>{status.transport}</code></span>
            <span><span className="text-gray-500">Tools:</span> <code>{status.tool_count}</code></span>
            <code className="ml-auto rounded bg-gray-100 px-2 py-0.5 text-xs">{status.standalone_runner}</code>
          </div>
        ) : <span className="text-gray-500">loading status…</span>}
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        <div className="rounded-lg border border-gray-200 bg-white shadow-sm overflow-y-auto" style={{ maxHeight: '70vh' }}>
          <div className="border-b border-gray-200 px-3 py-2 text-sm font-semibold text-gray-700">
            Tools ({tools.length})
          </div>
          <ul className="divide-y divide-gray-100">
            {tools.map(t => (
              <li
                key={t.name}
                onClick={() => { setSelected(t.name); setArgsText('{}'); setResult(null); setErr(null) }}
                className={`px-3 py-2 cursor-pointer hover:bg-gray-50 ${selected === t.name ? 'bg-purple-50' : ''}`}
              >
                <div className="flex items-center justify-between">
                  <span className="font-mono text-sm text-purple-700">{t.name}</span>
                  {t.last_call && (
                    <span className={`text-xs ${t.last_call.ok ? 'text-green-700' : 'text-red-700'}`}>
                      {t.last_call.latency_ms.toFixed(0)}ms · {t.last_call.ts.slice(11, 19)}
                    </span>
                  )}
                </div>
                <div className="text-xs text-gray-600 line-clamp-2">{t.description}</div>
              </li>
            ))}
          </ul>
        </div>

        <div className="rounded-lg border border-gray-200 bg-white p-3 shadow-sm space-y-3">
          {selectedTool ? (
            <>
              <div>
                <div className="text-xs uppercase tracking-wide text-gray-500">Tool</div>
                <div className="font-mono text-sm text-purple-700">{selectedTool.name}</div>
                <p className="mt-1 text-sm text-gray-700">{selectedTool.description}</p>
              </div>
              <div>
                <div className="text-xs uppercase tracking-wide text-gray-500 mb-1">Input schema</div>
                <pre className="rounded bg-gray-50 p-2 text-[11px] text-gray-700 overflow-x-auto">
                  {JSON.stringify(selectedTool.input_schema, null, 2)}
                </pre>
              </div>
              <div>
                <div className="text-xs uppercase tracking-wide text-gray-500 mb-1">Arguments (JSON)</div>
                <textarea
                  className="w-full rounded border border-gray-300 px-2 py-1 text-sm font-mono"
                  rows={6}
                  value={argsText}
                  onChange={e => setArgsText(e.target.value)}
                />
              </div>
              <div className="flex items-center gap-2">
                <button
                  onClick={() => void runTool()}
                  disabled={running}
                  className="rounded bg-purple-600 px-3 py-1.5 text-sm font-medium text-white hover:bg-purple-700 disabled:opacity-50"
                >
                  {running ? 'Running…' : 'Invoke'}
                </button>
                {err && <span className="text-xs text-red-600">{err}</span>}
              </div>
              {result !== null && (
                <div>
                  <div className="text-xs uppercase tracking-wide text-gray-500 mb-1">Result</div>
                  <pre className="rounded bg-gray-50 p-2 text-[11px] text-gray-700 overflow-x-auto max-h-72">
                    {JSON.stringify(result, null, 2)}
                  </pre>
                </div>
              )}
            </>
          ) : (
            <div className="text-sm text-gray-500">Select a tool from the list to test it.</div>
          )}
        </div>
      </div>
    </div>
  )
}
