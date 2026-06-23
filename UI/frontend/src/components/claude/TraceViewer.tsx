import { useState } from 'react'
import type { ToolUseTrace } from '../../types'

interface Props {
  trace: ToolUseTrace[]
}

export default function TraceViewer({ trace }: Props) {
  const [open, setOpen] = useState(false)
  if (!trace || trace.length === 0) return null

  const okCount = trace.filter(t => t.ok).length
  const errCount = trace.length - okCount

  return (
    <div className="rounded border border-gray-200 bg-gray-50 text-xs">
      <button
        onClick={() => setOpen(o => !o)}
        className="flex w-full items-center justify-between px-3 py-2 text-left text-gray-700 hover:bg-gray-100"
      >
        <span>
          <span className="font-medium">Tool calls: {trace.length}</span>
          {okCount > 0 && <span className="ml-2 text-green-700">✓ {okCount}</span>}
          {errCount > 0 && <span className="ml-2 text-red-700">✗ {errCount}</span>}
        </span>
        <span className="text-gray-400">{open ? '▾' : '▸'}</span>
      </button>
      {open && (
        <ol className="divide-y divide-gray-200 border-t border-gray-200">
          {trace.map((t, i) => (
            <li key={i} className="px-3 py-2">
              <div className="flex items-center justify-between">
                <span className={`font-mono ${t.ok ? 'text-purple-700' : 'text-red-700'}`}>
                  iter {t.iter}: {t.name}
                </span>
                {!t.ok && t.error && (
                  <span className="text-red-600">{t.error.slice(0, 100)}</span>
                )}
              </div>
              {Object.keys(t.args ?? {}).length > 0 && (
                <pre className="mt-1 overflow-x-auto rounded bg-white px-2 py-1 text-[10px] text-gray-600">
                  {JSON.stringify(t.args, null, 2)}
                </pre>
              )}
            </li>
          ))}
        </ol>
      )}
    </div>
  )
}
