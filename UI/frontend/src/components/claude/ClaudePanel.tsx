import { useState } from 'react'
import type { Config } from '../../types'
import ChatTab from './ChatTab'
import PredictionsTab from './PredictionsTab'
import AccuracyDashboardTab from './AccuracyDashboardTab'
import MCPToolsTab from './MCPToolsTab'
import ClaudeSettingsTab from './ClaudeSettingsTab'

type SubTab = 'chat' | 'predictions' | 'accuracy' | 'mcp' | 'settings'

const LABELS: Record<SubTab, string> = {
  chat: 'Chat',
  predictions: 'Predictions',
  accuracy: 'Accuracy Dashboard',
  mcp: 'MCP Tools',
  settings: 'Settings',
}

interface Props {
  config: Config
}

export default function ClaudePanel({ config }: Props) {
  const [sub, setSub] = useState<SubTab>('chat')

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <div className="flex gap-1 rounded-lg border border-gray-200 bg-white p-1 shadow-sm w-fit">
          {(['chat', 'predictions', 'accuracy', 'mcp', 'settings'] as SubTab[]).map(t => (
            <button
              key={t}
              onClick={() => setSub(t)}
              className={`rounded px-3 py-1.5 text-sm font-medium transition-colors ${
                sub === t ? 'bg-purple-600 text-white' : 'text-gray-600 hover:text-gray-900'
              }`}
            >
              {LABELS[t]}
            </button>
          ))}
        </div>
        {!config.anthropic_api_key && (
          <span className="rounded bg-amber-50 px-2 py-0.5 text-xs text-amber-700">
            Set ANTHROPIC_API_KEY in sidebar to enable Chat / Predict
          </span>
        )}
      </div>

      {sub === 'chat' && <ChatTab config={config} />}
      {sub === 'predictions' && <PredictionsTab />}
      {sub === 'accuracy' && <AccuracyDashboardTab />}
      {sub === 'mcp' && <MCPToolsTab />}
      {sub === 'settings' && <ClaudeSettingsTab config={config} />}
    </div>
  )
}
