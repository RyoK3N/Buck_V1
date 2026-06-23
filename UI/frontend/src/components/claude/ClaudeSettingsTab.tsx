import type { Config } from '../../types'

interface Props {
  config: Config
}

export default function ClaudeSettingsTab({ config }: Props) {
  const hasKey = !!config.anthropic_api_key

  return (
    <div className="space-y-4">
      <div className="rounded-lg border border-gray-200 bg-white p-4 shadow-sm space-y-2">
        <h3 className="text-sm font-semibold text-gray-700">Anthropic / Claude</h3>
        <div className="text-sm">
          API key: {hasKey
            ? <span className="text-green-700">configured ({config.anthropic_api_key.slice(0, 7)}…)</span>
            : <span className="text-red-700">not set — open the sidebar to set ANTHROPIC_API_KEY</span>}
        </div>
        <div className="text-sm">
          Model: <code className="rounded bg-gray-100 px-1.5 py-0.5">{config.claude_model || '(default)'}</code>
        </div>
        <p className="text-xs text-gray-500">
          Anthropic key + model can be edited in the left sidebar (sessionStorage-backed).
          To persist across sessions, set <code>ANTHROPIC_API_KEY</code> and <code>CLAUDE_MODEL</code> in your <code>.env</code>.
        </p>
      </div>

      <div className="rounded-lg border border-gray-200 bg-white p-4 shadow-sm space-y-2">
        <h3 className="text-sm font-semibold text-gray-700">MCP server</h3>
        <p className="text-sm text-gray-700">
          Run the standalone MCP server for Claude Desktop:
        </p>
        <pre className="rounded bg-gray-50 p-2 text-xs">python -m mcp_server.runner --transport stdio</pre>
        <p className="text-sm text-gray-700">
          To enable it inside the FastAPI process at <code>/mcp-sse</code> set <code>MOUNT_MCP_IN_API=true</code> in <code>.env</code> and restart.
        </p>
      </div>

      <div className="rounded-lg border border-gray-200 bg-white p-4 shadow-sm space-y-2">
        <h3 className="text-sm font-semibold text-gray-700">Accuracy scheduler</h3>
        <p className="text-sm text-gray-700">
          The intraday poller fetches latest prices for symbols with open predictions and reconciles
          evaluations. Tune the cadence with <code>ACCURACY_POLL_INTERVAL_MINUTES</code> in your <code>.env</code>.
        </p>
      </div>
    </div>
  )
}
