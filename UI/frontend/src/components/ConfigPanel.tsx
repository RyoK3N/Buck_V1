import { useCallback, useEffect, useRef, useState } from 'react'
import { getServerConfig } from '../api/client'
import type { Config } from '../types'

const SESSION_KEY = 'buck_config'
const LOADED_FLAG = 'buck_config_loaded'

const DEFAULTS: Config = {
  openai_api_key: '',
  indian_api_key: '',
  model: 'gpt-4o',
  base_url: '',
  anthropic_api_key: '',
  claude_model: 'claude-opus-4-5',
}

function load(): Config {
  try {
    const s = sessionStorage.getItem(SESSION_KEY)
    return s ? (JSON.parse(s) as Config) : { ...DEFAULTS }
  } catch {
    return { ...DEFAULTS }
  }
}

interface Props {
  onChange: (cfg: Config) => void
  onServerKeyStatus?: (status: { openai: boolean; indian: boolean }) => void
}

export default function ConfigPanel({ onChange, onServerKeyStatus }: Props) {
  const [cfg, setCfg] = useState<Config>(load)
  const [dotenvLoaded, setDotenvLoaded] = useState(false)
  // Whether the server already has its own key configured (from .env) — if
  // so, the key fields below can stay blank and requests will fall back to
  // it server-side. The server never sends the actual secret to the browser.
  const [serverHasOpenAIKey, setServerHasOpenAIKey] = useState(false)
  const [serverHasIndianKey, setServerHasIndianKey] = useState(false)
  const onChangeRef = useRef(onChange)
  onChangeRef.current = onChange
  const onServerKeyStatusRef = useRef(onServerKeyStatus)
  onServerKeyStatusRef.current = onServerKeyStatus

  // Auto-load from server .env on first mount. The server config endpoint
  // only reports whether a key is configured, not its value — model/base
  // URL are non-secret and still get pre-filled.
  useEffect(() => {
    const alreadyLoaded = sessionStorage.getItem(LOADED_FLAG)
    getServerConfig()
      .then(server => {
        setServerHasOpenAIKey(server.openai_api_key_configured)
        setServerHasIndianKey(server.indian_api_key_configured)
        onServerKeyStatusRef.current?.({
          openai: server.openai_api_key_configured,
          indian: server.indian_api_key_configured,
        })
        if (!alreadyLoaded) {
          const existing = load()
          const merged: Config = {
            openai_api_key: existing.openai_api_key || '',
            indian_api_key: existing.indian_api_key || '',
            model:          server.chat_model      || DEFAULTS.model,
            base_url:       server.openai_base_url || '',
            anthropic_api_key: existing.anthropic_api_key || '',
            claude_model:      existing.claude_model      || DEFAULTS.claude_model,
          }
          sessionStorage.setItem(SESSION_KEY, JSON.stringify(merged))
          sessionStorage.setItem(LOADED_FLAG, '1')
          setCfg(merged)
        }
        setDotenvLoaded(true)
      })
      .catch(() => {/* server may not be running yet */})
  }, [])

  // Notify parent whenever cfg changes
  useEffect(() => {
    sessionStorage.setItem(SESSION_KEY, JSON.stringify(cfg))
    onChangeRef.current(cfg)
  }, [cfg])

  const set = useCallback((key: keyof Config, value: string) => {
    setCfg(prev => ({ ...prev, [key]: value }))
  }, [])

  return (
    <aside className="flex flex-col gap-4 rounded-lg border border-gray-200 bg-white p-4 shadow-sm">
      <div className="flex items-center justify-between">
        <h2 className="text-sm font-semibold uppercase tracking-wide text-gray-500">
          Configuration
        </h2>
        {dotenvLoaded && (
          <span className="rounded bg-green-50 px-2 py-0.5 text-xs text-green-700">
            .env loaded
          </span>
        )}
      </div>

      <label className="flex flex-col gap-1">
        <span className="text-xs font-medium text-gray-700">
          OpenAI / OpenRouter API Key {serverHasOpenAIKey ? '' : '*'}
          {serverHasOpenAIKey && (
            <span className="ml-1 rounded bg-green-50 px-1.5 py-0.5 text-[10px] font-normal text-green-700">
              using server key
            </span>
          )}
        </span>
        <input
          type="password"
          className="rounded border border-gray-300 px-2 py-1.5 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
          placeholder={serverHasOpenAIKey ? 'leave blank to use the server key' : 'sk-... or sk-or-v1-...'}
          value={cfg.openai_api_key}
          onChange={e => set('openai_api_key', e.target.value)}
        />
      </label>

      <label className="flex flex-col gap-1">
        <span className="text-xs font-medium text-gray-700">
          Indian API Key
          {serverHasIndianKey && (
            <span className="ml-1 rounded bg-green-50 px-1.5 py-0.5 text-[10px] font-normal text-green-700">
              using server key
            </span>
          )}
        </span>
        <input
          type="password"
          className="rounded border border-gray-300 px-2 py-1.5 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
          placeholder={serverHasIndianKey ? 'leave blank to use the server key' : 'sk-live-... (optional)'}
          value={cfg.indian_api_key}
          onChange={e => set('indian_api_key', e.target.value)}
        />
      </label>

      <label className="flex flex-col gap-1">
        <span className="text-xs font-medium text-gray-700">Model</span>
        <input
          className="rounded border border-gray-300 px-2 py-1.5 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
          placeholder="gpt-4o or nvidia/nemotron-3-super-120b-a12b:free"
          value={cfg.model}
          onChange={e => set('model', e.target.value)}
        />
      </label>

      <label className="flex flex-col gap-1">
        <span className="text-xs font-medium text-gray-700">Base URL</span>
        <input
          className="rounded border border-gray-300 px-2 py-1.5 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
          placeholder="https://openrouter.ai/api/v1"
          value={cfg.base_url}
          onChange={e => set('base_url', e.target.value)}
        />
      </label>

      <div className="border-t border-gray-100 pt-3 mt-1" />

      <label className="flex flex-col gap-1">
        <span className="text-xs font-medium text-gray-700">Anthropic / Claude API Key</span>
        <input
          type="password"
          className="rounded border border-gray-300 px-2 py-1.5 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
          placeholder="sk-ant-... (only needed for Claude tab)"
          value={cfg.anthropic_api_key}
          onChange={e => set('anthropic_api_key', e.target.value)}
        />
      </label>

      <label className="flex flex-col gap-1">
        <span className="text-xs font-medium text-gray-700">Claude Model</span>
        <input
          className="rounded border border-gray-300 px-2 py-1.5 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
          placeholder="claude-opus-4-5"
          value={cfg.claude_model}
          onChange={e => set('claude_model', e.target.value)}
        />
      </label>

      <button
        className="rounded border border-gray-200 bg-gray-50 px-3 py-1.5 text-xs text-gray-600 hover:bg-gray-100"
        onClick={() => {
          sessionStorage.removeItem(SESSION_KEY)
          sessionStorage.removeItem(LOADED_FLAG)
          setCfg({ ...DEFAULTS })
        }}
      >
        Reload from .env
      </button>
    </aside>
  )
}
