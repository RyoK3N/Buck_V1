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
}

export default function ConfigPanel({ onChange }: Props) {
  const [cfg, setCfg] = useState<Config>(load)
  const [dotenvLoaded, setDotenvLoaded] = useState(false)
  const onChangeRef = useRef(onChange)
  onChangeRef.current = onChange

  // Auto-load from server .env on first mount.
  // Always overwrite with server values on first load of a new session.
  // After the first successful load we set a flag so subsequent mounts
  // (e.g. React strict-mode remount) don't clobber user edits.
  useEffect(() => {
    const alreadyLoaded = sessionStorage.getItem(LOADED_FLAG)
    getServerConfig()
      .then(server => {
        if (!alreadyLoaded) {
          // First load — server values take priority
          const merged: Config = {
            openai_api_key: server.openai_api_key || '',
            indian_api_key: server.indian_api_key || '',
            model:          server.chat_model     || DEFAULTS.model,
            base_url:       server.openai_base_url || '',
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
        <span className="text-xs font-medium text-gray-700">OpenAI / OpenRouter API Key *</span>
        <input
          type="password"
          className="rounded border border-gray-300 px-2 py-1.5 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
          placeholder="sk-... or sk-or-v1-..."
          value={cfg.openai_api_key}
          onChange={e => set('openai_api_key', e.target.value)}
        />
      </label>

      <label className="flex flex-col gap-1">
        <span className="text-xs font-medium text-gray-700">Indian API Key</span>
        <input
          type="password"
          className="rounded border border-gray-300 px-2 py-1.5 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
          placeholder="sk-live-... (optional)"
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
