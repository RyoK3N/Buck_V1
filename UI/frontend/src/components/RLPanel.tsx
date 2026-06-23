import { useState, useEffect, useCallback, useMemo } from 'react'
import { rlTrain, rlPredict, rlSimulate, rlGetModels, rlDeleteModel } from '../api/client'
import type {
  Config, RLTrainResponse, RLPredictResponse, RLSimulateResponse, RLModelInfo,
} from '../types'
import LoadingSpinner from './LoadingSpinner'

type RLTab = 'train' | 'evaluate' | 'simulate'

function fmtCurrency(n: number): string {
  return new Intl.NumberFormat('en-IN', { style: 'currency', currency: 'INR', maximumFractionDigits: 2 }).format(n)
}

function fmtPct(n: number): string {
  return (n >= 0 ? '+' : '') + n.toFixed(2) + '%'
}

function InfoIcon({ text }: { text: string }) {
  return (
    <span className="group relative ml-1 inline-flex cursor-help align-middle" title={text}>
      <svg className="h-3.5 w-3.5 text-gray-400" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
        <path strokeLinecap="round" strokeLinejoin="round" d="M13 16h-1v-4h-1m1-4h.01M12 2a10 10 0 100 20 10 10 0 000-20z" />
      </svg>
    </span>
  )
}

function Field({ label, info, children }: { label: string; info: string; children: React.ReactNode }) {
  return (
    <div>
      <label className="flex items-center text-xs font-medium text-gray-500 mb-1">
        {label}
        <InfoIcon text={info} />
      </label>
      {children}
    </div>
  )
}

function MiniLineChart({ data, width = 280, height = 80, color = '#3b82f6' }: { data: number[]; width?: number; height?: number; color?: string }) {
  if (data.length < 2) return <div className="text-xs text-gray-400">Insufficient data</div>
  const min = Math.min(...data)
  const max = Math.max(...data)
  const range = max - min || 1
  const pad = 5
  const pts = data.map((v, i) => {
    const x = pad + (i / (data.length - 1)) * (width - 2 * pad)
    const y = height - pad - ((v - min) / range) * (height - 2 * pad)
    return `${x},${y}`
  }).join(' ')
  const gradientId = `grad-${Math.random().toString(36).slice(2, 8)}`
  return (
    <svg width={width} height={height} className="w-full max-w-full" viewBox={`0 0 ${width} ${height}`}>
      <defs>
        <linearGradient id={gradientId} x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stopColor={color} stopOpacity="0.2" />
          <stop offset="100%" stopColor={color} stopOpacity="0.02" />
        </linearGradient>
      </defs>
      <polygon points={`${pad},${height - pad} ${pts} ${width - pad},${height - pad}`} fill={`url(#${gradientId})`} />
      <polyline points={pts} fill="none" stroke={color} strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  )
}

function MiniBarChart({ data, width = 280, height = 80, positiveColor = '#22c55e', negativeColor = '#ef4444' }: { data: { label: string; value: number }[]; width?: number; height?: number; positiveColor?: string; negativeColor?: string }) {
  if (data.length === 0) return <div className="text-xs text-gray-400">No data</div>
  const values = data.map(d => d.value)
  const absMax = Math.max(...values.map(Math.abs), 0.01)
  const barW = Math.min(20, (width - 20) / data.length - 2)
  const half = height / 2
  return (
    <svg width={width} height={height} className="w-full max-w-full" viewBox={`0 0 ${width} ${height}`}>
      <line x1="5" y1={half} x2={width - 5} y2={half} stroke="#e5e7eb" strokeWidth="1" />
      {data.map((d, i) => {
        const barH = (Math.abs(d.value) / absMax) * (half - 4)
        const x = 8 + i * (barW + 3)
        const y = d.value >= 0 ? half - barH : half
        const h = Math.max(barH, 1)
        return <rect key={i} x={x} y={y} width={barW} height={h} rx="1" fill={d.value >= 0 ? positiveColor : negativeColor} />
      })}
    </svg>
  )
}

interface Props { config: Config }

export default function RLPanel({ config }: Props) {
  const [tab, setTab] = useState<RLTab>('train')
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [models, setModels] = useState<RLModelInfo[]>([])
  const [trainResult, setTrainResult] = useState<RLTrainResponse | null>(null)
  const [predictResult, setPredictResult] = useState<RLPredictResponse | null>(null)
  const [simulateResult, setSimulateResult] = useState<RLSimulateResponse | null>(null)
  const [symbol, setSymbol] = useState('RELIANCE')
  const [startDate, setStartDate] = useState('2025-01-01')
  const [endDate, setEndDate] = useState('2026-05-29')
  const [interval, setInterval] = useState('1d')
  const [modelId, setModelId] = useState('dqn_model')
  const [episodes, setEpisodes] = useState(50)
  const [algorithm, setAlgorithm] = useState('dqn')
  const [hiddenDim, setHiddenDim] = useState(128)
  const [learningRate, setLearningRate] = useState(0.001)
  const [initialCapital, setInitialCapital] = useState(100000)

  const loadModels = useCallback(async () => {
    try { setModels((await rlGetModels()).models) } catch { }
  }, [])
  useEffect(() => { loadModels() }, [loadModels])

  const handleTrain = async () => {
    setLoading(true); setError(null); setTrainResult(null)
    try {
      const res = await rlTrain({
        symbol, start_date: startDate, end_date: endDate, interval,
        model_id: modelId, episodes, hidden_dim: hiddenDim,
        learning_rate: learningRate, initial_capital: initialCapital,
        algorithm,
        indian_api_key: config.indian_api_key || undefined,
      })
      setTrainResult(res); loadModels()
    } catch (e: unknown) {
      const detail = (e as { response?: { data?: { detail?: string } } })?.response?.data?.detail
      setError(detail ?? (e instanceof Error ? e.message : 'Training failed'))
    } finally { setLoading(false) }
  }

  const handlePredict = async () => {
    setLoading(true); setError(null); setPredictResult(null)
    try {
      const res = await rlPredict({
        symbol, start_date: startDate, end_date: endDate, interval,
        model_id: modelId, initial_capital: initialCapital,
        indian_api_key: config.indian_api_key || undefined,
      })
      setPredictResult(res)
    } catch (e: unknown) {
      const detail = (e as { response?: { data?: { detail?: string } } })?.response?.data?.detail
      setError(detail ?? (e instanceof Error ? e.message : 'Evaluation failed'))
    } finally { setLoading(false) }
  }

  const handleSimulate = async () => {
    setLoading(true); setError(null); setSimulateResult(null)
    try {
      const res = await rlSimulate({
        model_id: modelId, symbol, interval, initial_capital: initialCapital,
        indian_api_key: config.indian_api_key || undefined,
      })
      setSimulateResult(res)
    } catch (e: unknown) {
      const detail = (e as { response?: { data?: { detail?: string } } })?.response?.data?.detail
      setError(detail ?? (e instanceof Error ? e.message : 'Simulation failed'))
    } finally { setLoading(false) }
  }

  const handleDeleteModel = async (id: string) => {
    try { await rlDeleteModel(id); loadModels() } catch { }
  }

  const trainChartData = useMemo(() => {
    if (!trainResult?.episode_rewards?.length) return null
    const eps = trainResult.episode_rewards
    return {
      rewards: eps.map(e => e.total_reward),
      returns: eps.map(e => ({ label: `Ep${e.episode}`, value: e.return_pct })),
    }
  }, [trainResult])

  const eqCurve = trainResult?.equity_curve?.map(e => e.portfolio_value) ?? []
  const predCurve = predictResult?.equity_curve?.map(e => e.portfolio_value) ?? []

  const TAB_LABELS: Record<RLTab, string> = { train: 'Train', evaluate: 'Evaluate', simulate: 'Live Simulate' }

  return (
    <div className="space-y-4">
      <div className="flex gap-1 rounded-lg border border-gray-200 bg-white p-1 shadow-sm w-fit">
        {(['train', 'evaluate', 'simulate'] as RLTab[]).map(t => (
          <button key={t} onClick={() => setTab(t)}
            className={`rounded px-4 py-1.5 text-sm font-medium transition-colors ${tab === t ? 'bg-blue-600 text-white' : 'text-gray-600 hover:text-gray-900'}`}>
            {TAB_LABELS[t]}
          </button>
        ))}
      </div>

      <div className="rounded-lg border border-gray-200 bg-white p-4 shadow-sm space-y-4">
        <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
          <Field label="Symbol" info="NSE stock symbol (e.g. RELIANCE, TCS, INFY). .NS is auto-appended for Yahoo Finance.">
            <input className="w-full rounded border border-gray-300 px-2 py-1.5 text-sm" value={symbol} onChange={e => setSymbol(e.target.value.toUpperCase())} />
          </Field>
          <Field label="Model ID" info="Unique name to identify this trained model. Used to save, load, and evaluate later.">
            <input className="w-full rounded border border-gray-300 px-2 py-1.5 text-sm" value={modelId} onChange={e => setModelId(e.target.value)} />
          </Field>
          <Field label="Interval" info="Candlestick interval for training data. 1d=daily, 1h=hourly, 15m/5m/1m=intraday.">
            <select className="w-full rounded border border-gray-300 px-2 py-1.5 text-sm" value={interval} onChange={e => setInterval(e.target.value)}>
              <option value="1d">1 Day</option>
              <option value="1h">1 Hour</option>
              <option value="15m">15 Min</option>
              <option value="5m">5 Min</option>
              <option value="1m">1 Min</option>
            </select>
          </Field>
          <Field label="Capital (INR)" info="Starting cash balance for the simulation wallet. Default: ₹1,00,000.">
            <input type="number" className="w-full rounded border border-gray-300 px-2 py-1.5 text-sm" value={initialCapital} onChange={e => setInitialCapital(Number(e.target.value))} />
          </Field>
          <Field label="Start Date" info="Start of the training/evaluation date range (YYYY-MM-DD).">
            <input type="date" className="w-full rounded border border-gray-300 px-2 py-1.5 text-sm" value={startDate} onChange={e => setStartDate(e.target.value)} />
          </Field>
          <Field label="End Date" info="End of the training/evaluation date range (YYYY-MM-DD).">
            <input type="date" className="w-full rounded border border-gray-300 px-2 py-1.5 text-sm" value={endDate} onChange={e => setEndDate(e.target.value)} />
          </Field>
          {tab === 'train' && (
            <>
              <Field label="Episodes" info="Number of complete passes through the training data. Each episode trains the agent from scratch on the full price series. More episodes = more learning, but takes longer.">
                <input type="number" className="w-full rounded border border-gray-300 px-2 py-1.5 text-sm" value={episodes} onChange={e => setEpisodes(Number(e.target.value))} />
              </Field>
              <Field label="Algorithm" info="RL algorithm to use. DQN (Deep Q-Network) uses experience replay. PPO (Proximal Policy Optimization) clips updates for stability. A2C (Advantage Actor-Critic) combines policy and value learning.">
                <select className="w-full rounded border border-gray-300 px-2 py-1.5 text-sm" value={algorithm} onChange={e => setAlgorithm(e.target.value)}>
                  <option value="dqn">DQN</option>
                  <option value="ppo">PPO</option>
                  <option value="a2c">A2C</option>
                </select>
              </Field>
              <Field label="Hidden Dim" info="Size of the hidden layers in the DQN neural network. Larger values = more capacity but slower training (128-256 recommended).">
                <input type="number" className="w-full rounded border border-gray-300 px-2 py-1.5 text-sm" value={hiddenDim} onChange={e => setHiddenDim(Number(e.target.value))} />
              </Field>
              <Field label="Learning Rate" info="Step size for gradient descent during training. Lower = more stable but slower (0.0001-0.001 recommended).">
                <input type="number" step="0.0001" className="w-full rounded border border-gray-300 px-2 py-1.5 text-sm" value={learningRate} onChange={e => setLearningRate(Number(e.target.value))} />
              </Field>
            </>
          )}
        </div>

        <div className="flex gap-2">
          {tab === 'train' && (
            <button onClick={handleTrain} disabled={loading} className="rounded bg-blue-600 px-4 py-1.5 text-sm font-medium text-white hover:bg-blue-700 disabled:opacity-50">
              {loading ? 'Training...' : 'Train Agent'}
            </button>
          )}
          {tab === 'evaluate' && (
            <button onClick={handlePredict} disabled={loading} className="rounded bg-green-600 px-4 py-1.5 text-sm font-medium text-white hover:bg-green-700 disabled:opacity-50">
              {loading ? 'Evaluating...' : 'Evaluate Model'}
            </button>
          )}
          {tab === 'simulate' && (
            <button onClick={handleSimulate} disabled={loading} className="rounded bg-purple-600 px-4 py-1.5 text-sm font-medium text-white hover:bg-purple-700 disabled:opacity-50">
              {loading ? 'Simulating...' : 'Run Live Simulation'}
            </button>
          )}
        </div>
      </div>

      {error && (
        <div className="rounded-lg border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-700 whitespace-pre-wrap">
          <strong>Error:</strong> {error}
        </div>
      )}

      {loading && <LoadingSpinner />}

      {models.length > 0 && (
        <div className="rounded-lg border border-gray-200 bg-white p-4 shadow-sm">
          <h3 className="text-sm font-semibold text-gray-700 mb-2">Saved Models ({models.length})</h3>
          <div className="space-y-1 max-h-40 overflow-y-auto">
            {models.map(m => (
              <div key={m.id} className="flex items-center justify-between text-xs text-gray-600 py-1 px-2 hover:bg-gray-50 rounded">
                <div>
                  <span className="font-medium">{m.id}</span>
                  {m.created && <span className="ml-2 text-gray-400">{m.created}</span>}
                  {m.error && <span className="ml-2 text-red-400">corrupt</span>}
                  {m.train_steps > 0 && <span className="ml-2 text-gray-400">steps: {m.train_steps}</span>}
                </div>
                <button onClick={() => handleDeleteModel(m.id)} className="text-red-400 hover:text-red-600">Delete</button>
              </div>
            ))}
          </div>
        </div>
      )}

      {trainResult && (
        <div className="rounded-lg border border-gray-200 bg-white p-4 shadow-sm space-y-4">
          <h3 className="text-sm font-semibold text-gray-700">
            Training Results: <span className="text-blue-600">{trainResult.model_id}</span> on {trainResult.symbol}
          </h3>

          <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-6 gap-2 text-xs">
            <MetricBox label="Episodes" value={String(trainResult.episodes)} />
            <MetricBox label="Data Steps" value={String(trainResult.total_steps)} />
            <MetricBox label="Best Reward" value={trainResult.best_reward.toFixed(2)} />
            <MetricBox label="Final Return" value={fmtPct(trainResult.final_summary.total_return_pct)} color={trainResult.final_summary.total_return_pct >= 0 ? 'text-green-600' : 'text-red-600'} />
            <MetricBox label="Sharpe Ratio" value={trainResult.final_summary.sharpe_ratio.toFixed(3)} color={trainResult.final_summary.sharpe_ratio >= 0 ? 'text-green-600' : 'text-red-600'} />
            <MetricBox label="Max Drawdown" value={trainResult.final_summary.max_drawdown_pct.toFixed(2) + '%'} color="text-red-600" />
            <MetricBox label="Win Rate" value={fmtPct(trainResult.final_summary.win_rate_pct ?? 0)} />
            <MetricBox label="Portfolio" value={fmtCurrency(trainResult.final_summary.portfolio_value)} />
            <MetricBox label="Total Trades" value={String(trainResult.final_summary.total_trades)} />
            <MetricBox label="Algorithm" value={trainResult.algorithm ?? 'dqn'} />
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div className="bg-gray-50 rounded-lg p-3">
              <h4 className="text-xs font-semibold text-gray-600 mb-2">Episode Reward Progression</h4>
              {trainChartData ? (
                <>
                  <MiniLineChart data={trainChartData.rewards} color="#3b82f6" />
                  <div className="flex justify-between text-[10px] text-gray-400 mt-1">
                    <span>Ep 1</span>
                    <span>Best: {trainResult.best_reward.toFixed(1)}</span>
                    <span>Ep {trainResult.episodes}</span>
                  </div>
                </>
              ) : (
                <div className="text-xs text-gray-400 py-4 text-center">Insufficient training data</div>
              )}
            </div>

            <div className="bg-gray-50 rounded-lg p-3">
              <h4 className="text-xs font-semibold text-gray-600 mb-2">Episode Return %</h4>
              {trainChartData ? (
                <>
                  <MiniBarChart data={trainChartData.returns} />
                  <div className="flex justify-between text-[10px] text-gray-400 mt-1">
                    <span>Ep 1</span>
                    <span>Final: {fmtPct(trainResult.final_summary.total_return_pct)}</span>
                    <span>Ep {trainResult.episodes}</span>
                  </div>
                </>
              ) : (
                <div className="text-xs text-gray-400 py-4 text-center">No return data</div>
              )}
            </div>
          </div>

          {eqCurve.length > 0 && (
            <div className="bg-gray-50 rounded-lg p-3">
              <h4 className="text-xs font-semibold text-gray-600 mb-2">Equity Curve (Final Episode)</h4>
              <MiniLineChart data={eqCurve} color="#8b5cf6" />
              <div className="flex justify-between text-[10px] text-gray-400 mt-1">
                <span>Start: {fmtCurrency(trainResult.final_summary.initial_capital)}</span>
                <span>End: {fmtCurrency(trainResult.final_summary.portfolio_value)}</span>
              </div>
            </div>
          )}

          {trainResult.episode_rewards.length > 0 && (
            <div>
              <h4 className="text-xs font-semibold text-gray-600 mb-1">Episode Details</h4>
              <div className="max-h-52 overflow-y-auto text-xs space-y-0.5 font-mono">
                <div className="flex gap-3 text-gray-400 px-1 pb-1 border-b border-gray-100">
                  <span className="w-14">Episode</span>
                  <span className="w-20 text-right">Reward</span>
                  <span className="w-16 text-right">Return %</span>
                  <span className="w-14 text-right">Trades</span>
                  <span className="w-20 text-right">Portfolio</span>
                </div>
                {trainResult.episode_rewards.map((ep, i) => (
                  <div key={i} className={`flex gap-3 px-1 ${ep.total_reward === trainResult.best_reward ? 'bg-yellow-50 rounded' : ''}`}>
                    <span className="w-14 text-gray-500">Ep {ep.episode}</span>
                    <span className={`w-20 text-right ${ep.total_reward >= 0 ? 'text-green-600' : 'text-red-600'}`}>{ep.total_reward.toFixed(2)}</span>
                    <span className={`w-16 text-right ${ep.return_pct >= 0 ? 'text-green-600' : 'text-red-600'}`}>{fmtPct(ep.return_pct)}</span>
                    <span className="w-14 text-right text-gray-500">{ep.trades}</span>
                    <span className="w-20 text-right text-gray-600">{fmtCurrency(ep.portfolio_value)}</span>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      )}

      {predictResult && (
        <div className="rounded-lg border border-gray-200 bg-white p-4 shadow-sm space-y-4">
          <h3 className="text-sm font-semibold text-gray-700">
            Evaluation: <span className="text-green-600">{predictResult.model_id}</span> on {predictResult.symbol}
          </h3>

          <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-6 gap-2 text-xs">
            <MetricBox label="Total Return" value={fmtPct(predictResult.summary.total_return_pct)} color={predictResult.summary.total_return_pct >= 0 ? 'text-green-600' : 'text-red-600'} />
            <MetricBox label="Sharpe Ratio" value={predictResult.summary.sharpe_ratio.toFixed(3)} color={predictResult.summary.sharpe_ratio >= 0 ? 'text-green-600' : 'text-red-600'} />
            <MetricBox label="Max Drawdown" value={predictResult.summary.max_drawdown_pct.toFixed(2) + '%'} color="text-red-600" />
            <MetricBox label="Win Rate" value={fmtPct(predictResult.summary.win_rate_pct ?? 0)} />
            <MetricBox label="Total Trades" value={String(predictResult.summary.total_trades)} />
            <MetricBox label="Portfolio" value={fmtCurrency(predictResult.summary.portfolio_value)} />
            <MetricBox label="Cash" value={fmtCurrency(predictResult.summary.cash)} />
            <MetricBox label="Holdings" value={predictResult.summary.holdings.toFixed(4) + ' sh'} />
            <MetricBox label="Signals Generated" value={String(predictResult.total_signals)} />
          </div>

          {predCurve.length > 0 && (
            <div className="bg-gray-50 rounded-lg p-3">
              <h4 className="text-xs font-semibold text-gray-600 mb-2">Equity Curve</h4>
              <MiniLineChart data={predCurve} color="#22c55e" />
              <div className="flex justify-between text-[10px] text-gray-400 mt-1">
                <span>Start: {fmtCurrency(predictResult.summary.initial_capital)}</span>
                <span>End: {fmtCurrency(predictResult.summary.portfolio_value)}</span>
              </div>
            </div>
          )}

          {predictResult.signals.length > 0 && (
            <div>
              <h4 className="text-xs font-semibold text-gray-600 mb-1">Recent Signals (last {predictResult.signals.length})</h4>
              <div className="max-h-44 overflow-y-auto text-xs space-y-0.5 font-mono">
                <div className="flex gap-3 text-gray-400 px-1 pb-1 border-b border-gray-100">
                  <span className="w-12">Step</span>
                  <span className="w-14">Action</span>
                  <span className="w-20 text-right">Price</span>
                  <span className="w-24 text-right">Portfolio Value</span>
                </div>
                {predictResult.signals.map((s, i) => (
                  <div key={i} className="flex gap-3 px-1">
                    <span className="w-12 text-gray-500">#{s.step}</span>
                    <span className={`w-14 font-medium ${s.action === 'BUY' ? 'text-green-600' : s.action === 'SELL' ? 'text-red-600' : 'text-gray-500'}`}>{s.action}</span>
                    <span className="w-20 text-right text-gray-600">{fmtCurrency(s.price)}</span>
                    <span className="w-24 text-right text-gray-600">{fmtCurrency(s.portfolio_value)}</span>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      )}

      {simulateResult && (
        <div className="rounded-lg border border-gray-200 bg-white p-4 shadow-sm space-y-3">
          <h3 className="text-sm font-semibold text-gray-700">
            Live Simulation: <span className="text-purple-600">{simulateResult.symbol}</span> — {simulateResult.action}
          </h3>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-2 text-xs">
            <div className="bg-gray-50 rounded p-2">
              <div className="text-gray-500 mb-1">Live Price</div>
              <div className="font-semibold text-gray-800 text-sm">{fmtCurrency(simulateResult.price)}</div>
            </div>
            <div className="bg-gray-50 rounded p-2">
              <div className="text-gray-500 mb-1">Agent Action</div>
              <div className={`font-semibold text-sm ${simulateResult.action === 'BUY' ? 'text-green-600' : simulateResult.action === 'SELL' ? 'text-red-600' : 'text-gray-500'}`}>{simulateResult.action}</div>
            </div>
            <div className="bg-gray-50 rounded p-2">
              <div className="text-gray-500 mb-1">Portfolio Value</div>
              <div className="font-semibold text-gray-800 text-sm">{fmtCurrency(simulateResult.wallet.portfolio_value)}</div>
            </div>
            <div className="bg-gray-50 rounded p-2">
              <div className="text-gray-500 mb-1">Total Return</div>
              <div className={`font-semibold text-sm ${simulateResult.wallet.total_return_pct >= 0 ? 'text-green-600' : 'text-red-600'}`}>{fmtPct(simulateResult.wallet.total_return_pct)}</div>
            </div>
            <div className="bg-gray-50 rounded p-2">
              <div className="text-gray-500 mb-1">Cash</div>
              <div className="font-semibold text-gray-800 text-sm">{fmtCurrency(simulateResult.wallet.cash)}</div>
            </div>
            <div className="bg-gray-50 rounded p-2">
              <div className="text-gray-500 mb-1">Holdings</div>
              <div className="font-semibold text-gray-800 text-sm">{simulateResult.wallet.holdings.toFixed(4)} shares</div>
            </div>
            <div className="bg-gray-50 rounded p-2">
              <div className="text-gray-500 mb-1">Sharpe Ratio</div>
              <div className="font-semibold text-gray-800 text-sm">{simulateResult.wallet.sharpe_ratio.toFixed(3)}</div>
            </div>
            <div className="bg-gray-50 rounded p-2">
              <div className="text-gray-500 mb-1">Max Drawdown</div>
              <div className="font-semibold text-red-600 text-sm">{simulateResult.wallet.max_drawdown_pct.toFixed(2)}%</div>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}

function MetricBox({ label, value, color = 'text-gray-800' }: { label: string; value: string; color?: string }) {
  return (
    <div className="bg-gray-50 rounded p-2">
      <div className="text-gray-500 mb-0.5">{label}</div>
      <div className={`font-semibold text-sm ${color}`}>{value}</div>
    </div>
  )
}
