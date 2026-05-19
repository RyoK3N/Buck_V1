# tools/utility/ — Data Processing and Portfolio Utilities

Risk analytics, volatility profiling, data quality checks, and position sizing tools. These do not predict direction — they provide context that helps the agent calibrate confidence, size positions, and decide whether market conditions are safe for trading.

Every tool operates on the OHLCV DataFrame via `get_stock_data()` and follows the standard `BaseTool` + `@tool` pattern.

---

## Future Tools

### 1. `risk_metrics.py` — RiskMetricsTool

**What it does:** Computes a comprehensive risk profile from price returns: Sharpe ratio, Sortino ratio, maximum drawdown, Calmar ratio, and Value at Risk (both historical and parametric).

**Why it matters:** The LLM needs to contextualise directional signals against the asset's risk profile. A BUY signal on a stock with Sharpe < 0.3 and -40% max drawdown deserves less conviction than the same signal on a Sharpe 2.0 asset. The `risk_regime` label (LOW / MODERATE / HIGH / EXTREME) gives the LLM a single token to condition on.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `risk_free_rate` | 0.05 | Annualised risk-free rate |
| `var_confidence` | 0.95 | VaR confidence level |

**Key output fields:**
- `signal`, `strength`
- `sharpe_ratio`, `sortino_ratio`, `calmar_ratio`
- `max_drawdown`, `max_drawdown_duration_bars`
- `var_95_historical`, `var_95_parametric`, `cvar_95`
- `annualised_return`, `annualised_volatility`
- `risk_regime` — LOW / MODERATE / HIGH / EXTREME

**Signal convention:** Sharpe > 1.5 and drawdown > -10% = HOLD (favourable, don't over-trade). Sharpe < 0 = SELL (deteriorating risk-adjusted return). Else HOLD.

**Dependencies:** `numpy`, `scipy.stats`

---

### 2. `volatility_analyser.py` — VolatilityAnalyserTool

**What it does:** Calculates realised volatility at multiple horizons (5, 20, 60 bars), Garman-Klass volatility (uses full OHLC, not just close), EWMA volatility forecast, and volatility regime classification.

**Why it matters:** Volatility is a first-order trading input. `vol_percentile_1y` tells the agent where current vol sits in its annual distribution. The `vol_term_structure` label (CONTANGO = short vol < long vol = mean-reversion conditions; BACKWARDATION = short vol elevated = uncertainty) gives the LLM a regime signal. High `vol_of_vol` signals instability.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `short_window` | 5 | Short-term vol window |
| `medium_window` | 20 | Medium-term vol window |
| `long_window` | 60 | Long-term vol window |

**Key output fields:**
- `signal`, `strength`
- `realised_vol_5d`, `realised_vol_20d`, `realised_vol_60d`
- `garman_klass_vol` — OHLC-based estimator (more efficient than close-to-close)
- `ewma_vol_forecast`
- `vol_regime` — LOW / NORMAL / ELEVATED / EXTREME
- `vol_percentile_1y` — where current vol ranks in its 252-bar distribution
- `vol_term_structure` — CONTANGO / FLAT / BACKWARDATION
- `vol_of_vol` — volatility of volatility (instability measure)

**Signal convention:** `vol_percentile > 0.80` = HOLD (too risky to initiate). `vol_percentile < 0.20` = slight BUY lean (vol crush, calm market).

**Dependencies:** `numpy`, `pandas`

---

### 3. `correlation_analyser.py` — CorrelationAnalyserTool

**What it does:** Analyses serial autocorrelation of returns (mean-reversion vs momentum character), price-volume correlation, and the Hurst exponent.

**Why it matters:** The Hurst exponent is the key output. H > 0.55 = persistent/trending behaviour (favour momentum tool signals). H < 0.45 = mean-reverting (fade extremes, weight RSI oversold/overbought signals more). This tells the LLM which category of signals to trust more. A significant Ljung-Box p-value confirms returns are not IID, validating predictive tool use.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `max_lag` | 5 | Maximum autocorrelation lag |

**Key output fields:**
- `signal`, `strength`
- `autocorr_lag1` through `autocorr_lag5`
- `price_volume_corr` — Pearson correlation between returns and volume
- `hurst_exponent` — 0.5 = random walk, >0.5 = trending, <0.5 = mean-reverting
- `market_character` — TRENDING / MEAN_REVERTING / RANDOM_WALK
- `ljung_box_pvalue` — test for significant serial correlation
- `serial_correlation_significant` — bool

**Signal convention:** Hurst > 0.58 = BUY (trending, momentum favoured). Hurst < 0.42 = context-dependent (mean reversion confirms other bearish signals). Else HOLD.

**Dependencies:** `numpy`, `scipy.stats`

---

### 4. `position_sizer.py` — PositionSizerTool

**What it does:** Calculates recommended position sizes using Kelly Criterion, fixed-fraction risk, and ATR-based volatility-adjusted sizing. Estimates win rate and payoff ratio from the OHLCV data itself.

**Why it matters:** Without sizing, a BUY signal is incomplete. The Kelly fraction provides the mathematically optimal answer. Half-Kelly is the practical recommendation (full Kelly is too aggressive for real trading). ATR-based sizing accounts for current volatility directly.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `portfolio_value` | 100000 | Notional portfolio size |
| `risk_per_trade_pct` | 1.0 | Max risk per trade as % of portfolio |
| `atr_multiplier` | 2.0 | ATR multiplier for stop distance |

**Key output fields:**
- `signal`, `strength`
- `full_kelly_fraction`, `half_kelly_fraction`
- `fixed_risk_shares` — number of shares given risk budget
- `atr_stop_distance` — ATR-based stop loss distance
- `atr_position_size` — shares at given portfolio value
- `estimated_win_rate`, `estimated_payoff_ratio`
- `risk_reward_ratio`
- `recommended_fraction` — the conservative recommendation

**Signal convention:** Always HOLD — this tool does not predict direction. Strength reflects confidence in the sizing estimate (higher with more data).

**Dependencies:** `numpy`

---

### 5. `feature_engineer.py` — FeatureEngineerTool

**What it does:** A preprocessing utility that computes a standardised 20+ feature matrix from raw OHLCV data, runs data quality checks, and flags outlier bars. This is the "run first" tool that validates data before ML tools train on it.

**Why it matters:** Data quality gate. The LLM can see `recommended_min_bars_met: false` and decide not to trust ML predictions. `zero_volume_bars` catches halted periods. `outlier_bars` exposes events that could corrupt model training.

**No configurable parameters.**

**Key output fields:**
- `signal`, `strength`
- `features_computed` — count of engineered features
- `data_quality` — `{missing_values, zero_volume_bars, price_gaps, data_completeness}`
- `feature_statistics` — `{log_return_mean, log_return_std, volume_zscore_latest}`
- `outlier_bars` — list of `{index, type, magnitude}`
- `recommended_min_bars_met` — bool (true if >= 60 bars available)

**Signal convention:** Always HOLD. This is informational, not directional.

**Dependencies:** `numpy`, `pandas`, `scikit-learn` (StandardScaler)

---

### 6. `drawdown_analyser.py` — DrawdownAnalyserTool

**What it does:** Full decomposition of all historical drawdowns: depth, duration, recovery time, and current underwater position. Reports where the current drawdown sits in the historical distribution.

**Why it matters:** `drawdown_percentile` directly informs BUY decisions. A drawdown at the 5th percentile is a potential accumulation opportunity. One at the 95th percentile is a potential falling knife. `avg_recovery_bars` gives the LLM a time horizon expectation.

**No configurable parameters.**

**Key output fields:**
- `signal`, `strength`
- `current_drawdown` — current drawdown from peak (negative number)
- `is_in_drawdown` — bool
- `max_drawdown_ever`
- `current_drawdown_duration_bars`
- `drawdown_percentile` — where current drawdown ranks historically
- `avg_recovery_bars` — average bars to recover from drawdowns of similar depth
- `worst_3_drawdowns` — `[{depth, duration, recovery}, ...]`
- `drawdown_regime` — NONE / MINOR / MODERATE / SEVERE

**Signal convention:** `drawdown_percentile < 0.10` and starting to recover = BUY. `drawdown_percentile > 0.90` and still falling = SELL. Else HOLD.

**Dependencies:** `numpy`, `pandas`
