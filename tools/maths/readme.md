# tools/maths/ — Technical Analysis Indicators

Mathematical indicators that operate on OHLCV DataFrames and produce structured BUY / SELL / HOLD signals. These are the core deterministic tools the agent uses for every analysis run.

Each file exports a `BaseTool` subclass (`TOOL_CLASS`) and a LangChain `@tool` function (`TOOL_FUNC`) that the `ToolFactory` auto-discovers.

---

## Tools

### `moving_average.py` — MovingAverageTool

Calculates Simple Moving Average (SMA) and Exponential Moving Average (EMA) over configurable short and long windows. Generates a signal based on price position relative to the two averages and their crossover.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `short_window` | 10 | Short-term MA period |
| `long_window` | 50 | Long-term MA period |
| `moving_average_type` | `SMA` | `SMA` or `EMA` |

**Signal logic:** Short MA above long MA and price above short MA = BUY. Inverse = SELL. Otherwise HOLD.

---

### `rsi.py` — RSITool

Computes the Relative Strength Index using Wilder's smoothed average of gains vs losses.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `window` | 14 | RSI lookback period |

**Signal logic:** RSI > 70 = SELL (overbought). RSI < 30 = BUY (oversold). Between = HOLD (neutral).

---

### `macd.py` — MACDTool

Calculates the MACD line, signal line, and histogram from exponential moving averages.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `short_window` | 12 | Fast EMA period |
| `long_window` | 26 | Slow EMA period |
| `signal_window` | 9 | Signal line EMA period |

**Signal logic:** MACD above signal and histogram positive = BUY. MACD below signal and histogram negative = SELL. Otherwise HOLD.

---

### `obv.py` — OBVTool

Computes On-Balance Volume — a cumulative volume indicator that adds volume on up-days and subtracts on down-days.

**No configurable parameters.** Compares current OBV against its 10-bar moving average.

**Signal logic:** OBV above its MA (rising trend) = BUY. Below = SELL.

---

### `candlestick_patterns.py` — CandlestickPatternTool

Scans the last 20 candles for classical patterns: Doji, Hammer, Shooting Star, Bullish Engulfing, Bearish Engulfing. Aggregates bullish and bearish confidence scores.

**No configurable parameters.**

**Signal logic:** Net bullish score > bearish = BUY. Net bearish > bullish = SELL. Tied = HOLD.

---

### `support_resistance.py` — SupportResistanceTool

Identifies support and resistance levels using rolling window highs/lows. Reports nearest levels and proximity to current price.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `window` | 5 | Rolling window for level detection |

**Signal logic:** Price within 2% of support = BUY. Within 2% of resistance = SELL. Otherwise HOLD.

---

## Adding a New Maths Tool

1. Create `tools/maths/your_tool.py`
2. Subclass `BaseTool` from `agent_scripts.tools`
3. Implement `execute(self, data: pd.DataFrame, **kwargs) -> Dict[str, Any]`
4. Return at minimum: `signal` (BUY/SELL/HOLD), `strength` (0.0-1.0)
5. Add a `@tool`-decorated function that calls `get_stock_data()` and delegates to the class
6. Export `TOOL_CLASS = YourTool` and `TOOL_FUNC = your_tool`

The `ToolFactory` will auto-discover it on next import.
