# tools/buck_visualizer/ — Interactive Chart Scripts

Plotly-based visualization scripts that generate interactive HTML charts from stock data. These are standalone scripts (run via CLI or the `/visualize` API endpoint) — they do **not** export `TOOL_CLASS` / `TOOL_FUNC` and are intentionally skipped by the `ToolFactory` scanner.

Each script follows the same pattern:
1. Fetch OHLCV data via `DataVisualizationDownloader` (from `data_provider_viz.py`)
2. Build a Plotly figure with the relevant indicator overlaid on price data
3. Return the chart as a JSON-serialisable dict (for the API) or display it interactively (for CLI)

---

## Visualization Scripts

### `data_provider_viz.py` — DataVisualizationDownloader

Shared data-fetching utility used by all other scripts. Wraps the Yahoo and Indian data providers with a simplified async interface for visualization.

---

### `price_ma_plot.py`

Price chart with SMA/EMA overlays. Shows short and long moving averages against the close price, with crossover points highlighted.

---

### `candlestick_volume_plot.py`

Dual-pane chart: candlestick chart (top) with volume bars (bottom). Uses Plotly subplots with shared x-axis for time alignment.

---

### `rsi_plot.py`

Price chart (top) with RSI indicator (bottom). Marks the 30/70 overbought and oversold thresholds with horizontal reference lines.

---

### `macd_plot.py`

Three-pane view: price (top), MACD line vs signal line (middle), histogram (bottom). Histogram bars are colour-coded green (positive) / red (negative).

---

### `bollinger_plot.py`

Bollinger Bands overlay on the price chart: 20-period SMA with upper and lower bands at 2 standard deviations. Shaded fill between bands for visual clarity.

---

### `volatility_plot.py`

Rolling volatility chart showing realised volatility at multiple horizons (5, 10, 20 periods) plotted as line series below the price chart.

---

### `returns_histogram.py`

Histogram of daily/hourly returns with a fitted normal distribution overlay. Shows the empirical distribution of returns alongside theoretical expectations.

---

### `news_overlay_plot.py`

Price chart with news event markers: vertical lines or annotations at timestamps where news articles were published. Colour-coded by sentiment (green = positive, red = negative).

---

## Adding a New Visualizer

1. Create `tools/buck_visualizer/your_chart.py`
2. Use `DataVisualizationDownloader` for data fetching
3. Build a Plotly figure and return it via `fig.to_dict()`
4. Register the chart type in `UI/backend/visualizer.py` (`CHART_CATALOGUE` and `build_chart()`)
5. Do **not** add `TOOL_CLASS` or `TOOL_FUNC` — visualizer scripts as they are not agent tools, they are just visualization tools that are used.
