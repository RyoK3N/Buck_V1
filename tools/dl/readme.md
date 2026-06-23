# tools/dl/ — Deep Learning Tools

PyTorch-based neural network tools for price prediction. These models train on the input DataFrame at runtime, producing directional signals with probability estimates and accuracy metrics.

Every tool is context-engineered: the output dict includes the model's reasoning steps (features used, training/validation split, accuracy) so the LLM agent can assess trustworthiness and explain the prediction to the user.

---

## Tools

### `lstm_prediction.py` — LSTMPredictionTool

A 2-layer LSTM network that predicts next-bar price direction (UP / DOWN) from engineered features.

**Feature set (per timestep):**
- Log-return
- Rolling mean return (lookback window)
- Rolling volatility (lookback window)
- Momentum (close / close_lookback - 1)
- RSI-like up-day ratio
- Volume change ratio

| Parameter | Default | Description |
|-----------|---------|-------------|
| `lookback` | 10 | Feature engineering window |
| `seq_len` | 10 | LSTM sequence length |
| `epochs` | 50 | Training epochs |

**Architecture:** Input -> LSTM(hidden=32, layers=2, dropout=0.2) -> Linear(1) -> BCEWithLogitsLoss

**Signal logic:** P(up) > 0.6 = BUY. P(up) < 0.4 = SELL. Between = HOLD.

**Output includes:** `predicted_probability`, `predicted_direction`, `training_samples`, `validation_samples`, `accuracy`.

---

## Future Tools

### `transformer_prediction.py` — TransformerPredictionTool

A lightweight Transformer encoder (2 layers, 4 heads, d_model=32) for price direction prediction. Transformers can capture longer-range dependencies than LSTMs via self-attention, and the attention weights are interpretable — the agent can report which bars the model attended to most.

**Why:** Attention weights give the LLM a narrative ("the model focused on the volume spike 5 bars ago and the gap-up 12 bars ago"). This is the most LLM-friendly deep learning output possible.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `lookback` | 10 | Feature window |
| `seq_len` | 20 | Context length |
| `n_heads` | 4 | Attention heads |
| `epochs` | 50 | Training epochs |

**Dependencies:** `torch` (already in requirements)

---

### `cnn_pattern_detector.py` — CNNPatternDetectorTool

A 1D convolutional network that treats the OHLCV price series as a signal and learns local patterns (analogous to learned candlestick patterns). Conv1D layers with small kernels (3, 5, 7) capture patterns at different scales, pooled into a classification head.

**Why:** CNNs learn patterns that hand-coded candlestick rules miss. The multi-scale kernels detect both single-bar patterns (doji, hammer) and multi-bar formations (head-and-shoulders, flags) automatically.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `lookback` | 10 | Feature window |
| `window_size` | 30 | Input sequence length |
| `epochs` | 40 | Training epochs |

**Dependencies:** `torch`

---

### `autoencoder_anomaly.py` — AutoencoderAnomalyTool

A denoising autoencoder trained on OHLCV features. High reconstruction error on the latest bars indicates the current market state is unlike anything in recent history — a strong anomaly/regime-change signal.

**Why:** Complements the Isolation Forest in `tools/ml/`. The autoencoder learns a compressed representation of "normal" market behaviour and flags deviations. Unlike IF, it can capture non-linear feature interactions and temporal structure.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `encoding_dim` | 8 | Bottleneck dimension |
| `epochs` | 30 | Training epochs |
| `anomaly_threshold` | 95 | Percentile for anomaly cutoff |

**Dependencies:** `torch`

---

### `gru_volatility_forecast.py` — GRUVolatilityForecastTool

A GRU network trained on realised volatility and volume features to forecast next-period volatility. Outputs a volatility regime label and a numeric forecast that other tools (position_sizer) can consume.

**Why:** Volatility forecasting is a distinct task from direction prediction. A dedicated GRU for vol gives the agent a volatility view alongside the directional LSTM, enabling richer reasoning ("LSTM says BUY but GRU says volatility is about to spike — reduce position size").

| Parameter | Default | Description |
|-----------|---------|-------------|
| `lookback` | 20 | Feature window |
| `seq_len` | 15 | Sequence length |
| `epochs` | 40 | Training epochs |

**Dependencies:** `torch`

---

## Adding a New DL Tool

1. Create `tools/dl/your_model.py`
2. Define the PyTorch `nn.Module` as a private class (e.g., `_YourNet`)
3. Subclass `BaseTool`, implement `execute()` with feature engineering, training, and inference
4. Add a `@tool` function that calls `get_stock_data()` and delegates to the class
5. Export `TOOL_CLASS` and `TOOL_FUNC`
6. Return at minimum: `signal`, `strength`, `predicted_probability`, `accuracy`, `training_samples`
7. Ensure no look-ahead bias: labels at bar `i` use only data from bars `0..i`, target is bar `i+1`
