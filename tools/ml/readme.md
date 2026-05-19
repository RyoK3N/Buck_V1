# tools/ml/ — Classical Machine Learning Tools

Scikit-learn-based models that train on OHLCV data at runtime and produce directional signals. These complement the deep learning tools with faster training, better interpretability, and lower variance on small datasets.

All ML tools share a common feature engineering pipeline and follow the same `BaseTool` + `@tool` pattern. Every tool trains on the input DataFrame, holds out the last 20% for validation, and reports accuracy alongside its signal.

---

## Future Tools

### 1. `random_forest_classifier.py` — RandomForestClassifierTool

**What it does:** Trains a Random Forest on engineered OHLCV features (log-returns, multi-period momentum, RSI proxy, volume z-score, ATR, high-low range) to classify next-bar direction as UP or DOWN.

**Why it matters:** Random Forests handle noisy financial data well due to bagging. Out-of-bag scoring provides free validation without a held-out set. Feature importance output is directly LLM-readable — the agent can say "volume z-score was the most predictive feature."

| Parameter | Default | Description |
|-----------|---------|-------------|
| `n_estimators` | 200 | Number of trees |
| `max_depth` | 6 | Maximum tree depth |
| `lookback` | 10 | Feature engineering window |

**Key output fields:**
- `signal`, `strength`, `predicted_probability`, `predicted_direction`
- `feature_importances` — dict mapping feature names to importance scores
- `oob_score` — out-of-bag accuracy estimate
- `training_samples`

**Dependencies:** `scikit-learn`

---

### 2. `gradient_boosting_regime.py` — GradientBoostingRegimeTool

**What it does:** Uses a Gradient Boosting classifier to detect the current market regime: `TRENDING_UP`, `TRENDING_DOWN`, or `RANGING`. Regime labels are derived from rolling linear regression slope and R-squared over the past 20 bars.

**Why it matters:** Different strategies work in different regimes. A BUY signal from RSI in a RANGING market means something different than in a TRENDING_UP market. This tool gives the agent contextual awareness to weight other signals appropriately.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `n_estimators` | 100 | Boosting rounds |
| `max_depth` | 4 | Maximum tree depth |
| `regime_window` | 20 | Window for regime label derivation |

**Key output fields:**
- `signal`, `strength`, `regime`, `regime_confidence`
- `adx_proxy` — directional movement strength
- `bb_width_percentile` — Bollinger Band squeeze indicator
- `regime_history` — last 3 regime labels for trend detection

**Dependencies:** `scikit-learn`

---

### 3. `isolation_forest_anomaly.py` — IsolationForestAnomalyTool

**What it does:** Fits an Isolation Forest on features (log-return, volume ratio, intraday range, price gaps) to detect anomalous price/volume behaviour in recent bars.

**Why it matters:** Anomaly detection serves two purposes: (1) flagging unusual activity before major moves (earnings surprises, halts, gaps), and (2) defensive signalling — the agent should be cautious when anomaly scores are extreme, regardless of other tool outputs. No purely mathematical indicator covers this.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `contamination` | 0.05 | Expected anomaly fraction |
| `n_estimators` | 100 | Number of isolation trees |

**Key output fields:**
- `signal`, `strength`, `anomaly_detected` (bool)
- `anomaly_score` — raw isolation score for the latest bar
- `anomaly_type` — heuristic classification: `VOLUME_SPIKE`, `GAP_ANOMALY`, `RANGE_EXPANSION`, `NORMAL`
- `recent_anomalies` — scores for the last 5 bars

**Signal convention:** Anomaly detected in latest bar = HOLD with high strength (caution). No anomaly = HOLD with low strength (pass-through, defers to other tools).

**Dependencies:** `scikit-learn`

---

### 4. `svm_signal_classifier.py` — SVMSignalClassifierTool

**What it does:** An SVM with RBF kernel trained on the same feature set as the Random Forest, providing a non-linear decision boundary from a different model family.

**Why it matters:** Running SVM alongside Random Forest gives the LLM a second opinion from a fundamentally different algorithm. When both models agree ("RF and SVM both predict SELL"), the agent can express higher conviction. When they disagree, it signals uncertainty.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `C` | 1.0 | Regularisation parameter |
| `lookback` | 10 | Feature engineering window |

**Key output fields:**
- `signal`, `strength`, `predicted_probability`, `predicted_direction`
- `decision_function_score` — signed distance from the hyperplane
- `support_vectors_count`
- `cv_accuracy` — 5-fold cross-validation mean accuracy

**Dependencies:** `scikit-learn`

---

### 5. `logistic_regression_signal.py` — LogisticRegressionSignalTool

**What it does:** A regularised Logistic Regression for interpretable linear probability estimates of price direction.

**Why it matters:** The coefficients are the most interpretable ML output the LLM can reason about. A positive coefficient on `momentum_10` means "10-bar upward momentum is a bullish predictor." It is also the fastest model to train, making it a reliable fallback when data is sparse.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `C` | 1.0 | Inverse regularisation strength |
| `lookback` | 10 | Feature engineering window |

**Key output fields:**
- `signal`, `strength`, `predicted_probability`, `predicted_direction`
- `coefficients` — dict mapping feature names to signed weights
- `intercept`
- `training_accuracy`

**Dependencies:** `scikit-learn`

---

### 6. `knn_pattern_matcher.py` — KNNPatternMatcherTool

**What it does:** k-Nearest Neighbours on normalised price windows. Finds the historical windows in the same series most similar to the current window and aggregates their subsequent outcomes.

**Why it matters:** KNN is a historical analogue finder. Its output is highly intuitive for LLM reasoning: "the current 20-bar pattern resembles 10 historical windows, 7 of which went up." This is entirely non-parametric and degrades gracefully when the price series is non-stationary.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `k` | 10 | Number of neighbours |
| `window_size` | 20 | Pattern matching window |

**Key output fields:**
- `signal`, `strength`, `predicted_probability`
- `nearest_distances` — list of distances to the k neighbours
- `outcome_distribution` — `{"UP": 7, "DOWN": 3}`
- `window_size`

**Dependencies:** `scikit-learn`

---

## Shared Design Notes

- **Feature consistency:** All classification tools (RF, SVM, LR, KNN) should use the same feature engineering function to ensure the LLM can fairly compare their outputs. Consider creating a shared `_build_ml_features(data, lookback)` helper in a `tools/ml/_features.py` module.
- **Look-ahead bias:** Labels at bar `i` must use only data from bars `0..i`, predicting the direction of bar `i+1`.
- **Scaling:** SVM and LR require `StandardScaler`. RF and KNN do not, but scaling does not hurt them. Apply scaling uniformly.
- **Dependencies to add to `requirements.txt`:** `scikit-learn` (already present as `scipy` dependency; add explicitly).
