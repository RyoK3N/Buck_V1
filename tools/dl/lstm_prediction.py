"""PyTorch LSTM price-direction prediction tool."""

from __future__ import annotations
import json
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from langchain_core.tools import tool

from agent_scripts.tools import BaseTool, get_stock_data


# ── LSTM network ─────────────────────────────────────────────────────────────

class _LSTMNet(nn.Module):
    """Small LSTM for binary direction classification."""

    def __init__(self, input_size: int, hidden_size: int = 32, num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lstm_out, _ = self.lstm(x)
        last_hidden = lstm_out[:, -1, :]
        return self.fc(last_hidden).squeeze(-1)


# ── Tool class ───────────────────────────────────────────────────────────────

class LSTMPredictionTool(BaseTool):
    """PyTorch LSTM model for short-term price direction prediction."""

    def __init__(self):
        super().__init__(
            "lstm_prediction",
            "PyTorch LSTM model for short-term price direction prediction"
        )

    @staticmethod
    def _build_features(data: pd.DataFrame, lookback: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        """Create feature matrix and binary labels."""
        close = data['Close'].values.astype(np.float64)
        volume = data['Volume'].values.astype(np.float64)

        log_ret = np.diff(np.log(close + 1e-10))

        n = len(log_ret)
        feat_rows: list[np.ndarray] = []
        labels: list[int] = []

        for i in range(lookback, n - 1):
            window = log_ret[i - lookback:i]
            vol_window = volume[i - lookback + 1:i + 1]

            mean_ret = np.mean(window)
            volatility = np.std(window) + 1e-10
            momentum = close[i] / (close[i - lookback] + 1e-10) - 1.0
            up_ratio = np.sum(window > 0) / lookback
            vol_change = (volume[i] / (np.mean(vol_window) + 1e-10)) - 1.0

            feat_rows.append(np.array([
                log_ret[i], mean_ret, volatility, momentum, up_ratio, vol_change
            ]))
            labels.append(1 if log_ret[i] > 0 else 0)

        if len(feat_rows) == 0:
            return np.empty((0, 6)), np.empty(0)

        X = np.vstack(feat_rows)
        y = np.array(labels, dtype=np.float64)
        return X, y

    @staticmethod
    def _make_sequences(X: np.ndarray, y: np.ndarray, seq_len: int) -> Tuple[np.ndarray, np.ndarray]:
        if len(X) <= seq_len:
            return np.empty((0, seq_len, X.shape[1])), np.empty(0)
        Xs, ys = [], []
        for i in range(seq_len, len(X)):
            Xs.append(X[i - seq_len:i])
            ys.append(y[i])
        return np.array(Xs), np.array(ys)

    def execute(self, data: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        if not self._validate_data(data):
            raise ValueError("Invalid data structure")

        lookback: int = kwargs.get('lookback', 10)
        seq_len: int = kwargs.get('seq_len', 10)
        hidden_size: int = kwargs.get('hidden_size', 32)
        num_layers: int = kwargs.get('num_layers', 2)
        epochs: int = kwargs.get('epochs', 50)
        lr: float = kwargs.get('lr', 1e-3)

        device = torch.device('cpu')

        X_flat, y_flat = self._build_features(data, lookback=lookback)
        if X_flat.shape[0] < seq_len + 20:
            return {
                'signal': 'HOLD', 'strength': 0.0,
                'predicted_probability': 0.5,
                'predicted_direction': 'NEUTRAL',
                'training_samples': int(X_flat.shape[0]),
                'accuracy': 0.0,
                'note': 'Insufficient data for LSTM prediction',
            }

        mu = X_flat.mean(axis=0)
        sigma = X_flat.std(axis=0) + 1e-10
        X_norm = (X_flat - mu) / sigma

        X_seq, y_seq = self._make_sequences(X_norm, y_flat, seq_len)

        split = max(int(len(X_seq) * 0.8), 1)
        X_train, X_val = X_seq[:split], X_seq[split:]
        y_train, y_val = y_seq[:split], y_seq[split:]

        X_train_t = torch.tensor(X_train, dtype=torch.float32, device=device)
        y_train_t = torch.tensor(y_train, dtype=torch.float32, device=device)
        X_val_t = torch.tensor(X_val, dtype=torch.float32, device=device)

        n_features = X_train.shape[2]

        model = _LSTMNet(input_size=n_features, hidden_size=hidden_size, num_layers=num_layers).to(device)
        criterion = nn.BCEWithLogitsLoss()
        optimiser = torch.optim.Adam(model.parameters(), lr=lr)

        model.train()
        for _ in range(epochs):
            optimiser.zero_grad()
            logits = model(X_train_t)
            loss = criterion(logits, y_train_t)
            loss.backward()
            optimiser.step()

        model.eval()
        with torch.no_grad():
            val_logits = model(X_val_t)
            val_probs = torch.sigmoid(val_logits).cpu().numpy()
            val_preds = (val_probs > 0.5).astype(int)
            accuracy = float(np.mean(val_preds == y_val)) if len(y_val) > 0 else 0.0

            latest = torch.tensor(X_seq[-1:], dtype=torch.float32, device=device)
            prob_up = float(torch.sigmoid(model(latest)).cpu().item())

        if prob_up > 0.6:
            signal, direction = 'BUY', 'UP'
            strength = min((prob_up - 0.5) * 2, 1.0)
        elif prob_up < 0.4:
            signal, direction = 'SELL', 'DOWN'
            strength = min((0.5 - prob_up) * 2, 1.0)
        else:
            signal, direction = 'HOLD', 'NEUTRAL'
            strength = 0.3

        return {
            'signal': signal,
            'strength': strength,
            'predicted_probability': round(prob_up, 4),
            'predicted_direction': direction,
            'training_samples': int(X_train.shape[0]),
            'validation_samples': int(X_val.shape[0]),
            'accuracy': round(accuracy, 4),
            'lookback': lookback,
            'seq_len': seq_len,
            'epochs': epochs,
        }


@tool
def lstm_prediction(
    lookback: int = 10,
    seq_len: int = 10,
    epochs: int = 50,
) -> str:
    """Run a PyTorch LSTM model on the current stock data to predict short-term price direction. Returns a BUY/SELL/HOLD signal with probability and accuracy metrics."""
    data = get_stock_data()
    if data is None:
        return json.dumps({"error": "No stock data available"})
    t = LSTMPredictionTool()
    result = t.execute(data, lookback=lookback, seq_len=seq_len, epochs=epochs)
    return json.dumps(result, default=str)


TOOL_CLASS = LSTMPredictionTool
TOOL_FUNC = lstm_prediction
