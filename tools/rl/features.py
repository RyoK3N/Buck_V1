"""
tools.rl.features
─────────────────
Rich state extractor for the continuous-action RL stack.

Returns a fixed-length float32 vector for any (df, idx, position_state) triple.
Designed to be:
  * fast — pure numpy, no pandas in the hot loop after column extraction;
  * numerically safe — every divisor is + 1e-10, every log is on (x + eps),
    every clip is bounded so a single garbage candle can't blow up training;
  * stationary — every feature is either a ratio, z-score, or bounded in
    [-1, 1] / [0, 1] so the agent doesn't see "AAPL is at $193" or "$193 vs
    $0.50" as input.

The output dim is fixed at `STATE_DIM` regardless of where in the series you
are — when there isn't enough history for a long window the function clips
to whatever is available and pads with sensible neutral values (0 for
oscillators, 0 for momentum, 1.0 for ratios).
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

STATE_DIM = 30


def _safe(x: float, default: float = 0.0) -> float:
    if not np.isfinite(x):
        return default
    return float(x)


def _ema(arr: np.ndarray, span: int) -> np.ndarray:
    alpha = 2.0 / (span + 1.0)
    out = np.empty_like(arr, dtype=np.float64)
    out[0] = arr[0]
    for i in range(1, len(arr)):
        out[i] = alpha * arr[i] + (1.0 - alpha) * out[i - 1]
    return out


def _rsi(prices: np.ndarray, period: int) -> float:
    if len(prices) < period + 1:
        return 50.0  # neutral
    diff = np.diff(prices[-period - 1:])
    gains = np.where(diff > 0, diff, 0.0).sum() / period
    losses = -np.where(diff < 0, diff, 0.0).sum() / period
    if losses < 1e-12:
        return 100.0
    rs = gains / losses
    return 100.0 - 100.0 / (1.0 + rs)


def _macd(prices: np.ndarray) -> Tuple[float, float, float]:
    """Returns (macd_line_norm, signal_norm, histogram_norm) — each / price."""
    if len(prices) < 26:
        return 0.0, 0.0, 0.0
    ema12 = _ema(prices, 12)
    ema26 = _ema(prices, 26)
    macd = ema12 - ema26
    signal = _ema(macd, 9)
    hist = macd - signal
    p = max(prices[-1], 1e-6)
    return _safe(macd[-1] / p), _safe(signal[-1] / p), _safe(hist[-1] / p)


def _bollinger(prices: np.ndarray, period: int = 20) -> Tuple[float, float]:
    """Returns (position_in_band ∈ [0,1], width_ratio = (upper-lower)/mean)."""
    window = prices[-period:] if len(prices) >= period else prices
    if len(window) < 2:
        return 0.5, 0.0
    mean = float(np.mean(window))
    std = float(np.std(window))
    upper = mean + 2 * std
    lower = mean - 2 * std
    pos = (prices[-1] - lower) / (upper - lower + 1e-10)
    width = (upper - lower) / (abs(mean) + 1e-10)
    return _safe(np.clip(pos, 0.0, 1.0), 0.5), _safe(width, 0.0)


def _atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> Tuple[float, float]:
    """Returns (atr_absolute, atr_over_price). Simple-mean True Range over the last `period` bars."""
    if len(close) < 2:
        return 0.0, 0.0
    n = min(period, len(close) - 1)
    h = high[-n:]
    l = low[-n:]
    prev_close = close[-n - 1:-1]
    tr = np.maximum(h - l, np.maximum(np.abs(h - prev_close), np.abs(l - prev_close)))
    atr = float(np.mean(tr))
    return _safe(atr), _safe(atr / (close[-1] + 1e-10))


def _adx(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> float:
    """ADX in [0, 100] — measures trend strength regardless of direction."""
    n = min(period, len(close) - 1)
    if n < period:
        return 25.0  # neutral-ish
    up_move = np.diff(high[-n - 1:])
    down_move = -np.diff(low[-n - 1:])
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    tr_components = np.maximum(
        high[-n:] - low[-n:],
        np.maximum(np.abs(high[-n:] - close[-n - 1:-1]), np.abs(low[-n:] - close[-n - 1:-1])),
    )
    tr_sum = float(np.sum(tr_components)) + 1e-10
    plus_di = 100.0 * np.sum(plus_dm) / tr_sum
    minus_di = 100.0 * np.sum(minus_dm) / tr_sum
    dx = 100.0 * abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    return _safe(dx)


def _vwap_distance(prices: np.ndarray, volumes: np.ndarray, period: int = 20) -> float:
    """(price - VWAP) / price."""
    p = prices[-period:] if len(prices) >= period else prices
    v = volumes[-period:] if len(volumes) >= period else volumes
    if v.sum() < 1e-10:
        return 0.0
    vwap = float(np.sum(p * v) / np.sum(v))
    return _safe((prices[-1] - vwap) / (prices[-1] + 1e-10))


def _obv_trend(prices: np.ndarray, volumes: np.ndarray, period: int = 10) -> float:
    """log( mean(OBV last half) / mean(OBV prior half) )."""
    n = min(period, len(prices) - 1)
    if n < 4:
        return 0.0
    diff = np.diff(prices[-n - 1:])
    signed = np.where(diff > 0, volumes[-n:], np.where(diff < 0, -volumes[-n:], 0.0))
    obv = np.cumsum(signed)
    half = max(1, len(obv) // 2)
    recent = abs(obv[-half:].mean()) + 1.0
    prior = abs(obv[:half].mean()) + 1.0
    return _safe(np.log(recent / prior))


def _relative_volume(volumes: np.ndarray, period: int = 20) -> float:
    """log( current / mean(last `period`) )."""
    if len(volumes) < 2:
        return 0.0
    window = volumes[-period:] if len(volumes) >= period else volumes
    avg = float(np.mean(window)) + 1.0
    return _safe(np.log((volumes[-1] + 1.0) / avg))


def _time_features(index_value) -> Tuple[float, float, float, float]:
    """Returns (sin_hour, cos_hour, sin_dow, cos_dow). Intraday-only candles
    give a meaningful hour; daily candles always return (0, 1, sin_dow, cos_dow)."""
    try:
        ts = pd.Timestamp(index_value)
    except Exception:
        return 0.0, 1.0, 0.0, 1.0
    hour = ts.hour + ts.minute / 60.0
    dow = ts.dayofweek  # 0..6
    h_rad = 2 * np.pi * (hour / 24.0)
    d_rad = 2 * np.pi * (dow / 7.0)
    return float(np.sin(h_rad)), float(np.cos(h_rad)), float(np.sin(d_rad)), float(np.cos(d_rad))


def extract_rich_state(
    df: pd.DataFrame,
    idx: int,
    position_size: float,
    cash_ratio: float,
    unrealized_pnl_pct: float = 0.0,
    window: int = 30,
) -> np.ndarray:
    """Compute a STATE_DIM float32 feature vector at row `idx`.

    Args:
        df: OHLCV dataframe (must contain Close, Open, High, Low, Volume).
            Index is a DatetimeIndex (used for time-of-day features).
        idx: Row to extract the state for. Uses bars [0, idx] only — no
             lookahead.
        position_size: Fraction of portfolio currently in the stock, in [0, 1].
        cash_ratio: Fraction of portfolio currently in cash, in [0, 1].
        unrealized_pnl_pct: Open position's P&L vs. entry, in [-1, 1] approx.
        window: Lookback for windowed features (defaults to 30 bars).
    """
    n = idx + 1
    close = df['Close'].values[:n].astype(np.float64)
    high = df['High'].values[:n].astype(np.float64)
    low = df['Low'].values[:n].astype(np.float64)
    volume = df['Volume'].values[:n].astype(np.float64)

    prices = close[-window:] if n >= window else close

    # Returns + momentum
    log_ret_1 = _safe(np.log((prices[-1] + 1e-10) / (prices[-2] + 1e-10)) if len(prices) >= 2 else 0.0)
    log_ret_5 = _safe(np.log((prices[-1] + 1e-10) / (prices[-6] + 1e-10)) if len(prices) >= 6 else 0.0)
    mom_5 = _safe(prices[-1] / (prices[-6] + 1e-10) - 1.0 if len(prices) >= 6 else 0.0)
    mom_10 = _safe(prices[-1] / (prices[-11] + 1e-10) - 1.0 if len(prices) >= 11 else 0.0)
    mom_20 = _safe(prices[-1] / (prices[-21] + 1e-10) - 1.0 if len(close) >= 21 else 0.0)

    # Volatility
    realized_vol = _safe(np.std(np.diff(np.log(prices + 1e-10))) if len(prices) >= 3 else 0.0)

    # RSI at 3 windows
    rsi_7 = _rsi(close, 7) / 100.0
    rsi_14 = _rsi(close, 14) / 100.0
    rsi_21 = _rsi(close, 21) / 100.0

    # MACD
    macd, macd_sig, macd_hist = _macd(close)

    # Bollinger
    bb_pos, bb_width = _bollinger(close)

    # ATR
    atr_abs, atr_rel = _atr(high, low, close)

    # ADX (trend strength)
    adx_val = _adx(high, low, close) / 100.0

    # VWAP
    vwap_dist = _vwap_distance(close, volume)

    # OBV trend
    obv_t = _obv_trend(close, volume)

    # Relative volume
    rel_vol = _relative_volume(volume)

    # Bar shape
    high_low_range = _safe((high[-1] - low[-1]) / (close[-1] + 1e-10))
    upper_wick = _safe((high[-1] - max(close[-1], df['Open'].iloc[idx])) / (close[-1] + 1e-10))
    lower_wick = _safe((min(close[-1], df['Open'].iloc[idx]) - low[-1]) / (close[-1] + 1e-10))

    # Time
    sh, ch, sdow, cdow = _time_features(df.index[idx])

    # Position
    pos_flag = 1.0 if position_size > 1e-6 else 0.0

    feats = [
        log_ret_1, log_ret_5,
        mom_5, mom_10, mom_20,
        realized_vol,
        rsi_7, rsi_14, rsi_21,
        macd, macd_sig, macd_hist,
        bb_pos, bb_width,
        atr_rel,                       # absolute ATR is non-stationary; drop it
        adx_val,
        vwap_dist,
        obv_t,
        rel_vol,
        high_low_range, upper_wick, lower_wick,
        sh, ch, sdow, cdow,
        pos_flag, position_size, cash_ratio, unrealized_pnl_pct,
    ]
    arr = np.asarray(feats, dtype=np.float32)
    # Clip to a sane range so a single garbage value can't blow up the LSTM.
    np.clip(arr, -10.0, 10.0, out=arr)
    if arr.shape[0] != STATE_DIM:
        # Safety net — should never trigger; means STATE_DIM is out of sync.
        raise RuntimeError(f"feature vector dim mismatch: got {arr.shape[0]}, expected {STATE_DIM}")
    return arr


def feature_names() -> list[str]:
    """Stable ordered list of feature names — used by introspection tools."""
    return [
        "log_ret_1", "log_ret_5",
        "mom_5", "mom_10", "mom_20",
        "realized_vol",
        "rsi_7", "rsi_14", "rsi_21",
        "macd_norm", "macd_signal_norm", "macd_hist_norm",
        "bollinger_position", "bollinger_width",
        "atr_relative",
        "adx",
        "vwap_distance",
        "obv_trend",
        "relative_volume",
        "high_low_range", "upper_wick", "lower_wick",
        "sin_hour", "cos_hour", "sin_dayofweek", "cos_dayofweek",
        "position_flag", "position_size", "cash_ratio", "unrealized_pnl_pct",
    ]


if __name__ == "__main__":  # quick self-check
    import yfinance as yf
    df = yf.download("AAPL", period="3mo", interval="1d", progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    print(f"shape: {df.shape}, last 5 close: {df['Close'].tail().values}")
    state = extract_rich_state(df, len(df) - 1, position_size=0.0, cash_ratio=1.0)
    print(f"state dim: {state.shape[0]}, expected {STATE_DIM}")
    for name, val in zip(feature_names(), state):
        print(f"  {name:<24} {val:+.4f}")
