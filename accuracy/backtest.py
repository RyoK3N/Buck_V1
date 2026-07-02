"""
accuracy.backtest
─────────────────
Walk-forward validation for the technical-signal pipeline.

The live `accuracy/` tables only record predictions made going forward — they
can't tell you whether a threshold change helped before you ship it. This
module replays a historical OHLCV series through the analyzer one anchor at a
time: at each anchor the analyzer sees only data up to that bar, emits a
BUY/SELL/HOLD signal plus confidence, and the signal is then scored against
the realized forward return. That gives, offline and deterministically:

  • hit-rate per signal (was BUY actually followed by a rise?),
  • overall directional accuracy on actionable (non-HOLD) signals,
  • a confidence-calibration read (mean confidence on hits vs. misses —
    if these are equal, the confidence number carries no predictive
    information, whatever its absolute value).

Usage (library):

    from accuracy.backtest import walk_forward
    report = walk_forward(df)            # df: OHLCV DataFrame
    print(report["summary"])

Usage (CLI, fetches from Yahoo — network):

    python -m accuracy.backtest RELIANCE.NS --start 2024-01-01 --end 2024-06-30

By default only the fast indicator tools run (RSI, MACD, MAs, OBV,
support/resistance, candlestick patterns) — per-anchor LSTM training would
make a backtest take hours and defeat the point. Pass your own analyzer to
include it deliberately.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import pandas as pd

from agent_scripts.analyzers import TechnicalAnalyzer
from agent_scripts.interfaces import IAnalyzer, StockData
from agent_scripts.tools import ToolFactory
from agent_scripts.config import LOGGER

# Indicator tools cheap enough to run at every anchor.
FAST_TOOLS = [
    "rsi",
    "macd",
    "moving_average",
    "obv",
    "support_resistance",
    "candlestick_patterns",
]


def _fast_analyzer() -> TechnicalAnalyzer:
    available = set(ToolFactory.get_available_tools())
    tools = {name: ToolFactory.create_tool(name) for name in FAST_TOOLS if name in available}
    return TechnicalAnalyzer(tools=tools)


def walk_forward(
    data: pd.DataFrame,
    *,
    symbol: str = "BACKTEST",
    analyzer: Optional[IAnalyzer] = None,
    min_history: int = 60,
    step: int = 1,
    horizon: int = 1,
) -> Dict[str, Any]:
    """Replay `data` through the analyzer anchor-by-anchor and score signals.

    Parameters
    ----------
    data:
        OHLCV DataFrame (Open/High/Low/Close/Volume), chronological order.
    analyzer:
        Analyzer to validate. Defaults to a TechnicalAnalyzer restricted to
        the fast indicator tools (FAST_TOOLS).
    min_history:
        Bars the analyzer must see before the first anchor — below ~60 most
        indicators (e.g. the 50-bar MA) are still warming up.
    step:
        Bars between consecutive anchors (1 = every bar).
    horizon:
        Bars ahead whose close decides whether the signal was right.

    Returns
    -------
    dict with:
        records: per-anchor dicts (index, signal, confidence, forward_return, hit)
        summary: aggregate hit-rates and confidence calibration
    """
    if len(data) < min_history + horizon + 1:
        raise ValueError(
            f"Need at least min_history + horizon + 1 = {min_history + horizon + 1} "
            f"bars, got {len(data)}"
        )

    analyzer = analyzer or _fast_analyzer()
    close = data["Close"].astype(float)

    records: List[Dict[str, Any]] = []
    for anchor in range(min_history, len(data) - horizon, step):
        window = data.iloc[:anchor]
        stock = StockData(
            symbol=symbol,
            data=window,
            interval="backtest",
            start_date=str(window.index[0]),
            end_date=str(window.index[-1]),
        )
        result = analyzer.analyze(stock)
        rdata = result.get("data", {}) if isinstance(result, dict) else result["data"]
        summary = rdata.get("summary", {}) if isinstance(rdata, dict) else {}
        signal = summary.get("overall_signal", "HOLD")
        confidence = float(result["confidence"])

        entry = float(close.iloc[anchor - 1])
        exit_ = float(close.iloc[anchor - 1 + horizon])
        fwd_ret = (exit_ - entry) / entry

        hit: Optional[bool] = None
        if signal == "BUY":
            hit = fwd_ret > 0
        elif signal == "SELL":
            hit = fwd_ret < 0

        records.append(
            {
                "anchor": anchor,
                "timestamp": str(data.index[anchor - 1]),
                "signal": signal,
                "confidence": round(confidence, 4),
                "forward_return": round(fwd_ret, 6),
                "hit": hit,
            }
        )

    return {"symbol": symbol, "horizon": horizon, "records": records, "summary": _summarize(records)}


def _summarize(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    by_signal: Dict[str, Dict[str, float]] = {}
    for sig in ("BUY", "SELL", "HOLD"):
        sig_records = [r for r in records if r["signal"] == sig]
        scored = [r for r in sig_records if r["hit"] is not None]
        by_signal[sig] = {
            "n": len(sig_records),
            "hit_rate": (
                round(sum(1 for r in scored if r["hit"]) / len(scored), 4) if scored else None
            ),
        }

    scored = [r for r in records if r["hit"] is not None]
    hits = [r for r in scored if r["hit"]]
    misses = [r for r in scored if not r["hit"]]

    def _mean_conf(rs: List[Dict[str, Any]]) -> Optional[float]:
        return round(sum(r["confidence"] for r in rs) / len(rs), 4) if rs else None

    return {
        "anchors": len(records),
        "actionable": len(scored),
        "overall_hit_rate": round(len(hits) / len(scored), 4) if scored else None,
        "by_signal": by_signal,
        # Calibration: if mean confidence on hits ≈ mean confidence on misses,
        # the confidence score is not telling you anything about accuracy.
        "mean_confidence_on_hits": _mean_conf(hits),
        "mean_confidence_on_misses": _mean_conf(misses),
    }


def _cli() -> None:  # pragma: no cover — network path, exercised manually
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Walk-forward backtest of Buck's technical signals")
    parser.add_argument("symbol", help="Ticker, e.g. RELIANCE.NS")
    parser.add_argument("--start", required=True, help="YYYY-MM-DD")
    parser.add_argument("--end", required=True, help="YYYY-MM-DD")
    parser.add_argument("--interval", default="1d")
    parser.add_argument("--horizon", type=int, default=1)
    parser.add_argument("--step", type=int, default=1)
    parser.add_argument("--min-history", type=int, default=60)
    args = parser.parse_args()

    import yfinance as yf

    df = yf.download(args.symbol, start=args.start, end=args.end, interval=args.interval, progress=False)
    if df is None or df.empty:
        raise SystemExit(f"No data for {args.symbol}")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [str(c) for c in df.columns.get_level_values(0)]

    report = walk_forward(
        df,
        symbol=args.symbol,
        min_history=args.min_history,
        step=args.step,
        horizon=args.horizon,
    )
    LOGGER.info("Backtest finished: %d anchors", report["summary"]["anchors"])
    print(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":  # pragma: no cover
    _cli()
