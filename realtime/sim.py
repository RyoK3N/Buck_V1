"""
realtime.sim
────────────
The intraday online-learning loop.

Per observed bar the simulator:
  1. PREDICTS before the event — the agent picks a target position in [0, 1]
     for the *next* bar from the current state;
  2. OBSERVES the next bar (live poll or replay) and books the realised
     intraday PnL of holding that position;
  3. LEARNS — every `online_update_every` bars it rebuilds a small
     `TradingEnvironment` over the recent window and runs a PPO update on an
     in-session adaptive copy of the model (maximising the Sharpe-with-drawdown
     reward), never touching the base checkpoint;
  4. STREAMS the step summary through the headroom context-engineering pipeline
     (so the tokens Claude eventually reads are compressed + accounted for) and
     publishes it to `realtime.state` for the `rt_session_*` MCP tools.

On exit the adaptive model is saved to `<model_id>_live` and the run is persisted
as a training session so it can be visualised through the d3 tools.
"""

from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Any, Dict, Iterator, List, Optional

import numpy as np
import pandas as pd

from .state import LiveSessionState, register_session

# Min bars before we trust features / start updating.
_MIN_HISTORY = 35
_UPDATE_MIN_BARS = 45
TXN_COST = 0.0005  # 5 bps per unit of position change


def _signal(pos: float) -> str:
    return "BUY" if pos >= 0.66 else ("SELL" if pos <= 0.33 else "HOLD")


class IntradaySimulator:
    def __init__(
        self,
        symbol: str,
        model_id: str,
        *,
        interval: str = "1m",
        exchange: str = "NSE",
        poll_seconds: float = 30.0,
        capital: float = 100_000.0,
        replay: bool = False,
        replay_start: Optional[str] = None,
        replay_end: Optional[str] = None,
        online_update_every: int = 4,
        api_key: str = "",
        max_steps: int = 2000,
    ) -> None:
        self.symbol = symbol
        self.model_id = model_id
        self.interval = interval
        self.exchange = exchange
        self.poll_seconds = poll_seconds
        self.capital = capital
        self.replay = replay
        self.replay_start = replay_start
        self.replay_end = replay_end
        self.online_update_every = max(1, online_update_every)
        self.api_key = api_key
        self.max_steps = max_steps

        self.state = LiveSessionState(
            symbol=symbol, model_id=model_id, capital=capital,
            exchange=exchange, replay=replay, max_steps=max_steps,
        )
        self.df = pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])
        self.agent = None  # lazily loaded PPOContinuousAgent (in-session copy)
        self.pos = 0.0           # current target-position fraction in [0, 1]
        self.equity = capital
        self._bars_since_update = 0

    # ── model ────────────────────────────────────────────────────────────────
    def _load_agent(self):
        from tools.rl.ppo_continuous import PPOContinuousAgent

        agent = PPOContinuousAgent()
        loaded = agent.load(self.model_id)
        if not loaded:
            raise RuntimeError(
                f"model {self.model_id!r} not found or not a ppo_continuous checkpoint. "
                f"Train one first (rl_train algorithm='ppo_continuous')."
            )
        agent.reset_window()
        return agent

    # ── bar sources ────────────────────────────────────────────────────────────
    def _replay_bars(self) -> Iterator[Dict[str, Any]]:
        from tools.rl.rl_tool import fetch_historical_data

        df = fetch_historical_data(self.symbol, self.replay_start, self.replay_end, self.interval)
        if df is None or df.empty:
            raise RuntimeError(f"no replay data for {self.symbol} {self.replay_start}..{self.replay_end}")
        for ts, row in df.iterrows():
            yield {
                "timestamp": str(ts),
                "Open": float(row["Open"]), "High": float(row["High"]),
                "Low": float(row["Low"]), "Close": float(row["Close"]),
                "Volume": float(row.get("Volume", 0.0)),
            }

    def _live_bars(self) -> Iterator[Dict[str, Any]]:
        from accuracy.market_hours import is_market_open, time_until_next_open
        from tools.rl.rl_tool import fetch_live_data

        produced = 0
        while produced < self.max_steps:
            if not is_market_open(exchange=self.exchange):
                self.state.market_open = False
                self.state.status = "waiting_market"
                wait = min(self.poll_seconds * 4, max(30.0, time_until_next_open(exchange=self.exchange).total_seconds()))
                print(f"⏳ {self.symbol}: market closed ({self.exchange}); sleeping {wait:.0f}s")
                time.sleep(wait)
                continue
            self.state.market_open = True
            tick = fetch_live_data(self.symbol, self.api_key)
            if tick is None:
                print(f"⚠️  {self.symbol}: no live data this poll; retrying")
                time.sleep(self.poll_seconds)
                continue
            yield {
                "timestamp": tick.get("timestamp", datetime.now(timezone.utc).isoformat()),
                "Open": float(tick.get("open") or tick["price"]),
                "High": float(tick.get("high") or tick["price"]),
                "Low": float(tick.get("low") or tick["price"]),
                "Close": float(tick["price"]),
                "Volume": float(tick.get("volume", 0.0)),
            }
            produced += 1
            time.sleep(self.poll_seconds)

    def _bars(self) -> Iterator[Dict[str, Any]]:
        return self._replay_bars() if self.replay else self._live_bars()

    # ── feature state for prediction ─────────────────────────────────────────
    def _predict_target(self) -> Dict[str, float]:
        from tools.rl.features import extract_rich_state

        idx = len(self.df) - 1
        cash_ratio = 1.0 - self.pos
        state = extract_rich_state(self.df, idx=idx, position_size=self.pos,
                                   cash_ratio=cash_ratio, unrealized_pnl_pct=0.0)
        action, info = self.agent.act(state, eval_mode=True)
        return {"target_position": float(action), **info}

    # ── online learning ──────────────────────────────────────────────────────
    def _online_update(self) -> Optional[Dict[str, float]]:
        if len(self.df) < _UPDATE_MIN_BARS:
            return None
        from tools.rl.env import TradingEnvironment

        window_df = self.df.iloc[-min(len(self.df), 256):].reset_index(drop=True)
        env = TradingEnvironment(window_df, initial_capital=self.capital)
        env.reset()
        # Don't disturb the live prediction window; collect on a throwaway window.
        saved_window = self.agent._window_buf
        self.agent._window_buf = []
        try:
            rollout = self.agent.collect_rollout(env)
            if rollout["n_steps"] == 0:
                return None
            stats = self.agent.update(rollout)
        finally:
            self.agent._window_buf = saved_window
        return stats

    # ── headroom pipeline ────────────────────────────────────────────────────
    @staticmethod
    def _stream_through_headroom(step: Dict[str, Any]) -> None:
        """Push a step summary through the context-engineering pipeline so the
        live stream is compressed + accounted for (USAGE tracker)."""
        try:
            from mcp_server.context_engineering.compressor import compress_payload
            from mcp_server.context_engineering.cost_tracker import USAGE

            _, stats = compress_payload(step)
            USAGE.record("rt_step", stats.get("tokens_raw", 0), stats.get("tokens_compressed", 0),
                         compressed=stats.get("compressed", False))
        except Exception:
            pass

    # ── main loop ────────────────────────────────────────────────────────────
    def run(self) -> Dict[str, Any]:
        self.agent = self._load_agent()
        register_session(self.state)
        self.state.status = "running"
        print(f"▶️  intraday sim: {self.symbol} model={self.model_id} "
              f"{'REPLAY' if self.replay else 'LIVE'} interval={self.interval}")

        prev_close: Optional[float] = None
        pending: Optional[Dict[str, float]] = None  # prediction awaiting its outcome
        steps: List[Dict[str, Any]] = []

        try:
            for bar in self._bars():
                self.df.loc[len(self.df)] = [bar["Open"], bar["High"], bar["Low"], bar["Close"], bar["Volume"]]
                close = bar["Close"]

                # Settle the previous prediction against this bar's move.
                if pending is not None and prev_close:
                    tgt = pending["target_position"]
                    bar_ret = (close / prev_close) - 1.0
                    pos_change = abs(tgt - self.pos)
                    gross = tgt * bar_ret
                    cost = pos_change * TXN_COST
                    net = gross - cost
                    self.equity *= (1.0 + net)
                    self.pos = tgt

                    step = {
                        "ts": bar["timestamp"],
                        "price": round(close, 4),
                        "target_position": round(tgt, 4),
                        "signal": _signal(tgt),
                        "realized_return": round(net, 6),
                        "bar_return": round(bar_ret, 6),
                        "equity": round(self.equity, 2),
                        "mu": round(pending.get("mu", 0.0), 4),
                        "value": round(pending.get("value", 0.0), 4),
                    }
                    steps.append(step)
                    self._stream_through_headroom(step)
                    self.state.record_step(step)

                    # Online learning.
                    self._bars_since_update += 1
                    if self._bars_since_update >= self.online_update_every:
                        self._bars_since_update = 0
                        stats = self._online_update()
                        if stats:
                            self.state.record_update(stats)
                            print(f"  ↻ online update #{self.state.n_updates}: "
                                  f"policy_loss={stats.get('policy_loss', 0):.4f} "
                                  f"entropy={stats.get('entropy', 0):.4f}")

                    print(f"  {step['ts']}  px={close:.2f}  pos→{tgt:.2f} ({step['signal']})  "
                          f"ret={net*100:+.3f}%  equity={self.equity:,.0f}")

                # Predict for the NEXT bar (before the event) once we have history.
                prev_close = close
                if len(self.df) > _MIN_HISTORY:
                    pending = self._predict_target()

                if self.state.n_steps >= self.max_steps:
                    break
        except KeyboardInterrupt:
            print("\n🛑 interrupted — finalising session")
        finally:
            self.state.status = "stopped"
            self._finalise(steps)

        return self.state.status_dict()

    # ── finalisation ─────────────────────────────────────────────────────────
    def _finalise(self, steps: List[Dict[str, Any]]) -> None:
        live_id = f"{self.model_id}_live"
        try:
            if self.agent is not None:
                path = self.agent.save(live_id)
                print(f"💾 saved adaptive model → {path}")
        except Exception as exc:  # noqa: BLE001
            print(f"⚠️  could not save adaptive model: {exc}")

        # Persist as a training session so the d3 tools can visualise the run.
        try:
            from tools.rl.sessions import save_session

            ret_pct = round((self.equity / self.capital - 1.0) * 100.0, 4)
            result = {
                "model_id": live_id,
                "symbol": self.symbol,
                "algorithm": "ppo_continuous_live",
                "episodes": 1,
                "total_steps": len(steps),
                "episode_rewards": [{
                    "episode": 1,
                    "total_reward": round(self.equity - self.capital, 4),
                    "return_pct": ret_pct,
                    "n_updates": self.state.n_updates,
                }],
                "equity_curve": [{"portfolio_value": s["equity"]} for s in steps],
                "steps": steps,
                "final_summary": {
                    "final_portfolio_value": round(self.equity, 2),
                    "total_return_pct": ret_pct,
                    "n_online_updates": self.state.n_updates,
                },
            }
            sid = save_session(result)
            print(f"🗂️  session saved: {sid}  (return {ret_pct:+.2f}%, {self.state.n_updates} online updates)")
        except Exception as exc:  # noqa: BLE001
            print(f"⚠️  could not persist session: {exc}")
