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

import threading
import time
from datetime import datetime, timezone
from typing import Any, Dict, Iterator, List, Optional

import numpy as np
import pandas as pd

from .state import LiveSessionState, register_session

# Min bars before we trust features / start updating.
_MIN_HISTORY = 35
_UPDATE_MIN_BARS = 45
# Per-unit-of-position-change transaction cost. MUST match TradingEnvironment's
# default `transaction_cost` (tools/rl/env.py) so the live path's PnL matches the
# backtest/training math. (Replay uses TradingEnvironment directly.)
TXN_COST = 0.001  # 10 bps

# Wall-clock seconds represented by one bar of each interval — used to pace replay.
_BAR_PERIOD = {
    "1m": 60, "2m": 120, "5m": 300, "15m": 900, "30m": 1800,
    "60m": 3600, "90m": 5400, "1h": 3600, "1d": 86400,
}
# At/above this speed factor, replay runs with no pacing delay ("instant").
_INSTANT_SPEED = 1000.0


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
        online_update_every: int = 0,
        api_key: str = "",
        max_steps: int = 2000,
        speed: float = 1.0,
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
        # 0 disables online updates (pure inference → deterministic, matches backtest).
        self.online_update_every = max(0, int(online_update_every))
        self.api_key = api_key
        self.max_steps = max_steps
        # Replay fast-forward factor: per-bar delay = bar_period / speed.
        self.speed = max(0.0, float(speed))

        self.state = LiveSessionState(
            symbol=symbol, model_id=model_id, capital=capital,
            exchange=exchange, replay=replay, max_steps=max_steps,
        )
        self.df = pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])
        self.agent = None  # lazily loaded PPOContinuousAgent (in-session copy)
        self.pos = 0.0           # current target-position fraction in [0, 1]
        self.entry_price = 0.0   # avg entry price of the open position (live path)
        self.equity = capital
        self._bars_since_update = 0
        # Cooperative stop flag — set via request_stop() (e.g. from the REST layer)
        # to end the loop gracefully at the next bar / poll boundary.
        self._stop = threading.Event()

    def request_stop(self) -> None:
        """Ask the loop to finish at the next safe point (cooperative, thread-safe)."""
        self._stop.set()

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

    # ── pacing / position helpers ──────────────────────────────────────────────
    def _bar_period_seconds(self) -> float:
        return float(_BAR_PERIOD.get(self.interval, 60.0))

    def _replay_delay(self) -> float:
        """Wall-clock delay between replay bars given the fast-forward factor."""
        if self.speed <= 0 or self.speed >= _INSTANT_SPEED:
            return 0.0
        return self._bar_period_seconds() / self.speed

    def _live_unrealized(self, close: float) -> float:
        """Unrealized PnL % of the open position — mirrors env.unrealized_pnl_pct()."""
        if self.pos <= 1e-9 or self.entry_price <= 0:
            return 0.0
        return (close / self.entry_price - 1.0) * 100.0

    # ── bar source (live) ───────────────────────────────────────────────────────
    def _live_bars(self) -> Iterator[Dict[str, Any]]:
        from accuracy.market_hours import is_market_open, time_until_next_open
        from tools.rl.rl_tool import fetch_live_data

        produced = 0
        while produced < self.max_steps:
            if self._stop.is_set():
                return
            if not is_market_open(exchange=self.exchange):
                self.state.market_open = False
                self.state.status = "waiting_market"
                wait = min(self.poll_seconds * 4, max(30.0, time_until_next_open(exchange=self.exchange).total_seconds()))
                print(f"⏳ {self.symbol}: market closed ({self.exchange}); sleeping {wait:.0f}s")
                # Interruptible sleep so a stop request doesn't wait out the gate.
                if self._stop.wait(timeout=wait):
                    return
                continue
            self.state.market_open = True
            tick = fetch_live_data(self.symbol, self.api_key)
            if tick is None:
                print(f"⚠️  {self.symbol}: no live data this poll; retrying")
                if self._stop.wait(timeout=self.poll_seconds):
                    return
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
            if self._stop.wait(timeout=self.poll_seconds):
                return

    # ── feature state for prediction ─────────────────────────────────────────
    def _predict_target(self) -> Dict[str, float]:
        from tools.rl.features import extract_rich_state

        idx = len(self.df) - 1
        cash_ratio = 1.0 - self.pos
        close = float(self.df["Close"].iloc[idx])
        state = extract_rich_state(self.df, idx=idx, position_size=self.pos,
                                   cash_ratio=cash_ratio,
                                   unrealized_pnl_pct=self._live_unrealized(close))
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

    def _online_update_replay(self, df, upto_idx: int) -> Optional[Dict[str, float]]:
        """Online PPO update during replay over ONLY the bars seen so far (no future
        leakage). Same throwaway-env + window save/restore as _online_update."""
        if upto_idx + 1 < _UPDATE_MIN_BARS:
            return None
        from tools.rl.env import TradingEnvironment

        lo = max(0, upto_idx - 256)
        window_df = df.iloc[lo:upto_idx + 1].reset_index(drop=True)
        env = TradingEnvironment(window_df, initial_capital=self.capital)
        env.reset()
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

    # ── headroom accounting ──────────────────────────────────────────────────
    @staticmethod
    def _account_step(step: Dict[str, Any]) -> None:
        """Account a per-bar step in the USAGE tracker with a lightweight token
        count. Steps are tiny (~70 tokens) so we never invoke headroom compression
        here — that would be pure overhead in the hot loop and could trip the
        headroom circuit breaker over a long replay."""
        try:
            import json as _json
            from mcp_server.context_engineering.cost_tracker import USAGE, estimate_tokens

            n = estimate_tokens(_json.dumps(step, default=str))
            USAGE.record("rt_step", n, n, compressed=False)
        except Exception:
            pass

    # ── orchestration ──────────────────────────────────────────────────────────
    def run(self) -> Dict[str, Any]:
        self.agent = self._load_agent()
        register_session(self.state)
        self.state.status = "running"
        mode = "REPLAY" if self.replay else "LIVE"
        speed = f" speed×{self.speed:g}" if self.replay else ""
        print(f"▶️  intraday sim: {self.symbol} model={self.model_id} "
              f"{mode} interval={self.interval}{speed}")

        steps: List[Dict[str, Any]] = []
        try:
            if self.replay:
                self._run_replay(steps)
            else:
                self._run_live(steps)
        except KeyboardInterrupt:
            print("\n🛑 interrupted — finalising session")
        finally:
            self.state.status = "stopped"
            self._finalise(steps)

        return self.state.status_dict()

    # ── replay engine (env-backed, paced) ──────────────────────────────────────
    def _run_replay(self, steps: List[Dict[str, Any]]) -> None:
        """Stream a historical window through the SAME TradingEnvironment used by
        training / the /rl/predict backtest, so replay PnL is exact. Paced by the
        fast-forward factor so the UI can watch the agent bar-by-bar."""
        from tools.rl.rl_tool import fetch_historical_data
        from tools.rl.env import TradingEnvironment

        df = fetch_historical_data(self.symbol, self.replay_start, self.replay_end, self.interval)
        if df is None or df.empty:
            raise RuntimeError(f"no replay data for {self.symbol} {self.replay_start}..{self.replay_end}")
        self.df = df

        env = TradingEnvironment(df, initial_capital=self.capital)
        state = env.reset()
        self.agent.reset_window()
        delay = self._replay_delay()
        index = list(df.index)

        while not self._stop.is_set():
            if env.idx >= env.n - 1:
                break
            action, info = self.agent.act(state, eval_mode=True)
            price_now = float(env.close[env.idx])
            next_state, _reward, done, si = env.step(action)  # authoritative math
            bar_ret = (si.price / price_now - 1.0) if price_now else 0.0

            step = {
                "ts": str(index[si.step]) if si.step < len(index) else str(si.step),
                "price": round(si.price, 4),
                "target_position": round(si.target_position, 4),
                "signal": _signal(si.target_position),
                "realized_return": round(si.step_return, 6),
                "bar_return": round(bar_ret, 6),
                "equity": round(si.portfolio_value, 2),
                "mu": round(info.get("mu", 0.0), 4),
                "value": round(info.get("value", 0.0), 4),
                "forced_exit": si.forced_exit,
            }
            steps.append(step)
            self._account_step(step)
            self.state.record_step(step)
            self.pos = si.realized_position
            self.equity = si.portfolio_value

            # Optional online learning (off by default → deterministic backtest parity).
            if self.online_update_every:
                self._bars_since_update += 1
                if self._bars_since_update >= self.online_update_every:
                    self._bars_since_update = 0
                    stats = self._online_update_replay(df, si.step)
                    if stats:
                        self.state.record_update(stats)

            state = next_state
            if done or self.state.n_steps >= self.max_steps:
                break
            if delay and self._stop.wait(timeout=delay):
                break

        env.close_position()
        self.equity = env.portfolio_value()

    # ── live engine (incremental, poll-paced) ──────────────────────────────────
    def _run_live(self, steps: List[Dict[str, Any]]) -> None:
        prev_close: Optional[float] = None
        pending: Optional[Dict[str, float]] = None  # prediction awaiting its outcome

        for bar in self._live_bars():
            if self._stop.is_set():
                break
            self.df.loc[len(self.df)] = [bar["Open"], bar["High"], bar["Low"], bar["Close"], bar["Volume"]]
            close = bar["Close"]

            # Settle the previous prediction against this bar's move.
            if pending is not None and prev_close:
                tgt = pending["target_position"]
                bar_ret = (close / prev_close) - 1.0
                prev_pos = self.pos
                pos_change = abs(tgt - prev_pos)
                gross = tgt * bar_ret
                cost = pos_change * TXN_COST
                net = gross - cost
                self.equity *= (1.0 + net)
                self.pos = tgt
                # Track entry price so _predict_target sees real unrealized PnL.
                if tgt <= 1e-9:
                    self.entry_price = 0.0
                elif prev_pos <= 1e-9:
                    self.entry_price = close

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
                self._account_step(step)
                self.state.record_step(step)

                if self.online_update_every:
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
