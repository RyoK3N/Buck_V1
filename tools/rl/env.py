"""
tools.rl.env
────────────
Trading environment for the continuous-action RL stack.

Design notes:
  * Long-only (the Wallet class doesn't support shorts; clamping action to
    [0, 1] keeps the rest of the codebase honest).
  * Action is a *target position fraction* — 0.0 = all cash, 1.0 = all in.
    The env rebalances the wallet to that target each step, computing
    transaction cost on the absolute delta.
  * Reward is risk-adjusted: step_return / rolling_volatility, with explicit
    penalties for drawdown and trading friction. Maximizing this is much
    less reckless than maximizing raw PnL.
  * Stop-loss / take-profit fire automatically inside `step()` — they
    represent the env's risk-management contract with the agent, not a
    decision the agent makes.

The env is deliberately not gym-API-strict (no Dict spaces, no seeding hooks);
Buck doesn't need that yet and pulling in `gymnasium` for a single env is
overkill.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

from .features import STATE_DIM, extract_rich_state


@dataclass
class StepInfo:
    """Info dict for each env.step() call. Returned alongside (state, reward, done)."""
    step: int
    price: float
    target_position: float          # what the agent asked for, clamped to [0, 1]
    realized_position: float        # what the wallet actually ended up at
    portfolio_value: float
    cash: float
    holdings: float
    step_return: float
    rolling_vol: float
    drawdown: float
    transaction_cost: float
    forced_exit: Optional[str]      # None | "stop_loss" | "take_profit"


class TradingEnvironment:
    """Single-asset, long-only trading env for RL.

    Args:
        df:                  OHLCV dataframe. Must contain Open/High/Low/Close/Volume.
        initial_capital:     Starting cash.
        transaction_cost:    Round-trip fraction per dollar traded (e.g. 0.001).
        stop_loss:           If unrealized loss exceeds this fraction, force-exit
                             (e.g. 0.05 = 5%). Set to None to disable.
        take_profit:         Analogous for gains. None disables.
        reward_lookback:     Number of recent returns used to compute the
                             rolling volatility denominator. Default 20.
        drawdown_penalty:    λ in reward; penalty = -λ * drawdown². Default 1.0.
        risk_free_rate:      Per-step risk-free rate (rarely meaningful for
                             intraday; default 0 keeps things tidy).
        min_history:         How many bars the env burns at the start before
                             the agent gets to act. Lets the feature extractor
                             warm up. Default 5 — features.py degrades gracefully
                             past that. Bump to 21+ if you have plenty of data
                             and want all indicators fully warm before training.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        initial_capital: float = 100_000.0,
        transaction_cost: float = 0.001,
        stop_loss: Optional[float] = 0.05,
        take_profit: Optional[float] = 0.10,
        reward_lookback: int = 20,
        drawdown_penalty: float = 1.0,
        risk_free_rate: float = 0.0,
        min_history: int = 5,
    ):
        required = {"Open", "High", "Low", "Close", "Volume"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"df missing required columns: {sorted(missing)}")
        if len(df) <= min_history + 1:
            raise ValueError(f"df too short ({len(df)}) for min_history={min_history}")

        self.df = df
        self.close = df["Close"].values.astype(np.float64)
        self.n = len(df)
        self.initial_capital = float(initial_capital)
        self.transaction_cost = float(transaction_cost)
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        self.reward_lookback = int(reward_lookback)
        self.drawdown_penalty = float(drawdown_penalty)
        self.risk_free_step = float(risk_free_rate)
        self.min_history = int(min_history)

        # State (set up in reset)
        self.cash: float = 0.0
        self.holdings: float = 0.0
        self.entry_price: float = 0.0
        self.idx: int = 0
        self.peak_pv: float = 0.0
        self.recent_returns: deque[float] = deque(maxlen=self.reward_lookback)
        self.trades: list[Dict[str, Any]] = []
        self.equity_curve: list[Dict[str, Any]] = []

    # ── Properties ──────────────────────────────────────────────────────────

    @property
    def state_dim(self) -> int:
        return STATE_DIM

    def portfolio_value(self, price: Optional[float] = None) -> float:
        p = price if price is not None else self.close[self.idx]
        return self.cash + self.holdings * p

    def position_size(self, price: Optional[float] = None) -> float:
        """Holdings as a fraction of current portfolio value."""
        pv = self.portfolio_value(price)
        if pv <= 0 or self.holdings <= 1e-9:
            return 0.0
        p = price if price is not None else self.close[self.idx]
        return float(self.holdings * p / pv)

    def unrealized_pnl_pct(self) -> float:
        if self.holdings <= 1e-9 or self.entry_price <= 0:
            return 0.0
        return float(self.close[self.idx] / self.entry_price - 1.0)

    # ── Lifecycle ───────────────────────────────────────────────────────────

    def reset(self) -> np.ndarray:
        self.cash = self.initial_capital
        self.holdings = 0.0
        self.entry_price = 0.0
        self.idx = self.min_history
        self.peak_pv = self.initial_capital
        self.recent_returns.clear()
        self.trades.clear()
        self.equity_curve.clear()
        self._snapshot()
        return self._state()

    def step(self, action: float) -> Tuple[np.ndarray, float, bool, StepInfo]:
        """Apply target position fraction, advance one bar, compute reward."""
        if self.idx >= self.n - 1:
            raise RuntimeError("env exhausted — call reset() before stepping again")

        target = float(np.clip(action, 0.0, 1.0))
        price_now = float(self.close[self.idx])
        pv_before = self.portfolio_value(price_now)
        forced: Optional[str] = None

        # Stop-loss / take-profit run BEFORE the agent's action — they're the
        # env's safety net, not the agent's choice.
        if self.holdings > 1e-9:
            unr = self.unrealized_pnl_pct()
            if self.stop_loss is not None and unr <= -abs(self.stop_loss):
                target = 0.0
                forced = "stop_loss"
            elif self.take_profit is not None and unr >= abs(self.take_profit):
                target = 0.0
                forced = "take_profit"

        # Rebalance to target. Trade is the dollar delta between current and
        # target position; transaction cost is applied to its absolute value.
        current_pos_value = self.holdings * price_now
        target_pos_value = target * pv_before
        delta_value = target_pos_value - current_pos_value
        tc = abs(delta_value) * self.transaction_cost

        # Apply trade
        if delta_value > 0:  # buy
            buy_value = min(delta_value, self.cash - tc)
            if buy_value > 0:
                qty = buy_value / price_now
                self.cash -= (buy_value + tc * (buy_value / max(abs(delta_value), 1e-9)))
                self.holdings += qty
                # Volume-weighted entry price update
                if self.entry_price > 0:
                    prev_qty = self.holdings - qty
                    self.entry_price = (self.entry_price * prev_qty + price_now * qty) / max(self.holdings, 1e-9)
                else:
                    self.entry_price = price_now
                self.trades.append({"step": self.idx, "type": "BUY",
                                    "price": price_now, "value": round(buy_value, 2),
                                    "fee": round(tc, 4), "forced": forced})
        elif delta_value < 0:  # sell
            sell_value = min(-delta_value, current_pos_value)
            if sell_value > 0:
                qty = sell_value / price_now
                self.cash += (sell_value - tc * (sell_value / max(abs(delta_value), 1e-9)))
                self.holdings -= qty
                if self.holdings < 1e-9:
                    self.holdings = 0.0
                    self.entry_price = 0.0
                self.trades.append({"step": self.idx, "type": "SELL",
                                    "price": price_now, "value": round(sell_value, 2),
                                    "fee": round(tc, 4), "forced": forced})

        # Advance one bar
        self.idx += 1
        price_next = float(self.close[self.idx])
        pv_after = self.portfolio_value(price_next)
        step_return = (pv_after / pv_before - 1.0) if pv_before > 1e-9 else 0.0
        self.recent_returns.append(step_return)

        # Drawdown bookkeeping
        if pv_after > self.peak_pv:
            self.peak_pv = pv_after
        drawdown = (self.peak_pv - pv_after) / max(self.peak_pv, 1e-9)

        # Risk-adjusted reward
        if len(self.recent_returns) >= 5:
            rolling_vol = float(np.std(self.recent_returns)) + 1e-4
        else:
            rolling_vol = 0.01  # neutral until we have enough history
        risk_adj_return = (step_return - self.risk_free_step) / rolling_vol
        dd_penalty = self.drawdown_penalty * drawdown * drawdown
        tc_penalty = tc / max(pv_before, 1e-9)
        reward = risk_adj_return - dd_penalty - tc_penalty

        realized = self.position_size(price_next)
        info = StepInfo(
            step=self.idx,
            price=price_next,
            target_position=target,
            realized_position=realized,
            portfolio_value=pv_after,
            cash=self.cash,
            holdings=self.holdings,
            step_return=step_return,
            rolling_vol=rolling_vol,
            drawdown=drawdown,
            transaction_cost=tc,
            forced_exit=forced,
        )
        self._snapshot()

        done = self.idx >= self.n - 1
        return self._state(), float(reward), done, info

    def close_position(self) -> float:
        """Force-exit any open position at the current price (no further stepping)."""
        if self.holdings <= 1e-9:
            return self.portfolio_value()
        price = float(self.close[self.idx])
        sell_value = self.holdings * price
        tc = sell_value * self.transaction_cost
        self.cash += sell_value - tc
        self.holdings = 0.0
        self.entry_price = 0.0
        self.trades.append({"step": self.idx, "type": "SELL", "price": price,
                            "value": round(sell_value, 2), "fee": round(tc, 4),
                            "forced": "episode_end"})
        return self.portfolio_value()

    # ── Reporting ───────────────────────────────────────────────────────────

    def summary(self) -> Dict[str, Any]:
        price = float(self.close[self.idx])
        pv = self.portfolio_value(price)
        total_return = (pv / self.initial_capital - 1.0) * 100.0
        max_dd = self._max_drawdown_pct()
        sharpe = self._annualized_sharpe()
        wins, losses = self._win_loss_counts()
        return {
            "initial_capital": round(self.initial_capital, 2),
            "final_portfolio_value": round(pv, 2),
            "total_return_pct": round(total_return, 4),
            "annualized_sharpe": round(sharpe, 4),
            "max_drawdown_pct": round(max_dd, 4),
            "total_trades": len(self.trades),
            "winning_trades": wins,
            "losing_trades": losses,
            "win_rate_pct": round(100.0 * wins / max(wins + losses, 1), 2),
            "transaction_cost": self.transaction_cost,
        }

    # ── Internals ───────────────────────────────────────────────────────────

    def _state(self) -> np.ndarray:
        price = float(self.close[self.idx])
        pv = self.portfolio_value(price)
        cash_ratio = float(self.cash / max(pv, 1e-9))
        return extract_rich_state(
            df=self.df,
            idx=self.idx,
            position_size=self.position_size(price),
            cash_ratio=cash_ratio,
            unrealized_pnl_pct=self.unrealized_pnl_pct(),
        )

    def _snapshot(self) -> None:
        price = float(self.close[self.idx])
        self.equity_curve.append({
            "step": self.idx,
            "price": price,
            "cash": round(self.cash, 2),
            "holdings": round(self.holdings, 6),
            "portfolio_value": round(self.portfolio_value(price), 2),
        })

    def _max_drawdown_pct(self) -> float:
        if not self.equity_curve:
            return 0.0
        pv = np.array([e["portfolio_value"] for e in self.equity_curve], dtype=np.float64)
        peak = np.maximum.accumulate(pv)
        dd = (peak - pv) / np.maximum(peak, 1e-9)
        return float(dd.max() * 100.0)

    def _annualized_sharpe(self) -> float:
        if len(self.equity_curve) < 5:
            return 0.0
        pv = np.array([e["portfolio_value"] for e in self.equity_curve], dtype=np.float64)
        rets = np.diff(pv) / np.maximum(pv[:-1], 1e-9)
        if rets.std() < 1e-9:
            return 0.0
        return float((rets.mean() - self.risk_free_step) / (rets.std() + 1e-9) * np.sqrt(252))

    def _win_loss_counts(self) -> Tuple[int, int]:
        wins = losses = 0
        last_buy_price = None
        last_buy_qty = 0.0
        for t in self.trades:
            if t["type"] == "BUY":
                last_buy_price = t["price"]
                last_buy_qty = t["value"] / max(t["price"], 1e-9)
            elif t["type"] == "SELL" and last_buy_price is not None:
                pnl = (t["price"] - last_buy_price) * last_buy_qty - t["fee"]
                if pnl > 0:
                    wins += 1
                else:
                    losses += 1
                last_buy_price = None
                last_buy_qty = 0.0
        return wins, losses


if __name__ == "__main__":  # quick smoke
    import yfinance as yf
    df = yf.download("AAPL", period="6mo", interval="1d", progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    env = TradingEnvironment(df)
    state = env.reset()
    print(f"state shape: {state.shape}, dim: {env.state_dim}")
    total_reward = 0.0
    # Random-ish policy: alternate full long / flat
    for s in range(50):
        action = 1.0 if s % 5 < 2 else 0.0
        state, reward, done, info = env.step(action)
        total_reward += reward
        if done:
            break
    env.close_position()
    print(f"steps: {env.idx}, total reward: {total_reward:.4f}")
    print(env.summary())
