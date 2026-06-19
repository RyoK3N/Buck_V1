from __future__ import annotations
import numpy as np
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple


class Wallet:
    def __init__(self, initial_capital: float = 100000.0, transaction_cost: float = 0.001):
        self.initial_capital = initial_capital
        self.transaction_cost = transaction_cost
        self.reset()

    def reset(self):
        self.cash = self.initial_capital
        self.holdings = 0.0
        self.trades: List[Dict[str, Any]] = []
        self.equity_curve: List[Dict[str, Any]] = []
        self._record_snapshot(None, self.initial_capital)

    def _record_snapshot(self, price: Optional[float], pv: Optional[float] = None):
        self.equity_curve.append({
            'timestamp': datetime.now().isoformat(),
            'cash': self.cash,
            'holdings': self.holdings,
            'price': price,
            'portfolio_value': pv if pv is not None else (self.cash + self.holdings * price if price else self.cash),
        })

    def buy(self, price: float, timestamp: Optional[str] = None) -> Tuple[bool, float, float]:
        max_cost = self.cash / (1 + self.transaction_cost)
        if max_cost < price * 0.5:
            return False, 0.0, 0.0
        quantity = max_cost / price
        cost = quantity * price
        fee = cost * self.transaction_cost
        self.cash -= (cost + fee)
        self.holdings += quantity
        self.trades.append({
            'timestamp': timestamp or datetime.now().isoformat(),
            'type': 'BUY',
            'price': round(price, 2),
            'quantity': round(quantity, 4),
            'cost': round(cost, 2),
            'fee': round(fee, 2),
        })
        pv = self.cash + self.holdings * price
        self._record_snapshot(price, pv)
        return True, quantity, fee

    def sell(self, price: float, timestamp: Optional[str] = None) -> Tuple[bool, float, float]:
        if self.holdings < 1e-6:
            return False, 0.0, 0.0
        quantity = self.holdings
        revenue = quantity * price
        fee = revenue * self.transaction_cost
        self.cash += (revenue - fee)
        self.holdings = 0.0
        self.trades.append({
            'timestamp': timestamp or datetime.now().isoformat(),
            'type': 'SELL',
            'price': round(price, 2),
            'quantity': round(quantity, 4),
            'revenue': round(revenue, 2),
            'fee': round(fee, 2),
        })
        pv = self.cash
        self._record_snapshot(price, pv)
        return True, quantity, fee

    def close_position(self, price: float, timestamp: Optional[str] = None) -> Tuple[bool, float, float]:
        return self.sell(price, timestamp)

    def get_portfolio_value(self, current_price: float) -> float:
        return self.cash + self.holdings * current_price

    def get_total_return_pct(self, current_price: float) -> float:
        pv = self.get_portfolio_value(current_price)
        return ((pv - self.initial_capital) / self.initial_capital) * 100

    def get_sharpe_ratio(self, risk_free_rate: float = 0.05) -> float:
        if len(self.equity_curve) < 5:
            return 0.0
        values = np.array([e['portfolio_value'] for e in self.equity_curve])
        returns = np.diff(values) / (values[:-1] + 1e-10)
        if len(returns) < 2 or np.std(returns) < 1e-10:
            return 0.0
        return float((np.mean(returns) - risk_free_rate / 252) / (np.std(returns) + 1e-10) * np.sqrt(252))

    def get_max_drawdown_pct(self) -> float:
        if len(self.equity_curve) < 2:
            return 0.0
        values = [e['portfolio_value'] for e in self.equity_curve]
        peak = values[0]
        max_dd = 0.0
        for v in values:
            if v > peak:
                peak = v
            dd = (peak - v) / peak if peak > 0 else 0
            max_dd = max(max_dd, dd)
        return max_dd * 100

    def get_win_rate(self) -> float:
        buys = [t for t in self.trades if t['type'] == 'BUY']
        sells = [t for t in self.trades if t['type'] == 'SELL']
        if not sells:
            return 0.0
        wins = 0
        buy_idx = 0
        for sell in sells:
            if buy_idx < len(buys):
                buy = buys[buy_idx]
                net_profit = sell['revenue'] - sell['fee'] - buy['cost'] - buy['fee']
                if net_profit > 0:
                    wins += 1
                buy_idx += 1
        return (wins / len(sells)) * 100.0

    def get_summary(self, current_price: float) -> Dict[str, Any]:
        pv = self.get_portfolio_value(current_price)
        total_return = ((pv - self.initial_capital) / self.initial_capital) * 100
        return {
            'initial_capital': round(self.initial_capital, 2),
            'cash': round(self.cash, 2),
            'holdings': round(self.holdings, 4),
            'current_price': round(current_price, 2),
            'portfolio_value': round(pv, 2),
            'total_return_pct': round(total_return, 2),
            'total_trades': len(self.trades),
            'win_rate_pct': round(self.get_win_rate(), 2),
            'sharpe_ratio': round(self.get_sharpe_ratio(), 4),
            'max_drawdown_pct': round(self.get_max_drawdown_pct(), 2),
            'transaction_cost': self.transaction_cost,
        }

    def to_dict(self) -> Dict[str, Any]:
        return {
            'initial_capital': self.initial_capital,
            'cash': self.cash,
            'holdings': self.holdings,
            'transaction_cost': self.transaction_cost,
            'trades': self.trades[-200:],
            'equity_curve': self.equity_curve,
            'summary': self.get_summary(self.trades[-1]['price'] if self.trades else 0) if self.trades else self.get_summary(0),
        }
