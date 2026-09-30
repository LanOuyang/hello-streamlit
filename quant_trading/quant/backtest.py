"""Vectorised single-asset backtester."""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from .metrics import summarize
from .risk import apply_stop_loss, volatility_target
from .strategies import Strategy


@dataclass
class BacktestResult:
    equity: pd.Series
    returns: pd.Series
    positions: pd.Series
    benchmark: pd.Series
    costs: pd.Series
    metrics: dict = field(default_factory=dict)

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame({
            "equity": self.equity,
            "returns": self.returns,
            "position": self.positions,
            "costs": self.costs,
            "benchmark": self.benchmark,
        })


@dataclass
class Backtester:
    """Runs a strategy on close-to-close returns.

    The signal computed at the close of bar ``t`` is held over bar ``t+1``,
    which avoids look-ahead bias. Trading costs (commission + slippage, in
    basis points of traded notional) are charged when the position changes.

    Optional overlays, applied in order: stop-loss, volatility targeting.
    """

    initial_capital: float = 100_000.0
    commission_bps: float = 0.0
    slippage_bps: float = 0.0
    stop_loss: float | None = None
    vol_target: float | None = None
    vol_window: int = 20
    max_leverage: float = 1.0

    def target_positions(self, prices: pd.DataFrame, strategy: Strategy) -> pd.Series:
        close = prices["close"]
        target = strategy.generate_positions(prices).reindex(prices.index).fillna(0.0)
        if self.stop_loss:
            target = apply_stop_loss(target, close, self.stop_loss)
        if self.vol_target:
            target = volatility_target(target, close, self.vol_target,
                                       self.vol_window, self.max_leverage)
        return target

    def run(self, prices: pd.DataFrame, strategy: Strategy) -> BacktestResult:
        if "close" not in prices:
            raise ValueError("prices must contain a 'close' column")
        if len(prices) < 2:
            raise ValueError("need at least 2 bars")
        asset_ret = prices["close"].pct_change().fillna(0.0)
        positions = self.target_positions(prices, strategy).shift(1).fillna(0.0)
        turnover = positions.diff().abs().fillna(positions.abs())
        costs = turnover * (self.commission_bps + self.slippage_bps) / 10_000
        strat_ret = positions * asset_ret - costs
        equity = self.initial_capital * (1 + strat_ret).cumprod()
        benchmark = self.initial_capital * (1 + asset_ret).cumprod()
        metrics = summarize(equity, strat_ret, positions)
        metrics["total_costs"] = float(costs.sum())
        return BacktestResult(equity.rename("equity"), strat_ret.rename("returns"),
                              positions.rename("position"), benchmark.rename("benchmark"),
                              costs.rename("costs"), metrics)
