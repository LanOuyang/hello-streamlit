"""Vectorised single-asset backtester."""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from .metrics import summarize
from .strategies import Strategy


@dataclass
class BacktestResult:
    equity: pd.Series
    returns: pd.Series
    positions: pd.Series
    benchmark: pd.Series
    metrics: dict = field(default_factory=dict)

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame({
            "equity": self.equity,
            "returns": self.returns,
            "position": self.positions,
            "benchmark": self.benchmark,
        })


class Backtester:
    """Runs a strategy on close-to-close returns.

    The signal computed at the close of bar ``t`` is held over bar ``t+1``,
    which avoids look-ahead bias.
    """

    def __init__(self, initial_capital: float = 100_000.0):
        self.initial_capital = initial_capital

    def run(self, prices: pd.DataFrame, strategy: Strategy) -> BacktestResult:
        if "close" not in prices:
            raise ValueError("prices must contain a 'close' column")
        asset_ret = prices["close"].pct_change().fillna(0.0)
        target = strategy.generate_positions(prices).reindex(prices.index).fillna(0.0)
        positions = target.shift(1).fillna(0.0)
        strat_ret = positions * asset_ret
        equity = self.initial_capital * (1 + strat_ret).cumprod()
        benchmark = self.initial_capital * (1 + asset_ret).cumprod()
        metrics = summarize(equity, strat_ret, positions)
        return BacktestResult(equity.rename("equity"), strat_ret.rename("returns"),
                              positions.rename("position"), benchmark.rename("benchmark"),
                              metrics)
