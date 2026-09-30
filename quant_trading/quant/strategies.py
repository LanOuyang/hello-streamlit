"""Trading strategies.

A strategy maps a price DataFrame to a target position series in [-1, 1]
(1 = fully long, 0 = flat, -1 = fully short) computed using information
available at the *close* of each bar. The backtester is responsible for
lagging positions so trades execute on the next bar (no look-ahead).
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


class Strategy:
    name: str = "base"

    def generate_positions(self, prices: pd.DataFrame) -> pd.Series:
        raise NotImplementedError


@dataclass
class SMACrossover(Strategy):
    """Long when the fast SMA is above the slow SMA, otherwise flat (or short)."""

    fast: int = 20
    slow: int = 50
    allow_short: bool = False
    name: str = "sma_crossover"

    def __post_init__(self) -> None:
        if not 0 < self.fast < self.slow:
            raise ValueError("require 0 < fast < slow")

    def generate_positions(self, prices: pd.DataFrame) -> pd.Series:
        close = prices["close"]
        fast = close.rolling(self.fast).mean()
        slow = close.rolling(self.slow).mean()
        pos = pd.Series(0.0, index=close.index)
        pos[fast > slow] = 1.0
        if self.allow_short:
            pos[fast < slow] = -1.0
        pos[slow.isna()] = 0.0
        return pos.rename("position")
