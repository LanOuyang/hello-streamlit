"""Trading strategies.

A strategy maps a price DataFrame to a target position series in [-1, 1]
(1 = fully long, 0 = flat, -1 = fully short) computed using information
available at the *close* of each bar. The backtester is responsible for
lagging positions so trades execute on the next bar (no look-ahead).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
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


@dataclass
class Momentum(Strategy):
    """Time-series momentum: long if the trailing ``lookback`` return is positive."""

    lookback: int = 126
    allow_short: bool = False
    name: str = "momentum"

    def __post_init__(self) -> None:
        if self.lookback < 1:
            raise ValueError("lookback must be >= 1")

    def generate_positions(self, prices: pd.DataFrame) -> pd.Series:
        ret = prices["close"].pct_change(self.lookback)
        pos = pd.Series(0.0, index=prices.index)
        pos[ret > 0] = 1.0
        if self.allow_short:
            pos[ret < 0] = -1.0
        return pos.rename("position")


@dataclass
class BollingerMeanReversion(Strategy):
    """Enter long below the lower band, exit when price reverts to the mean.

    With ``allow_short`` it also shorts above the upper band.
    """

    window: int = 20
    num_std: float = 2.0
    allow_short: bool = False
    name: str = "bollinger"

    def __post_init__(self) -> None:
        if self.window < 2 or self.num_std <= 0:
            raise ValueError("require window >= 2 and num_std > 0")

    def generate_positions(self, prices: pd.DataFrame) -> pd.Series:
        close = prices["close"]
        mid = close.rolling(self.window).mean()
        std = close.rolling(self.window).std()
        upper, lower = mid + self.num_std * std, mid - self.num_std * std
        return _stateful_bands(close, mid, lower, upper, self.allow_short)


@dataclass
class RSIReversion(Strategy):
    """Long when RSI is oversold, exit when RSI crosses back above ``exit_level``."""

    period: int = 14
    oversold: float = 30.0
    overbought: float = 70.0
    exit_level: float = 50.0
    allow_short: bool = False
    name: str = "rsi"

    def __post_init__(self) -> None:
        if not 0 < self.oversold < self.exit_level < self.overbought < 100:
            raise ValueError("require 0 < oversold < exit_level < overbought < 100")

    def generate_positions(self, prices: pd.DataFrame) -> pd.Series:
        r = rsi(prices["close"], self.period)
        level = pd.Series(self.exit_level, index=r.index)
        return _stateful_bands(r, level, pd.Series(self.oversold, index=r.index),
                               pd.Series(self.overbought, index=r.index), self.allow_short)


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Wilder's Relative Strength Index."""
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = gain / loss
    out = 100 - 100 / (1 + rs)
    out[(loss == 0) & gain.notna()] = 100.0
    return out


def _stateful_bands(x, mid, lower, upper, allow_short) -> pd.Series:
    """Enter when ``x`` breaks a band, hold until it crosses ``mid``."""
    pos = np.zeros(len(x))
    state = 0.0
    xv, mv, lv, uv = x.to_numpy(), mid.to_numpy(), lower.to_numpy(), upper.to_numpy()
    for i in range(len(xv)):
        if np.isnan(mv[i]) or np.isnan(xv[i]):
            state = 0.0
        elif state == 0.0:
            if xv[i] < lv[i]:
                state = 1.0
            elif allow_short and xv[i] > uv[i]:
                state = -1.0
        elif state == 1.0 and xv[i] >= mv[i]:
            state = 0.0
        elif state == -1.0 and xv[i] <= mv[i]:
            state = 0.0
        pos[i] = state
    return pd.Series(pos, index=x.index, name="position")


STRATEGIES: dict[str, type[Strategy]] = {
    "sma_crossover": SMACrossover,
    "momentum": Momentum,
    "bollinger": BollingerMeanReversion,
    "rsi": RSIReversion,
}


def make_strategy(name: str, **params) -> Strategy:
    """Instantiate a registered strategy by name."""
    try:
        cls = STRATEGIES[name]
    except KeyError:
        raise ValueError(f"unknown strategy {name!r}; choose from {sorted(STRATEGIES)}") from None
    return cls(**params)
