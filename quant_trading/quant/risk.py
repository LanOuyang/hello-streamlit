"""Position sizing and risk management overlays.

All functions take *target* positions (decided at the close of bar ``t``)
and only use data available up to and including bar ``t``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .metrics import TRADING_DAYS


def volatility_target(positions: pd.Series, close: pd.Series, target_vol: float,
                      window: int = 20, max_leverage: float = 1.0) -> pd.Series:
    """Scale positions so the expected annualised volatility is ``target_vol``."""
    if target_vol <= 0 or window < 2 or max_leverage <= 0:
        raise ValueError("require target_vol > 0, window >= 2, max_leverage > 0")
    realized = close.pct_change().rolling(window).std() * np.sqrt(TRADING_DAYS)
    scale = (target_vol / realized).clip(upper=max_leverage)
    scale = scale.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return (positions * scale).rename("position")


def apply_stop_loss(positions: pd.Series, close: pd.Series, stop_loss: float) -> pd.Series:
    """Trailing stop: exit when price moves ``stop_loss`` against the best level
    since entry. After a stop, stay flat until the raw signal changes."""
    if not 0 < stop_loss < 1:
        raise ValueError("stop_loss must be in (0, 1)")
    pos = positions.to_numpy(dtype=float)
    px = close.to_numpy(dtype=float)
    out = np.zeros_like(pos)
    best = np.nan
    stopped_signal = None
    prev_sig = 0.0
    for i in range(len(pos)):
        sig = pos[i]
        side = np.sign(sig)
        if side != np.sign(prev_sig):
            best = px[i]
            stopped_signal = None
        prev_sig = sig
        if side == 0 or stopped_signal is not None:
            continue
        best = max(best, px[i]) if side > 0 else min(best, px[i])
        adverse = (px[i] / best - 1) * side
        if adverse <= -stop_loss:
            stopped_signal = side
            continue
        out[i] = sig
    return pd.Series(out, index=positions.index, name="position")
