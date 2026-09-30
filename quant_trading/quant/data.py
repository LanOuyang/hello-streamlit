"""Market data sources."""

from __future__ import annotations

import numpy as np
import pandas as pd


def generate_gbm_prices(
    n_days: int = 1000,
    start_price: float = 100.0,
    mu: float = 0.08,
    sigma: float = 0.2,
    seed: int | None = 42,
    start: str = "2020-01-01",
) -> pd.DataFrame:
    """Generate synthetic daily OHLCV bars following geometric Brownian motion.

    ``mu`` and ``sigma`` are annualised drift and volatility.
    """
    if n_days < 2:
        raise ValueError("n_days must be >= 2")
    rng = np.random.default_rng(seed)
    dt = 1 / 252
    log_ret = (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * rng.standard_normal(n_days)
    log_ret[0] = 0.0
    close = start_price * np.exp(np.cumsum(log_ret))
    open_ = np.concatenate([[start_price], close[:-1]])
    noise = np.abs(rng.standard_normal(n_days)) * sigma * np.sqrt(dt) * close
    high = np.maximum(open_, close) + noise
    low = np.minimum(open_, close) - noise
    volume = rng.integers(100_000, 1_000_000, n_days)
    index = pd.bdate_range(start=start, periods=n_days, name="date")
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=index,
    )


def load_csv(path: str) -> pd.DataFrame:
    """Load OHLCV bars from a CSV with a ``date`` column and at least ``close``.

    Column names are case-insensitive (e.g. ``Date,Open,High,Low,Close,Volume``).
    """
    df = pd.read_csv(path)
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    if "adj_close" in df.columns:
        df["close"] = df["adj_close"]
    if "date" not in df.columns or "close" not in df.columns:
        raise ValueError("CSV must contain 'date' and 'close' columns")
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()
    df = df[~df.index.duplicated(keep="last")]
    return df.dropna(subset=["close"])
