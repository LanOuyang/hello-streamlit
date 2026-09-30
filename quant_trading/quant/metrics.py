"""Performance metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def total_return(equity: pd.Series) -> float:
    return float(equity.iloc[-1] / equity.iloc[0] - 1)


def cagr(equity: pd.Series, periods_per_year: int = TRADING_DAYS) -> float:
    years = (len(equity) - 1) / periods_per_year
    if years <= 0 or equity.iloc[-1] <= 0:
        return 0.0
    return float((equity.iloc[-1] / equity.iloc[0]) ** (1 / years) - 1)


def annual_volatility(returns: pd.Series, periods_per_year: int = TRADING_DAYS) -> float:
    return float(returns.std(ddof=1) * np.sqrt(periods_per_year))


def sharpe_ratio(returns: pd.Series, risk_free: float = 0.0,
                 periods_per_year: int = TRADING_DAYS) -> float:
    excess = returns - risk_free / periods_per_year
    std = excess.std(ddof=1)
    if not std or np.isnan(std):
        return 0.0
    return float(excess.mean() / std * np.sqrt(periods_per_year))


def sortino_ratio(returns: pd.Series, periods_per_year: int = TRADING_DAYS) -> float:
    downside = returns[returns < 0]
    dd = np.sqrt((downside**2).sum() / max(len(returns), 1))
    if dd == 0:
        return 0.0
    return float(returns.mean() / dd * np.sqrt(periods_per_year))


def drawdown(equity: pd.Series) -> pd.Series:
    return equity / equity.cummax() - 1


def max_drawdown(equity: pd.Series) -> float:
    return float(drawdown(equity).min())


def calmar_ratio(equity: pd.Series) -> float:
    mdd = max_drawdown(equity)
    return 0.0 if mdd == 0 else float(cagr(equity) / abs(mdd))


def summarize(equity: pd.Series, returns: pd.Series, positions: pd.Series) -> dict:
    trades = positions.diff().abs().fillna(positions.abs())
    active = returns[positions != 0]
    return {
        "total_return": total_return(equity),
        "cagr": cagr(equity),
        "volatility": annual_volatility(returns),
        "sharpe": sharpe_ratio(returns),
        "sortino": sortino_ratio(returns),
        "max_drawdown": max_drawdown(equity),
        "calmar": calmar_ratio(equity),
        "win_rate": float((active > 0).mean()) if len(active) else 0.0,
        "exposure": float((positions != 0).mean()),
        "turnover": float(trades.sum()),
    }


def format_metrics(metrics: dict) -> str:
    pct = {"total_return", "cagr", "volatility", "max_drawdown", "win_rate", "exposure"}
    lines = []
    for k, v in metrics.items():
        lines.append(f"{k:>14}: {v:8.2%}" if k in pct else f"{k:>14}: {v:8.3f}")
    return "\n".join(lines)
