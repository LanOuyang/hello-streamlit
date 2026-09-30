"""Parameter search with out-of-sample validation."""

from __future__ import annotations

import itertools
from dataclasses import dataclass

import pandas as pd

from .backtest import Backtester
from .strategies import make_strategy


@dataclass
class OptimizationResult:
    best_params: dict
    table: pd.DataFrame          # in-sample results for every combination
    train_metrics: dict
    test_metrics: dict


def split_train_test(prices: pd.DataFrame, train_frac: float = 0.7):
    if not 0 < train_frac < 1:
        raise ValueError("train_frac must be in (0, 1)")
    cut = int(len(prices) * train_frac)
    if cut < 2 or len(prices) - cut < 2:
        raise ValueError("not enough data to split")
    return prices.iloc[:cut], prices.iloc[cut:]


def grid_search(prices: pd.DataFrame, strategy_name: str, grid: dict[str, list],
                backtester: Backtester | None = None, metric: str = "sharpe",
                train_frac: float = 0.7, fixed: dict | None = None) -> OptimizationResult:
    """Evaluate every parameter combination on the training window, pick the best by
    ``metric`` and report its performance on the untouched test window.

    The test window is evaluated with the preceding training data included as
    warm-up so indicators are fully formed, but only test-period returns count.
    """
    backtester = backtester or Backtester()
    fixed = fixed or {}
    train, test = split_train_test(prices, train_frac)
    keys = list(grid)
    rows = []
    for values in itertools.product(*(grid[k] for k in keys)):
        params = dict(zip(keys, values))
        try:
            strategy = make_strategy(strategy_name, **fixed, **params)
        except ValueError:
            continue  # invalid combination, e.g. fast >= slow
        rows.append({**params, **backtester.run(train, strategy).metrics})
    if not rows:
        raise ValueError("no valid parameter combinations")
    table = pd.DataFrame(rows).sort_values(metric, ascending=False).reset_index(drop=True)
    best = {k: _py(table.loc[0, k]) for k in keys}

    strategy = make_strategy(strategy_name, **fixed, **best)
    train_metrics = backtester.run(train, strategy).metrics
    test_metrics = _evaluate_window(prices, test.index[0], strategy, backtester)
    return OptimizationResult(best, table, train_metrics, test_metrics)


def _evaluate_window(prices, start, strategy, backtester) -> dict:
    full = backtester.run(prices, strategy)
    # restrict to the test window; the first test bar's return is already
    # driven by a signal formed on the last training bar, which is fine.
    mask = prices.index >= start
    from .metrics import summarize
    returns = full.returns[mask]
    equity = backtester.initial_capital * (1 + returns).cumprod()
    equity = pd.concat([pd.Series([backtester.initial_capital],
                                  index=[prices.index[~mask][-1]]), equity])
    returns = pd.concat([pd.Series([0.0], index=equity.index[:1]), returns])
    positions = pd.concat([pd.Series([0.0], index=equity.index[:1]), full.positions[mask]])
    metrics = summarize(equity, returns, positions)
    metrics["total_costs"] = float(full.costs[mask].sum())
    return metrics


def _py(value):
    return value.item() if hasattr(value, "item") else value
