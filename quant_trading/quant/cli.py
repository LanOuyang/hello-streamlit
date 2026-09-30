"""Command line entry point: ``python -m quant.cli``."""

from __future__ import annotations

import argparse

import pandas as pd

from .backtest import Backtester
from .data import generate_gbm_prices, load_csv
from .metrics import format_metrics, summarize
from .optimize import METRIC_NAMES, grid_search
from .strategies import STRATEGIES, make_strategy


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Backtest a trading strategy.")
    p.add_argument("--csv", help="OHLCV CSV file (default: synthetic GBM data)")
    p.add_argument("--days", type=int, default=1000, help="synthetic data length")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--strategy", default="sma_crossover", choices=sorted(STRATEGIES))
    p.add_argument("--param", action="append", default=[], metavar="KEY=VALUE",
                   help="strategy parameter, repeatable (e.g. --param fast=10)")
    p.add_argument("--short", action="store_true", help="allow short positions")
    p.add_argument("--capital", type=float, default=100_000.0)
    p.add_argument("--commission-bps", type=float, default=1.0)
    p.add_argument("--slippage-bps", type=float, default=2.0)
    p.add_argument("--stop-loss", type=float, help="trailing stop, e.g. 0.1 for 10%%")
    p.add_argument("--vol-target", type=float, help="annualised vol target, e.g. 0.15")
    p.add_argument("--max-leverage", type=float, default=1.0)
    p.add_argument("--grid", action="append", default=[], metavar="KEY=V1,V2,...",
                   help="optimise a parameter over values (repeatable); "
                        "runs a train/test grid search instead of a single backtest")
    p.add_argument("--train-frac", type=float, default=0.7)
    p.add_argument("--metric", default="sharpe", choices=sorted(METRIC_NAMES),
                   help="metric to maximise in grid search")
    return p


def _coerce(value: str):
    for cast in (int, float):
        try:
            return cast(value)
        except ValueError:
            pass
    if value.lower() in ("true", "false"):
        return value.lower() == "true"
    return value


def parse_params(items: list[str]) -> dict:
    params = {}
    for item in items:
        key, sep, value = item.partition("=")
        if not sep or not key:
            raise SystemExit(f"invalid --param {item!r}, expected KEY=VALUE")
        params[key.strip()] = _coerce(value.strip())
    return params


def parse_grid(items: list[str]) -> dict:
    grid = {}
    for item in items:
        key, sep, values = item.partition("=")
        if not sep or not key or not values:
            raise SystemExit(f"invalid --grid {item!r}, expected KEY=V1,V2,...")
        grid[key.strip()] = [_coerce(v.strip()) for v in values.split(",") if v.strip()]
    return grid


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    prices = load_csv(args.csv) if args.csv else generate_gbm_prices(args.days, seed=args.seed)
    params = {"allow_short": args.short, **parse_params(args.param)}
    backtester = Backtester(
        initial_capital=args.capital,
        commission_bps=args.commission_bps,
        slippage_bps=args.slippage_bps,
        stop_loss=args.stop_loss,
        vol_target=args.vol_target,
        max_leverage=args.max_leverage,
    )
    if args.grid:
        return _run_grid(args, prices, backtester, params)
    strategy = make_strategy(args.strategy, **params)
    result = backtester.run(prices, strategy)

    bench_ret = result.benchmark.pct_change().fillna(0.0)
    bench = summarize(result.benchmark, bench_ret, pd.Series(1.0, index=bench_ret.index))
    print(f"Strategy: {strategy}")
    print(f"Period: {prices.index[0].date()} -> {prices.index[-1].date()} ({len(prices)} bars)\n")
    print("== Strategy ==")
    print(format_metrics(result.metrics))
    print("\n== Buy & Hold ==")
    print(format_metrics(bench))
    return 0


def _run_grid(args, prices, backtester, params) -> int:
    grid = parse_grid(args.grid)
    fixed = {k: v for k, v in params.items() if k not in grid}
    res = grid_search(prices, args.strategy, grid, backtester, metric=args.metric,
                      train_frac=args.train_frac, fixed=fixed)
    print(f"Grid search for {args.strategy} ({len(res.table)} combinations, "
          f"maximising {args.metric})\n")
    cols = list(grid) + ["sharpe", "cagr", "max_drawdown"]
    print(res.table[cols].head(10).to_string(index=False))
    print(f"\nBest params: {res.best_params}\n")
    print("== In-sample ==")
    print(format_metrics(res.train_metrics))
    print("\n== Out-of-sample ==")
    print(format_metrics(res.test_metrics))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
