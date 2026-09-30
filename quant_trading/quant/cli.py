"""Command line entry point: ``python -m quant.cli``."""

from __future__ import annotations

import argparse

from .backtest import Backtester
from .data import generate_gbm_prices, load_csv
from .metrics import format_metrics, summarize
from .strategies import SMACrossover


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Backtest a trading strategy.")
    p.add_argument("--csv", help="OHLCV CSV file (default: synthetic GBM data)")
    p.add_argument("--days", type=int, default=1000, help="synthetic data length")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--fast", type=int, default=20)
    p.add_argument("--slow", type=int, default=50)
    p.add_argument("--short", action="store_true", help="allow short positions")
    p.add_argument("--capital", type=float, default=100_000.0)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    prices = load_csv(args.csv) if args.csv else generate_gbm_prices(args.days, seed=args.seed)
    strategy = SMACrossover(args.fast, args.slow, allow_short=args.short)
    result = Backtester(args.capital).run(prices, strategy)

    bench_ret = result.benchmark.pct_change().fillna(0.0)
    bench = summarize(result.benchmark, bench_ret, bench_ret * 0 + 1)
    print(f"Strategy: {strategy}")
    print(f"Period: {prices.index[0].date()} -> {prices.index[-1].date()} ({len(prices)} bars)\n")
    print("== Strategy ==")
    print(format_metrics(result.metrics))
    print("\n== Buy & Hold ==")
    print(format_metrics(bench))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
