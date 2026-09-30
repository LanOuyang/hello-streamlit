"""Command line entry point: ``python -m quant.cli``."""

from __future__ import annotations

import argparse

from .backtest import Backtester
from .data import generate_gbm_prices, load_csv
from .metrics import format_metrics, summarize
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


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    prices = load_csv(args.csv) if args.csv else generate_gbm_prices(args.days, seed=args.seed)
    strategy = make_strategy(args.strategy, allow_short=args.short, **parse_params(args.param))
    backtester = Backtester(
        initial_capital=args.capital,
        commission_bps=args.commission_bps,
        slippage_bps=args.slippage_bps,
        stop_loss=args.stop_loss,
        vol_target=args.vol_target,
        max_leverage=args.max_leverage,
    )
    result = backtester.run(prices, strategy)

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
