# Quant Trading System

A small, modular, tested backtesting system for single-asset, daily-bar trading
strategies. It is self-contained and can be moved to its own repository
unchanged (`git subtree split --prefix quant_trading` preserves its history).

## Goal

Build a research tool that lets you go from idea → honest out-of-sample
estimate quickly:

* **Correct** – no look-ahead bias (signals formed at close *t* trade over bar *t+1*).
* **Realistic** – commissions, slippage, position sizing and stop-losses.
* **Comparable** – standard metrics (CAGR, Sharpe, Sortino, max drawdown,
  Calmar, win rate, exposure, turnover) against buy & hold.
* **Robust** – parameter search on a training window, evaluation on unseen data.
* **Usable** – CLI for scripting, Streamlit dashboard for exploration.

## Staged plan

| Stage | Scope | Status |
|------:|-------|--------|
| 1 | MVP: synthetic GBM / CSV data, SMA crossover, vectorised backtester, metrics, CLI | ✅ |
| 2 | Strategies: time-series momentum, Bollinger mean reversion, RSI; strategy registry | ✅ |
| 3 | Realism: commission + slippage, trailing stop-loss, volatility targeting / leverage cap | ✅ |
| 4 | Grid search with train/test split and out-of-sample report | ✅ |
| 5 | Streamlit dashboard | ✅ |
| next | Walk-forward optimisation, multi-asset portfolios, live data adapters, paper trading | ⏳ |

## Layout

```
quant/
  data.py        synthetic GBM prices, CSV loader
  strategies.py  Strategy base class, SMA/momentum/Bollinger/RSI, registry
  risk.py        stop-loss and volatility-target overlays
  backtest.py    Backtester (costs, lagged execution, overlays)
  metrics.py     performance statistics
  optimize.py    grid search with out-of-sample evaluation
  cli.py         command line interface
app.py           Streamlit dashboard
tests/           pytest suite
```

## Usage

```bash
pip install -r requirements.txt

# Stage 1: simplest run (synthetic data, SMA 20/50)
python -m quant.cli

# Other strategies / parameters / your own data
python -m quant.cli --strategy bollinger --param window=30 --param num_std=1.5
python -m quant.cli --csv prices.csv --strategy momentum --short

# Costs and risk management
python -m quant.cli --commission-bps 1 --slippage-bps 5 --stop-loss 0.1 --vol-target 0.15

# Parameter search (train 70% / test 30%)
python -m quant.cli --grid fast=5,10,20 --grid slow=50,100,200 --metric sharpe

# Dashboard
streamlit run app.py

# Tests
pip install -r requirements-dev.txt && python -m pytest -q
```

CSV files need a `date` column and a `close` column (case-insensitive;
`Adj Close` is used when present).

## Adding a strategy

Subclass `Strategy`, implement `generate_positions(prices) -> pd.Series` returning
target exposure in `[-1, 1]` using only data up to each bar, and register it in
`STRATEGIES`. The backtester handles execution lag, costs and risk overlays.

> For research and education only – not investment advice.
