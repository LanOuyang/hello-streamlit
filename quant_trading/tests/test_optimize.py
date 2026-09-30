import pytest

from quant.cli import main, parse_grid
from quant.data import generate_gbm_prices
from quant.optimize import grid_search, split_train_test


def test_split():
    prices = generate_gbm_prices(100)
    train, test = split_train_test(prices, 0.7)
    assert len(train) == 70 and len(test) == 30
    assert train.index[-1] < test.index[0]
    with pytest.raises(ValueError):
        split_train_test(prices, 1.0)


def test_grid_search_picks_best_and_skips_invalid():
    prices = generate_gbm_prices(600, seed=4)
    res = grid_search(prices, "sma_crossover", {"fast": [5, 10, 60], "slow": [30, 50]})
    # fast=60 is invalid for both slow values
    assert len(res.table) == 4
    assert res.table["sharpe"].is_monotonic_decreasing
    assert res.best_params == {k: res.table.loc[0, k] for k in ("fast", "slow")}
    assert res.train_metrics["sharpe"] == pytest.approx(res.table.loc[0, "sharpe"])
    assert set(res.test_metrics) >= {"sharpe", "cagr", "max_drawdown"}


def test_grid_search_no_valid_combos():
    with pytest.raises(ValueError):
        grid_search(generate_gbm_prices(200), "sma_crossover", {"fast": [50], "slow": [10]})


def test_parse_grid():
    assert parse_grid(["fast=5,10", "num_std=1.5,2"]) == {"fast": [5, 10], "num_std": [1.5, 2]}
    with pytest.raises(SystemExit):
        parse_grid(["fast="])


def test_cli_grid(capsys):
    assert main(["--days", "500", "--strategy", "momentum", "--grid", "lookback=20,60"]) == 0
    out = capsys.readouterr().out
    assert "Out-of-sample" in out and "Best params" in out
