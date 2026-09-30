import numpy as np
import pandas as pd
import pytest

from quant.backtest import Backtester
from quant.cli import main, parse_params
from quant.data import generate_gbm_prices
from quant.strategies import (STRATEGIES, BollingerMeanReversion, Momentum,
                              RSIReversion, make_strategy, rsi)


def _prices(values):
    idx = pd.bdate_range("2021-01-01", periods=len(values))
    return pd.DataFrame({"close": np.asarray(values, dtype=float)}, index=idx)


def test_momentum_trending_up_is_long():
    prices = _prices(np.linspace(100, 200, 50))
    pos = Momentum(lookback=10).generate_positions(prices)
    assert (pos.iloc[:10] == 0).all()
    assert (pos.iloc[10:] == 1).all()


def test_momentum_short():
    prices = _prices(np.linspace(200, 100, 50))
    pos = Momentum(lookback=10, allow_short=True).generate_positions(prices)
    assert (pos.iloc[10:] == -1).all()


def test_bollinger_enters_on_drop_and_exits_on_revert():
    vals = [100.0] * 20 + [100.5, 99.5] * 5 + [90.0, 95.0, 101.0, 101.0]
    pos = BollingerMeanReversion(window=10, num_std=2).generate_positions(_prices(vals))
    i_drop = 30
    assert pos.iloc[i_drop] == 1.0
    assert pos.iloc[-1] == 0.0
    assert pos.iloc[:10].eq(0).all()


def test_rsi_bounds_and_extremes():
    up = rsi(pd.Series(np.arange(1, 40, dtype=float)))
    assert up.dropna().eq(100).all()
    r = rsi(generate_gbm_prices(300)["close"]).dropna()
    assert r.between(0, 100).all()


def test_rsi_validation():
    with pytest.raises(ValueError):
        RSIReversion(oversold=60, exit_level=50)


def test_registry_and_all_strategies_backtest():
    prices = generate_gbm_prices(400, seed=9)
    for name in STRATEGIES:
        res = Backtester().run(prices, make_strategy(name))
        assert np.isfinite(res.equity).all()
        assert res.positions.between(-1, 1).all()
    with pytest.raises(ValueError):
        make_strategy("nope")


def test_parse_params():
    assert parse_params(["a=1", "b=2.5", "c=true", "d=x"]) == {"a": 1, "b": 2.5, "c": True, "d": "x"}
    with pytest.raises(SystemExit):
        parse_params(["bad"])


@pytest.mark.parametrize("name", sorted(STRATEGIES))
def test_cli_each_strategy(name, capsys):
    assert main(["--days", "300", "--strategy", name]) == 0
    assert "Buy & Hold" in capsys.readouterr().out
