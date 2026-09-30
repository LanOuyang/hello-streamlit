import numpy as np
import pandas as pd
import pytest

from quant import metrics
from quant.backtest import Backtester
from quant.cli import main
from quant.data import generate_gbm_prices, load_csv
from quant.strategies import SMACrossover, Strategy


class AlwaysLong(Strategy):
    def generate_positions(self, prices):
        return pd.Series(1.0, index=prices.index)


class Peek(Strategy):
    """Uses the same bar's return; would be perfect if not lagged by the backtester."""

    def generate_positions(self, prices):
        return np.sign(prices["close"].pct_change()).fillna(0.0)


def test_gbm_shape_and_ohlc_consistency():
    df = generate_gbm_prices(300, seed=1)
    assert len(df) == 300
    assert (df["high"] >= df[["open", "close"]].max(axis=1)).all()
    assert (df["low"] <= df[["open", "close"]].min(axis=1)).all()
    pd.testing.assert_frame_equal(df, generate_gbm_prices(300, seed=1))


def test_load_csv(tmp_path):
    p = tmp_path / "p.csv"
    p.write_text("Date,Close\n2020-01-02,11\n2020-01-01,10\n")
    df = load_csv(str(p))
    assert list(df["close"]) == [10, 11]


def test_always_long_matches_buy_and_hold():
    prices = generate_gbm_prices(200, seed=3)
    res = Backtester(1000).run(prices, AlwaysLong())
    # first bar has no position (signal lagged), then fully invested
    expected = 1000 * prices["close"] / prices["close"].iloc[0]
    np.testing.assert_allclose(res.equity.values, expected.values)


def test_no_lookahead():
    prices = generate_gbm_prices(500, seed=5)
    res = Backtester().run(prices, Peek())
    # Trading on bar t's close must only earn bar t+1's return.
    assert res.metrics["sharpe"] < 2


def test_sma_validation_and_warmup():
    with pytest.raises(ValueError):
        SMACrossover(50, 20)
    prices = generate_gbm_prices(100, seed=0)
    pos = SMACrossover(5, 30).generate_positions(prices)
    assert (pos.iloc[:29] == 0).all()
    assert set(pos.unique()) <= {0.0, 1.0}


def test_metrics_basic():
    eq = pd.Series([100, 120, 90, 110.0])
    assert metrics.total_return(eq) == pytest.approx(0.10)
    assert metrics.max_drawdown(eq) == pytest.approx(-0.25)
    assert metrics.sharpe_ratio(pd.Series([0.0, 0.0])) == 0.0


def test_cli_runs(capsys):
    assert main(["--days", "300", "--param", "fast=5", "--param", "slow=20"]) == 0
    assert "sharpe" in capsys.readouterr().out
