import numpy as np
import pandas as pd
import pytest

from quant.backtest import Backtester
from quant.cli import main
from quant.data import generate_gbm_prices
from quant.risk import apply_stop_loss, volatility_target
from quant.strategies import SMACrossover, Strategy


class Toggle(Strategy):
    def generate_positions(self, prices):
        return pd.Series(np.arange(len(prices)) % 2, index=prices.index, dtype=float)


def _series(values):
    return pd.Series(values, index=pd.bdate_range("2021-01-01", periods=len(values)), dtype=float)


def test_costs_charged_per_unit_turnover():
    prices = generate_gbm_prices(100, seed=2)
    free = Backtester().run(prices, Toggle())
    paid = Backtester(commission_bps=5, slippage_bps=5).run(prices, Toggle())
    turnover = free.positions.diff().abs().fillna(free.positions.abs()).sum()
    assert paid.metrics["total_costs"] == pytest.approx(turnover * 0.001)
    assert paid.equity.iloc[-1] < free.equity.iloc[-1]


def test_stop_loss_exits_and_waits_for_new_signal():
    close = _series([100, 110, 105, 98, 120, 130, 130, 140])
    sig = _series([1, 1, 1, 1, 1, 0, 1, 1])
    out = apply_stop_loss(sig, close, 0.10)
    # peak 110 -> 98 is -10.9%: stopped at bar 3, stays flat through bar 4,
    # re-enters when the signal resets (bar 6).
    assert list(out) == [1, 1, 1, 0, 0, 0, 1, 1]


def test_stop_loss_short_side():
    close = _series([100, 90, 100])
    out = apply_stop_loss(_series([-1, -1, -1]), close, 0.10)
    assert list(out) == [-1, -1, 0]


def test_stop_loss_validation():
    with pytest.raises(ValueError):
        apply_stop_loss(_series([1]), _series([1]), 1.5)


def test_vol_target_scales_and_caps():
    prices = generate_gbm_prices(500, sigma=0.4, seed=1)
    pos = pd.Series(1.0, index=prices.index)
    scaled = volatility_target(pos, prices["close"], 0.10, window=20, max_leverage=1.0)
    assert (scaled.iloc[:20] == 0).all()
    assert scaled.between(0, 1).all()
    assert scaled.iloc[20:].mean() < 0.5  # 40% vol asset scaled toward 10%
    res = Backtester(vol_target=0.10).run(prices, SMACrossover(10, 30))
    assert res.metrics["volatility"] < 0.2


def test_cli_with_risk_options(capsys):
    assert main(["--days", "300", "--stop-loss", "0.1", "--vol-target", "0.15"]) == 0
    assert "total_costs" in capsys.readouterr().out
