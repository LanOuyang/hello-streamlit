"""Streamlit dashboard: ``streamlit run app.py``."""

import os
import sys

import pandas as pd
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from quant.backtest import Backtester  # noqa: E402
from quant.data import generate_gbm_prices, load_csv  # noqa: E402
from quant.metrics import drawdown, summarize  # noqa: E402
from quant.strategies import STRATEGIES, make_strategy  # noqa: E402

st.set_page_config(page_title="Quant Backtester", page_icon="📈", layout="wide")
st.title("📈 Quant Trading Backtester")

sb = st.sidebar
sb.header("Data")
upload = sb.file_uploader("OHLCV CSV (date, close, ...)", type="csv")
if upload is None:
    days = sb.slider("Synthetic days", 200, 3000, 1000, 100)
    seed = sb.number_input("Seed", value=42, step=1)
    mu = sb.slider("Annual drift", -0.3, 0.5, 0.08, 0.01)
    sigma = sb.slider("Annual volatility", 0.05, 0.8, 0.2, 0.01)
    prices = generate_gbm_prices(days, mu=mu, sigma=sigma, seed=int(seed))
else:
    prices = load_csv(upload)

sb.header("Strategy")
name = sb.selectbox("Strategy", list(STRATEGIES))
params = {"allow_short": sb.checkbox("Allow short")}
if name == "sma_crossover":
    params["fast"] = sb.slider("Fast SMA", 2, 100, 20)
    params["slow"] = sb.slider("Slow SMA", 5, 300, 50)
elif name == "momentum":
    params["lookback"] = sb.slider("Lookback", 5, 252, 126)
elif name == "bollinger":
    params["window"] = sb.slider("Window", 5, 100, 20)
    params["num_std"] = sb.slider("Std devs", 0.5, 4.0, 2.0, 0.1)
else:
    params["period"] = sb.slider("RSI period", 2, 50, 14)
    params["oversold"] = sb.slider("Oversold", 5, 45, 30)
    params["overbought"] = sb.slider("Overbought", 55, 95, 70)

sb.header("Execution & risk")
commission = sb.number_input("Commission (bps)", 0.0, 100.0, 1.0)
slippage = sb.number_input("Slippage (bps)", 0.0, 100.0, 2.0)
use_stop = sb.checkbox("Trailing stop-loss")
stop = sb.slider("Stop %", 1, 50, 10) / 100 if use_stop else None
use_vt = sb.checkbox("Volatility targeting")
vt = sb.slider("Target vol %", 1, 50, 15) / 100 if use_vt else None
lev = sb.slider("Max leverage", 0.5, 3.0, 1.0, 0.1) if use_vt else 1.0

try:
    strategy = make_strategy(name, **params)
    bt = Backtester(commission_bps=commission, slippage_bps=slippage,
                    stop_loss=stop, vol_target=vt, max_leverage=lev)
    result = bt.run(prices, strategy)
except ValueError as exc:
    st.error(str(exc))
    st.stop()

bench_ret = result.benchmark.pct_change().fillna(0.0)
bench = summarize(result.benchmark, bench_ret, pd.Series(1.0, index=bench_ret.index))
m = result.metrics
c = st.columns(5)
c[0].metric("Total return", f"{m['total_return']:.1%}", f"{m['total_return'] - bench['total_return']:+.1%} vs B&H")
c[1].metric("CAGR", f"{m['cagr']:.1%}")
c[2].metric("Sharpe", f"{m['sharpe']:.2f}", f"{m['sharpe'] - bench['sharpe']:+.2f} vs B&H")
c[3].metric("Max drawdown", f"{m['max_drawdown']:.1%}")
c[4].metric("Exposure", f"{m['exposure']:.0%}")

st.subheader("Equity curve")
st.line_chart(pd.DataFrame({"Strategy": result.equity, "Buy & Hold": result.benchmark}))
left, right = st.columns(2)
left.subheader("Drawdown")
left.area_chart(drawdown(result.equity).rename("drawdown"))
right.subheader("Position")
right.line_chart(result.positions)

st.subheader("Metrics")
st.dataframe(pd.DataFrame({"Strategy": m, "Buy & Hold": bench}).round(4))
st.download_button("Download results CSV", result.to_frame().to_csv(), "backtest.csv")
