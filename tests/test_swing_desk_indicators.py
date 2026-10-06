import math

import numpy as np
import pandas as pd
import pytest

from stock_agent.features import indicators as base
from stock_agent.features.swing_scan import _rsi
from tests.swing_desk_support import frame, uptrend, with_dip


def test_rsi2_matches_swing_scan_exactly():
    from stock_agent.swing_desk.indicators import compute_ticker
    df = frame(with_dip(uptrend(260)))
    ind = compute_ticker(df)
    assert ind["rsi2"] == pytest.approx(float(_rsi(df.close, 2).iloc[-1]), abs=1e-12)


def test_reused_indicators_equal_feature_library():
    from stock_agent.swing_desk.indicators import compute_ticker
    df = frame(uptrend(260))
    ind = compute_ticker(df)
    assert ind["sma50"] == pytest.approx(float(base.sma(df.close, 50).iloc[-1]))
    assert ind["sma200"] == pytest.approx(float(base.sma(df.close, 200).iloc[-1]))
    assert ind["ema50"] == pytest.approx(float(base.ema(df.close, 50).iloc[-1]))
    assert ind["atr14"] == pytest.approx(float(base.atr(df, 14).iloc[-1]))
    assert ind["rsi14"] == pytest.approx(float(base.rsi(df.close, 14).iloc[-1]))
    assert ind["atr_pct"] == pytest.approx(ind["atr14"] / ind["close"] * 100)
    assert ind["adx14"] == pytest.approx(float(base.adx(df, 14)["adx14"].iloc[-1]))
    assert ind["ret_1d"] == pytest.approx((df.close.iloc[-1] / df.close.iloc[-2] - 1) * 100)
    assert ind["avg_value_20_bn"] == pytest.approx(float((df.close * df.volume).tail(20).mean() / 1e9))


def test_indicators_are_causal_when_a_future_bar_is_appended():
    from stock_agent.swing_desk.indicators import compute_ticker
    df = frame(uptrend(261))
    as_of = df.date.iloc[-2]
    full = compute_ticker(df, as_of=as_of)
    trimmed = compute_ticker(df.iloc[:-1])
    for key, value in trimmed.items():
        if isinstance(value, float) and math.isnan(value):
            assert math.isnan(full[key])
        else:
            assert full[key] == value, key


def test_limit_proxy_and_up_streak():
    from stock_agent.swing_desk.indicators import compute_ticker
    closes = uptrend(260)
    closes[-1] = closes[-2] * 1.068
    ind = compute_ticker(frame(closes))
    assert ind["limit_up_t"] is True and ind["limit_down_t"] is False
    closes[-3:] = [100, 101, 102]
    closes[-4] = 103
    assert compute_ticker(frame(closes))["up_streak"] == 2


def test_locked_excursion_uses_entry_session_and_next_only():
    from stock_agent.swing_desk.excursion import excursion_stats
    n = 12
    opens = np.full(n, 100.0)
    closes = np.full(n, 100.0)
    highs = np.full(n, 101.0)
    lows = np.full(n, 99.0)
    lows[3] = 95.0      # e=2 (via e+1) and e=3 (via e) see -5%
    lows[6] = 90.0      # deeper low two sessions after e=4 -> outside e=4's locked window
    df = frame(closes, opens=opens, highs=highs, lows=lows)
    stats = excursion_stats(df, lookback=250)
    locked = stats["_locked_mae"]
    # entries e = 0..n-2 (need e+1)
    assert len(locked) == n - 1
    assert locked[2] == pytest.approx(-5.0) and locked[3] == pytest.approx(-5.0)
    assert locked[4] == pytest.approx(-1.0)
    assert locked[5] == pytest.approx(-10.0) and locked[6] == pytest.approx(-10.0)
    assert stats["locked_mae_p50"] == pytest.approx(-1.0)
    assert stats["locked_mae_p90"] <= stats["locked_mae_p80"] <= stats["locked_mae_p50"]
    # day2 return given breach: entries 2,3,5,6 breached -3%; C_{e+2}/O_e - 1 = 0
    assert stats["day2_ret_given_breach"] == pytest.approx(0.0)
    assert stats["n_breach"] == 4


def test_mfe_mae_five_session_window():
    from stock_agent.swing_desk.excursion import excursion_stats
    n = 10
    opens = np.full(n, 100.0)
    highs = np.full(n, 101.0)
    highs[4] = 110.0
    df = frame(np.full(n, 100.0), opens=opens, highs=highs, lows=np.full(n, 98.0))
    stats = excursion_stats(df)
    assert stats["n_window5"] == n - 4
    # entries 0..4 see the 110 high within e..e+4; entries 5 sees no 110
    assert stats["mfe5_p50"] == pytest.approx(10.0)
    assert stats["mae5_p50"] == pytest.approx(-2.0)
