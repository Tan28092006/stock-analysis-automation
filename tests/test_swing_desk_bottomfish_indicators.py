"""Bottom-fish indicator fields (T12 spec §3): floors, stabilisation, episode, beta, ticks, causality."""
import math

import numpy as np
import pandas as pd
import pytest

from tests.swing_desk_support import frame, uptrend


def cfg():
    from stock_agent.swing_desk.config import load_bf_config
    return load_bf_config()


def bars(warm, tail):
    """warm closes (flat OHLC +-0.5%) followed by explicit (o, h, l, c[, v]) tail bars."""
    o = list(warm)
    h = [x * 1.005 for x in warm]
    lo = [x * 0.995 for x in warm]
    c = list(warm)
    v = [2e6] * len(warm)
    for bar in tail:
        o.append(bar[0]), h.append(bar[1]), lo.append(bar[2]), c.append(bar[3])
        v.append(bar[4] if len(bar) > 4 else 2e6)
    return frame(c, opens=o, highs=h, lows=lo, volume=np.array(v))


def locked(prev, pct=0.93):
    from stock_agent.swing_desk.bf_indicators import floor_price
    p = floor_price(prev)
    return (p, p, p, p)


def compute(df, index_df=None, **kw):
    from stock_agent.swing_desk.bf_indicators import compute_bottomfish_fields
    return compute_bottomfish_fields(df, index_df, cfg=cfg(), **kw)


@pytest.mark.parametrize("ref,expected", [
    (10_740.0, 9_990.0),     # raw 9,988.2 -> tick 10 -> 9,990
    (10_750.0, 10_000.0),    # raw 9,997.5 -> tick 10 -> 10,000
    (53_710.0, 50_000.0),    # raw 49,950.3 -> tick 50 -> 50,000
    (53_700.0, 49_950.0),    # raw 49,941 -> tick 50 -> 49,950
    (53_800.0, 50_100.0),    # raw 50,034 -> tick 100 -> 50,100
    (24_750.0, 23_050.0),    # PNJ 01/10 -> 02/10 floor
    (21_650.0, 20_150.0),    # PNJ 05/10 -> 06/10 floor
])
def test_floor_price_rounds_inside_the_band_to_hose_tick(ref, expected):
    from stock_agent.swing_desk.bf_indicators import floor_price, tick_size
    got = floor_price(ref)
    assert got == expected
    assert got >= ref * 0.93 - 1e-6 and got % tick_size(got) == 0


def test_floor_and_locked_floor_detection():
    warm = [30000.0] * 80
    p1 = locked(30000.0)
    df = bars(warm, [p1, (p1[3] * 0.97, p1[3] * 0.99, p1[3] * 0.93, p1[3] * 0.932)])
    f = compute(df)
    assert f["floor_t"] and not f["locked_floor_t"]          # -6.8% close but traded
    assert f["floor_streak"] == 2 and f["locked_floor_streak"] == 0
    g = compute(df.iloc[:-1])
    assert g["floor_t"] and g["locked_floor_t"] and g["locked_floor_streak"] == 1
    assert f["floors_in_last_5"] == 2 and f["sessions_since_floor"] == 0


def test_stab_state_transitions():
    warm = list(uptrend(80, 30000.0, seed=2))
    p = warm[-1]
    a = locked(p)
    b = locked(a[3])
    c = (b[3] * 0.95, b[3] * 0.98, b[3] * 0.925, b[3] * 0.931)         # -6.9%: floor, traded
    d = (c[3] * 1.01, c[3] * 1.03, c[3] * 0.999, c[3] * 1.02)          # first non-floor, low above c
    e = (d[3] * 1.0, d[3] * 1.02, d[3] * 0.995, d[3] * 1.01)           # higher low, no new low
    df = bars(warm, [a, b, c, d, e])
    states = [compute(df.iloc[:len(df) - k])["stab_state"] for k in (3, 2, 1, 0)]
    assert states == ["LOCKED", "FALLING", "STAB1", "STAB2"]
    weak = bars(warm, [a, b, c, d, (d[3] * 0.99, d[3] * 1.0, c[2] * 0.99, d[3] * 0.995)])  # new episode low
    w = compute(weak)
    assert w["stab_state"] == "WEAK" and w["new_episode_low_2"] and not w["higher_low_t"]
    calm = compute(frame(uptrend(120, 20000.0, seed=4)))
    assert calm["stab_state"] == "NONE" and calm["sessions_since_floor"] is None


def test_panic_low_and_episode_reset():
    warm = [30000.0] * 80
    fall = [(30000 * k, 30000 * k * 1.01, 30000 * k * 0.98, 30000 * k) for k in (0.94, 0.92, 0.9, 0.88)]
    rec = [(30000 * k, 30000 * k * 1.01, 30000 * k * 0.99, 30000 * k) for k in (0.92, 0.96, 0.99)]
    df = bars(warm, fall)
    f = compute(df)
    assert f["episode_sessions"] == 4
    assert f["panic_low"] == pytest.approx(30000 * 0.88 * 0.98)
    assert f["episode_depth_pct"] == pytest.approx((0.88 - 1) * 100)
    assert f["episode_start_date"] == str(df["date"].iloc[-4])
    g = compute(bars(warm, fall + rec))
    assert g["episode_sessions"] == 0 and g["panic_low"] is None    # 0.99 x high_60 >= 0.95: reset


def test_beta_excludes_last_ten_sessions_and_idio_arithmetic():
    rng = np.random.default_rng(5)
    n = 120
    ir = rng.normal(0, 0.01, n)
    sr = 2.0 * ir
    sr[-10:] = rng.normal(-0.02, 0.03, 10)            # fall window: not part of the beta estimate
    idx = frame(1000 * np.cumprod(1 + ir), volume=5e8)
    stock = frame(20000 * np.cumprod(1 + sr))
    f = compute(stock, idx)
    assert f["beta_60"] == pytest.approx(2.0, abs=1e-9)
    assert f["idio_ret_10d"] == pytest.approx(f["ret_10d"] - 2.0 * f["idx_ret_10d"])
    assert f["idx_ret_5d"] == pytest.approx((idx["close"].iloc[-1] / idx["close"].iloc[-6] - 1) * 100)
    short = compute(stock.tail(45), idx.tail(45))
    assert short["beta_60"] is None and short["idio_ret_10d"] is None
    misaligned = compute(stock, idx.iloc[:-1])
    assert misaligned["beta_60"] is None and "align" in misaligned["index_status"]


def test_vol_ratio_prior_excludes_today_and_capitulation():
    warm = [30000.0] * 40
    df = bars(warm, [(30000, 30100, 28500, 28700, 8e6)])     # -4.3% on 4x volume
    f = compute(df)
    assert f["vol_ratio_20_prior"] == pytest.approx(4.0)
    assert f["capitulation_t"]
    g = compute(bars(warm, [(30000, 30100, 29000, 29200, 8e6)]))    # -2.7%: no
    assert not g["capitulation_t"]
    fl = locked(30000.0)
    h = compute(bars(warm, [(fl[0], 30000 * 0.98, fl[0], 30000 * 0.97, 9e6)]))   # opened at floor, then traded
    assert h["capitulation_t"]


def test_index_atr_ratio_and_flows():
    idx = frame(uptrend(300, 1200.0, seed=7), volume=5e8)
    f = compute(frame(uptrend(300, 20000.0, seed=8)), idx)
    assert math.isfinite(f["index_atr_ratio"]) and 0.3 < f["index_atr_ratio"] < 3
    df = frame(uptrend(60, 20000.0, seed=9))
    days = [str(d) for d in df["date"]]
    flows = pd.DataFrame({"symbol": "AAA", "date": days[-25:],
                          "f_net_val": [-1.0] * 24 + [2.0], "f_net_vol": [-100.0] * 24 + [500.0],
                          "td_net_val": 0.0})
    g = compute(df, flows=flows, symbol="AAA")
    assert g["f_sell_streak"] == 0 and g["f_net_turn_positive"] and g["flow_status"] == "OK"
    assert g["f_net_1d_pct_vol"] == pytest.approx(500.0 / df["volume"].iloc[-1] * 100)


def test_causality_future_bar_changes_nothing():
    idx = frame(uptrend(300, 1200.0, seed=7), volume=5e8)
    stock = frame(uptrend(300, 20000.0, seed=3))
    as_of = stock["date"].iloc[-2]
    a = compute(stock.iloc[:-1], idx.iloc[:-1])
    b = compute(stock, idx, as_of=as_of)
    for k in a:
        va, vb = a[k], b[k]
        if isinstance(va, float) and math.isnan(va):
            assert isinstance(vb, float) and math.isnan(vb), k
        else:
            assert va == vb, k
