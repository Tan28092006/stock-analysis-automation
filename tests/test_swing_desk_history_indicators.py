"""Long-window fields come from the long series; short-window fields are unchanged (pure functions)."""
import json

import numpy as np
import pytest

from tests.swing_desk_support import frame, market_frames

ROOT = __import__("pathlib").Path(__file__).resolve().parents[1]


def _same(a, b):
    if isinstance(a, float) and isinstance(b, float) and np.isnan(a) and np.isnan(b):
        return True
    return a == b


def test_swing_long_window_fields_come_from_the_long_series():
    from stock_agent.swing_desk.indicators import LONG_WINDOW_FIELDS, compute_ticker
    long = market_frames(n=600)["BBB"]
    short = long.tail(150).reset_index(drop=True)
    got = compute_ticker(short, long_df=long)
    full, only_short = compute_ticker(long), compute_ticker(short)
    assert got["sma200"] == pytest.approx(long["close"].tail(200).mean(), rel=1e-12)
    assert got["high_250"] == long["high"].tail(250).max() and got["n_locked"] == 250
    assert only_short["n_locked"] == 149 and np.isnan(only_short["sma200"]) and np.isnan(only_short["high_250"])
    for key in LONG_WINDOW_FIELDS:
        assert _same(got[key], full[key]), key
    for key in set(got) - set(LONG_WINDOW_FIELDS):
        assert _same(got[key], only_short[key]), key


def test_swing_results_identical_where_long_and_short_series_agree():
    from stock_agent.swing_desk.indicators import compute_ticker
    for sym, long in market_frames(n=600).items():
        short = long.tail(300).reset_index(drop=True)      # every window fits inside 300 sessions
        a, b = compute_ticker(short, long_df=long), compute_ticker(short)
        assert set(a) == set(b)
        assert all(_same(a[k], b[k]) for k in a), sym


def test_long_series_must_end_on_the_short_series_session():
    from stock_agent.swing_desk.indicators import compute_ticker
    long = market_frames(n=600)["AAA"]
    with pytest.raises(ValueError, match="long series"):
        compute_ticker(long.tail(150).reset_index(drop=True), long_df=long.iloc[:-1])


def _bf_cfg():
    return json.loads((ROOT / "configs/swing_desk/bottomfish_v1.json").read_text(encoding="utf-8"))


def _long_decline(n=600):
    """Peak at session 100, then a steady -0.3%/session slide that never recovers to 95% of the 60-day high."""
    c = np.concatenate([np.linspace(20000, 40000, 101), 40000 * np.cumprod(np.full(n - 101, 0.997))])
    return frame(c, volume=3e6)


def test_bottomfish_episode_uses_the_long_series():
    from stock_agent.swing_desk.bf_indicators import BF_LONG_WINDOW_FIELDS, compute_bottomfish_fields
    cfg = _bf_cfg()
    long = _long_decline()
    idx = market_frames(n=600)["VNINDEX"]
    short, short_idx = long.tail(150).reset_index(drop=True), idx.tail(150).reset_index(drop=True)
    got = compute_bottomfish_fields(short, short_idx, cfg=cfg, long_df=long)
    full = compute_bottomfish_fields(long, idx, cfg=cfg)
    only_short = compute_bottomfish_fields(short, short_idx, cfg=cfg)
    assert got["episode_sessions"] > 400 and got["episode_start_date"] < str(short["date"].iloc[0])
    assert only_short["episode_sessions"] < 150          # truncated by the 150-session window
    for key in BF_LONG_WINDOW_FIELDS:
        assert _same(got[key], full[key]), key
    for key in set(got) - set(BF_LONG_WINDOW_FIELDS):
        assert _same(got[key], only_short[key]), key


def test_bottomfish_identical_where_series_agree_and_episode_fits():
    from stock_agent.swing_desk.bf_indicators import compute_bottomfish_fields
    cfg = _bf_cfg()
    frames = market_frames(n=600)
    idx = frames["VNINDEX"].tail(300).reset_index(drop=True)
    for sym in ("AAA", "BBB", "VCB"):
        short = frames[sym].tail(300).reset_index(drop=True)
        a = compute_bottomfish_fields(short, idx, cfg=cfg, long_df=frames[sym])
        b = compute_bottomfish_fields(short, idx, cfg=cfg)
        assert all(_same(a[k], b[k]) for k in a), sym
