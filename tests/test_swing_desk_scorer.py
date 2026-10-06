from datetime import date

import numpy as np
import pandas as pd
import pytest

from tests.swing_desk_support import dates

T0 = date(2026, 9, 1)


def bars(rows):
    """rows: list of (open, high, low, close); dates are consecutive sessions ending 2026-09-21."""
    d = dates(len(rows))
    o, h, l, c = zip(*rows)
    return pd.DataFrame({"date": d, "open": o, "high": h, "low": l, "close": c, "volume": 1e6})


def rec(df, *, limit=100.0, stop=97.0, target=106.0, hold=8, kind="limit", p=None, idx=1):
    return {"id": "x", "source": "desk", "ticker": "AAA", "action": "BUY", "target_session": str(df.date.iloc[idx]),
            "entry": {"type": kind, "limit_price": limit}, "stop": stop, "target": target,
            "max_hold_sessions": hold, "p_win_stated": p}


FLAT = (100, 101, 99.5, 100)


def test_limit_not_reached_is_no_fill():
    from stock_agent.swing_desk.scorer import score_row
    df = bars([FLAT, (102, 103, 101, 102)] + [FLAT] * 10)
    assert score_row(rec(df), df)["status"] == "no_fill"


def test_fill_at_min_open_limit_and_ato():
    from stock_agent.swing_desk.scorer import score_row
    df = bars([FLAT, (99, 101, 98.5, 100)] + [FLAT] * 10)
    assert score_row(rec(df), df)["fill"] == 99
    assert score_row(rec(df, kind="ato"), df)["fill"] == 99


def test_stop_hit_while_locked_exits_at_day2_close():
    from stock_agent.swing_desk.scorer import score_row
    df = bars([FLAT, FLAT, (99, 99.5, 95, 96), (96.5, 98, 96, 97.5)] + [FLAT] * 10)
    r = score_row(rec(df), df)
    assert r["status"] == "resolved" and r["exit_reason"] == "stop_locked"
    assert r["exit_price"] == 97.5 and r["holding_sessions"] == 2


def test_gap_through_stop_fills_at_open():
    from stock_agent.swing_desk.scorer import score_row
    df = bars([FLAT, FLAT, FLAT, FLAT, (95, 96, 94, 95.5)] + [FLAT] * 8)
    r = score_row(rec(df), df)
    assert r["exit_reason"] == "stop_gap" and r["exit_price"] == 95 and r["holding_sessions"] == 3


def test_intraday_stop_fills_at_stop_and_beats_same_bar_target():
    from stock_agent.swing_desk.scorer import score_row
    df = bars([FLAT, FLAT, FLAT, FLAT, (100, 107, 96, 100)] + [FLAT] * 8)
    r = score_row(rec(df), df)
    assert r["exit_reason"] == "stop" and r["exit_price"] == 97


def test_limit_down_day_carries_to_next_session():
    from stock_agent.swing_desk.scorer import score_row
    df = bars([FLAT, FLAT, FLAT, FLAT, (99, 99, 93, 93), (92, 94, 91, 93)] + [FLAT] * 8)
    r = score_row(rec(df), df)
    assert r["exit_reason"] == "stop_gap" and r["exit_price"] == 92 and r["holding_sessions"] == 4
    assert r["limit_down_carry"] == 1


def test_target_hit_and_gap_above_target():
    from stock_agent.swing_desk.scorer import score_row
    df = bars([FLAT, FLAT, FLAT, FLAT, (101, 106.5, 100.5, 106)] + [FLAT] * 8)
    r = score_row(rec(df), df)
    assert r["exit_reason"] == "target" and r["exit_price"] == 106
    df = bars([FLAT, FLAT, FLAT, FLAT, (107, 108, 106.5, 107)] + [FLAT] * 8)
    assert score_row(rec(df), df)["exit_price"] == 107


def test_time_exit_costs_r_and_excursions():
    from stock_agent.swing_desk.scorer import score_row
    rows = [FLAT, FLAT] + [(100, 102, 98, 101)] * 8 + [FLAT] * 3
    df = bars(rows)
    r = score_row(rec(df, hold=8, p=0.6), df)
    assert r["exit_reason"] == "time" and r["holding_sessions"] == 8 and r["exit_price"] == 101
    assert r["net_pct"] == pytest.approx((101 * 0.9975 / (100 * 1.0015) - 1) * 100)
    assert r["net_pct_stress"] == pytest.approx(1.0 - 0.6)
    assert r["R"] == pytest.approx(r["net_pct"] / 3.0)
    assert r["mae_pct"] == pytest.approx(-2.0) and r["mfe_pct"] == pytest.approx(2.0)
    assert r["mae_R"] == pytest.approx(-2.0 / 3.0)
    assert r["win"] is True


def test_pending_rows_stay_pending_without_look_ahead():
    from stock_agent.swing_desk.scorer import score_row
    df = bars([FLAT, FLAT, FLAT, FLAT])
    assert score_row(rec(df), df)["status"] == "pending"
    future = rec(df)
    future["target_session"] = "2026-09-22"
    assert score_row(future, df)["status"] == "pending"


def test_summary_brier_activity_and_pairing():
    from stock_agent.swing_desk.scorer import summarize
    scored = [
        {"source": "desk", "status": "resolved", "net_pct": 2.0, "net_pct_stress": 1.8, "R": .6, "win": True,
         "p_win_stated": 0.8, "target_session": "2026-09-08", "ticker": "A"},
        {"source": "desk", "status": "resolved", "net_pct": -3.0, "net_pct_stress": -3.2, "R": -1, "win": False,
         "p_win_stated": 0.6, "target_session": "2026-09-09", "ticker": "B"},
        {"source": "desk", "status": "no_fill", "target_session": "2026-09-10", "ticker": "C"},
        {"source": "baseline", "status": "resolved", "net_pct": 1.0, "net_pct_stress": .8, "R": .3, "win": True,
         "target_session": "2026-09-08", "ticker": "A"},
        {"source": "baseline", "status": "pending", "target_session": "2026-10-08", "ticker": "D"},
    ]
    s = summarize(scored, activity_floor=4)
    d = s["arms"]["desk"]
    assert d["resolved"] == 2 and d["no_fill"] == 1 and d["no_fill_rate"] == pytest.approx(100 / 3)
    assert d["brier"] == pytest.approx(((0.8 - 1) ** 2 + (0.6 - 0) ** 2) / 2)
    assert d["brier_reference"] == pytest.approx(0.25)
    assert s["arms"]["baseline"]["pending"] == 1
    assert s["paired"]["desk_minus_baseline"]["sessions"] == 1
    assert s["paired"]["desk_minus_baseline"]["mean_diff_pct"] == pytest.approx(1.0)
    act = s["activity"]["desk"]["2026-09"]
    assert act["ideas"] == 3 and act["floor_met"] is False
    assert s["conclusion"].startswith("NOT PROVEN")
