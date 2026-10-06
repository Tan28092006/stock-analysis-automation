"""Bottom-fish evaluation arms: freezing (bf_baseline / bf_knife / bf_random, bf_desk) and BF scoring."""
from datetime import datetime
import random

import numpy as np
import pytest

from stock_agent.data.exchange_calendar import VN_TIMEZONE
from tests.swing_desk_bf_support import panic_desk
from tests.swing_desk_support import NOW_EOD, frame


def built(tmp_path, monkeypatch):
    from stock_agent.swing_desk import service
    paths = panic_desk(tmp_path, monkeypatch)
    service.write_state(paths, service.compute_state(paths, NOW_EOD, s2_override=[], with_flows_data=False))
    return paths


def test_freeze_code_arms_same_session_with_episode_and_mode(tmp_path, monkeypatch):
    from stock_agent.swing_desk import bf_arms, reclog
    paths = built(tmp_path, monkeypatch)
    out = bf_arms.freeze_bf_arms(paths, now=NOW_EOD)
    assert set(out) == {"bf_baseline", "bf_knife", "bf_random"}
    assert [r["ticker"] for r in out["bf_baseline"]] == ["AAA"]
    rows = reclog.read_rows(paths.rec_log)
    assert {r["source"] for r in rows} == {"bf_baseline", "bf_knife", "bf_random"}
    assert all(r["mode"] == "bottomfish" and r["episode_id"].startswith(r["ticker"] + "-") for r in rows)
    assert all(r["target_session"] == "2026-09-22" and r["protocol"] == "bottomfish_prospective_v1" for r in rows)
    knife = out["bf_knife"][0]
    assert knife["entry"]["type"] == "ato" and knife["entry"]["session"] <= "2026-09-21"
    base = out["bf_baseline"][0]
    assert base["entry"]["valid_session"] == "2026-09-22" and base["invalidation"] == base["panic_low"]
    assert base["price_only_classification"] is False and base["bf_news_sha"]
    assert reclog.verify_chain(paths.rec_log)["ok"]
    with pytest.raises(ValueError, match="already frozen"):
        bf_arms.freeze_bf_arms(paths, now=NOW_EOD)


def test_freeze_refuses_after_0900_and_after_an_analyst_row(tmp_path, monkeypatch):
    from stock_agent.swing_desk import bf_arms
    paths = built(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="09:00"):
        bf_arms.freeze_bf_arms(paths, now=datetime(2026, 9, 22, 9, 30, tzinfo=VN_TIMEZONE))
    idea = {"ticker": "AAA", "action": "BUY", "mode": "bottomfish", "bf_type": "a", "profile": "P1",
            "episode_id": "AAA-20260914", "catastrophe_stop_pct": 12.0, "time_stop_sessions": 10,
            "data_as_of": "2026-09-21", "entry": {"type": "limit", "limit_price": 20000.0},
            "p_win_stated": 0.5, "sentiment": "neutral, cited S-01"}
    with pytest.raises(ValueError, match="type a"):
        bf_arms.freeze_bf_desk(paths, [{**idea, "bf_type": "c"}], now=NOW_EOD, model_id="m", n_rollouts=4)
    with pytest.raises(ValueError, match="at most 1"):
        bf_arms.freeze_bf_desk(paths, [idea, idea], now=NOW_EOD, model_id="m", n_rollouts=4)
    bf_arms.freeze_bf_desk(paths, [idea], now=NOW_EOD, model_id="m", n_rollouts=4)
    with pytest.raises(ValueError, match="analyst"):
        bf_arms.freeze_bf_arms(paths, now=NOW_EOD)


def test_freeze_needs_registered_protocol(tmp_path, monkeypatch):
    from stock_agent.swing_desk import bf_arms
    paths = built(tmp_path, monkeypatch)
    paths.bf_protocol.unlink()
    with pytest.raises(ValueError, match="registered"):
        bf_arms.freeze_bf_arms(paths, now=NOW_EOD)


def test_bf_random_is_seeded_by_session_date(tmp_path):
    from stock_agent.swing_desk import bf_arms
    from stock_agent.swing_desk.config import DeskPaths, load_bf_config
    paths = DeskPaths(repo=tmp_path, data_root=tmp_path, desk=tmp_path / "swing_desk")
    bf = load_bf_config()
    plan = {"effective_loss_pct": 19.6, "earliest_sell": "x", "qty": 1, "risk_vnd": 1.0}
    rows = [{"ticker": f"T{i:02d}", "fall_type_auto": "IDIO", "episode_id": f"T{i:02d}-20260901", "plan_p1": plan,
             "structural_verdict": "BLOCK", "structural_failed": [],
             "fields": {"close": 100.0, "ret_10d": -12.0, "panic_low": 90.0, "stab_state": "WEAK"}} for i in range(12)]
    scan = {"rows": rows, "as_of": "2026-09-21", "snapshot_manifest_sha256": "s", "trade_session": "2026-09-22"}
    a = [r["ticker"] for r in bf_arms.candidate_bf_rows(paths, scan, "bf_random", bf)]
    b = [r["ticker"] for r in bf_arms.candidate_bf_rows(paths, scan, "bf_random", bf)]
    assert a == b and len(a) == 3
    expected = sorted(r["ticker"] for r in random.Random(20260922).sample(sorted(rows, key=lambda r: r["ticker"]), 3))
    assert a == expected
    other = [r["ticker"] for r in bf_arms.candidate_bf_rows(paths, {**scan, "trade_session": "2026-09-23"}, "bf_random", bf)]
    assert other == sorted(r["ticker"] for r in random.Random(20260923).sample(sorted(rows, key=lambda r: r["ticker"]), 3))


def test_score_log_scores_bf_rows_separately(tmp_path, monkeypatch):
    from stock_agent.swing_desk import baseline, bf_arms
    paths = built(tmp_path, monkeypatch)
    bf_arms.freeze_bf_arms(paths, now=NOW_EOD)
    res = baseline.score_log(paths, now=NOW_EOD)
    assert res["summary"]["arms"] == {} and res["bf_rows"]
    knife = [r for r in res["bf_rows"] if r["source"] == "bf_knife"]
    assert knife and knife[0]["status"] in ("pending", "resolved")
    assert all(r["status"] == "pending" for r in res["bf_rows"] if r["source"] != "bf_knife")
    assert res["bf_summary"]["conclusion"].startswith("NOT PROVEN")


# ------------------------------------------------------------------ scorer

def bars(closes, opens=None, lows=None, highs=None):
    closes = np.asarray(closes, float)
    opens = closes if opens is None else np.asarray(opens, float)
    lows = np.minimum(opens, closes) * 0.99 if lows is None else np.asarray(lows, float)
    highs = np.maximum(opens, closes) * 1.01 if highs is None else np.asarray(highs, float)
    return frame(closes, opens=opens, highs=highs, lows=lows)


def row(df, i, **over):
    day = str(df["date"].iloc[i])
    base = {"id": "x", "source": "bf_baseline", "ticker": "AAA", "action": "BUY", "mode": "bottomfish",
            "target_session": day, "episode_id": "AAA-1", "bf_type": "a", "profile": "P1",
            "entry": {"type": "limit", "limit_price": 100.5, "valid_session": day}, "invalidation": 90.0,
            "catastrophe_stop_pct": 12.0, "time_stop_sessions": 10, "exit_signal": None}
    base.update(over)
    return base


def score(df, r):
    from stock_agent.swing_desk.scorer import score_bf_row
    return score_bf_row(r, df)


def test_fill_no_fill_and_pending():
    df = bars([100.0] * 30)
    assert score(df, row(df, 20, entry={"type": "limit", "limit_price": 98.0}))["status"] == "no_fill"
    r = score(df, row(df, 25))
    assert r["status"] == "pending" and r["fill"] == 100.0          # time stop beyond the data: no look-ahead
    future = row(df, 0, target_session="2099-01-02", entry={"type": "limit", "limit_price": 100.5})
    assert score(df, future)["status"] == "pending"


def test_lock_two_sessions_then_invalidation_exits_next_open_with_costs():
    c = [100.0] * 20 + [100.0, 85.0, 86.0, 84.0, 88.0, 89.0] + [90.0] * 5
    o = list(c)
    o[23] = 85.5
    df = bars(c, opens=o)
    r = score(df, row(df, 20))
    assert r["status"] == "resolved" and r["exit_reason"] == "invalidation"
    assert r["signal_date"] == str(df["date"].iloc[22])               # first allowed close is e+2
    assert r["exit_date"] == str(df["date"].iloc[23]) and r["exit_price"] == 85.5
    assert r["net_pct"] == pytest.approx((85.5 * 0.9975 / (100 * 1.0015) - 1) * 100)
    assert r["net_pct_stress"] == pytest.approx((85.5 / 100 - 1) * 100 - 0.6)


def test_floor_close_defers_exit_and_counts_floor_lock_days():
    c = [100.0] * 20 + [100.0, 99.0, 87.0, 80.0, 74.0, 76.0] + [76.0] * 5
    df = bars(c)
    r = score(df, row(df, 20, invalidation=None))
    assert r["exit_reason"] == "catastrophe" and r["signal_date"] == str(df["date"].iloc[22])
    assert r["floor_lock_days"] == 2 and r["exit_date"] == str(df["date"].iloc[25])


def test_time_stop_and_exit_signal():
    df = bars([100.0] * 40)
    r = score(df, row(df, 20, invalidation=None))
    assert r["exit_reason"] == "time" and r["exit_date"] == str(df["date"].iloc[31])
    c = [100.0] * 20 + [100.0, 97.0, 96.0, 99.0] + [99.0] * 10
    df = bars(c)
    r = score(df, row(df, 20, invalidation=None, exit_signal="close_above_sma5_or_ema21"))
    assert r["exit_reason"] == "exit_signal" and r["signal_date"] == str(df["date"].iloc[23])
