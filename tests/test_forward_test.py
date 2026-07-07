"""Unit tests for the forward-test ledger scorer — the trade-replay logic that will
score the user's multi-month paper-trade. Covers stop/target/time precedence, the T+2
settlement lock, next-open fill, pending (not-enough-future-bars), and logger idempotency.
"""
import json
from pathlib import Path

import pandas as pd
import pytest

from stock_agent.pipeline import forward_test as ft


def _frame(bars, start="2026-01-05"):
    """bars: list of (open, high, low, close). Dates are consecutive weekdays."""
    dates = pd.bdate_range(start, periods=len(bars)).astype(str).str.slice(0, 10)
    return pd.DataFrame({
        "date": dates,
        "open": [b[0] for b in bars], "high": [b[1] for b in bars],
        "low": [b[2] for b in bars], "close": [b[3] for b in bars],
    })


# entry is filled at open[i+1]; T+2 lock skips k=0,1; first evaluated bar is k=2 (j=i+3)
def test_target_hit():
    # signal at i=0, entry=open[1]=100, target 110 hit at i=3 high
    f = _frame([(99, 99, 99, 99), (100, 101, 99, 100), (100, 102, 99, 101),
                (101, 103, 100, 102), (102, 111, 101, 110), (110, 112, 108, 111)])
    status, net, reason = ft._replay_mr(f, 0, stop=95, target=110, max_hold=15)
    assert status == "resolved" and reason == "TARGET"
    assert net == pytest.approx((110 - 100) / 100 * 100 - ft.COST_PCT, abs=0.01)


def test_stop_hit():
    # entry=100, stop=95 breached at i=3 low=94
    f = _frame([(99, 99, 99, 99), (100, 101, 99, 100), (100, 102, 99, 101),
                (101, 103, 94, 96), (96, 97, 90, 92), (92, 93, 90, 91)])
    status, net, reason = ft._replay_mr(f, 0, stop=95, target=200, max_hold=15)
    assert status == "resolved" and reason == "STOP"
    assert net == pytest.approx((95 - 100) / 100 * 100 - ft.COST_PCT, abs=0.01)


def test_time_exit():
    # neither stop nor target; exit at close on the max_hold-th bar after entry
    bars = [(99, 99, 99, 99)] + [(100 + i * 0.1, 101, 99, 100 + i * 0.1) for i in range(8)]
    f = _frame(bars)
    status, net, reason = ft._replay_mr(f, 0, stop=90, target=200, max_hold=3)
    assert status == "resolved" and reason == "TIME"


def test_t2_lock_ignores_early_stop():
    # stop would trigger on k=0 and k=1 (low=90 < stop 95) but T+2 lock must ignore them;
    # afterwards price recovers and hits target -> TARGET, not STOP
    f = _frame([(99, 99, 99, 99), (100, 101, 90, 100), (100, 102, 90, 101),
                (101, 111, 100, 110), (110, 112, 108, 111)])
    status, net, reason = ft._replay_mr(f, 0, stop=95, target=110, max_hold=15)
    assert reason == "TARGET"  # early sub-stop lows on k=0/1 were correctly ignored


def test_pending_when_not_enough_bars():
    # signal is the last bar -> no next-open to fill -> pending
    f = _frame([(99, 99, 99, 99), (100, 101, 99, 100)])
    status, net, reason = ft._replay_mr(f, 1, stop=95, target=110, max_hold=15)
    assert status == "pending"
    # entry fills but max_hold not yet elapsed and no exit -> still pending
    f2 = _frame([(99, 99, 99, 99), (100, 101, 99, 100), (100, 101, 99, 100)])
    status2, _, _ = ft._replay_mr(f2, 0, stop=90, target=200, max_hold=15)
    assert status2 == "pending"


def test_logger_idempotent(tmp_path, monkeypatch):
    ledger = tmp_path / "ft.jsonl"
    monkeypatch.setattr(ft, "LEDGER_PATH", ledger)
    mr = {"data_date": "2026-07-03", "rules_hash": "h", "market": {"state": "PANIC_BEAR"},
          "buys": [{"symbol": "PNJ", "date": "2026-07-03", "entry_reference": 54.6,
                    "stop_loss": 50.0, "take_profit": 58.0, "max_hold_days": 15,
                    "win_prob": 0.57, "reward_risk": 1.2, "close": 54.6, "gate": "vn30"}],
          "prob_buys": [], "watches": []}
    mom = {"data_date": "2026-07-03", "market": {"state": "PANIC_BEAR"}, "exposure_pct": 100,
           "variant": "all", "picks": [{"symbol": "VIC", "close": 204.7, "weight_pct": 8.6,
                                        "qty": 100, "momentum_12_1_pct": 300.0, "vn30": True}]}
    r1 = ft.log_recommendations(mr, mom)
    r2 = ft.log_recommendations(mr, mom)
    assert r1["appended"] == 2 and r2["appended"] == 0
    lines = [json.loads(x) for x in ledger.read_text(encoding="utf-8").splitlines()]
    assert len(lines) == 2
    assert {l["engine"] for l in lines} == {"mr", "momentum"}
