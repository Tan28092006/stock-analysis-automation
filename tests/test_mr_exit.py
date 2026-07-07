"""Acceptance tests for the canonical MR exit replay (features/mr_exit.simulate_mr_exit)
and that the callers which wrap it agree. Covers: stop, target, T+15 time-exit, the T+2
settlement lock (an exit signal inside the lock must be ignored), same-bar stop/target
precedence, and the pending (horizon beyond data) edge.
"""
import pandas as pd

from stock_agent.features.mr_exit import simulate_mr_exit


def _frame(rows):
    return pd.DataFrame(rows, columns=["date", "open", "high", "low", "close"])


def test_stop_hit_after_lock_and_lock_is_respected():
    # entry at idx1. stop=90. idx2 low=85 would hit stop BUT is inside the T+2 lock
    # (bars entry_idx..entry_idx+1) -> must be ignored. First tradable bar is idx3.
    f = _frame([
        ("d0", 100, 101, 99, 100),   # signal bar
        ("d1", 100, 102, 98, 101),   # entry bar (idx1)
        ("d2", 100, 101, 85, 100),   # inside lock -> stop at 85 IGNORED
        ("d3", 100, 101, 95, 100),   # tradable, no hit
        ("d4", 100, 101, 88, 100),   # low 88 <= stop 90 -> STOP here
        ("d5", 100, 130, 80, 100),
    ])
    assert simulate_mr_exit(f, 1, stop=90, target=120, max_hold=10, settle_lock=2) == (4, 90.0, "stop", True)


def test_target_hit():
    f = _frame([
        ("d0", 100, 101, 99, 100),
        ("d1", 100, 102, 98, 101),
        ("d2", 100, 101, 99, 100),
        ("d3", 100, 125, 99, 100),   # high 125 >= target 120 -> TARGET
        ("d4", 100, 101, 88, 100),
    ])
    assert simulate_mr_exit(f, 1, 90, 120, 10, 2) == (3, 120.0, "target", True)


def test_time_exit_at_horizon():
    # no stop/target; max_hold=3 -> horizon = entry(1)+3 = 4 -> exit close[4]=107
    f = _frame([
        ("d0", 100, 101, 99, 100),
        ("d1", 100, 102, 98, 101),
        ("d2", 100, 101, 99, 100),
        ("d3", 100, 101, 99, 100),
        ("d4", 100, 101, 99, 107),
        ("d5", 100, 101, 99, 100),
    ])
    assert simulate_mr_exit(f, 1, 50, 200, 3, 2) == (4, 107.0, "time", True)


def test_same_bar_stop_and_target_stop_wins():
    f = _frame([
        ("d0", 100, 101, 99, 100),
        ("d1", 100, 102, 98, 101),
        ("d2", 100, 101, 99, 100),
        ("d3", 100, 125, 88, 100),   # both stop (88<=90) and target (125>=120) -> stop wins
    ])
    idx, px, reason, resolved = simulate_mr_exit(f, 1, 90, 120, 10, 2)
    assert reason == "stop" and px == 90.0


def test_pending_when_horizon_beyond_data():
    # horizon = 1+10 = 11 but only 6 bars, no stop/target -> unresolved, partial mark at last
    f = _frame([("d%d" % k, 100, 101, 99, 100) for k in range(6)])
    idx, px, reason, resolved = simulate_mr_exit(f, 1, 50, 200, 10, 2)
    assert resolved is False and idx == 5


def test_forward_test_wrapper_matches():
    # forward_test._replay_mr: signal at i=0, entry open[1]=100, stop 90 hit at idx4.
    from stock_agent.pipeline.forward_test import _replay_mr, COST_PCT
    f = _frame([
        ("d0", 100, 101, 99, 100),
        ("d1", 100, 102, 98, 101),
        ("d2", 100, 101, 85, 100),   # inside lock, ignored
        ("d3", 100, 101, 95, 100),
        ("d4", 100, 101, 88, 100),
        ("d5", 100, 130, 80, 100),
    ])
    status, net, reason = _replay_mr(f, 0, stop=90, target=120, max_hold=10)
    assert status == "resolved" and reason == "STOP"
    assert net == round((90 - 100) / 100 * 100 - COST_PCT, 2)   # -10.60


def test_position_manager_wrapper_matches(tmp_path, monkeypatch):
    # check_positions: recorded position entered on d0 (entry bar = idx0), stop 90.
    # Anchored on entry_date (not signal+1) — but uses the SAME exit loop.
    from stock_agent.features import position_manager as pm
    f = _frame([
        ("d0", 100, 102, 98, 100),   # entry_date bar (idx0)
        ("d1", 100, 101, 99, 100),   # inside lock
        ("d2", 100, 101, 88, 100),   # idx2 = entry+2, first tradable -> stop 90 hit
        ("d3", 100, 130, 80, 100),
    ])
    monkeypatch.setattr(pm, "PRICES_DIR", tmp_path)
    f.to_csv(tmp_path / "TEST.csv", index=False)
    store = pm.PositionStore(path=tmp_path / "pos.json")
    store.add(symbol="TEST", entry_date="d0", entry_price=100.0, stop=90.0,
              target=120.0, max_hold_days=10, qty=100)
    res = pm.check_positions(store)
    assert len(res) == 1
    p = res[0]
    assert p["live_status"] == "SELL" and p["sell_reason"] == "STOP_LOSS"
    assert p["exit_price"] == 90.0
