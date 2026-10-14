from datetime import datetime

import pytest

from stock_agent.data.exchange_calendar import VN_TIMEZONE
from tests.swing_desk_support import NOW_EOD
from tests.test_swing_desk_check import setup_desk


def build(paths):
    from stock_agent.swing_desk import service
    state = service.compute_state(paths, NOW_EOD, s2_override=[], with_flows_data=False)
    service.write_state(paths, state)
    return state


def test_baseline_and_control_rows_from_same_state(tmp_path, monkeypatch):
    from stock_agent.swing_desk import baseline
    paths, _ = setup_desk(tmp_path, monkeypatch)
    state = build(paths)
    rows_b = baseline.candidate_rows(state, "baseline")
    assert [r["ticker"] for r in rows_b] == ["AAA"]
    r = rows_b[0]
    assert r["setup"] == ["S1"] and r["gate"] == "PASS" and r["entry"]["type"] == "limit"
    assert r["stop"] < r["entry"]["limit_price"] < r["target"]
    assert r["entry"]["valid_session"] == "2026-09-22" and r["data_as_of"] == "2026-09-21"
    c1 = baseline.candidate_rows(state, "control")
    c2 = baseline.candidate_rows(state, "control")
    assert c1 == c2 and len(c1) <= 3
    pool = {t for t, v in state["tickers"].items() if v["structural_gate_ex_setup"] == "PASS"}
    assert {r["ticker"] for r in c1} <= pool


def test_freeze_arms_appends_both_arms_before_0900(tmp_path, monkeypatch):
    from stock_agent.swing_desk import baseline, reclog
    paths, _ = setup_desk(tmp_path, monkeypatch)
    build(paths)
    out = baseline.freeze_code_arms(paths, now=NOW_EOD)
    assert set(out) == {"baseline", "control"}
    rows = reclog.read_rows(paths.rec_log)
    assert {r["source"] for r in rows} <= {"baseline", "control"} and rows
    assert all(r["target_session"] == "2026-09-22" and r["model_id"] == "rules" for r in rows)
    assert reclog.verify_chain(paths.rec_log)["ok"]
    with pytest.raises(ValueError, match="already frozen"):
        baseline.freeze_code_arms(paths, now=NOW_EOD)
    late = datetime(2026, 9, 22, 9, 30, tzinfo=VN_TIMEZONE)
    with pytest.raises(ValueError):
        baseline.freeze_code_arms(paths, now=late)


def test_freeze_analyst_file_requires_fresh_state(tmp_path, monkeypatch):
    from stock_agent.swing_desk import baseline
    paths, _ = setup_desk(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="state"):
        baseline.freeze_analyst(paths, [], source="desk", now=NOW_EOD, model_id="m", n_rollouts=4)


def test_score_log_resolves_with_newest_snapshot(tmp_path, monkeypatch):
    from stock_agent.swing_desk import baseline
    paths, _ = setup_desk(tmp_path, monkeypatch)
    build(paths)
    baseline.freeze_code_arms(paths, now=NOW_EOD)
    result = baseline.score_log(paths, now=NOW_EOD)
    assert result["chain"]["ok"]
    assert all(r["status"] == "pending" for r in result["rows"])
    assert (paths.scores / "2026-09-21.json").is_file()
    assert result["summary"]["conclusion"].startswith("NOT PROVEN")
