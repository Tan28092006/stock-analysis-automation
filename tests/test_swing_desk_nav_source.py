"""Desk state sizing uses the journal NAV when portfolio.txt has one, and says which NAV it used."""
from stock_agent.swing_desk.sizing import plan as size_plan
from tests.swing_desk_support import NOW_EOD
from tests.test_swing_desk_check import setup_desk


def _state(paths):
    from stock_agent.swing_desk import service
    return service.compute_state(paths, NOW_EOD, s2_override=[], with_flows_data=False)


def test_state_plan_uses_journal_nav(tmp_path, monkeypatch):
    from stock_agent.swing_desk.config import load_config
    paths, _ = setup_desk(tmp_path, monkeypatch)
    paths.desk.mkdir(parents=True, exist_ok=True)
    paths.portfolio.write_text("2026-09-01|NAV||11200000||||||\n", encoding="utf-8")
    state = _state(paths)
    t = state["tickers"]["AAA"]
    plan = t["plan_default_nav"]
    expected = size_plan(t["indicators"]["close"], 11_200_000.0, load_config(paths.config))
    assert plan["nav"] == 11_200_000.0
    assert plan["nav_source"] == "journal"
    assert plan["qty"] == expected["qty"]
    assert state["manifest"]["nav_source"] == "journal"
    from stock_agent.swing_desk import baseline
    row = baseline.candidate_rows(state, "baseline")[0]
    assert row["size"]["qty"] == expected["qty"] and row["size"]["nav_source"] == "journal"


def test_state_plan_falls_back_to_default_and_says_so(tmp_path, monkeypatch):
    paths, _ = setup_desk(tmp_path, monkeypatch)
    state = _state(paths)
    plan = state["tickers"]["AAA"]["plan_default_nav"]
    assert plan["nav"] == 1_000_000_000.0
    assert plan["nav_source"].startswith("default")
    assert state["manifest"]["nav_source"].startswith("default")
