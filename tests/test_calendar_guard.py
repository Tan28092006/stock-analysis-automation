"""Warn before the maintained HOSE holiday table runs out; never invent holidays."""
from __future__ import annotations

from datetime import date, datetime, timezone
import json

from stock_agent.data import exchange_calendar as cal


def _vn(y, m, d, h=18):
    return datetime(y, m, d, h - 7, tzinfo=timezone.utc)


def test_no_warning_while_today_and_horizon_are_covered():
    last = max(d.year for d in cal.HOSE_HOLIDAYS)
    assert cal.calendar_coverage_warning(_vn(last, 10, 6)) is None


def test_warns_when_horizon_crosses_last_table_year():
    last = max(d.year for d in cal.HOSE_HOLIDAYS)
    warning = cal.calendar_coverage_warning(_vn(last, 12, 10))
    assert warning and warning["last_holiday_year"] == last
    assert warning["today_covered"] is True and warning["horizon_covered"] is False
    assert str(last + 1) in warning["message"]


def test_warns_when_today_is_past_last_table_year_and_never_adds_holidays():
    before = set(cal.HOSE_HOLIDAYS)
    last = max(before)
    warning = cal.calendar_coverage_warning(_vn(last.year + 1, 1, 5))
    assert warning and warning["today_covered"] is False
    assert "holiday" in warning["message"].lower()
    assert set(cal.HOSE_HOLIDAYS) == before


def test_horizon_is_configurable_and_uses_vietnam_date():
    last = max(d.year for d in cal.HOSE_HOLIDAYS)
    # 31 Dec 23:30 VN is already 1 Jan in no timezone; 17:00 UTC 31 Dec = 00:00 VN 1 Jan.
    assert cal.calendar_coverage_warning(datetime(last, 12, 31, 17, 30, tzinfo=timezone.utc),
                                         horizon_days=0)["today_covered"] is False
    assert cal.calendar_coverage_warning(datetime(last, 12, 31, 16, 0, tzinfo=timezone.utc),
                                         horizon_days=0) is None


def test_foreign_refresh_status_surfaces_calendar_warning(tmp_path, monkeypatch, capsys):
    from stock_agent.pipeline import foreign_refresh as job
    monkeypatch.setattr(job, "calendar_coverage_warning",
                        lambda now=None, horizon_days=30: {"message": "HOSE holiday table ends 2026"})
    job.main(["--root", str(tmp_path), "--symbols", "MBB", "--status"])
    printed = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert printed["calendar_warning"]["message"] == "HOSE holiday table ends 2026"


def test_paper_runner_surfaces_calendar_warning_in_output_and_latest(tmp_path, monkeypatch, capsys):
    from stock_agent.pipeline import paper_runner as r
    from tests.test_market_runtime_hardening import make_snapshot
    manifest = make_snapshot(tmp_path / "snapshot")
    monkeypatch.setattr(r, "load_universe", lambda: {"symbols": ["AAA"]})
    real_now = r._now
    monkeypatch.setattr(r, "_now", lambda value=None: real_now(value or datetime(2026, 9, 21, 10, tzinfo=timezone.utc)))
    monkeypatch.setattr(r, "calendar_coverage_warning",
                        lambda now=None, horizon_days=30: {"message": "HOSE holiday table ends 2026"})
    out = tmp_path / "out"
    assert r.main(["--manifest", str(manifest), "--output-dir", str(out), "--check"]) == 0
    assert "HOSE holiday table ends 2026" in capsys.readouterr().out
    latest = json.loads((out / "latest.json").read_text(encoding="utf-8"))
    assert latest["calendar_warning"]["message"] == "HOSE holiday table ends 2026"


def test_paper_record_itself_does_not_carry_operational_warnings(tmp_path, monkeypatch):
    from stock_agent.pipeline import paper_runner as r
    from tests.test_market_runtime_hardening import make_snapshot
    manifest = make_snapshot(tmp_path / "snapshot")
    monkeypatch.setattr(r, "load_universe", lambda: {"symbols": ["AAA"]})
    real_now = r._now
    monkeypatch.setattr(r, "_now", lambda value=None: real_now(value or datetime(2026, 9, 21, 10, tzinfo=timezone.utc)))
    monkeypatch.setattr(r, "calendar_coverage_warning", lambda now=None, horizon_days=30: {"message": "x"})
    out = tmp_path / "out"
    r.main(["--manifest", str(manifest), "--output-dir", str(out), "--run"])
    record = json.loads((out / "runs/2026-09-21/paper.json").read_text(encoding="utf-8"))
    assert "calendar_warning" not in record
