"""Report expected trading sessions with no paper record; never backfill."""
from __future__ import annotations

from datetime import datetime, timezone
import json

import pytest

from stock_agent.pipeline import paper_runner as r

RECORDED = ["2026-09-25", "2026-09-28", "2026-09-29", "2026-09-30", "2026-10-02", "2026-10-05"]


def _record(out, session):
    folder = out / "runs" / session
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "paper.json").write_text('{"status": "recorded"}', encoding="utf-8")


def test_closed_window_without_record_is_reported_as_missed(tmp_path):
    for session in RECORDED:
        _record(tmp_path, session)
    (tmp_path / "runs" / "2026-10-01" / ".write-lock").mkdir(parents=True)  # interrupted writer, no record
    report = r.missed_sessions(tmp_path, now=datetime(2026, 10, 6, 10, tzinfo=timezone.utc))  # 17:00 VN
    assert report["since"] == "2026-09-25" and report["through"] == "2026-10-06"
    assert report["missed"] == ["2026-10-01"]
    assert report["window_open_unrecorded"] == ["2026-10-06"]
    assert report["recorded_sessions"] == len(RECORDED)
    assert "backfill" in report["note"]


def test_open_window_is_not_yet_missed_and_closes_at_0900_next_session(tmp_path):
    for session in RECORDED[:-1]:
        _record(tmp_path, session)
    before = r.missed_sessions(tmp_path, now=datetime(2026, 10, 6, 1, 59, tzinfo=timezone.utc))  # 08:59 VN
    assert before["window_open_unrecorded"] == ["2026-10-05"] and "2026-10-05" not in before["missed"]
    after = r.missed_sessions(tmp_path, now=datetime(2026, 10, 6, 2, 0, tzinfo=timezone.utc))  # 09:00 VN
    assert after["missed"] == ["2026-10-01", "2026-10-05"] and after["window_open_unrecorded"] == []


def test_no_records_means_nothing_to_compare(tmp_path):
    report = r.missed_sessions(tmp_path, now=datetime(2026, 10, 6, 10, tzinfo=timezone.utc))
    assert report["since"] is None and report["missed"] == []


@pytest.fixture
def cli(tmp_path, monkeypatch):
    from tests.test_market_runtime_hardening import make_snapshot
    manifest = make_snapshot(tmp_path / "snapshot")
    monkeypatch.setattr(r, "load_universe", lambda: {"symbols": ["AAA"]})
    real_now = r._now
    monkeypatch.setattr(r, "_now", lambda value=None: real_now(value or datetime(2026, 9, 21, 10, tzinfo=timezone.utc)))
    out = tmp_path / "out"
    _record(out, "2026-09-17")
    return manifest, out


def test_run_reports_missed_sessions_in_latest_and_output_without_backfill(cli, capsys, monkeypatch):
    from stock_agent.pipeline import paper_scoring
    manifest, out = cli
    # The placeholder 2026-09-17 record is not scoreable; scoring is not under test here.
    monkeypatch.setattr(paper_scoring, "score_paper_runs", lambda *a, **kw: {
        "status": "ok", "as_of": "2026-09-21", "pending": 0, "resolved": 0, "errors": [],
        "volume_revision_rows": 0, "volume_revision_runs": 0})
    assert r.main(["--manifest", str(manifest), "--output-dir", str(out), "--run"]) == 0
    latest = json.loads((out / "latest.json").read_text(encoding="utf-8"))
    assert latest["missed_sessions"]["missed"] == ["2026-09-18"]
    assert latest["missed_sessions"]["window_open_unrecorded"] == []  # 2026-09-21 was just recorded
    assert '"missed": ["2026-09-18"]' in capsys.readouterr().out
    assert not (out / "runs" / "2026-09-18").exists()
    record = json.loads((out / "runs/2026-09-21/paper.json").read_text(encoding="utf-8"))
    assert "missed_sessions" not in record


def test_failed_run_still_reports_missed_sessions(cli, monkeypatch):
    manifest, out = cli
    monkeypatch.setattr(r, "build_market_snapshot", lambda *a, **kw: {"status": "blocked", "failures": ["offline"]})
    assert r.main(["--refresh", "--run", "--output-dir", str(out)]) == 2
    latest = json.loads((out / "latest.json").read_text(encoding="utf-8"))
    assert latest["status"] == "failed" and latest["missed_sessions"]["missed"] == ["2026-09-18"]


def test_status_mode_is_read_only(cli, capsys):
    manifest, out = cli
    before = sorted(p.relative_to(out) for p in out.rglob("*"))
    assert r.main(["--status", "--output-dir", str(out)]) == 0
    printed = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert printed["missed_sessions"]["missed"] == ["2026-09-18"]
    assert printed["missed_sessions"]["window_open_unrecorded"] == ["2026-09-21"]
    assert sorted(p.relative_to(out) for p in out.rglob("*")) == before
    assert r.main(["--status", "--run", "--output-dir", str(out)]) == 2
