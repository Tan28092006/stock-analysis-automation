"""Per-session EOD step status: observability only, exit codes unchanged."""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path

import pytest

from stock_agent.pipeline import eod_status
from stock_agent.pipeline import foreign_refresh as flow
from stock_agent.pipeline import paper_runner as r


@pytest.fixture
def status_dir(tmp_path, monkeypatch):
    path = tmp_path / "eod_status"
    monkeypatch.setenv("VN30_EOD_STATUS_DIR", str(path))
    return path


def _read(status_dir: Path, session: str) -> dict:
    return json.loads((status_dir / f"{session}.json").read_text(encoding="utf-8"))


def test_steps_merge_into_one_atomic_session_record(status_dir):
    eod_status.record_step("2026-10-05", "foreign_refresh", {"outcome": "ready", "exit_code": 0})
    eod_status.record_step("2026-10-05", "paper_runner", {"outcome": "recorded", "exit_code": 0})
    eod_status.record_step("2026-10-05", "paper_runner", {"outcome": "already_recorded", "exit_code": 0})
    data = _read(status_dir, "2026-10-05")
    assert data["schema_version"] == 1 and data["session"] == "2026-10-05"
    assert data["steps"]["foreign_refresh"]["outcome"] == "ready"
    assert data["steps"]["paper_runner"]["outcome"] == "already_recorded"
    assert data["steps"]["paper_runner"]["attempt"] == 2
    assert sorted(p.name for p in status_dir.iterdir()) == ["2026-10-05.json"]


def test_invalid_session_name_is_rejected(status_dir):
    with pytest.raises(ValueError):
        eod_status.record_step("../evil", "paper_runner", {})


@pytest.fixture
def cli(tmp_path, monkeypatch, status_dir):
    from tests.test_market_runtime_hardening import make_snapshot
    manifest = make_snapshot(tmp_path / "snapshot")
    monkeypatch.setattr(r, "load_universe", lambda: {"symbols": ["AAA"]})
    real_now = r._now
    monkeypatch.setattr(r, "_now", lambda value=None: real_now(value or datetime(2026, 9, 21, 10, tzinfo=timezone.utc)))
    return manifest, tmp_path / "out", status_dir


def test_recorded_run_is_visible_as_success(cli):
    manifest, out, status_dir = cli
    assert r.main(["--manifest", str(manifest), "--output-dir", str(out), "--run"]) == 0
    step = _read(status_dir, "2026-09-21")["steps"]["paper_runner"]
    assert step["outcome"] == "recorded" and step["paper_recorded"] is True and step["exit_code"] == 0
    assert Path(step["paper_record_path"]).name == "paper.json" and Path(step["paper_record_path"]).exists()
    assert step["scoring_status"] == "ok"
    assert step["started_at"].endswith("+00:00") and step["finished_at"].endswith("+00:00")


def test_scoring_block_does_not_hide_a_recorded_paper_run(cli):
    manifest, out, status_dir = cli
    placeholder = out / "runs" / "2026-09-18"
    placeholder.mkdir(parents=True)
    (placeholder / "paper.json").write_text("{}", encoding="utf-8")  # unscoreable -> scoring blocked
    assert r.main(["--manifest", str(manifest), "--output-dir", str(out), "--run"]) == 2  # unchanged
    step = _read(status_dir, "2026-09-21")["steps"]["paper_runner"]
    assert step["outcome"] == "recorded_scoring_blocked" and step["exit_code"] == 2
    assert step["paper_recorded"] is True and step["paper_record_status"] == "recorded"
    assert step["scoring_status"] == "blocked" and step["scoring_errors"] >= 1


def test_blocked_snapshot_is_a_failed_step_without_record(cli, monkeypatch):
    manifest, out, status_dir = cli
    monkeypatch.setattr(r, "build_market_snapshot", lambda *a, **kw: {"status": "blocked", "failures": ["offline"]})
    assert r.main(["--refresh", "--run", "--output-dir", str(out)]) == 2
    # --refresh derives its session from the real clock, as before this change.
    (only,) = list(status_dir.glob("*.json"))
    step = json.loads(only.read_text(encoding="utf-8"))["steps"]["paper_runner"]
    assert step["outcome"] == "failed" and step["paper_recorded"] is False and step["mode"] == "refresh_run"
    assert step["snapshot_status"] == "blocked" and "Snapshot blocked" in step["error"]


def test_preview_modes_write_no_status(cli):
    manifest, out, status_dir = cli
    assert r.main(["--manifest", str(manifest), "--output-dir", str(out), "--check"]) == 0
    assert not status_dir.exists()


def test_status_write_failure_never_changes_exit_code(cli, capsys):
    manifest, out, status_dir = cli
    status_dir.parent.mkdir(parents=True, exist_ok=True)
    status_dir.write_text("not a directory")
    assert r.main(["--manifest", str(manifest), "--output-dir", str(out), "--run"]) == 0
    assert "eod_status" in capsys.readouterr().err


def test_flow_retry_records_status_with_unchanged_exit_code(status_dir, tmp_path, monkeypatch):
    monkeypatch.setattr(flow, "refresh_with_retries", lambda symbols, root: dict(
        status="partial", expected_session="2026-10-05", manifest_path="m.json", attempts=4,
        attempt_manifest_paths=["a", "b"], retry_exhausted=True, errors=[],
        missing_latest={"foreign": ["MBB"], "proprietary": []}))
    assert flow.main(["--root", str(tmp_path), "--symbols", "MBB", "--retry-stale"]) == 2
    step = _read(status_dir, "2026-10-05")["steps"]["foreign_refresh"]
    assert step["outcome"] == "partial" and step["exit_code"] == 2 and step["mode"] == "retry_stale"
    assert step["retry_exhausted"] is True and step["missing_latest_count"] == 1
    assert step["attempt_manifest_paths"] == ["a", "b"]


def test_flow_exception_records_blocked_status(status_dir, tmp_path, monkeypatch):
    def boom(symbols, root):
        raise ValueError("Concurrent or interrupted collector; investigate lock")
    monkeypatch.setattr(flow, "refresh_with_retries", boom)
    monkeypatch.setattr(flow.ff, "aware", lambda value=None: datetime(2026, 10, 5, 10, tzinfo=timezone.utc)
                        if value is None else datetime.fromisoformat(str(value)) if isinstance(value, str) else value)
    assert flow.main(["--root", str(tmp_path), "--symbols", "MBB", "--retry-stale"]) == 2
    step = _read(status_dir, "2026-10-05")["steps"]["foreign_refresh"]
    assert step["outcome"] == "blocked" and step["error"] == "ValueError"


def test_flow_status_mode_writes_no_status(status_dir, tmp_path):
    flow.main(["--root", str(tmp_path), "--symbols", "MBB", "--status"])
    assert not status_dir.exists()
