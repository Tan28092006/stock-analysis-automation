"""Interrupted flow collector state: read-only diagnosis and guarded quarantine.

Fixtures are synthetic directories under tmp_path; no collector or network runs.
"""
from __future__ import annotations

from datetime import date, datetime, timezone
import json
import os
from pathlib import Path
import time

import pytest

from stock_agent.data import foreign_flows as ff
from stock_agent.pipeline import foreign_refresh as job

ORPHAN = "20261001T130512000000Z_deadbeef"
HOURS = 3600


def _age(path: Path, seconds: float) -> None:
    stamp = time.time() - seconds
    for item in [*path.rglob("*"), path]:
        os.utime(item, (stamp, stamp))


def make_store(root: Path, *, lock_age=None, orphan_age=5 * HOURS, committed=True) -> Path:
    runs = root / "runs"
    orphan = runs / ORPHAN
    (orphan / "raw").mkdir(parents=True)
    (orphan / "raw" / "foreign_MBB.json").write_bytes(b'{"partial": true}')
    _age(orphan, orphan_age)
    if committed:
        good = runs / "20260930T100000000000Z_cafef00d"
        good.mkdir()
        (good / "manifest.json").write_text("{}", encoding="utf-8")
    if lock_age is not None:
        lock = root / ".collect-lock"
        lock.mkdir()
        _age(lock, lock_age)
    return orphan


def _printed(capsys) -> dict:
    return json.loads(capsys.readouterr().out.strip().splitlines()[-1])


@pytest.fixture
def no_verified_records(monkeypatch):
    # The committed fixture manifest is a placeholder; health() itself is not under test here.
    monkeypatch.setattr(ff, "verified_records", lambda root=ff.STORE: [])


def test_status_names_stale_lock_and_orphan_run(tmp_path, capsys, no_verified_records):
    make_store(tmp_path, lock_age=5 * HOURS)
    assert job.main(["--root", str(tmp_path), "--symbols", "MBB", "--status"]) == 2
    out = _printed(capsys)
    assert out["lock"]["present"] and out["lock"]["stale"] is True
    assert 4.9 * HOURS <= out["lock"]["age_seconds"] <= 5.1 * HOURS
    assert out["collector_plausibly_running"] is False
    assert [o["name"] for o in out["orphan_runs"]] == [ORPHAN]
    assert out["orphan_runs"][0]["files"] == 1
    assert out["incomplete_runs"] == [ORPHAN]
    assert "quarantine-orphans" in out["orphan_assessment"]


def test_status_flags_young_lock_as_possibly_running(tmp_path, capsys, no_verified_records):
    make_store(tmp_path, lock_age=60, orphan_age=30)
    job.main(["--root", str(tmp_path), "--symbols", "MBB", "--status"])
    out = _printed(capsys)
    assert out["lock"]["stale"] is False and out["collector_plausibly_running"] is True


def test_quarantine_is_dry_run_by_default(tmp_path, capsys):
    orphan = make_store(tmp_path, lock_age=5 * HOURS)
    assert job.main(["--root", str(tmp_path), "--quarantine-orphans"]) == 0
    out = _printed(capsys)
    assert out["status"] == "dry_run" and out["would_move"] == [ORPHAN]
    assert out["would_remove_stale_lock"] is True
    assert orphan.is_dir() and (tmp_path / ".collect-lock").is_dir()
    assert not (tmp_path / "quarantine").exists()


def test_quarantine_refuses_young_lock_even_with_apply(tmp_path, capsys):
    orphan = make_store(tmp_path, lock_age=10 * 60)
    assert job.main(["--root", str(tmp_path), "--quarantine-orphans", "--apply"]) == 2
    out = _printed(capsys)
    assert out["status"] == "refused" and "running" in out["reason"]
    assert orphan.is_dir() and (tmp_path / ".collect-lock").is_dir()
    assert not (tmp_path / "quarantine").exists()


def test_quarantine_refuses_recent_orphan_activity_under_stale_lock(tmp_path, capsys):
    orphan = make_store(tmp_path, lock_age=5 * HOURS, orphan_age=60)
    assert job.main(["--root", str(tmp_path), "--quarantine-orphans", "--apply"]) == 2
    assert _printed(capsys)["status"] == "refused" and orphan.is_dir()


def test_apply_moves_never_deletes_and_health_stops_counting(tmp_path, capsys, no_verified_records):
    orphan = make_store(tmp_path, lock_age=5 * HOURS)
    before = ff.health(["MBB"], root=tmp_path)
    assert before["incomplete_runs"] == [ORPHAN] and before["collection_in_progress"]
    assert job.main(["--root", str(tmp_path), "--quarantine-orphans", "--apply"]) == 0
    out = _printed(capsys)
    assert out["status"] == "quarantined" and out["moved"] == [ORPHAN]
    assert not orphan.exists()
    destination = Path(out["destination"])
    assert destination.parent == tmp_path / "quarantine"
    assert (destination / ORPHAN / "raw" / "foreign_MBB.json").read_bytes() == b'{"partial": true}'
    reason = json.loads((destination / "reason.json").read_text(encoding="utf-8"))
    assert reason["status"] == "quarantined" and "manifest.json" in reason["reason"]
    assert reason["stale_lock_removed"]["age_seconds"] >= 4.9 * HOURS
    assert [m["name"] for m in reason["moved"]] == [ORPHAN]
    assert not (tmp_path / ".collect-lock").exists()
    assert (tmp_path / "runs" / "20260930T100000000000Z_cafef00d" / "manifest.json").exists()
    after = ff.health(["MBB"], root=tmp_path)
    assert after["incomplete_runs"] == [] and after["collection_in_progress"] is False


def test_apply_without_lock_moves_orphan_and_leaves_no_lock(tmp_path, capsys):
    orphan = make_store(tmp_path, lock_age=None, orphan_age=60)
    assert job.main(["--root", str(tmp_path), "--quarantine-orphans", "--apply"]) == 0
    out = _printed(capsys)
    assert out["status"] == "quarantined" and out["stale_lock_removed"] is None
    assert not orphan.exists() and not (tmp_path / ".collect-lock").exists()


def test_apply_with_nothing_to_do_creates_nothing(tmp_path, capsys):
    (tmp_path / "runs").mkdir()
    assert job.main(["--root", str(tmp_path), "--quarantine-orphans", "--apply"]) == 0
    assert _printed(capsys)["status"] == "nothing_to_do"
    assert not (tmp_path / "quarantine").exists()


def test_apply_flag_requires_quarantine_mode(tmp_path):
    with pytest.raises(SystemExit) as exc:
        job.main(["--root", str(tmp_path), "--status", "--apply"])
    assert exc.value.code == 2


def test_collector_cannot_start_while_quarantine_holds_lock(tmp_path, monkeypatch):
    from stock_agent.pipeline import flow_orphans
    make_store(tmp_path, lock_age=None)
    seen = {}
    real_rename = flow_orphans.os.rename

    def spy(src, dst):
        seen["lock_held"] = (tmp_path / ".collect-lock").is_dir()
        with pytest.raises(ValueError, match="lock"):
            ff.collect_flows(["MBB"], root=tmp_path, fetcher=lambda *a: b"",
                             clock=lambda: datetime(2026, 10, 1, 11, tzinfo=timezone.utc))
        return real_rename(src, dst)

    monkeypatch.setattr(flow_orphans.os, "rename", spy)
    result = flow_orphans.quarantine_orphans(tmp_path, apply=True)
    assert result["status"] == "quarantined" and seen["lock_held"]
    assert not (tmp_path / ".collect-lock").exists()


def test_foreign_flows_runbook_documents_quarantine():
    text = (Path(__file__).resolve().parents[1] / "docs" / "FOREIGN-FLOWS.md").read_text(encoding="utf-8")
    assert "--quarantine-orphans" in text and "--apply" in text
    assert "quarantine/" in text
