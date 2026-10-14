"""Retention is a report only: list old snapshot/flow run dirs with sizes, delete nothing."""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path

from stock_agent.pipeline import retention_report as rr

NOW = datetime(2026, 10, 6, 12, tzinfo=timezone.utc)


def _dir(path: Path, files: dict[str, bytes]) -> Path:
    path.mkdir(parents=True)
    for name, data in files.items():
        (path / name).parent.mkdir(parents=True, exist_ok=True)
        (path / name).write_bytes(data)
    return path


def make_tree(tmp_path):
    paper, flows = tmp_path / "paper", tmp_path / "flows"
    old_ref = _dir(paper / "snapshots" / "20260801T100000000000Z", {"manifest.json": b"{}", "prices/AAA.csv": b"x" * 100})
    _dir(paper / "snapshots" / "20260802T100000000000Z", {"manifest.json": b"{}", "raw/AAA.json": b"y" * 50})
    _dir(paper / "snapshots" / "20261005T100000000000Z", {"manifest.json": b"{}"})
    record = paper / "runs" / "2026-08-01"
    record.mkdir(parents=True)
    (record / "paper.json").write_text(json.dumps({"snapshot_manifest": str(old_ref / "manifest.json")}))
    _dir(flows / "runs" / "20260810T100000000000Z_aaaa1111", {"manifest.json": b"{}", "records.jsonl": b"z" * 30})
    _dir(flows / "runs" / "20260811T100000000000Z_bbbb2222", {"raw/foreign_MBB.json": b"w" * 10})  # orphan
    _dir(flows / "runs" / "20261005T100000000000Z_cccc3333", {"manifest.json": b"{}"})
    return paper, flows


def _listing(root: Path):
    return sorted((str(p.relative_to(root)), p.stat().st_size if p.is_file() else -1) for p in root.rglob("*"))


def test_lists_only_dirs_older_than_n_days_with_sizes_and_references(tmp_path):
    paper, flows = make_tree(tmp_path)
    before = _listing(tmp_path)
    report = rr.retention_report(paper_dir=paper, flow_root=flows, days=30, now=NOW)
    snaps = {s["name"]: s for s in report["snapshots"]["older"]}
    assert set(snaps) == {"20260801T100000000000Z", "20260802T100000000000Z"}
    assert snaps["20260801T100000000000Z"]["bytes"] == 102 and snaps["20260801T100000000000Z"]["files"] == 2
    assert snaps["20260801T100000000000Z"]["referenced_by_paper_record"] is True
    assert snaps["20260802T100000000000Z"]["referenced_by_paper_record"] is False
    assert report["snapshots"]["older_count"] == 2 and report["snapshots"]["older_bytes"] == 154
    assert report["snapshots"]["total_count"] == 3
    runs = {r["name"]: r for r in report["flow_runs"]["older"]}
    assert set(runs) == {"20260810T100000000000Z_aaaa1111", "20260811T100000000000Z_bbbb2222"}
    assert runs["20260811T100000000000Z_bbbb2222"]["has_manifest"] is False
    assert report["flow_runs"]["older_bytes"] == 42 and report["days"] == 30
    assert "nothing deleted" in report["note"]
    assert _listing(tmp_path) == before


def test_cli_prints_report_and_exits_zero(tmp_path, capsys):
    paper, flows = make_tree(tmp_path)
    assert rr.main(["--paper-dir", str(paper), "--flow-root", str(flows), "--days", "0"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["snapshots"]["older_count"] == 3 and out["flow_runs"]["older_count"] == 3


def test_missing_folders_are_empty_not_errors(tmp_path):
    report = rr.retention_report(paper_dir=tmp_path / "none", flow_root=tmp_path / "none2", days=30, now=NOW)
    assert report["snapshots"]["older"] == [] and report["flow_runs"]["total_count"] == 0


def test_module_has_no_deletion_path():
    source = Path(rr.__file__).read_text(encoding="utf-8")
    for verb in ("unlink", "rmtree", "rmdir", "os.remove", "shutil.move", "os.rename", "os.replace"):
        assert verb not in source
