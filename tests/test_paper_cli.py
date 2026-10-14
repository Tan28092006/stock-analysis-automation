from datetime import datetime, timezone
import json
import subprocess
import sys

import pytest

from stock_agent.pipeline import paper_runner as r
from tests.test_market_runtime_hardening import make_snapshot


@pytest.fixture
def setup(tmp_path, monkeypatch):
    manifest = make_snapshot(tmp_path / "snapshot")
    monkeypatch.setattr(r, "load_universe", lambda: {"symbols": ["AAA"]})
    real_now = r._now
    monkeypatch.setattr(r, "_now", lambda value=None: real_now(value or datetime(2026, 9, 21, 10, tzinfo=timezone.utc)))
    return manifest, tmp_path / "out"


def test_cli_run_scores_without_touching_record_on_retry(setup):
    manifest, out = setup
    args = ["--manifest", str(manifest), "--output-dir", str(out), "--run"]
    assert r.main(args) == 0
    paper = out / "runs/2026-09-21/paper.json"
    before = paper.read_bytes()
    assert json.loads(before)["status"] == "recorded"
    assert r.main(args) == 0
    assert paper.read_bytes() == before
    scores = json.loads((out / "scores/latest.json").read_text(encoding="utf-8"))
    assert scores["status"] == "ok" and scores["pending"] >= 1


def test_cli_score_only_preserves_scan_report(setup):
    manifest, out = setup
    args = ["--manifest", str(manifest), "--output-dir", str(out)]
    assert r.main(args + ["--check"]) == 0
    before = (out / "latest.json").read_bytes()
    assert r.main(args + ["--score"]) == 0
    assert (out / "latest.json").read_bytes() == before


def test_cli_preview_window_exit_and_source_failures(setup, monkeypatch):
    manifest, out = setup
    monkeypatch.setattr(r, "_now", lambda value=None: value or datetime(2026, 9, 22, 4, tzinfo=timezone.utc))
    args = ["--manifest", str(manifest), "--output-dir", str(out)]
    assert r.main(args + ["--run"]) == 3
    assert r.main(args + ["--check"]) == 0
    assert r.main(args + ["--run", "--check"]) == 2
    assert r.main(["--prices-dir", str(manifest.parent / "prices"), "--output-dir", str(out)]) == 2
    assert r.main(args + ["--refresh"]) == 2


def test_cli_refresh_blocked_and_success(setup, monkeypatch):
    manifest, out = setup
    monkeypatch.setattr(r, "build_market_snapshot", lambda *a, **kw: {"status": "blocked", "failures": ["offline"]})
    assert r.main(["--refresh", "--run", "--output-dir", str(out)]) == 2
    assert json.loads((out / "latest.json").read_text())["status"] == "failed"
    def refresh(symbols, folder, **kwargs):
        make_snapshot(folder)
        return {"status": "verified"}
    monkeypatch.setattr(r, "build_market_snapshot", refresh)
    assert r.main(["--refresh", "--run", "--output-dir", str(out)]) == 0


def test_score_errors_are_visible_without_overwriting_scan(setup):
    manifest, out = setup
    args = ["--manifest", str(manifest), "--output-dir", str(out)]
    assert r.main(args + ["--run"]) == 0
    (out / "runs/2026-09-21/paper.json").write_text("{")
    assert r.main(args + ["--score"]) == 2
    assert r.main(["--score", "--output-dir", str(out)]) == 2


def test_corrupted_existing_record_cannot_pass_idempotency(setup):
    manifest, out = setup
    now = datetime(2026, 9, 21, 10, tzinfo=timezone.utc)
    payload = r.run_paper(manifest.parent / "prices", ["AAA"], manifest_path=manifest, now=now)
    r.commit_paper_run(payload, out, now=now)
    path = out / "runs/2026-09-21/paper.json"
    previous = json.loads(path.read_text(encoding="utf-8"))
    previous["recommendations"] = []
    path.write_text(json.dumps(previous))
    with pytest.raises(ValueError, match="corrupt"):
        r.commit_paper_run(payload, out, now=now)


def test_cli_scores_rolled_source_without_rewriting_signal(tmp_path):
    from tests.test_paper_scoring import record, rolling_snapshot
    record(tmp_path, track="momentum")
    out = tmp_path / "out"
    scan = out / "latest.json"
    scan.write_text('{"scan": "preserve-existing-report"}', encoding="utf-8")
    paper = out / "runs/2026-09-21/paper.json"
    before_scan, before_paper = scan.read_bytes(), paper.read_bytes()
    current = rolling_snapshot(tmp_path / "current")
    assert r.main(["--manifest", str(current), "--output-dir", str(out), "--score"]) == 0
    assert scan.read_bytes() == before_scan and paper.read_bytes() == before_paper
    scores = json.loads((out / "scores/latest.json").read_text(encoding="utf-8"))
    assert scores["pending"] == 1 and scores["resolved"] == 0
    assert scores["records"][0]["history_check"]["archived_prefix_rows_not_reobserved"] == 2


@pytest.mark.parametrize("mode", ["--score", "--run", "subprocess"])
def test_cli_exposes_volume_revisions_without_recomputing_old_signal(tmp_path, monkeypatch, capsys, mode):
    from tests.test_paper_scoring import record, revise_volume, rolling_snapshot
    record(tmp_path, track="momentum")
    current = rolling_snapshot(tmp_path / "current", transform=lambda rows: revise_volume(rows, 999999))
    out = tmp_path / "out"
    scan = out / "latest.json"
    scan.write_text('{"scan":"keep"}', encoding="utf-8")
    paper = out / "runs/2026-09-21/paper.json"
    before_scan, before_paper = scan.read_bytes(), paper.read_bytes()

    def no_scan(*args, **kwargs):
        pytest.fail("Score-only must not run a signal scan")

    # --run exercises CLI integration with an already-completed synthetic run.
    completed = {"status": "recorded", "source_verified": True}
    monkeypatch.setattr(r, "run_paper", (lambda *a, **kw: completed) if mode == "--run" else no_scan)
    args = ["--manifest", str(current), "--output-dir", str(out)]
    if mode == "subprocess":
        process = subprocess.run([sys.executable, "-m", "stock_agent.pipeline.paper_runner", *args, "--score"],
                                 text=True, capture_output=True, timeout=60)
        assert process.returncode == 0, process.stderr + process.stdout
        printed = process.stdout
    else:
        assert r.main(args + [mode]) == 0
        printed = capsys.readouterr().out
    assert "ok_with_revisions" in printed and "volume_revision_rows" in printed
    scores = json.loads((out / "scores/latest.json").read_text(encoding="utf-8"))
    assert scores["status"] == "ok_with_revisions" and scores["volume_revision_rows"] == 1
    assert scores["pending"] == 1 and scores["resolved"] == 0
    assert paper.read_bytes() == before_paper
    if mode != "--run":
        assert scan.read_bytes() == before_scan


@pytest.mark.parametrize("mode", ["--score", "--run"])
def test_cli_still_blocks_shared_price_revision(tmp_path, monkeypatch, mode):
    from tests.test_paper_scoring import record, rolling_snapshot
    record(tmp_path)

    def revise_price(rows):
        if rows[0]["symbol"] == "AAA":
            rows[0]["c"][-5] += 1
        return rows

    current = rolling_snapshot(tmp_path / "current", transform=revise_price)
    monkeypatch.setattr(r, "run_paper", lambda *a, **kw: {"status": "recorded", "source_verified": True})
    assert r.main(["--manifest", str(current), "--output-dir", str(tmp_path / "out"), mode]) == 2
    result = json.loads((tmp_path / "out/scores/latest.json").read_text(encoding="utf-8"))
    assert result["status"] == "blocked" and result["errors"] and not result["records"]
