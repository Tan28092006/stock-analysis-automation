from datetime import datetime, timezone
import json

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
