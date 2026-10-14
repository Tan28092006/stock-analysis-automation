from pathlib import Path
import shutil
import subprocess

import pytest

from stock_agent.swing_desk.config import DeskPaths
from tests.swing_desk_support import NOW_EOD
from tests.test_swing_desk_check import ready, setup_desk

REPO = Path(__file__).resolve().parents[1]
JOURNAL = ("2026-08-03|NAV||987654321||||||\n"
           "2026-08-04|BUY|BBB||20000|1000|S1|PASS|calm||secret-note-xyz\n"
           "2026-08-07|SELL|BBB||21000|1000|target||calm||\n"
           "2026-08-10|BUY|VCB||21000|1000|forum|NONE|fomo||\n"
           "2026-08-14|SELL|VCB||20000|1000|stop||fear||\n"
           "2026-09-17|BUY|AAA||70000|500|S1|WARN|hope||\n"
           "2026-09-18|BUY|AAA||69000|200|S1|WARN|hope||\n")


@pytest.mark.skipif(shutil.which("git") is None, reason="git not installed")
@pytest.mark.parametrize("path,ignored", [
    ("swing_desk/portfolio.txt", True), ("swing_desk/notes.txt", True),
    ("swing_desk/state/2026-09-21/market.json", True), ("swing_desk/snapshots/x/manifest.json", True),
    ("swing_desk/checks/checks.jsonl", True), ("swing_desk/recommendations/recommendations.jsonl", True),
    ("swing_desk/sentiment/2026-09-21/a1.md", True), ("swing_desk/runs/2026-09-21/MISSION.md", True),
    ("swing_desk/events/events.json", True), ("swing_desk/logs/refresh.log", True),
    ("swing_desk/portfolio.example.txt", False), ("swing_desk/README.md", False),
])
def test_private_runtime_paths_are_git_ignored(path, ignored):
    result = subprocess.run(["git", "check-ignore", "-q", path], cwd=REPO)
    assert (result.returncode == 0) is ignored, path


def test_review_groups_by_gate_and_emotion_without_nav_or_raw_lines(tmp_path, monkeypatch):
    from stock_agent.swing_desk import review, service
    paths, _ = setup_desk(tmp_path, monkeypatch)
    ready(paths, JOURNAL)
    rep = review.build_review(paths, now=NOW_EOD)
    text = rep["markdown"]
    assert "987.654.321" not in text and "987,654,321" not in text and "987654321" not in text
    assert "secret-note-xyz" not in text and "|BUY|" not in text
    assert rep["by_gate"]["PASS"]["n"] == 1 and rep["by_gate"]["NONE"]["n"] == 1
    assert rep["by_emotion"]["fomo"]["n"] == 1 and rep["by_emotion"]["calm"]["win_rate"] == 100.0
    assert rep["activity"]["2026-08"]["new_entries"] == 2
    assert rep["activity"]["2026-09"]["new_entries"] == 1 and rep["activity"]["2026-09"]["add_on_buys"] == 1
    pos = rep["open_positions"][0]
    assert pos["ticker"] == "AAA" and pos["qty"] == 700 and "sellable" in pos
    shown = review.build_review(paths, now=NOW_EOD, show_nav=True)["markdown"]
    assert "NAV: 987." in shown and "NAV: 987." not in text and "987.490" not in text


def test_state_files_never_contain_journal_lines(tmp_path, monkeypatch):
    from stock_agent.swing_desk import review, service
    paths, _ = setup_desk(tmp_path, monkeypatch)
    ready(paths, JOURNAL)
    state = service.compute_state(paths, NOW_EOD, s2_override=[], with_flows_data=False)
    rep = review.build_review(paths, now=NOW_EOD)
    folder = service.write_state(paths, state, extra_files={"portfolio_review.md": rep["markdown"]})
    for f in folder.rglob("*"):
        if f.is_file() and f.name != "portfolio_review.md":
            body = f.read_text(encoding="utf-8")
            assert "secret-note-xyz" not in body and "987654321" not in body, f
