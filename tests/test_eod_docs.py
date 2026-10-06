"""Operator docs must describe the scheduled 17:05 job that actually runs.

Since 7d788cb the launcher runs only the flow collector and the paper runner,
with no Git publishing; eod_update is a manual legacy path.
"""
from __future__ import annotations

from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
DEPLOY = (ROOT / "docs" / "DEPLOY.md").read_text(encoding="utf-8")
RENDER_FREE = (ROOT / "docs" / "RENDER-FREE.md").read_text(encoding="utf-8")


def _section(text: str, number: int) -> str:
    match = re.search(rf"^## {number}\. .*?(?=^## \d+\. |\Z)", text, flags=re.M | re.S)
    assert match, f"section {number} missing"
    return match.group(0)


def test_deploy_section1_names_the_real_scheduled_commands():
    s1 = _section(DEPLOY, 1)
    assert "foreign_refresh --retry-stale" in s1
    assert "paper_runner --refresh --run" in s1
    # eod_update may be listed, but never as the daily scheduled job.
    for line in s1.splitlines():
        if "pipeline.eod_update" in line:
            assert "17:05" not in line and "daily" not in line.lower()
    assert "manual" in s1.lower()
    assert "prices_hist" in s1 and "stale" in s1.lower()


def test_deploy_free_tier_no_longer_promises_git_push_refresh():
    s4 = _section(DEPLOY, 4)
    assert "run_eod_update.bat` + `git push" not in s4
    assert "7d788cb" in s4


def test_deploy_section6_cron_mirrors_the_launcher_not_eod_update():
    s6 = _section(DEPLOY, 6)
    cron_lines = [l for l in s6.splitlines() if re.match(r"^\s*5 10 \* \* 1-5", l)]
    assert cron_lines and all("eod_update " not in l and "eod_update\n" not in l for l in cron_lines)
    assert any("foreign_refresh --retry-stale" in l and "paper_runner --refresh --run" in l for l in cron_lines)
    assert "no git push" in s6.lower() or "không push" in s6.lower()


def test_deploy_has_daily_health_check_runbook():
    assert re.search(r"^##+ .*Daily health check", DEPLOY, flags=re.M)
    block = DEPLOY[DEPLOY.index("Daily health check"):]
    for needle in ("foreign_refresh --status", "eod_update.log", "paper/latest.json"):
        assert needle in block


def test_render_free_states_push_was_removed_and_data_is_frozen():
    assert "File bat giờ có thêm bước `git push`" not in RENDER_FREE
    assert "7d788cb" in RENDER_FREE
    assert "eod_update" in RENDER_FREE
    assert "prices_hist" in RENDER_FREE
