"""On-demand refresh: reuse a fresh verified snapshot or fetch one into the desk's isolated
output dir through the existing paper_runner (preview only), optionally collect foreign flows
through the existing collector (outside 16:30-18:30 VN only), then build desk state atomically.

Never passes `--run` (no paper record), never edits schedules/BAT files, never writes
`data/paper/runs` or `data/raw/prices_hist`.
"""
from __future__ import annotations

from datetime import datetime, time
from pathlib import Path
import subprocess
import sys

from ..data.exchange_calendar import VN_TIMEZONE, completed_session_date
from ..data.reconciliation import verify_snapshot
from . import data as dl
from .config import DeskPaths, load_universe_symbols

FLOW_BLACKOUT = (time(16, 30), time(18, 30))   # scheduled 17:05 job holds .collect-lock


def run_process(cmd: list[str], cwd: Path) -> int:
    return subprocess.run(cmd, cwd=cwd).returncode


def in_flow_blackout(now: datetime) -> bool:
    local = now.astimezone(VN_TIMEZONE).time()
    return FLOW_BLACKOUT[0] <= local < FLOW_BLACKOUT[1]


def snapshot_command(paths: DeskPaths, python: str = sys.executable) -> list[str]:
    return [python, "-m", "stock_agent.pipeline.paper_runner", "--refresh", "--check",
            "--output-dir", str(paths.desk / "snapshots")]


def flows_command(paths: DeskPaths, python: str = sys.executable) -> list[str]:
    return [python, "-m", "stock_agent.pipeline.foreign_refresh", "--root", str(paths.flows_root)]


def _flows_status(paths: DeskPaths, now: datetime) -> str:
    from ..data import foreign_flows as ff
    if not paths.flows_root.is_dir():
        return "UNVERIFIED"
    try:
        return ff.health(load_universe_symbols(paths), root=paths.flows_root, now=now)["status"]
    except Exception:
        return "UNVERIFIED"


def refresh(paths: DeskPaths, *, now: datetime, with_flows: bool = False, runner=run_process,
            python: str = sys.executable, build_state: bool = True, s2_override=None) -> dict:
    expected = str(completed_session_date(now))
    report = {"expected_session": expected, "snapshot": None, "snapshot_action": None,
              "flows_action": None, "state": None}
    manifest = dl.newest_verified_manifest(paths, as_of=expected)
    if manifest is not None:
        try:
            verify_snapshot(manifest)
            report["snapshot_action"] = "reused"
        except Exception as exc:
            report["snapshot_reuse_error"] = str(exc)
            manifest = None
    if manifest is None:
        cmd = snapshot_command(paths, python)
        report["snapshot_command"] = cmd
        report["snapshot_exit_code"] = runner(cmd, paths.repo)
        manifest = dl.newest_verified_manifest(paths, as_of=expected)
        report["snapshot_action"] = "fetched" if manifest is not None else "failed"
    report["snapshot"] = str(manifest) if manifest else None

    status = _flows_status(paths, now)
    report["flows_status_before"] = status
    if status == "ready":
        report["flows_action"] = "ready"
    elif not with_flows:
        report["flows_action"] = "not_requested (flows stay UNVERIFIED/stale; pass --with-flows to collect)"
    elif in_flow_blackout(now):
        report["flows_action"] = "skipped: 16:30-18:30 VN blackout (scheduled collection window)"
    else:
        cmd = flows_command(paths, python)
        report["flows_command"] = cmd
        report["flows_exit_code"] = runner(cmd, paths.repo)
        report["flows_action"] = "collected" if report["flows_exit_code"] == 0 else "collector_failed_closed"

    if build_state:
        if manifest is None:
            report["state"] = "not built: no verified snapshot for the completed session"
            return report
        from . import review, service
        state = service.compute_state(paths, now, s2_override=s2_override)
        extra = {}
        if paths.portfolio.is_file():
            extra["portfolio_review.md"] = review.build_review(paths, now=now)["markdown"]
        report["state"] = str(service.write_state(paths, state, extra_files=extra))
    return report
