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


# ------------------------------------------------------------------ refresh --held (bottom-fish)

HELD_HISTORY_DAYS = 550   # about 370 sessions: 250-session high + 60-session beta window + warm-up


def held_outside_vn30(paths: DeskPaths) -> tuple[list[str], list[dict]]:
    """Open journal positions whose symbol is not in the VN30 universe (read-only)."""
    from .journal import load_journal
    journal = load_journal(paths.portfolio)
    if journal is None:
        return [], []
    universe = set(load_universe_symbols(paths))
    return sorted(t for t, p in journal.positions.items() if p.get("qty") and t not in universe), journal.errors


def refresh_held(paths: DeskPaths, *, now: datetime, fetcher=None, announce=None) -> dict:
    """Isolated, verified VCI snapshot for held symbols outside VN30 (+VNINDEX) for check/review only.

    Writes a NEW directory under swing_desk/snapshots/extra/<UTC>/ through
    reconciliation.build_market_snapshot (same VCI fetcher and rate limit); never writes `data/`,
    never changes schedules. These symbols can only get HOLD/SELL/AVOID, never BUY (BF-G1).
    """
    from datetime import timedelta, timezone
    from ..data.reconciliation import build_market_snapshot
    expected = completed_session_date(now)
    symbols, errors = held_outside_vn30(paths)
    report = {"expected_session": str(expected), "held_outside_vn30": symbols, "journal_errors": errors,
              "action": None, "snapshot": None}
    if not symbols:
        report["action"] = "nothing to fetch (no open position outside VN30 in swing_desk/portfolio.txt)"
        return report
    fresh = {s: dl.extra_freshness(paths, now, s) for s in symbols}
    if all(f["ok"] for f in fresh.values()):
        report.update(action="reused", snapshot=sorted({f["manifest"] for f in fresh.values()}))
        return report
    request = symbols + ["VNINDEX"]
    out = paths.extra_snapshots / now.astimezone(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    if announce:
        announce(f"refresh --held: will fetch {', '.join(request)} from VCI for {expected} into {out} "
                 "(new directory; data/ and schedules untouched)")
    manifest = build_market_snapshot(request, out, expected - timedelta(days=HELD_HISTORY_DAYS), expected,
                                     fetcher=fetcher)
    report.update(action="fetched" if manifest["status"] == "verified" else "blocked",
                  snapshot=str(out / "manifest.json"), status=manifest["status"], failures=manifest["failures"],
                  requested=request)
    return report
