"""On-demand refresh: reuse a fresh verified snapshot or fetch one into the desk's isolated
output dir through the existing paper_runner (preview only), optionally collect foreign flows
through the existing collector (outside 16:30-18:30 VN only), optionally (`--history`) fetch and
verify a long history store through `stock_agent.data.history_store`, then build desk state
atomically and report the history-store status and per-ticker long-window source.

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


HISTORY_YEARS = 10
HISTORY_IN_PROGRESS_MINUTES = 60   # a store dir without manifest younger than this is a running fetch


def held_symbols(paths: DeskPaths) -> list[str]:
    """Every open journal position (portfolio.txt), VN30 or not."""
    from .journal import load_journal
    journal = load_journal(paths.portfolio) if paths.portfolio.is_file() else None
    return sorted(t for t, p in (journal.positions if journal else {}).items() if p.get("qty"))


def history_fetch_command(paths: DeskPaths, as_of: str, held: list[str], python: str = sys.executable) -> list[str]:
    cmd = [python, "-m", "stock_agent.data.history_store", "fetch", "--years", str(HISTORY_YEARS),
           "--universe", "vn30", "vnindex"]
    if held:
        cmd += ["--symbols", *held]
    return cmd + ["--as-of", as_of, "--out", str(paths.history_root)]


def history_verify_command(store: Path, python: str = sys.executable) -> list[str]:
    return [python, "-m", "stock_agent.data.history_store", "verify", str(store)]


def _running_fetches(paths: DeskPaths, now: datetime) -> list[str]:
    from datetime import timezone
    out = []
    root = paths.history_root
    for d in (root.iterdir() if root.is_dir() else []):
        if not d.is_dir() or (d / "manifest.json").is_file():
            continue
        try:
            started = datetime.strptime(d.name.split("_")[0], "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc)
        except ValueError:
            continue
        if 0 <= (now - started).total_seconds() < HISTORY_IN_PROGRESS_MINUTES * 60:
            out.append(d.name)
    return sorted(out)


def history_needs_fetch(paths: DeskPaths, expected: str, symbols: list[str]) -> tuple[bool, str, str | None]:
    """(needed, reason, covering store). Fresh = verified store(s) for the completed session hold every
    required symbol with status ok/partial (per symbol, as the desk selects them)."""
    from .history import HistoryResolver
    res = HistoryResolver(paths, expected)
    stores = [s for s in res.stores() if res._ok(s)]
    if not stores:
        skipped = f" (failed verify: {[s['store_id'] for s in res.skipped]})" if res.skipped else ""
        return True, f"no verified history store with as_of {expected}{skipped}", None
    missing = [s for s in symbols if not any(st["manifest"]["symbols"].get(s, {}).get("status") in ("ok", "partial")
                                             for st in stores)]
    if missing:
        return True, f"not ok/partial in any verified store for {expected}: {missing}", stores[0]["store_id"]
    return False, f"verified store(s) for {expected} cover {len(symbols)} symbols", stores[0]["store_id"]


def fetch_history_store(paths: DeskPaths, *, now: datetime, expected: str, runner=run_process,
                  python: str = sys.executable) -> dict:
    """`refresh --history`: fetch VN30 + VNINDEX + held into a NEW store when stale, then `verify` it.

    history_store paces every VCI call (>= 3.1 s) and stops on HTTP 429; no schedule is changed. A store
    directory without a manifest younger than an hour is treated as a running fetch: no second fetch.
    """
    held = held_symbols(paths)
    symbols = sorted(set(load_universe_symbols(paths)) | {"VNINDEX"} | set(held))
    needed, reason, store = history_needs_fetch(paths, expected, symbols)
    out = {"action": None, "reason": reason, "store": store, "symbols": symbols, "held": held}
    if not needed:
        out["action"] = "fresh"
        return out
    running = _running_fetches(paths, now)
    if running:
        out.update(action="skipped_in_progress",
                   reason=f"{reason}; a fetch appears to be running ({running}); not starting a second one")
        return out
    root = paths.history_root
    before = {p.name for p in root.iterdir()} if root.is_dir() else set()
    cmd = history_fetch_command(paths, expected, held, python)
    out["fetch_command"] = cmd
    out["fetch_exit_code"] = runner(cmd, paths.repo)    # 1 when some symbol failed: the store may still be usable
    new = sorted(p for p in root.iterdir()
                 if p.name not in before and (p / "manifest.json").is_file()) if root.is_dir() else []
    if not new:
        out.update(action="fetch_failed", store=store)
        return out
    out["store"] = new[-1].name
    vcmd = history_verify_command(new[-1], python)
    out["verify_command"] = vcmd
    out["verify_exit_code"] = runner(vcmd, paths.repo)
    out["action"] = "fetched" if out["verify_exit_code"] == 0 else "fetched_verify_failed"
    return out


def history_report(paths: DeskPaths, now: datetime, snapshot_as_of: str | None, coverage: dict | None) -> dict:
    """History-store line + per-ticker long-window source (from the state just built)."""
    from .history import store_line
    line = store_line(paths, now, snapshot_as_of)
    line["coverage"] = coverage
    fallbacks = sorted(t for t, c in (coverage or {}).items() if c.get("history_source") != "history_store")
    line["fallbacks"] = fallbacks
    if line["stale"] or not line["matches_snapshot_as_of"] or fallbacks:
        line["hint"] = ("long windows fall back to the 650-day snapshot for " + (", ".join(fallbacks) or "all tickers")
                        + "; run `refresh --history` to fetch VN30 + VNINDEX + held (paced VCI calls)")
    return line


def _flows_status(paths: DeskPaths, now: datetime) -> str:
    from ..data import foreign_flows as ff
    if not paths.flows_root.is_dir():
        return "UNVERIFIED"
    try:
        return ff.health(load_universe_symbols(paths), root=paths.flows_root, now=now)["status"]
    except Exception:
        return "UNVERIFIED"


def refresh(paths: DeskPaths, *, now: datetime, with_flows: bool = False, runner=run_process,
            python: str = sys.executable, build_state: bool = True, s2_override=None,
            fetch_history: bool = False) -> dict:
    expected = str(completed_session_date(now))
    report = {"expected_session": expected, "snapshot": None, "snapshot_action": None,
              "flows_action": None, "history_fetch": None, "state": None}
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

    if fetch_history:
        report["history_fetch"] = fetch_history_store(paths, now=now, expected=expected, runner=runner,
                                                    python=python)

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
            report["history"] = history_report(paths, now, None, None)
            return report
        from . import review, service
        state = service.compute_state(paths, now, s2_override=s2_override)
        extra = {}
        if paths.portfolio.is_file():
            extra["portfolio_review.md"] = review.build_review(paths, now=now)["markdown"]
        report["state"] = str(service.write_state(paths, state, extra_files=extra))
        report["history"] = history_report(paths, now, state["manifest"]["as_of"],
                                           state["manifest"]["history"]["per_ticker"])
    else:
        report["history"] = history_report(paths, now, expected if manifest is not None else None, None)
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
