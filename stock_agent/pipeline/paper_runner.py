"""Source-audited EOD observation runner. No broker, legacy ledger or model writes."""
from __future__ import annotations

import argparse
from datetime import date, datetime, time, timedelta, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import uuid

import pandas as pd

from ..config import load_universe
from ..data.exchange_calendar import (
    VN_TIMEZONE, completed_session_date, is_trading_day, next_trading_day,
    symbol_trading_days_between,
)
from ..data.reconciliation import _invalid, build_market_snapshot, verify_snapshot
from ..features import mr_scan, momentum_scan

DEFAULT_OUTPUT = Path("data/paper")


def _now(value: datetime | None = None) -> datetime:
    value = value or datetime.now(timezone.utc)
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("now must be timezone-aware")
    return value


def _digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def _window(session: date, now: datetime) -> bool:
    return (is_trading_day(session)
            and datetime.combine(session, time(16), VN_TIMEZONE) <= now
            < datetime.combine(next_trading_day(session), time(9), VN_TIMEZONE))


def assess_market_data(prices_dir: Path, symbols: list[str], *, now=None) -> dict:
    now = _now(now)
    session = completed_session_date(now)
    issues, coverage = {}, {}
    if not symbols or len(set(symbols)) != len(symbols) or any(
            not re.fullmatch(r"[A-Z0-9]{2,12}", s) or s == "VNINDEX" for s in symbols):
        raise ValueError("A distinct stock universe is required")
    expected = set(symbols) | {"VNINDEX"}
    extras = {p.stem for p in Path(prices_dir).glob("*.csv")} - expected
    if extras:
        issues["universe"] = [f"unexpected symbols: {sorted(extras)}"]
    for symbol in sorted(expected):
        try:
            frame = pd.read_csv(Path(prices_dir) / f"{symbol}.csv")
            frame["date"] = pd.to_datetime(frame.date, format="%Y-%m-%d", errors="raise").dt.date
            if frame.empty or len(frame) < 90:
                raise ValueError("insufficient history: at least 90 completed bars required")
            if _invalid(frame).any() or not frame.date.is_monotonic_increasing:
                raise ValueError("invalid or unordered OHLCV")
            if any(not is_trading_day(d) or d > session for d in frame.date):
                raise ValueError("non-trading or incomplete/future session")
            if frame.date.iloc[-1] != session:
                raise ValueError(f"stale: latest {frame.date.iloc[-1]}, expected {session}")
            if set(symbol_trading_days_between(symbol, frame.date.iloc[0], session)) - set(frame.date):
                raise ValueError("missing internal trading sessions; investigate listing/suspension/source")
            if float(frame.volume.iloc[-1]) <= 0:
                raise ValueError("latest session has no executable volume")
            coverage[symbol] = {"rows": len(frame), "start": str(frame.date.iloc[0]),
                                "end": str(frame.date.iloc[-1])}
        except Exception as exc:
            issues[symbol] = [str(exc)]
    return {"session": str(session), "data_ready": not issues, "issues": issues,
            "coverage": coverage, "universe": sorted(symbols),
            "recording_window_open": _window(session, now)}


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, ensure_ascii=False, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def commit_paper_run(payload: dict, output_dir: Path, *, now=None) -> dict:
    """One immutable session record; conflicting reruns fail, never overwrite."""
    now = _now(now)
    if payload.get("mode") != "paper" or not payload.get("data_ready") or not payload.get("source_verified"):
        raise ValueError("Only source-verified, data-ready paper runs may be recorded")
    session = date.fromisoformat(payload["session"])
    if not _window(session, now):
        raise ValueError("Outside ex-ante recording window; preview only")
    if not payload.get("snapshot_manifest"):
        raise ValueError("Source manifest is required at the recording boundary")
    manifest_path = Path(payload["snapshot_manifest"])
    checked = _source(manifest_path.parent / "prices", manifest_path,
                      payload.get("universe", []), payload["session"], now=now)
    if any(checked[k] != payload.get(k) for k in checked):
        raise ValueError("Paper run conflict: verified source identity changed")
    identity = _digest({k: payload.get(k) for k in (
        "session", "input_snapshot", "rules_hash", "code_hash", "universe", "recommendations", "model")})
    folder = Path(output_dir) / "runs" / str(session)
    folder.mkdir(parents=True, exist_ok=True)
    lock = folder / ".write-lock"
    try:
        lock.mkdir()
    except FileExistsError as exc:
        raise ValueError("Concurrent or interrupted paper writer; investigate lock before retry") from exc
    path = folder / "paper.json"
    try:
        if path.exists():
            previous = json.loads(path.read_text(encoding="utf-8"))
            if previous.get("record_identity") != identity:
                raise ValueError("Paper run conflict: session already contains different inputs or decisions")
            return {"status": "already_recorded", "path": str(path.resolve())}
        record = {**payload, "recorded_at": now.isoformat(), "record_identity": identity,
                  "recording_contract": "signal_day_16VN_to_next_session_09VN_v1"}
        _atomic_json(path, record)
        return {"status": "recorded", "path": str(path.resolve())}
    finally:
        lock.rmdir()


def _source(prices_dir: Path, manifest_path: Path | None, symbols: list[str], session: str, *, now=None) -> dict:
    if manifest_path is None:
        raise ValueError("No source manifest: unverified local cache is preview-only")
    manifest_path = Path(manifest_path).resolve()
    manifest = verify_snapshot(manifest_path)
    for stamp in [manifest["fetched_at"], manifest["completed_at"],
                  *(item["fetched_at"] for item in manifest["files"].values())]:
        if _now(datetime.fromisoformat(stamp)) > _now(now):
            raise ValueError("Source fetched after the claimed prediction time")
    if Path(prices_dir).resolve() != manifest_path.parent / "prices":
        raise ValueError("Manifest is not bound to the supplied prices directory")
    if set(manifest["symbols"]) != set(symbols) | {"VNINDEX"} or manifest["as_of"] != session:
        raise ValueError("Manifest universe/as-of mismatch")
    return {"source_verified": True, "snapshot_manifest": str(manifest_path),
            "snapshot_manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
            "input_snapshot": _digest({s: item["sha256"] for s, item in manifest["files"].items()}),
            "source": manifest["source"], "adjustment_basis": manifest["adjustment_basis"],
            "universe_policy": manifest["universe_policy"]}


def run_paper(prices_dir: Path, symbols: list[str], *, output_dir: Path = DEFAULT_OUTPUT,
              manifest_path: Path | None = None, record: bool = False, now=None) -> dict:
    fixed_clock = now is not None
    now = _now(now)
    readiness = assess_market_data(prices_dir, symbols, now=now)
    result = {**readiness, "mode": "paper", "generated_at": now.isoformat(),
              "source_verified": False, "status": "blocked", "live_orders_enabled": False}
    try:
        result.update(_source(prices_dir, manifest_path, symbols, readiness["session"], now=now))
    except Exception as exc:
        result["source_error"] = str(exc)
    if not readiness["data_ready"]:
        return result
    mr = mr_scan._compute(0, .55, prices_dir=Path(prices_dir), use_model=False, include_positions=False)
    momentum = momentum_scan._compute(10, prices_dir=Path(prices_dir), include_positions=False)
    if (mr["scan_errors"] or set(mr["scanned_symbols"]) != set(symbols)
            or set(momentum["scanned_symbols"]) != set(symbols)
            or mr["data_date"] != readiness["session"] or momentum["data_date"] != readiness["session"]):
        result["issues"]["scan"] = ["Scanner failed or silently omitted requested symbols", mr["scan_errors"]]
        result["data_ready"] = False
        return result
    sources = sorted(Path("stock_agent/features").glob("*.py")) + [
        Path(__file__), Path("stock_agent/data/exchange_calendar.py"), Path("configs/universe_vn30.json")]
    result.update({"rules_hash": mr["rules_hash"],
                   "code_hash": _digest({str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}),
                   "model": {"mode": "rules_only", "legacy_artifact_loaded": False},
                   "mr": mr, "momentum": momentum,
                   "recommendations": ([{"track": "mr", **p} for p in mr["buys"]]
                                       + [{"track": "momentum", "date": readiness["session"], **p}
                                          for p in momentum["picks"]]),
                   "status": "preview_only",
                   "limitations": ["Paper signals, no orders or verified execution/profitability.",
                                   "Momentum returns are informational, not a funded portfolio.",
                                   "Current fixed universe; historical adjustment/PIT lineage unverified."]})
    if result["source_verified"]:
        checked = _source(prices_dir, manifest_path, symbols, readiness["session"], now=now)
        if checked != {key: result[key] for key in checked}:
            raise ValueError("Source changed during scan")
    if record and result["source_verified"] and readiness["recording_window_open"]:
        result.update(commit_paper_run(result, output_dir, now=now if fixed_clock else _now()))
    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refresh", action="store_true", help="Fetch a NEW source-backed snapshot")
    parser.add_argument("--run", action="store_true", help="Record only within the ex-ante EOD window")
    parser.add_argument("--check", action="store_true", help="Check/preview without recording")
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--prices-dir", type=Path)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    try:
        if args.check and args.run:
            raise ValueError("--check and --run are mutually exclusive")
        symbols = list(load_universe()["symbols"])
        manifest = args.manifest
        prices_dir = args.prices_dir
        if args.refresh:
            if manifest or prices_dir:
                raise ValueError("--refresh cannot reuse supplied source paths")
            session = completed_session_date()
            folder = args.output_dir / "snapshots" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
            print(f"Fetching {len(symbols) + 1} symbols through {session}", flush=True)
            snapshot = build_market_snapshot(symbols + ["VNINDEX"], folder,
                                             start=session - timedelta(days=650), as_of=session)
            manifest = folder / "manifest.json"
            if snapshot["status"] != "verified":
                raise ValueError(f"Snapshot blocked: {snapshot['failures']}; inspect {manifest}")
        prices_dir = prices_dir or (manifest.parent / "prices" if manifest else Path("data/raw/prices_hist"))
        result = run_paper(prices_dir, symbols, output_dir=args.output_dir,
                           manifest_path=manifest, record=args.run)
        _atomic_json(args.output_dir / "latest.json", result)
        print(json.dumps({k: result.get(k) for k in (
            "status", "session", "data_ready", "source_verified", "recording_window_open", "path", "issues")},
                         ensure_ascii=False), flush=True)
        if result["status"] == "blocked" or not result["source_verified"]:
            return 2
        return 3 if args.run and result["status"] == "preview_only" else 0
    except Exception as exc:
        _atomic_json(args.output_dir / "latest.json", {"status": "failed", "error": str(exc),
                                                      "generated_at": _now().isoformat()})
        print(f"FAILED: {exc}", flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
