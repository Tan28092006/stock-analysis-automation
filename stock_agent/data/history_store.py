"""Long (default 10-year) daily price history store, separate from the 650-day paper snapshot.

Engineering artifact only. Each fetch writes a NEW immutable directory
`<out>/<UTC>_<id>/` with `raw/`, `prices/`, `issues/` and, last and atomically,
`manifest.json`. A directory without a manifest is an incomplete run and is ignored.

Rules (shared with `reconciliation.parse_vci_history`) are applied per row: invalid
rows are quarantined in `issues/<SYM>.json` with a reason instead of failing the whole
symbol. Rows are never filled, repaired or re-adjusted.

Limits: prices are provider-adjusted at fetch vintage (not point-in-time certified);
the universe is a union of current/recorded memberships (survivorship-biased before
2022); the exchange calendar only verifies sessions from 2021-01-01. Using this store in
research_ml/R5 or research_gate is a data-contract change that requires preregistration.

CLI: python -m stock_agent.data.history_store {fetch,verify,status} --help
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import date, datetime, time as daytime, timedelta, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import time
import uuid

import numpy as np
import pandas as pd
import requests

from . import reconciliation as rec
from .exchange_calendar import (VN_TIMEZONE, completed_session_date, is_trading_day,
                                symbol_trading_days_between, trading_days_between)

REPO = Path(__file__).resolve().parents[2]
DEFAULT_ROOT = Path("data/history")
CURRENT_VN30 = Path("configs/universe_vn30.json")
VN30_HISTORY_CONFIGS = (Path("configs/research/vn30_membership_2022_2026.json"),
                        Path("configs/research/vn30_membership_2025_2026.json"))
VN100_CONFIG = Path("configs/research/vn100_membership_2026_h1.json")
UNIVERSE_SETS = ("vn30", "vn30-history", "vn100", "vnindex")
INDEX_SYMBOLS = {"VNINDEX", "VN30", "HNXINDEX", "UPCOMINDEX"}
SYMBOL_RE = re.compile(r"[A-Z0-9]{2,12}")
FIELD_KEYS = {"date": "t", "open": "o", "high": "h", "low": "l", "close": "c", "volume": "v"}
PRICE_COLUMNS = rec.PRICE_COLUMNS

PACING_SECONDS = 3.1  # same bound as reconciliation: <= 19 requests/minute, every call incl. retries
MAX_ATTEMPTS = 3  # only timeouts/connection errors/5xx are retried; never 429 or other 4xx
MAX_CONSECUTIVE_FAILURES = 5  # stop calling the provider after this many failed symbols in a row
CALENDAR_VERIFIED_FROM = date(2021, 1, 1)  # HOSE_HOLIDAYS coverage starts here
CALENDAR_VERIFIED_TO = date(2026, 12, 31)  # ... and ends here
MISSING_SAMPLE = 20

ADJUSTMENT_BASIS = "provider_adjusted_as_retrieved_unverified"
UNIVERSE_POLICY = ("union_of_current_and_recorded_memberships_not_pit; "
                   "survivorship_biased_before_2022 (no point-in-time membership configs)")
VERIFICATION_SCOPE = ("source bytes, sha256, per-row OHLCV/session validity and calendar coverage only; "
                      "not corporate-action or point-in-time adjustment certification")
RESEARCH_GATE_NOTE = ("Engineering artifact. Not registered for research_ml/R5 or research_gate; "
                      "using it there is a data-contract change that needs preregistration.")


class RateLimited(RuntimeError):
    """Provider returned 429: never retried, and the run stops calling the provider."""


class Pacer:
    """Guarantee at least `interval` seconds between the starts of consecutive calls."""

    def __init__(self, interval: float = PACING_SECONDS, clock=time.monotonic, sleep=time.sleep):
        self.interval, self.clock, self.sleep = interval, clock, sleep
        self.last: float | None = None

    def wait(self) -> None:
        if self.last is not None:
            while (remaining := self.interval - (self.clock() - self.last)) > 0:
                self.sleep(remaining)
        self.last = self.clock()


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _write_bytes(path: Path, data: bytes) -> None:
    with open(path, "xb") as handle:  # never overwrite an existing file
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())


def _atomic_write(path: Path, data: bytes) -> None:
    tmp = path.with_name(path.name + ".tmp")
    try:
        _write_bytes(tmp, data)
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def _json_bytes(value) -> bytes:
    return (json.dumps(value, indent=2, ensure_ascii=False) + "\n").encode("utf-8")


def csv_bytes(frame: pd.DataFrame) -> bytes:
    """Same serialisation as the paper snapshot CSVs."""
    return frame.to_csv(index=False, lineterminator="\n").encode("utf-8")


def _unit(symbol: str) -> str:
    return "index_points" if symbol in INDEX_SYMBOLS else "VND"


# --- universe ------------------------------------------------------------------------------------

def _config_members(path: Path) -> set[str]:
    """Every symbol a dated membership config ever lists (initial, added or removed)."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    members = set(data.get("initial_members", []))
    for change in data.get("changes", []):
        members.update(change.get("add", []))
        members.update(change.get("remove", []))
    return {s.upper() for s in members}


def resolve_universe(sets, extra=(), repo: Path = REPO) -> tuple[list[str], dict[str, list[str]]]:
    """Deduplicated sorted union and, per symbol, the sorted labels that brought it in.

    No historical list is invented: only symbols present in the repo configs are used.
    """
    repo = Path(repo)
    requested: list[str] = []
    for item in sets:
        requested.extend(part.strip().lower() for part in str(item).split(",") if part.strip())
    if "all" in requested:
        requested = [s for s in requested if s != "all"] + list(UNIVERSE_SETS)
    unknown = sorted(set(requested) - set(UNIVERSE_SETS))
    if unknown or not requested:
        raise ValueError(f"unknown universe set(s) {unknown}; choose from all,{','.join(UNIVERSE_SETS)}")
    sources: dict[str, set[str]] = {}
    if "vn30" in requested:
        current = json.loads((repo / CURRENT_VN30).read_text(encoding="utf-8"))["symbols"]
        sources["vn30"] = {s.upper() for s in current}
    if "vn30-history" in requested:
        sources["vn30-history"] = set().union(*(_config_members(repo / p) for p in VN30_HISTORY_CONFIGS))
    if "vn100" in requested:
        sources["vn100"] = _config_members(repo / VN100_CONFIG)
    if "vnindex" in requested:
        sources["vnindex"] = {"VNINDEX"}
    if extra:
        sources["extra"] = {str(s).strip().upper() for s in extra}
    labels: dict[str, list[str]] = {}
    for label, members in sources.items():
        for symbol in members:
            if not SYMBOL_RE.fullmatch(symbol):
                raise ValueError(f"invalid market symbol {symbol!r} in {label}")
            labels.setdefault(symbol, []).append(label)
    symbols = sorted(labels)
    return symbols, {s: sorted(labels[s]) for s in symbols}


# --- non-strict parse ----------------------------------------------------------------------------

def parse_history_rows(payload: object, symbol: str, start: date, as_of: date) -> tuple[pd.DataFrame, dict]:
    """Return (valid rows, issues). Same rules as `reconciliation.parse_vci_history`, per row.

    A structural problem (wrong/ambiguous symbol, missing or ragged arrays) or zero valid
    rows raises ValueError: the symbol fails. Invalid rows are quarantined with reasons
    and their raw values; valid rows are returned unchanged. Rows outside [start, as_of]
    (e.g. an intraday bar or extra countBack history) are counted, not quarantined.
    """
    records = payload.get("data", []) if isinstance(payload, dict) else payload
    if (not isinstance(records, list) or len(records) != 1 or not isinstance(records[0], dict)
            or records[0].get("symbol") != symbol):
        raise ValueError(f"{symbol}: missing or ambiguous VCI symbol response")
    item = records[0]
    columns = {}
    for name, key in FIELD_KEYS.items():
        if not isinstance(item.get(key), list):
            raise ValueError(f"{symbol}: VCI field {key!r} missing or not a list")
        columns[name] = item[key]
    if len({len(v) for v in columns.values()}) != 1:
        raise ValueError(f"{symbol}: VCI arrays have different lengths")
    raw = pd.DataFrame(columns, dtype=object)
    seconds = pd.to_numeric(raw["date"], errors="coerce")
    stamps = pd.to_datetime(seconds, unit="s", utc=True, errors="coerce")
    bad_time = stamps.isna()
    dates = pd.Series([None] * len(raw), index=raw.index, dtype=object)
    dates[~bad_time] = stamps[~bad_time].dt.tz_convert(VN_TIMEZONE).dt.date
    in_window = ~bad_time & dates.map(lambda d: d is not None and start <= d <= as_of)
    excluded = int((~bad_time & ~in_window).sum())

    window = raw.loc[in_window].copy()
    window["date"] = dates[in_window]
    values = window[PRICE_COLUMNS].apply(pd.to_numeric, errors="coerce").astype(float)
    finite = np.isfinite(values).all(axis=1)
    rules = {
        "non_finite": ~finite,
        "non_positive_price": (values[["open", "high", "low", "close"]] <= 0).any(axis=1),
        "negative_volume": values.volume < 0,
        "high_below_ohlc": values.high < values[["open", "low", "close"]].max(axis=1),
        "low_above_ohlc": values.low > values[["open", "high", "close"]].min(axis=1),
        "duplicate_date": window.date.duplicated(keep=False),
        "non_session": ~window.date.map(is_trading_day).astype(bool),
    }
    reasons: dict[int, list[str]] = {int(i): ["bad_timestamp"] for i in raw.index[bad_time]}
    for name, mask in rules.items():
        for index in window.index[mask.to_numpy(dtype=bool)]:
            reasons.setdefault(int(index), []).append(name)

    good = window.loc[[i for i in window.index if int(i) not in reasons]].copy()
    good[PRICE_COLUMNS] = good[PRICE_COLUMNS].apply(pd.to_numeric, errors="coerce")
    good = good.sort_values("date").reset_index(drop=True)
    if good.empty:
        raise ValueError(f"{symbol}: no valid completed bars in requested interval")
    # Invariant: what survives must pass the strict snapshot rules exactly.
    if rec._invalid(good).any() or not good.date.map(is_trading_day).all():
        raise RuntimeError(f"{symbol}: quarantine rules diverged from reconciliation rules")

    quarantined = []
    for index in sorted(reasons):
        row = raw.loc[index]
        day = dates[index]
        quarantined.append({"row": index, "t": row["date"], "date": None if day is None else str(day),
                            "reasons": reasons[index], **{c: row[c] for c in PRICE_COLUMNS}})
    issues = {"symbol": symbol, "start": str(start), "as_of": str(as_of), "source_rows": len(raw),
              "excluded_out_of_window": excluded, "quarantined": quarantined}
    return good, issues


# --- coverage ------------------------------------------------------------------------------------

def coverage_report(symbol: str, present_dates, as_of: date) -> dict:
    """Expected HOSE sessions vs present valid rows, from the symbol's first valid row.

    Only [CALENDAR_VERIFIED_FROM, CALENDAR_VERIFIED_TO] is checked against the maintained
    holiday table. Outside it, weekday gaps may be holidays: reported as UNVERIFIED, never
    as missing. The first valid row is not a verified listing date.
    """
    present = set(present_dates)
    if not present:
        return {}
    first = min(present)
    report: dict = {"first_present": str(first), "through": str(as_of),
                    "calendar_verified_from": str(CALENDAR_VERIFIED_FROM),
                    "calendar_verified_to": str(CALENDAR_VERIFIED_TO)}
    v_start, v_end = max(first, CALENDAR_VERIFIED_FROM), min(as_of, CALENDAR_VERIFIED_TO)
    if v_start <= v_end:
        expected = symbol_trading_days_between(symbol, v_start, v_end)
        missing = [d for d in expected if d not in present]
        report["verified_calendar"] = {
            "status": "VERIFIED_CALENDAR", "start": str(v_start), "end": str(v_end),
            "expected_sessions": len(expected), "present_sessions": len(expected) - len(missing),
            "missing_sessions": len(missing), "missing_dates_sample": [str(d) for d in missing[:MISSING_SAMPLE]]}

    def unverified(left: date, right: date) -> dict:
        weekdays = [left + timedelta(days=i) for i in range((right - left).days + 1)]
        weekdays = [d for d in weekdays if d.weekday() < 5]
        rows = sum(1 for d in present if left <= d <= right)
        return {"status": "UNVERIFIED", "start": str(left), "end": str(right), "weekdays": len(weekdays),
                "present_rows": rows, "weekday_gaps_unverified": len(weekdays) - rows,
                "note": "no maintained HOSE holiday table here: gaps may be holidays, not missing data"}

    if first < CALENDAR_VERIFIED_FROM:
        report["unverified_pre_calendar"] = unverified(first, min(as_of, CALENDAR_VERIFIED_FROM - timedelta(days=1)))
    if as_of > CALENDAR_VERIFIED_TO:
        report["unverified_post_calendar"] = unverified(max(first, CALENDAR_VERIFIED_TO + timedelta(days=1)), as_of)
    return report


# --- network -------------------------------------------------------------------------------------

def request_payload(symbol: str, start: date, as_of: date) -> dict:
    """Identical request shape to `reconciliation.fetch_vci_history`."""
    return {"timeFrame": "ONE_DAY", "symbols": [symbol],
            "to": int(datetime.combine(as_of + timedelta(days=1), daytime.min, VN_TIMEZONE).timestamp()),
            "countBack": len(pd.bdate_range(start, as_of)) + 5}


def fetch_raw(symbol: str, start: date, as_of: date, *, post, pacer: Pacer, sleep=time.sleep) -> tuple[bytes, dict]:
    """One paced POST per attempt; retry only timeouts/connection errors/5xx (bounded)."""
    payload = request_payload(symbol, start, as_of)
    for attempt in range(1, MAX_ATTEMPTS + 1):
        pacer.wait()
        try:
            response = post(rec.VCI_HISTORY_URL, json=payload, headers=rec.HEADERS, timeout=(10, 30))
        except (requests.Timeout, requests.ConnectionError):
            if attempt == MAX_ATTEMPTS:
                raise
            sleep(PACING_SECONDS * attempt)
            continue
        status = int(getattr(response, "status_code", 0))
        if status == 429:
            raise RateLimited(f"{symbol}: HTTP 429 rate limited (not retried)")
        if status in {500, 502, 503, 504} and attempt < MAX_ATTEMPTS:
            sleep(PACING_SECONDS * attempt)
            continue
        if status != 200:
            raise RuntimeError(f"{symbol}: HTTP {status} (not retried)" if status < 500
                               else f"{symbol}: HTTP {status} after {attempt} attempts")
        return response.content, {"fetched_at": datetime.now(timezone.utc).isoformat(),
                                  "request": payload, "attempts": attempt}
    raise AssertionError("unreachable")


# --- build ---------------------------------------------------------------------------------------

def _counts(entries: dict) -> dict:
    tally = Counter(e["status"] for e in entries.values())
    return {k: tally.get(k, 0) for k in ("ok", "partial", "failed")}


def _overall(counts: dict) -> str:
    total = sum(counts.values())
    if total and counts["ok"] == total:
        return "ok"
    return "failed" if counts["failed"] == total else "partial"


def build_store(symbols, out_root: Path, *, start: date, as_of: date, now: datetime | None = None,
                post=None, clock=time.monotonic, sleep=time.sleep, store_id: str | None = None,
                universe: dict | None = None, years: float | None = None) -> tuple[Path, dict]:
    """Fetch every symbol into a NEW directory; manifest.json is written last, atomically."""
    now = now or datetime.now(timezone.utc)
    symbols = [str(s).upper() for s in symbols]
    if not symbols or len(set(symbols)) != len(symbols) or any(not SYMBOL_RE.fullmatch(s) for s in symbols):
        raise ValueError("history store requires distinct valid symbols")
    if as_of > completed_session_date(now) or not is_trading_day(as_of):
        raise ValueError("as_of must be a completed trading session")
    if start > as_of:
        raise ValueError("start is after as_of")
    stamp = now.astimezone(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    store = Path(out_root).resolve() / f"{stamp}_{store_id or uuid.uuid4().hex[:8]}"
    store.parent.mkdir(parents=True, exist_ok=True)
    store.mkdir(exist_ok=False)
    for sub in ("raw", "prices", "issues"):
        (store / sub).mkdir()

    post = post or requests.post
    pacer = Pacer(PACING_SECONDS, clock=clock, sleep=sleep)
    entries: dict[str, dict] = {}
    abort_reason, consecutive_failures = None, 0
    for symbol in symbols:
        entry: dict = {"status": "failed", "unit": _unit(symbol), "error": None}
        entries[symbol] = entry
        if abort_reason:
            entry["error"] = f"not attempted: {abort_reason}"
            continue
        try:
            raw, meta = fetch_raw(symbol, start, as_of, post=post, pacer=pacer, sleep=sleep)
        except RateLimited as exc:
            entry["error"] = f"{type(exc).__name__}: {exc}"
            abort_reason = "provider rate limit (HTTP 429) earlier in this run"
            continue
        except Exception as exc:  # network/HTTP failure: this symbol only
            entry["error"] = f"{type(exc).__name__}: {exc}"
            consecutive_failures += 1
            if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                abort_reason = f"{MAX_CONSECUTIVE_FAILURES} consecutive fetch failures"
            continue
        consecutive_failures = 0
        entry.update(meta)
        entry["raw_path"] = f"raw/{symbol}.json"
        entry["raw_sha256"] = _sha(raw)
        _write_bytes(store / entry["raw_path"], raw)
        try:
            frame, issues = parse_history_rows(json.loads(raw), symbol, start, as_of)
        except Exception as exc:  # structural/empty response: raw kept for forensics
            entry["error"] = f"{type(exc).__name__}: {exc}"
            continue
        data, issue_bytes = csv_bytes(frame), _json_bytes(issues)
        entry.update({"path": f"prices/{symbol}.csv", "sha256": _sha(data),
                      "issues_path": f"issues/{symbol}.json", "issues_sha256": _sha(issue_bytes)})
        _write_bytes(store / entry["path"], data)
        _write_bytes(store / entry["issues_path"], issue_bytes)
        reasons = Counter(r for q in issues["quarantined"] for r in q["reasons"])
        entry.update({"status": "partial" if issues["quarantined"] else "ok",
                      "rows": len(frame), "first_date": str(frame.date.iloc[0]), "last_date": str(frame.date.iloc[-1]),
                      "ends_before_as_of": frame.date.iloc[-1] < as_of,
                      "quarantined_rows": len(issues["quarantined"]), "quarantine_reasons": dict(sorted(reasons.items())),
                      "excluded_out_of_window": issues["excluded_out_of_window"],
                      "coverage": coverage_report(symbol, frame.date.tolist(), as_of)})

    counts = _counts(entries)
    manifest = {"schema_version": 1, "kind": "price_history_store", "store_id": store.name,
                "status": _overall(counts), "counts": counts, "source": "VCI", "endpoint": rec.VCI_HISTORY_URL,
                "created_at": now.astimezone(timezone.utc).isoformat(),
                "completed_at": datetime.now(timezone.utc).isoformat(),
                "start": str(start), "as_of": str(as_of), "years": years,
                "universe": universe or {"symbols": symbols}, "symbols": entries,
                "pacing_seconds": PACING_SECONDS, "max_attempts": MAX_ATTEMPTS, "aborted": abort_reason,
                "adjustment_basis": ADJUSTMENT_BASIS, "universe_policy": UNIVERSE_POLICY,
                "verification_scope": VERIFICATION_SCOPE, "research_gate": RESEARCH_GATE_NOTE}
    _atomic_write(store / "manifest.json", _json_bytes(manifest))
    return store, manifest


# --- verify / status -----------------------------------------------------------------------------

def _resolve_manifest(path: Path) -> Path:
    path = Path(path).resolve()
    return path / "manifest.json" if path.is_dir() else path


def verify_store(path: Path) -> dict:
    """Re-hash every file and re-derive CSV + issues from the preserved raw bytes."""
    manifest_path = _resolve_manifest(path)
    root = manifest_path.parent
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("kind") != "price_history_store":
        raise ValueError("not a price history store manifest")
    if manifest.get("counts") != _counts(manifest["symbols"]):
        raise ValueError("manifest counts do not match symbol entries")
    start, as_of = date.fromisoformat(manifest["start"]), date.fromisoformat(manifest["as_of"])
    checked = 0
    for symbol, entry in manifest["symbols"].items():
        for key, hash_key in (("raw_path", "raw_sha256"), ("path", "sha256"), ("issues_path", "issues_sha256")):
            if key not in entry:
                continue
            file = (root / entry[key]).resolve()
            if not file.is_relative_to(root):
                raise ValueError(f"{symbol}: path escaped the store root")
            if _sha(file.read_bytes()) != entry[hash_key]:
                raise ValueError(f"{symbol}: {key} hash mismatch")
        if entry["status"] == "failed":
            continue
        frame, issues = parse_history_rows(json.loads((root / entry["raw_path"]).read_bytes()), symbol, start, as_of)
        if _sha(csv_bytes(frame)) != entry["sha256"] or _sha(_json_bytes(issues)) != entry["issues_sha256"]:
            raise ValueError(f"{symbol}: CSV/issues vs raw source mismatch")
        expected_status = "partial" if issues["quarantined"] else "ok"
        if (entry["status"], entry["rows"], entry["quarantined_rows"]) != (expected_status, len(frame), len(issues["quarantined"])):
            raise ValueError(f"{symbol}: manifest row counts/status disagree with raw source")
        checked += 1
    return {"store": str(root), "status": manifest["status"], "counts": manifest["counts"],
            "symbols_checked": checked, "as_of": manifest["as_of"], "verified": True}


def store_status(root: Path = DEFAULT_ROOT, now: datetime | None = None) -> dict:
    """Newest complete store (by as_of, then completed_at) and its staleness. No hashing here."""
    now = now or datetime.now(timezone.utc)
    completed = completed_session_date(now)
    root = Path(root)
    best, key, incomplete = None, None, []
    for directory in sorted(p for p in root.iterdir() if p.is_dir()) if root.is_dir() else []:
        manifest_path = directory / "manifest.json"
        if not manifest_path.is_file():
            incomplete.append(directory.name)
            continue
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            incomplete.append(directory.name)
            continue
        if manifest.get("kind") != "price_history_store":
            continue
        k = (str(manifest.get("as_of")), str(manifest.get("completed_at")))
        if key is None or k > key:
            best, key = (directory, manifest), k
    report = {"root": str(root), "completed_session": str(completed), "incomplete_dirs": incomplete}
    if best is None:
        return {**report, "store": None, "stale": True, "reason": "no complete history store"}
    directory, manifest = best
    as_of = date.fromisoformat(manifest["as_of"])
    behind = len(trading_days_between(as_of + timedelta(days=1), completed))
    return {**report, "store": str(directory), "as_of": manifest["as_of"], "start": manifest.get("start"),
            "status": manifest.get("status"), "counts": manifest.get("counts"),
            "failed_symbols": sorted(s for s, e in manifest["symbols"].items() if e["status"] == "failed"),
            "partial_symbols": sorted(s for s, e in manifest["symbols"].items() if e["status"] == "partial"),
            "sessions_behind": behind, "stale": behind > 0,
            "note": "status does not re-hash files; run `verify <dir>` before relying on contents"}


# --- CLI -----------------------------------------------------------------------------------------

def _years_before(day: date, years: float) -> date:
    whole = int(years)
    try:
        anchor = day.replace(year=day.year - whole)
    except ValueError:  # 29 February
        anchor = day.replace(year=day.year - whole, day=28)
    return anchor - timedelta(days=round((years - whole) * 365.25))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="python -m stock_agent.data.history_store", description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    fetch = sub.add_parser("fetch", help="fetch a new immutable history store (network)")
    fetch.add_argument("--years", type=float, default=10.0)
    fetch.add_argument("--universe", nargs="+", default=["all"],
                       help=f"all or any of {', '.join(UNIVERSE_SETS)} (space/comma separated)")
    fetch.add_argument("--symbols", nargs="*", default=[], help="extra symbols, e.g. holdings PNJ TPB")
    fetch.add_argument("--out", type=Path, default=DEFAULT_ROOT)
    fetch.add_argument("--as-of", type=date.fromisoformat, default=None, help="default: completed session")
    fetch.add_argument("--dry-run", action="store_true", help="print universe and duration; no network")
    verify = sub.add_parser("verify", help="re-hash and re-parse a store directory")
    verify.add_argument("store", type=Path)
    status = sub.add_parser("status", help="newest store, counts and staleness")
    status.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    args = parser.parse_args(argv)

    if args.command == "fetch":
        if not 0 < args.years <= 30:
            parser.error("--years must be in (0, 30]")
        symbols, labels = resolve_universe(args.universe, extra=args.symbols)
        as_of = args.as_of or completed_session_date()
        start = _years_before(as_of, args.years)
        universe = {"sets": args.universe, "extra": sorted({s.upper() for s in args.symbols}),
                    "symbols": symbols, "membership_of": labels}
        if args.dry_run:
            print(json.dumps({"dry_run": True, "start": str(start), "as_of": str(as_of), "count": len(symbols),
                              "symbols": symbols, "estimated_seconds_min": round(len(symbols) * PACING_SECONDS, 1)},
                             indent=2))
            return 0
        store, manifest = build_store(symbols, args.out, start=start, as_of=as_of, universe=universe, years=args.years)
        print(json.dumps({"store": str(store), "status": manifest["status"], "counts": manifest["counts"],
                          "aborted": manifest["aborted"],
                          "partial": {s: e["quarantine_reasons"] for s, e in manifest["symbols"].items()
                                      if e["status"] == "partial"},
                          "failed": {s: e["error"] for s, e in manifest["symbols"].items()
                                     if e["status"] == "failed"}}, indent=2, ensure_ascii=False))
        return 1 if manifest["counts"]["failed"] else 0
    if args.command == "verify":
        try:
            print(json.dumps(verify_store(args.store), indent=2))
        except (ValueError, OSError, KeyError) as exc:
            print(json.dumps({"verified": False, "error": f"{type(exc).__name__}: {exc}"}, indent=2))
            return 1
        return 0
    report = store_status(args.root)
    print(json.dumps(report, indent=2))
    return 1 if report["stale"] else 0


if __name__ == "__main__":
    sys.exit(main())
