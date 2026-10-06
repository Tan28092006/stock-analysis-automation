"""Locate and verify the newest desk snapshot; load prices and flows (read-only).

Primary prices: newest *verified* VCI snapshot under `data/paper/snapshots/` or the desk's
isolated `swing_desk/snapshots/snapshots/`. Legacy `data/raw/prices_hist` is never used for
the live desk; `legacy_prices_freshness` only reports how stale it is. Legacy foreign rows
(`data/raw/foreign`) are quarantined and never read.
"""
from __future__ import annotations

from datetime import datetime
import hashlib
import json
import math
from pathlib import Path

import pandas as pd

from ..data.exchange_calendar import completed_session_date
from ..data.reconciliation import verify_snapshot
from .config import DeskPaths, load_universe_symbols


def _manifests(paths: DeskPaths) -> list[Path]:
    found = []
    for root in (paths.paper_snapshots, paths.desk_snapshots):
        if root.is_dir():
            found.extend(p for p in root.glob("*/manifest.json") if p.is_file())
    return found


def newest_verified_manifest(paths: DeskPaths, *, as_of: str | None = None) -> Path | None:
    """Newest manifest with status=verified (by as_of, then fetched_at). Hashes not checked here."""
    best, key = None, None
    for path in _manifests(paths):
        try:
            m = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if m.get("kind") != "market_snapshot" or m.get("status") != "verified":
            continue
        if as_of is not None and m.get("as_of") != as_of:
            continue
        k = (str(m.get("as_of")), str(m.get("completed_at") or m.get("fetched_at")))
        if key is None or k > key:
            best, key = path, k
    return best


def manifest_sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def freshness(paths: DeskPaths, now: datetime) -> dict:
    """G0: newest verified snapshot re-verified now, as_of == completed session, 30 VN30 + VNINDEX."""
    expected = str(completed_session_date(now))
    out = {"ok": False, "expected_session": expected, "manifest": None, "as_of": None,
           "manifest_sha256": None, "reason": ""}
    path = newest_verified_manifest(paths)
    if path is None:
        out["reason"] = "no verified snapshot found; run `refresh`"
        return out
    out["manifest"] = str(path)
    out["manifest_sha256"] = manifest_sha256(path)
    try:
        manifest = verify_snapshot(path)
    except Exception as exc:  # hash mismatch, missing file, bad source bytes
        out["reason"] = f"snapshot failed re-verification: {exc}"
        return out
    out["as_of"] = manifest["as_of"]
    for stamp in [manifest.get("completed_at"), manifest.get("fetched_at")]:
        if stamp and datetime.fromisoformat(stamp) > now:
            out["reason"] = "snapshot fetched after the stated time (clock/look-ahead)"
            return out
    universe = set(load_universe_symbols(paths)) | {"VNINDEX"}
    if set(manifest["files"]) != universe:
        missing = sorted(universe - set(manifest["files"]))
        extra = sorted(set(manifest["files"]) - universe)
        out["reason"] = f"snapshot universe mismatch (missing {missing}, extra {extra})"
        return out
    if manifest["as_of"] != expected:
        out["reason"] = f"stale: snapshot as_of {manifest['as_of']} != completed session {expected}; run `refresh`"
        return out
    out["ok"] = True
    out["reason"] = f"verified snapshot as_of {manifest['as_of']}"
    out["file_sha256"] = {s: e["sha256"] for s, e in manifest["files"].items()}
    out["fetched_at"] = manifest.get("fetched_at")
    return out


def prices_dir(manifest_path: Path) -> Path:
    return Path(manifest_path).parent / "prices"


def load_frames(manifest_path: Path) -> dict[str, pd.DataFrame]:
    frames = {}
    for csv in sorted(prices_dir(manifest_path).glob("*.csv")):
        df = pd.read_csv(csv)
        df["date"] = pd.to_datetime(df["date"].astype(str).str[:10]).dt.date
        frames[csv.stem] = df.sort_values("date").reset_index(drop=True)
    return frames


def close_lookup(frames: dict[str, pd.DataFrame]):
    def lookup(ticker, day):
        df = frames.get(ticker)
        if df is None:
            return None
        row = df[df["date"] == day]
        return float(row["close"].iloc[0]) if len(row) else None
    return lookup


def s2_symbols(manifest_path: Path, as_of: str) -> tuple[list[str], str]:
    """VN30 MR gate buys for session `as_of`, reusing mr_scan (never re-implemented)."""
    from ..features import mr_scan
    try:
        res = mr_scan._compute(0, .55, prices_dir=prices_dir(manifest_path), use_model=False,
                               include_positions=False)
    except Exception as exc:
        return [], f"UNVERIFIED: mr_scan failed ({type(exc).__name__})"
    if res.get("data_date") != as_of:
        return [], f"UNVERIFIED: mr_scan data_date {res.get('data_date')} != {as_of}"
    return sorted({b["symbol"] for b in res["buys"] if b.get("date") == as_of}), "OK"


def market_summary(manifest_path: Path, frames: dict[str, pd.DataFrame], universe: list[str], cfg: dict) -> dict:
    """Regime/breadth exactly as mr_scan._market_state + VNINDEX stats and limit counts."""
    from ..features import mr_scan
    from .indicators import compute_ticker
    eod = {s: df.assign(date=df["date"].astype(str)) for s, df in frames.items() if s != "VNINDEX"}
    state = mr_scan._market_state(prices_dir(manifest_path), eod)
    out = {"regime": state["state"], "state": state["state"], "breadth_pct": state["breadth_pct"],
           "index_close": state["index_close"], "index_ema50": state["index_ema50"], "date": state["date"]}
    idx = frames.get("VNINDEX")
    if idx is not None and len(idx) > 20:
        ind = compute_ticker(idx)
        out.update(index_ret_1d=ind["ret_1d"], index_ret_5d=ind["ret_5d"], index_atr_pct=ind["atr_pct"])
    lim = cfg["gate"]["limit_proxy_pct"]
    ups = downs = 0
    for s in universe:
        df = frames.get(s)
        if df is None or len(df) < 2:
            continue
        r = (df["close"].iloc[-1] / df["close"].iloc[-2] - 1) * 100
        ups += r >= lim
        downs += r <= -lim
    out.update(limit_up_count=int(ups), limit_down_count=int(downs))
    return out


def flows_snapshot(paths: DeskPaths, symbols: list[str], now: datetime, as_of_time: str | None):
    """(health dict, wide flows frame or None). Read-only; errors -> UNVERIFIED, never zero."""
    from ..data import foreign_flows as ff
    if not paths.flows_root.is_dir():
        return {"status": "UNVERIFIED", "reason": "no data/foreign_flows_v2 store"}, None
    try:
        health = ff.health(symbols, root=paths.flows_root, now=now)
        flows = ff.load_flows(root=paths.flows_root, as_of=as_of_time or now)
    except Exception as exc:
        return {"status": "UNVERIFIED", "reason": f"{type(exc).__name__}: {exc}"}, None
    slim = {k: health.get(k) for k in ("status", "expected_session", "rows", "collection_in_progress",
                                       "incomplete_runs", "strategy_approved")}
    slim["missing_latest_count"] = {k: len(v) for k, v in (health.get("missing_latest") or {}).items()}
    return slim, flows


def legacy_prices_freshness(paths: DeskPaths, now: datetime) -> dict:
    path = paths.data_root / "data" / "raw" / "prices_hist" / "VNINDEX.csv"
    if not path.is_file():
        return {"present": False}
    try:
        last = pd.read_csv(path, usecols=["date"])["date"].astype(str).str[:10].max()
    except Exception as exc:
        return {"present": True, "error": type(exc).__name__}
    expected = str(completed_session_date(now))
    return {"present": True, "latest": last, "stale": last != expected,
            "warning": "legacy prices_hist is NOT used by the desk" + (f"; stale since {last}" if last != expected else "")}


def finite_or_none(value):
    return value if isinstance(value, (int, float)) and math.isfinite(value) else None
