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
from .config import DeskPaths, load_primary_symbols, load_universe_symbols


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
    primary = set(load_primary_symbols(paths))
    universe = primary | {"VNINDEX"}
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
    others = [s for s in load_universe_symbols(paths) if s not in primary]
    if others:   # extended universe (VN100): per-ticker daily-data status
        tickers = {s: {"ok": True, "source": "primary", "manifest": out["manifest"],
                       "manifest_sha256": out["manifest_sha256"], "reason": out["reason"]}
                   for s in sorted(primary)}
        tickers.update(universe_ticker_freshness(paths, now, others, expected=expected, primary=manifest,
                                                 primary_path=path))
        out["tickers"] = tickers
        for sym, t in tickers.items():
            if t["source"] == "universe" and t["ok"]:
                out["file_sha256"][sym] = t["file_sha256"]
        out["universe_manifests"] = {t["manifest"]: t["manifest_sha256"] for t in tickers.values()
                                     if t["source"] == "universe" and t["ok"]}
    return out


# ------------------------------------------------------------------ universe snapshots (VN100 names outside VN30)

def universe_manifests(paths: DeskPaths) -> list[tuple[Path, dict]]:
    """Verified-status universe snapshots under <desk>/snapshots/universe/, newest first (as_of, completed_at)."""
    root = paths.universe_snapshots
    found = []
    for path in (root.glob("*/manifest.json") if root.is_dir() else []):
        try:
            m = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if m.get("kind") == "market_snapshot" and m.get("status") == "verified":
            found.append((path, m))
    found.sort(key=lambda pm: (str(pm[1].get("as_of")), str(pm[1].get("completed_at") or pm[1].get("fetched_at")),
                               pm[0].parent.name), reverse=True)
    return found


def _index_tail(manifest_path: Path, manifest: dict, n: int = 5) -> list[float] | None:
    entry = manifest["files"].get("VNINDEX")
    if entry is None:
        return None
    df = pd.read_csv(Path(manifest_path).parent / entry["path"])
    return [round(float(v), 6) for v in df["close"].tail(n)]


def universe_ticker_freshness(paths: DeskPaths, now: datetime, symbols: list[str], *, expected: str,
                              primary: dict | None = None, primary_path: Path | None = None) -> dict:
    """Per symbol: the newest verified universe snapshot holding it, re-verified now; ok only when its as_of is
    the completed session, it was not fetched after `now`, and its VNINDEX tail equals the paper snapshot's."""
    manifests = universe_manifests(paths)
    primary_tail = _index_tail(primary_path, primary) if primary is not None and primary_path is not None else None
    checked: dict[str, tuple[str | None, dict | None]] = {}

    def check(path: Path, m: dict) -> str | None:
        key = str(path)
        if key not in checked:
            try:
                verified = verify_snapshot(path)
            except Exception as exc:
                checked[key] = (f"universe snapshot failed re-verification: {exc}", None)
                return checked[key][0]
            reason = None
            if any(s and datetime.fromisoformat(s) > now for s in (verified.get("completed_at"), verified.get("fetched_at"))):
                reason = "universe snapshot fetched after the stated time (clock/look-ahead)"
            elif verified["as_of"] != expected:
                reason = (f"stale: universe snapshot as_of {verified['as_of']} != completed session {expected}; "
                          "run `refresh`")
            elif "VNINDEX" not in verified["files"]:
                reason = "universe snapshot has no VNINDEX (cannot cross-check against the paper snapshot)"
            elif primary_tail is not None and _index_tail(path, verified) != primary_tail:
                reason = "VNINDEX in the universe snapshot differs from the paper snapshot (provider revision?)"
            checked[key] = (reason, verified)
        return checked[key][0]

    out = {}
    for sym in symbols:
        hit = next(((p, m) for p, m in manifests if sym in m.get("files", {})), None)
        base = {"ok": False, "source": "universe", "manifest": None, "manifest_sha256": None, "as_of": None}
        if hit is None:
            out[sym] = {**base, "source": "none",
                        "reason": f"{sym} has no verified daily snapshot (outside the paper VN30 snapshot); run `refresh`"}
            continue
        path, m = hit
        reason = check(path, m)
        entry = {**base, "manifest": str(path), "manifest_sha256": manifest_sha256(path), "as_of": m.get("as_of")}
        if reason:
            out[sym] = {**entry, "reason": reason}
        else:
            out[sym] = {**entry, "ok": True, "reason": f"verified universe snapshot as_of {m['as_of']}",
                        "file_sha256": m["files"][sym]["sha256"]}
    return out


def primary_frames(frames: dict, primary: list[str]) -> dict:
    """The paper-basket frames (+VNINDEX): regime/breadth and limit counts keep their VN30 definition."""
    keep = set(primary) | {"VNINDEX"}
    return {s: f for s, f in frames.items() if s in keep}


def load_desk_frames(fresh: dict) -> dict[str, pd.DataFrame]:
    """Paper snapshot frames plus, with an extended universe, each fresh ticker from its universe snapshot."""
    frames = load_frames(Path(fresh["manifest"]))
    for sym, t in (fresh.get("tickers") or {}).items():
        if t["source"] != "universe" or not t["ok"]:
            continue
        m = json.loads(Path(t["manifest"]).read_text(encoding="utf-8"))
        df = pd.read_csv(Path(t["manifest"]).parent / m["files"][sym]["path"])
        df["date"] = pd.to_datetime(df["date"].astype(str).str[:10]).dt.date
        frames[sym] = df.sort_values("date").reset_index(drop=True)
    return frames


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


# ------------------------------------------------------------------ extra snapshots (`refresh --held`)

def newest_extra_manifest(paths: DeskPaths, ticker: str) -> Path | None:
    """Newest verified extra snapshot (desk-owned, outside data/) that contains `ticker`."""
    root = paths.extra_snapshots
    best, key = None, None
    if not root.is_dir():
        return None
    for path in root.glob("*/manifest.json"):
        try:
            m = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if m.get("kind") != "market_snapshot" or m.get("status") != "verified" or ticker not in m.get("files", {}):
            continue
        k = (str(m.get("as_of")), str(m.get("completed_at") or m.get("fetched_at")))
        if key is None or k > key:
            best, key = path, k
    return best


def extra_freshness(paths: DeskPaths, now: datetime, ticker: str) -> dict:
    """BF-G0 for a held symbol outside VN30: same checks as `freshness` on its extra snapshot."""
    expected = str(completed_session_date(now))
    out = {"ok": False, "expected_session": expected, "manifest": None, "as_of": None,
           "manifest_sha256": None, "reason": "", "source": "extra"}
    path = newest_extra_manifest(paths, ticker)
    if path is None:
        out["reason"] = f"{ticker} is not in the VN30 snapshot and has no verified extra snapshot; run `refresh --held`"
        return out
    out["manifest"] = str(path)
    out["manifest_sha256"] = manifest_sha256(path)
    try:
        manifest = verify_snapshot(path)
    except Exception as exc:
        out["reason"] = f"extra snapshot failed re-verification: {exc}"
        return out
    out["as_of"] = manifest["as_of"]
    for stamp in [manifest.get("completed_at"), manifest.get("fetched_at")]:
        if stamp and datetime.fromisoformat(stamp) > now:
            out["reason"] = "extra snapshot fetched after the stated time (clock/look-ahead)"
            return out
    if "VNINDEX" not in manifest["files"]:
        out["reason"] = "extra snapshot has no VNINDEX (beta / fall type cannot be computed)"
        return out
    if manifest["as_of"] != expected:
        out["reason"] = f"stale: extra snapshot as_of {manifest['as_of']} != completed session {expected}; run `refresh --held`"
        return out
    out.update(ok=True, reason=f"verified extra snapshot as_of {manifest['as_of']}",
               file_sha256={s: e["sha256"] for s, e in manifest["files"].items()})
    return out
