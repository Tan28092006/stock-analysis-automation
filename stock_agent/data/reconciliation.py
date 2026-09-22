"""Immutable, source-backed market snapshots; never silently repair raw history.

`verified` is a transport/schema/freshness verdict, NOT proof of historical
point-in-time adjustments, membership, execution quality, or model performance.
"""
from __future__ import annotations

from datetime import date, datetime, time as daytime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import re
import time

import numpy as np
import pandas as pd
import requests

from .exchange_calendar import VN_TIMEZONE, completed_session_date, is_trading_day

VCI_HISTORY_URL = "https://trading.vietcap.com.vn/api/chart/OHLCChart/gap-chart"
PRICE_COLUMNS = ["open", "high", "low", "close", "volume"]
HEADERS = {"Content-Type": "application/json", "Origin": "https://trading.vietcap.com.vn",
           "Referer": "https://trading.vietcap.com.vn/", "User-Agent": "Mozilla/5.0"}


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _invalid(frame: pd.DataFrame) -> pd.Series:
    values = frame[PRICE_COLUMNS].apply(pd.to_numeric, errors="coerce")
    return (~np.isfinite(values).all(axis=1)
            | (values[["open", "high", "low", "close"]] <= 0).any(axis=1)
            | (values.volume < 0)
            | (values.high < values[["open", "low", "close"]].max(axis=1))
            | (values.low > values[["open", "high", "close"]].min(axis=1))
            | frame.date.duplicated(keep=False))


def parse_vci_history(payload: object, symbol: str, start: date, as_of: date) -> pd.DataFrame:
    """VCI native values are VND for stocks and points for indices: no guessing."""
    records = payload.get("data", []) if isinstance(payload, dict) else payload
    if not isinstance(records, list) or len(records) != 1 or records[0].get("symbol") != symbol:
        raise ValueError(f"{symbol}: missing or ambiguous VCI symbol response")
    item = records[0]
    frame = pd.DataFrame({name: item[key] for name, key in
                          {"date": "t", "open": "o", "high": "h", "low": "l", "close": "c", "volume": "v"}.items()})
    frame["date"] = pd.to_datetime(pd.to_numeric(frame.date, errors="raise"), unit="s", utc=True).dt.tz_convert(VN_TIMEZONE).dt.date
    frame = frame.loc[(frame.date >= start) & (frame.date <= as_of)].copy()
    frame[PRICE_COLUMNS] = frame[PRICE_COLUMNS].apply(pd.to_numeric, errors="coerce")
    if frame.empty:
        raise ValueError(f"{symbol}: no completed bars in requested interval")
    bad = _invalid(frame) | ~frame.date.map(is_trading_day)
    if bad.any():
        raise ValueError(f"{symbol}: invalid source OHLCV/session rows {frame.loc[bad, 'date'].astype(str).tolist()[:8]}")
    return frame.sort_values("date").reset_index(drop=True)


def fetch_vci_history(symbol: str, start: date, as_of: date) -> tuple[pd.DataFrame, dict]:
    """Public read-only query, bounded timeout, no shared provider/cache mutation."""
    symbol = symbol.upper()
    if not re.fullmatch(r"[A-Z0-9]{2,12}", symbol):
        raise ValueError("Invalid market symbol")
    cutoff = min(as_of, completed_session_date())
    if start > cutoff:
        raise ValueError("start is after completed-session cutoff")
    payload = {"timeFrame": "ONE_DAY", "symbols": [symbol],
               "to": int(datetime.combine(cutoff + timedelta(days=1), daytime.min, VN_TIMEZONE).timestamp()),
               "countBack": len(pd.bdate_range(start, cutoff)) + 5}
    response = requests.post(VCI_HISTORY_URL, json=payload, headers=HEADERS, timeout=(10, 30))
    response.raise_for_status()
    fetched_at = datetime.now(timezone.utc).isoformat()
    raw = response.content
    return parse_vci_history(json.loads(raw), symbol, start, cutoff), {
        "source": "VCI", "endpoint": VCI_HISTORY_URL, "fetched_at": fetched_at,
        "request": payload, "raw_bytes": raw, "raw_sha256": _sha(raw)}


def build_market_snapshot(symbols: list[str], output_dir: Path, start: date, as_of: date,
                          min_rows: int = 120, fetcher=None) -> dict:
    """Write to a NEW directory only. Any failed symbol blocks the snapshot."""
    if as_of > completed_session_date() or not is_trading_day(as_of):
        raise ValueError("as_of must be a completed trading session")
    symbols = [s.upper() for s in symbols]
    if not symbols or len(set(symbols)) != len(symbols) or any(not re.fullmatch(r"[A-Z0-9]{2,12}", s) for s in symbols):
        raise ValueError("Snapshot requires distinct valid symbols")
    if min_rows < 1 or start > as_of:
        raise ValueError("Invalid snapshot interval or minimum rows")
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "prices").mkdir()
    (output_dir / "raw").mkdir()
    manifest = {"schema_version": 1, "kind": "market_snapshot", "status": "blocked", "source": "VCI",
                "fetched_at": datetime.now(timezone.utc).isoformat(), "as_of": str(as_of), "start": str(start),
                "price_dir": str(output_dir / "prices"), "symbols": symbols, "min_rows": min_rows,
                "files": {}, "failures": [], "adjustment_basis": "provider_adjusted_as_retrieved_unverified",
                "universe_policy": "current_fixed_universe_not_historical_pit",
                "verification_scope": "source bytes, hash, OHLCV validity and completed-session freshness only"}
    fetch = fetcher or fetch_vci_history
    for index, symbol in enumerate(symbols):
        if fetcher is None and index:
            time.sleep(3.1)  # bounded 19 req/min; never evade provider limits
        try:
            frame, source = fetch(symbol, start, as_of)
            raw = source["raw_bytes"]
            # Always validate the preserved source bytes, not a fetcher's silently repaired frame.
            checked = parse_vci_history(json.loads(raw), symbol, start, as_of)
            pd.testing.assert_frame_equal(frame.reset_index(drop=True), checked, check_dtype=False)
            raw_path = f"raw/{symbol}.json"
            (output_dir / raw_path).write_bytes(raw)
            if len(frame) < min_rows or frame.date.iloc[-1] != as_of:
                raise ValueError(f"{symbol}: insufficient or stale bars (rows={len(frame)}, latest={frame.date.iloc[-1]})")
            csv_bytes = frame.to_csv(index=False, lineterminator="\n").encode("utf-8")
            relative_path = f"prices/{symbol}.csv"
            (output_dir / relative_path).write_bytes(csv_bytes)
            manifest["files"][symbol] = {"path": relative_path, "sha256": _sha(csv_bytes), "rows": len(frame),
                "start": str(frame.date.iloc[0]), "end": str(frame.date.iloc[-1]),
                "unit": "index_points" if symbol in {"VNINDEX", "VN30", "HNXINDEX", "UPCOMINDEX"} else "VND",
                "raw_path": raw_path, "raw_sha256": _sha(raw), "fetched_at": source["fetched_at"],
                "endpoint": source.get("endpoint", VCI_HISTORY_URL), "request": source.get("request")}
        except (Exception,) as exc:
            manifest["failures"].append({"symbol": symbol, "error": f"{type(exc).__name__}: {exc}"})
    manifest["status"] = "blocked" if manifest["failures"] else "verified"
    manifest["completed_at"] = datetime.now(timezone.utc).isoformat()
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return manifest


def verify_snapshot(manifest_path: Path) -> dict:
    """Validate snapshot contents without trusting mutable absolute price_dir."""
    manifest_path = Path(manifest_path).resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("kind") != "market_snapshot" or manifest.get("status") != "verified":
        raise ValueError("Snapshot is blocked or not a market snapshot")
    if set(manifest["files"]) != set(manifest["symbols"]):
        raise ValueError("Incomplete snapshot symbols")
    for symbol, entry in manifest["files"].items():
        for key, hash_key in (("path", "sha256"), ("raw_path", "raw_sha256")):
            path = (manifest_path.parent / entry[key]).resolve()
            if not path.is_relative_to(manifest_path.parent):
                raise ValueError("Snapshot path escaped its root")
            if _sha(path.read_bytes()) != entry[hash_key]:
                raise ValueError(f"{symbol}: snapshot hash mismatch")
        frame = parse_vci_history(json.loads((manifest_path.parent / entry["raw_path"]).read_bytes()), symbol,
                                  date.fromisoformat(manifest["start"]), date.fromisoformat(manifest["as_of"]))
        if _sha(frame.to_csv(index=False, lineterminator="\n").encode("utf-8")) != entry["sha256"]:
            raise ValueError(f"{symbol}: CSV/source mismatch")
        if len(frame) < manifest["min_rows"] or str(frame.date.iloc[-1]) != manifest["as_of"]:
            raise ValueError(f"{symbol}: snapshot is stale or insufficient")
    return manifest


def reconcile_invalid_rows(original: pd.DataFrame, source: pd.DataFrame) -> tuple[pd.DataFrame, list[dict]]:
    """Return staged copy only; require exact neighboring source basis agreement.

    No prices are synthesized, no valid original bars are replaced, and original
    rows remain quarantined when both immediate neighbors cannot be verified.
    """
    candidate = original.copy(deep=True)
    bad = _invalid(original)
    decisions = []
    for position in np.flatnonzero(bad.to_numpy()):
        original_index = original.index[position]
        day = original.loc[original_index, "date"]
        replacement = source.loc[source.date == day]
        neighbors = [position - 1, position + 1]
        verified = len(replacement) == 1 and not _invalid(replacement).any()
        for neighbor in neighbors:
            if neighbor < 0 or neighbor >= len(original) or bad.iloc[neighbor]:
                verified = False
                continue
            row = original.iloc[neighbor]
            match = source.loc[source.date == row.date]
            if len(match) != 1 or not np.allclose(row[PRICE_COLUMNS].astype(float), match.iloc[0][PRICE_COLUMNS].astype(float), rtol=0, atol=1e-8):
                verified = False
        if verified:
            candidate.loc[original_index, PRICE_COLUMNS] = replacement.iloc[0][PRICE_COLUMNS].to_numpy()
        decisions.append({"date": str(day), "status": "staged_verified_neighbors" if verified else "quarantine",
                          "reason": "source row + both neighbors agree" if verified else "source missing/invalid or adjustment basis unverified"})
    return candidate, decisions
