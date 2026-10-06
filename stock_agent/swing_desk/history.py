"""Long-window price series for the desk from the verified 10-year history store (read-only).

The daily verified snapshot stays the freshness authority (G0 / BF-G0) and the source of every
short-window indicator. A history store (`stock_agent.data.history_store`) only supplies the
longer series for long-window fields, and only when, per ticker:

1. mode is "on" (`--history off` restores the snapshot-only behaviour);
2. the store is complete (manifest present), its `as_of` equals the snapshot's `as_of`, and
   `history_store.verify_store` passes (re-hash + re-parse from raw bytes);
3. the ticker's status in that store is ok or partial (quarantined rows stay out and are listed
   as data issues). Selection is per ticker: among qualifying stores a non-aborted run (manifest
   `aborted` empty) is preferred, then the newest; the chosen store_id is recorded per ticker;
4. over the snapshot's window the store has exactly the snapshot's sessions and the same OHLCV
   (exact for VND prices and volume; 1e-6 points for indices).

Otherwise the snapshot series is used with `history_source: "snapshot_650d"` and a WARN note. A
mismatch (e.g. a corporate-action re-adjustment between fetch times) is never stitched: it is
flagged `history_mismatch`. Sources are never mixed inside one series.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .config import DeskPaths

SNAPSHOT_SOURCE = "snapshot_650d"
STORE_SOURCE = "history_store"
MODES = ("on", "off")
FIELDS = ("open", "high", "low", "close", "volume")
PRICE_TOL = {"VND": 0.0, "index_points": 1e-6}   # absolute; volume is always exact
SAMPLE = 10

_VERIFY_CACHE: dict[tuple[str, str], str | None] = {}   # (store dir, manifest sha) -> None (ok) or error


def clear_cache() -> None:
    _VERIFY_CACHE.clear()


def _sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _verify(directory: Path, manifest_sha: str) -> str | None:
    """None when `verify_store` passes, else the error text. Cached per process and manifest hash."""
    from ..data.history_store import verify_store
    key = (str(directory), manifest_sha)
    if key not in _VERIFY_CACHE:
        try:
            verify_store(directory)
            _VERIFY_CACHE[key] = None
        except Exception as exc:   # hash mismatch, missing file, unparseable raw bytes
            _VERIFY_CACHE[key] = f"{type(exc).__name__}: {exc}"
    return _VERIFY_CACHE[key]


def complete_stores(root: Path) -> list[dict]:
    """Every complete price-history store under `root`, newest first (completed_at, then name)."""
    out = []
    root = Path(root)
    if not root.is_dir():
        return out
    for directory in root.iterdir():
        path = directory / "manifest.json"
        if not directory.is_dir() or not path.is_file():
            continue
        try:
            manifest = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if manifest.get("kind") != "price_history_store":
            continue
        out.append({"dir": directory, "store_id": directory.name, "manifest": manifest,
                    "manifest_sha256": _sha(path)})
    out.sort(key=lambda s: (str(s["manifest"].get("completed_at")), s["store_id"]), reverse=True)
    return out


def _read_prices(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"].astype(str).str[:10]).dt.date
    return df.sort_values("date").reset_index(drop=True)


def compare(store_df: pd.DataFrame, snap_df: pd.DataFrame, unit: str = "VND") -> dict | None:
    """None when the store equals the snapshot over the snapshot's window, else a mismatch detail."""
    snap = snap_df.sort_values("date").reset_index(drop=True)
    first = snap["date"].iloc[0]
    last = max(snap["date"].iloc[-1], store_df["date"].iloc[-1])
    window = store_df[(store_df["date"] >= first) & (store_df["date"] <= last)].reset_index(drop=True)
    s_days, h_days = set(snap["date"]), set(window["date"])
    missing, extra = sorted(s_days - h_days), sorted(h_days - s_days)
    detail = {"window": [str(first), str(last)], "snapshot_sessions": len(snap),
              "sessions_missing_in_store": [str(d) for d in missing[:SAMPLE]],
              "sessions_extra_in_store": [str(d) for d in extra[:SAMPLE]], "value_diffs": 0, "first_diff": None}
    joined = snap.merge(window, on="date", suffixes=("_s", "_h"))
    tol = PRICE_TOL.get(unit, 0.0)
    for field in FIELDS:
        a, b = joined[f"{field}_s"].to_numpy(dtype=float), joined[f"{field}_h"].to_numpy(dtype=float)
        bad = ~(np.abs(a - b) <= (0.0 if field == "volume" else tol))
        detail["value_diffs"] += int(bad.sum())
        if bad.any() and detail["first_diff"] is None:
            i = int(np.argmax(bad))
            detail["first_diff"] = {"date": str(joined["date"].iloc[i]), "field": field,
                                    "snapshot": float(a[i]), "store": float(b[i])}
    if missing or extra or detail["value_diffs"]:
        detail["sessions_missing_count"], detail["sessions_extra_count"] = len(missing), len(extra)
        return detail
    return None


class HistoryResolver:
    """Per-ticker long series for one snapshot session. Lazy: a store is verified only when needed."""

    def __init__(self, paths: DeskPaths, as_of: str | None):
        self.paths, self.as_of = paths, as_of
        self.mode = paths.history if paths.history in MODES else "on"
        self._stores: list[dict] | None = None
        self.skipped: list[dict] = []

    def stores(self) -> list[dict]:
        """Complete stores whose as_of equals the snapshot's (not yet verified): non-aborted runs
        first, then newest first. An aborted run (e.g. 5 consecutive fetch failures) is still a
        valid store for the symbols it did fetch, but loses to a complete run of the same session."""
        if self._stores is None:
            matching = [s for s in complete_stores(self.paths.history_root)
                        if self.as_of is not None and s["manifest"].get("as_of") == self.as_of]
            self._stores = sorted(matching, key=lambda s: bool(s["manifest"].get("aborted")))  # stable
        return self._stores

    def _ok(self, store: dict) -> bool:
        error = _verify(store["dir"], store["manifest_sha256"])
        if error and not any(s["store_id"] == store["store_id"] for s in self.skipped):
            self.skipped.append({"store_id": store["store_id"], "reason": f"verify failed: {error}"})
        return error is None

    def _base(self, snap_df: pd.DataFrame) -> dict:
        return {"history_mode": self.mode, "history_source": SNAPSHOT_SOURCE, "history_store_id": None,
                "history_store_manifest_sha256": None, "history_store_as_of": None, "history_store_aborted": None,
                "history_first_date": str(snap_df["date"].iloc[0]) if len(snap_df) else None,
                "history_rows": int(len(snap_df)), "history_mismatch": False, "history_data_issues": [],
                "history_notes": []}

    def series(self, symbol: str, snap_df: pd.DataFrame) -> tuple[pd.DataFrame | None, dict]:
        """(long frame or None, provenance). None means: use the snapshot series for long windows too."""
        prov = self._base(snap_df)
        warn = lambda text: prov["history_notes"].append({"level": "WARN", "text": text})  # noqa: E731
        if self.mode == "off":
            prov["history_notes"].append({"level": "INFO", "text": "history off: long windows use the 650-day snapshot"})
            return None, prov
        stores = [s for s in self.stores() if self._ok(s)]
        if not stores:
            warn(f"no verified history store with as_of {self.as_of} under {self.paths.history_root}; "
                 "long windows use the 650-day snapshot (run `refresh --history`)")
            return None, prov
        chosen = next((s for s in stores if s["manifest"]["symbols"].get(symbol, {}).get("status") in ("ok", "partial")),
                      None)
        if chosen is None:
            states = {s["store_id"]: (s["manifest"]["symbols"].get(symbol) or {}).get("status", "absent") for s in stores}
            what = "failed" if "failed" in states.values() else "not in the store"
            warn(f"{symbol} {what} in every verified history store for {self.as_of} ({states}); "
                 "long windows use the 650-day snapshot")
            return None, prov
        entry = chosen["manifest"]["symbols"][symbol]
        store_df = _read_prices(chosen["dir"] / entry["path"])
        issues = json.loads((chosen["dir"] / entry["issues_path"]).read_text(encoding="utf-8"))
        prov.update(history_store_id=chosen["store_id"], history_store_manifest_sha256=chosen["manifest_sha256"],
                    history_store_as_of=chosen["manifest"]["as_of"],
                    history_store_aborted=chosen["manifest"].get("aborted") or None,
                    history_data_issues=[{"date": q.get("date"), "reasons": q.get("reasons")}
                                         for q in issues.get("quarantined", [])])
        mismatch = compare(store_df, snap_df, entry.get("unit", "VND"))
        if mismatch is not None:
            prov.update(history_mismatch=True, history_mismatch_detail=mismatch)
            warn(f"history_mismatch: store {chosen['store_id']} differs from the snapshot over the snapshot window "
                 f"({mismatch['value_diffs']} value diffs, {mismatch['sessions_missing_count']} sessions missing, "
                 f"{mismatch['sessions_extra_count']} extra; e.g. a re-adjustment between fetch times). "
                 "Not stitched: long windows use the 650-day snapshot")
            return None, prov
        prov.update(history_source=STORE_SOURCE, history_first_date=str(store_df["date"].iloc[0]),
                    history_rows=int(len(store_df)))
        if prov["history_data_issues"]:
            prov["history_notes"].append({"level": "INFO", "text": f"{len(prov['history_data_issues'])} quarantined "
                                          "store row(s) excluded (listed in history_data_issues)"})
        return store_df, prov

    def summary(self, provs: dict[str, dict]) -> dict:
        """State-manifest block: mode, overall source, per-ticker provenance, stores used, skipped stores."""
        sources = {p["history_source"] for p in provs.values()}
        stores = {p["history_store_id"]: p["history_store_manifest_sha256"] for p in provs.values()
                  if p["history_source"] == STORE_SOURCE}
        overall = sources.pop() if len(sources) == 1 else ("per_ticker" if sources else None)
        keys = ("history_source", "history_store_id", "history_store_manifest_sha256", "history_first_date",
                "history_rows", "history_mismatch")
        return {"history_mode": self.mode, "history_source": overall, "history_root": str(self.paths.history_root),
                "snapshot_as_of": self.as_of, "stores": stores, "skipped_stores": list(self.skipped),
                "mismatches": sorted(t for t, p in provs.items() if p["history_mismatch"]),
                "fallbacks": sorted(t for t, p in provs.items() if p["history_source"] != STORE_SOURCE),
                "per_ticker": {t: {k: p.get(k) for k in keys} for t, p in sorted(provs.items())}}


def provenance_fields(prov: dict | None) -> dict:
    """Flat keys copied into ticker JSON, log rows and recommendation rows."""
    prov = prov or {}
    return {k: prov.get(k) for k in ("history_mode", "history_source", "history_store_id",
                                     "history_store_manifest_sha256", "history_first_date", "history_rows",
                                     "history_mismatch")}
