"""Shared, read-only boundary for live EOD consumers (not a training transform)."""
from __future__ import annotations

from datetime import date
import hashlib
from pathlib import Path

import pandas as pd

from . import exchange_calendar as calendar


def completed_bars(frame: pd.DataFrame, end: date | None = None) -> pd.DataFrame:
    cutoff = calendar.completed_session_date()
    if end is not None:
        cutoff = min(cutoff, end)
    dates = pd.to_datetime(frame["date"], errors="coerce").dt.date
    keep = dates.notna() & (dates <= cutoff)
    keep &= dates.map(lambda d: calendar.is_trading_day(d) if pd.notna(d) else False)
    return frame.loc[keep].sort_values("date").reset_index(drop=True).copy()


def read_eod_csv(path: Path, end: date | None = None) -> pd.DataFrame:
    frame = pd.read_csv(path)
    frame["date"] = frame["date"].astype(str).str[:10]
    return completed_bars(frame, end)


def scan_input_snapshot(prices_dir: Path) -> str:
    """Invalidate cached scans on data, universe, artifact or EOD-boundary changes.

    Content hashes also provide a reproducible identifier for future ledger rows.
    No assertion that those bytes are clean/independent is implied by a hash.
    """
    h = hashlib.sha256(b"eod-scan-contract-v1")
    h.update(str(calendar.completed_session_date()).encode())
    paths = sorted(prices_dir.glob("*.csv")) + [
        Path("configs/universe_vn30.json"), Path("data/models/win_prob_mr.pkl")]
    for path in paths:
        h.update(str(path).encode())
        h.update(hashlib.sha256(path.read_bytes()).digest() if path.exists() else b"missing")
    return h.hexdigest()
