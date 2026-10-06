"""Offline fixtures for swing desk tests (imported, not collected)."""
from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
import json
from pathlib import Path

import numpy as np
import pandas as pd

from stock_agent.data.exchange_calendar import VN_TIMEZONE, trading_days_between

AS_OF = date(2026, 9, 21)          # a Monday, completed session for NOW_EOD
NOW_EOD = datetime(2026, 9, 21, 17, 0, tzinfo=VN_TIMEZONE)   # after 16:00 VN
UNIVERSE = ["AAA", "BBB", "VCB"]


def dates(n: int = 300, end: date = AS_OF) -> list[date]:
    return trading_days_between(end - timedelta(days=n * 2), end)[-n:]


def frame(closes, *, end: date = AS_OF, opens=None, highs=None, lows=None, volume=2e6) -> pd.DataFrame:
    closes = np.asarray(closes, dtype=float)
    days = dates(len(closes), end)
    opens = closes.copy() if opens is None else np.asarray(opens, dtype=float)
    highs = np.maximum(opens, closes) * 1.005 if highs is None else np.asarray(highs, dtype=float)
    lows = np.minimum(opens, closes) * 0.995 if lows is None else np.asarray(lows, dtype=float)
    vol = np.full(len(closes), float(volume)) if np.isscalar(volume) else np.asarray(volume, dtype=float)
    return pd.DataFrame({"date": days, "open": opens, "high": highs, "low": lows, "close": closes, "volume": vol})


def uptrend(n: int = 300, start: float = 20000.0, drift: float = 0.0015, wiggle: float = 0.004, seed: int = 1):
    rng = np.random.default_rng(seed)
    steps = drift + wiggle * rng.standard_normal(n)
    return start * np.cumprod(1 + steps)


def with_dip(closes, pct_days=(-0.012, -0.012, -0.012)):
    """Replace the last len(pct_days) closes with consecutive small declines (RSI2 dip)."""
    closes = np.array(closes, dtype=float)
    k = len(pct_days)
    base = closes[-k - 1]
    for i, pct in enumerate(pct_days):
        base *= 1 + pct
        closes[-k + i] = base
    return closes


def payload(symbol: str, df: pd.DataFrame) -> list[dict]:
    ts = [str(int(pd.Timestamp(str(d), tz="UTC").timestamp())) for d in df["date"]]
    return [{"symbol": symbol, "t": ts, "o": df.open.round(2).tolist(), "h": df.high.round(2).tolist(),
             "l": df.low.round(2).tolist(), "c": df.close.round(2).tolist(), "v": df.volume.tolist()}]


def write_snapshot(root: Path, frames: dict[str, pd.DataFrame], monkeypatch, *, as_of: date = AS_OF,
                   name: str = "20260921T100000000000Z") -> Path:
    """Build a VERIFIED snapshot through the real reconciliation code with a fake fetcher."""
    from stock_agent.data import reconciliation as rec
    monkeypatch.setattr(rec, "completed_session_date", lambda: as_of)

    def fetch(symbol, start, end):
        raw = json.dumps(payload(symbol, frames[symbol])).encode()
        return rec.parse_vci_history(json.loads(raw), symbol, start, end), {
            "source": "VCI", "fetched_at": datetime.combine(end, datetime.min.time(), timezone.utc).replace(hour=9, minute=25).isoformat(),
            "raw_bytes": raw}

    out = Path(root) / name
    start = min(frames[s]["date"].iloc[0] for s in frames)
    manifest = rec.build_market_snapshot(list(frames), out, start, as_of, min_rows=30, fetcher=fetch)
    assert manifest["status"] == "verified", manifest["failures"]
    # build_market_snapshot stamps completed_at with the real clock; pin it to fixture time
    # (manifest.json itself is not part of the hashed file set).
    path = out / "manifest.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    stamp = datetime.combine(as_of, datetime.min.time(), timezone.utc).replace(hour=9, minute=30).isoformat()
    data["fetched_at"] = data["completed_at"] = stamp
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return path


def make_repo(tmp_path: Path, universe=UNIVERSE) -> Path:
    """Minimal repo layout: desk config + universe + rules_mr (for mr_scan) copied from the real repo."""
    repo = tmp_path / "repo"
    (repo / "configs" / "swing_desk").mkdir(parents=True)
    real = Path(__file__).resolve().parents[1]
    (repo / "configs/swing_desk/desk_v1.json").write_bytes((real / "configs/swing_desk/desk_v1.json").read_bytes())
    (repo / "configs/universe_vn30.json").write_text(json.dumps({"symbols": list(universe)}), encoding="utf-8")
    return repo


def market_frames(universe=UNIVERSE, *, dip: str | None = "AAA", n: int = 300) -> dict[str, pd.DataFrame]:
    frames = {"VNINDEX": frame(uptrend(n, 1200.0, seed=7), volume=5e8)}
    for i, sym in enumerate(universe):
        closes = uptrend(n, 20000.0 + 1000 * i, seed=11 + i)
        if sym == dip:
            closes = with_dip(uptrend(n, 20000.0, drift=0.008, wiggle=0.002, seed=3), (-0.03,) * 4)
        frames[sym] = frame(closes, volume=3e6)
    return frames
