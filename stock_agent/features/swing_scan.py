"""SWING scan — Connors-style short-term reversal (validated: scratch/connors_tune.py).

A DIFFERENT game from the MR "bat day" bottom-fisher: don't try to catch a crash bottom
(a ~48% coin-flip, proven across 7 crashes). Instead swing-trade short oversold dips ONLY
inside an uptrend, and exit fast on the bounce:

  signal = RSI(2) < 5  AND  close > SMA50   (buy the dip only while the trend is up)
  exit   = close > SMA5 (the bounce, ~2-4 sessions) / max 8 sessions / -8% disaster stop

Backtest 2018-2026 (all ~100 syms, next-open fills): win 64%, +0.52%/trade at 0.4% cost,
POSITIVE every single year incl uptrends (2021/24/25) AND crashes (2022 59%, 2026 56%) —
because the SMA50 filter sits out sustained downtrends. HONEST caveats: the edge is THIN
(+0.5%/trade, per-trade Sharpe ~0.11, cost-sensitive: ~+0.3% at 0.6% cost), it's a
high-frequency small-edge tactic, and the current-basket survivorship caveat still applies.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import compute_rules_hash, load_json
from .mr_scan import PRICES_DIR, MR_RULES_PATH, _load_frames, _market_state
from .position_manager import money_cfg, suggest_size

RSI2_MAX = 5.0
STOP_PCT = 0.08          # -8% disaster stop
MAX_HOLD = 8
CACHE_PATH = Path("data/pipeline/swing_scan_cache.json")


def _rsi(series: pd.Series, n: int) -> pd.Series:
    d = series.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + up / dn.replace(0, np.nan))


def _compute(rsi2_max: float = RSI2_MAX) -> dict:
    rules = load_json(MR_RULES_PATH)
    cfg = money_cfg(rules)
    frames = _load_frames(PRICES_DIR)
    market = _market_state(PRICES_DIR, frames)

    picks = []
    for sym, df in frames.items():
        c = df["close"]
        if len(c) < 210:
            continue
        rsi2 = float(_rsi(c, 2).iloc[-1])
        sma5 = float(c.rolling(5).mean().iloc[-1])
        sma50 = float(c.rolling(50).mean().iloc[-1])
        close = float(c.iloc[-1])
        if not (np.isfinite(rsi2) and np.isfinite(sma50)):
            continue
        if rsi2 < rsi2_max and close > sma50:
            stop = round(close * (1 - STOP_PCT), 2)
            size = suggest_size(cfg["account_nav"], close, stop, cfg)
            picks.append({
                "symbol": sym, "rsi2": round(rsi2, 1), "close": round(close, 2),
                "entry_reference": round(close, 2), "sma5": round(sma5, 2),
                "sma50": round(sma50, 2), "stop_loss": stop,
                "exit_rule": "Bán khi close > SMA5 (thường 2-4 phiên)", "size": size,
            })
    picks.sort(key=lambda p: p["rsi2"])   # most oversold first

    return {
        "mode": "swing_rsi2",
        "rules_hash": compute_rules_hash(rules),
        "data_date": market.get("date"),
        "market": market,
        "rsi2_max": rsi2_max,
        "picks": picks,
        "note": (f"Lướt sóng: RSI(2) < {int(rsi2_max)} & giá > SMA50 (mua dip trong uptrend). "
                 f"Bán khi close > SMA5 (~2-4 phiên) / tối đa {MAX_HOLD} phiên / stop -{int(STOP_PCT*100)}%. "
                 f"Edge mỏng ~+0.5%/lệnh (kiểm 2018-2026, dương mọi năm) — high-freq, nhạy phí."),
    }


def swing_scan(force: bool = False) -> dict:
    rules_hash = compute_rules_hash(load_json(MR_RULES_PATH))
    if not force and CACHE_PATH.exists():
        try:
            cached = json.loads(CACHE_PATH.read_text(encoding="utf-8"))
            latest = None
            idx = PRICES_DIR / "VNINDEX.csv"
            if idx.exists():
                latest = str(pd.read_csv(idx)["date"].astype(str).str.slice(0, 10).max())
            if cached.get("rules_hash") == rules_hash and cached.get("data_date") == latest:
                return cached
        except Exception:
            pass
    payload = _compute()
    try:
        CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        CACHE_PATH.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    except Exception:
        pass
    return payload
