"""CORE momentum scan — QUANT-GRADE (validated: scratch/quant_momentum.py).

Techniques: 12-1 momentum (rank by return t-252 -> t-21, skip last month), inverse-vol
(risk-parity) target weights, vol-TARGETING exposure (scale total exposure by
target_vol / market_vol -> auto de-risk when vol is high), buffering (a held name stays
until it drops out of the top 2N -> low turnover). VN100 universe for breadth.

Backtest on the fixed current-30-VN30 basket (next-open fills, cost 0.4%): FULL +157-190%,
Sharpe ~0.9. But this is HEAVILY SURVIVORSHIP-INFLATED: on a point-in-time universe
(top-30 by trailing traded value, membership rotating) it drops to ~+33% / Sharpe ~0.31 /
DD ~-56% — roughly index-like. Treat the fixed-basket numbers as an optimistic upper bound.
Other honest caveats: momentum-crash risk in fast crashes (hard kill-switch, %-drop
circuit-breaker, per-position stops, and an excess-momentum gate were ALL tested and
REJECTED — kill-switches/stops whipsaw; the excess gate only "worked" on the exact survivor
basket and vanished on point-in-time / VN100 / random-subset). No RISK_ON gate: vol-targeting
handles risk. See memory: momentum-excess-gate, crash-response-by-shape.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import compute_rules_hash, load_json
from .indicators import ema
from .mr_scan import PRICES_DIR, MR_RULES_PATH, _load_frames, _market_state
from .position_manager import money_cfg

LB_LONG = 252            # 12 months
LB_SKIP = 21             # skip last 1 month (short-term reversal)
VOL_WIN = 60
TARGET_VOL = 0.20        # annualized portfolio vol target
DEFAULT_TOP_N = 10
BUFFER_MULT = 2          # hold a name until it leaves the top (2 x top_n)
CACHE_PATH = Path("data/pipeline/momentum_scan_cache.json")


def _vn30() -> set:
    try:
        from ..config import load_universe
        return {s.upper() for s in load_universe()["symbols"]}
    except Exception:
        return set()


def _market_vol() -> float:
    """VNINDEX 20-day realized vol, annualized (latest)."""
    idx = PRICES_DIR / "VNINDEX.csv"
    if not idx.exists():
        return TARGET_VOL
    df = pd.read_csv(idx)
    r = df["close"].pct_change()
    v = r.rolling(20, min_periods=10).std().iloc[-1] * math.sqrt(252)
    return float(v) if np.isfinite(v) else TARGET_VOL


def _rank(frames: dict) -> list[tuple]:
    """Return [(symbol, mom_12_1, vol_ann, close)] sorted by momentum desc, above-history only."""
    scored = []
    for sym, df in frames.items():
        c = df["close"].to_numpy()
        if len(c) < LB_LONG + 2:
            continue
        c_now, c_then = float(c[-1 - LB_SKIP]), float(c[-1 - LB_LONG])
        r = pd.Series(c).pct_change()
        vol = float(r.iloc[-VOL_WIN:].std() * math.sqrt(252))
        if c_then > 0 and c_now > 0 and np.isfinite(vol) and vol > 0:
            scored.append((sym, c_now / c_then - 1, vol, float(c[-1])))
    scored.sort(key=lambda x: -x[1])
    return scored


def _compute(top_n: int) -> dict:
    rules = load_json(MR_RULES_PATH)
    cfg = money_cfg(rules)
    frames = _load_frames(PRICES_DIR)
    market = _market_state(PRICES_DIR, frames)
    vn30 = _vn30()

    mvol = _market_vol()
    exposure = min(1.0, TARGET_VOL / max(mvol, 1e-6))   # de-risk when vol high
    ranked = _rank(frames)
    top = ranked[:top_n]
    buffer_syms = {s for s, *_ in ranked[:top_n * BUFFER_MULT]}

    inv = {s: 1.0 / max(v, 0.05) for s, m, v, c in top}
    wsum = sum(inv.values()) or 1.0
    picks = []
    for sym, mom, vol, close in top:
        w = inv[sym] / wsum * exposure       # inverse-vol weight, scaled by exposure
        qty = int(np.floor(cfg["account_nav"] * w / close / cfg["lot_size"]) * cfg["lot_size"])
        picks.append({
            "symbol": sym, "vn30": sym in vn30,
            "momentum_12_1_pct": round(mom * 100, 1), "vol_ann_pct": round(vol * 100, 0),
            "close": round(close, 2), "weight_pct": round(w * 100, 1), "qty": qty,
            "trend_exit": "rớt khỏi top-momentum (rebal tháng)",
        })

    # open momentum positions + alerts (exit = dropped out of the top-2N buffer)
    positions, sell_alerts = [], []
    try:
        from .position_manager import PositionStore, check_momentum_positions
        positions = check_momentum_positions(PositionStore(), buffer_syms, True)
        sell_alerts = [p for p in positions if p.get("live_status") == "SELL"]
    except Exception:
        pass

    return {
        "mode": "quant_momentum_12_1",
        "rules_hash": compute_rules_hash(rules),
        "data_date": market.get("date"),
        "active": True,           # always on; vol-targeting handles risk (no RISK_ON gate)
        "market": market,
        "top_n": top_n,
        "exposure_pct": round(exposure * 100, 0),
        "market_vol_pct": round(mvol * 100, 0),
        "buffer_symbols": sorted(buffer_syms),
        "picks": picks,
        "positions": positions,
        "sell_alerts": sell_alerts,
        "note": (f"12-1 momentum · trọng số nghịch-vol · exposure {round(exposure*100)}% "
                 f"(vol thị trường {round(mvol*100)}%/năm, target {int(TARGET_VOL*100)}%). "
                 f"Rebal tháng, giữ tới khi rớt top-{top_n*BUFFER_MULT}."),
    }


def momentum_scan(top_n: int = DEFAULT_TOP_N, force: bool = False) -> dict:
    rules_hash = compute_rules_hash(load_json(MR_RULES_PATH))
    if not force and CACHE_PATH.exists():
        try:
            cached = json.loads(CACHE_PATH.read_text(encoding="utf-8"))
            latest = None
            idx = PRICES_DIR / "VNINDEX.csv"
            if idx.exists():
                latest = str(pd.read_csv(idx)["date"].astype(str).str.slice(0, 10).max())
            if cached.get("rules_hash") == rules_hash and cached.get("data_date") == latest and cached.get("top_n") == top_n:
                try:
                    from .position_manager import PositionStore, check_momentum_positions
                    buf = set(cached.get("buffer_symbols", []))
                    cached["positions"] = check_momentum_positions(PositionStore(), buf, True)
                    cached["sell_alerts"] = [p for p in cached["positions"] if p.get("live_status") == "SELL"]
                except Exception:
                    pass
                return cached
        except Exception:
            pass
    payload = _compute(top_n)
    try:
        CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        CACHE_PATH.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    except Exception:
        pass
    return payload
