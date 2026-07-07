"""Win-probability model for mean-reversion (bat day) candidates — META-LABELING.

Loosen the entry gate to a broad dip-zone, label each historical candidate by the ACTUAL
trade outcome (net of costs), train LightGBM on features known at signal close, and
CALIBRATE (isotonic) so predicted P matches realized win rate. Validated in
scratch/win_probability.py: calibration is exact; discrimination is weak overall
(AUC ~0.50) but real in crash/recovery regimes (2022 AUC 0.557); used as a FILTER the
loose money-losing pool becomes 56% win / +1.78%/trade at P>=0.55.

Features are read from the SAME production indicator frame the signal engine scores
(add_indicators output), so there is zero train/inference feature drift.
"""
from __future__ import annotations

import pickle
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from .indicators import add_indicators, ema, sma
from .mr_exit import simulate_mr_exit

ARTIFACT_PATH = Path("data/models/win_prob_mr.pkl")

FEATURES = [
    "rsi14", "rsi_below_30", "dist_below_bblo", "bb_pos", "vol_ratio", "atr_pct",
    "ret_1d", "ret_5d", "ret_20d", "reversal", "vsa_stop",
    "dist_to_kijun", "dist_to_ema21", "cloud_block", "adx14", "di_diff",
    "rr_to_kijun", "market_regime", "breadth", "rs_20d",
]

# Trade-simulation constants (must match the production MR exit logic / rules_mr.json)
COST = 0.60
STOP_ATR = 3.0
MAX_HOLD = 15
T2_LOCK = 2
CAND_RSI_MAX = 40.0        # loose dip-zone candidate gate
CAND_BAND_MULT = 1.05


def _prep(df: pd.DataFrame) -> pd.DataFrame:
    d = df.sort_values("date").reset_index(drop=True).copy()
    d["date"] = d["date"].astype(str).str.slice(0, 10)
    out = add_indicators(d, atr_period=14)
    return out


def feature_row(f: pd.DataFrame, i: int, market_regime: float, breadth: float,
                idx_ret20: float) -> dict | None:
    """Extract the win-prob feature vector at bar i of a prepared indicator frame.
    Returns None if the bar is unusable. Uses only data <= i (causal)."""
    if i < 20 or i >= len(f):
        return None
    close = float(f.at[i, "close"])
    bbl = float(f.at[i, "bb_lower"]) if pd.notna(f.at[i, "bb_lower"]) else np.nan
    bbm = float(f.at[i, "bb_mid"]) if pd.notna(f.at[i, "bb_mid"]) else close
    rsi = float(f.at[i, "rsi14"]) if pd.notna(f.at[i, "rsi14"]) else np.nan
    atrv = float(f.at[i, "atr"]) if pd.notna(f.at[i, "atr"]) else np.nan
    if not (np.isfinite(bbl) and np.isfinite(rsi) and np.isfinite(atrv) and atrv > 0 and close > 0):
        return None
    kijun = float(f.at[i, "kijun"]) if pd.notna(f.at[i, "kijun"]) else float(f.at[i, "ema21"])
    ema21 = float(f.at[i, "ema21"]) if pd.notna(f.at[i, "ema21"]) else close
    sa = float(f.at[i, "senkou_a"]) if pd.notna(f.at[i, "senkou_a"]) else 0.0
    sb = float(f.at[i, "senkou_b"]) if pd.notna(f.at[i, "senkou_b"]) else 0.0
    cloud_bottom = min(sa, sb) if (sa and sb) else 0.0
    prev_close = float(f.at[i - 1, "close"])
    c5 = float(f.at[i - 5, "close"]); c20 = float(f.at[i - 20, "close"])
    return {
        "rsi14": rsi,
        "rsi_below_30": max(0.0, 30 - rsi),
        "dist_below_bblo": (bbl - close) / close,
        "bb_pos": (close - bbm) / (bbm - bbl) if (bbm - bbl) else 0.0,
        "vol_ratio": float(f.at[i, "volume_ratio_20"]) if pd.notna(f.at[i, "volume_ratio_20"]) else 1.0,
        "atr_pct": atrv / close,
        "ret_1d": float(f.at[i, "return_1d"]) if pd.notna(f.at[i, "return_1d"]) else 0.0,
        "ret_5d": close / c5 - 1 if c5 else 0.0,
        "ret_20d": close / c20 - 1 if c20 else 0.0,
        "reversal": int(close > prev_close),
        "vsa_stop": int(f.at[i, "vsa_stopping_volume"]) if pd.notna(f.at[i, "vsa_stopping_volume"]) else 0,
        "dist_to_kijun": (kijun - close) / close,
        "dist_to_ema21": (ema21 - close) / close,
        "cloud_block": int(bool(cloud_bottom) and close < cloud_bottom < kijun),
        "adx14": float(f.at[i, "adx14"]) if pd.notna(f.at[i, "adx14"]) else 0.0,
        "di_diff": (float(f.at[i, "plus_di14"]) - float(f.at[i, "minus_di14"]))
                    if pd.notna(f.at[i, "plus_di14"]) else 0.0,
        "rr_to_kijun": (kijun - close) / (STOP_ATR * atrv),
        "market_regime": float(market_regime),
        "breadth": float(breadth),
        "rs_20d": (close / c20 - 1 if c20 else 0.0) - (idx_ret20 or 0.0),
        # non-feature helpers for candidate gating / labeling:
        "_close": close, "_bbl": bbl, "_rsi": rsi, "_atr": atrv, "_kijun": kijun,
    }


def is_candidate(fr: dict) -> bool:
    return fr["_rsi"] < CAND_RSI_MAX and fr["_close"] <= fr["_bbl"] * CAND_BAND_MULT


def _context(prices_dir: Path):
    """market_regime + breadth + index 20d return, keyed by date string."""
    idx = pd.read_csv(prices_dir / "VNINDEX.csv")
    idx["date"] = idx["date"].astype(str).str.slice(0, 10)
    idx = idx.sort_values("date").reset_index(drop=True)
    e50 = ema(idx["close"], 50)
    regime = {d: (1.0 if c > e else 0.0) for d, c, e in zip(idx["date"], idx["close"], e50) if np.isfinite(e)}
    idx_ret20 = dict(zip(idx["date"], idx["close"].pct_change(20)))
    return regime, idx_ret20


def _breadth_map(frames: dict[str, pd.DataFrame]) -> dict:
    above, tot = {}, {}
    for f in frames.values():
        m = sma(f["close"], 20)
        for dt, c, mv in zip(f["date"].astype(str), f["close"], m):
            if np.isfinite(mv):
                tot[dt] = tot.get(dt, 0) + 1
                above[dt] = above.get(dt, 0) + (1 if c > mv else 0)
    return {dt: above.get(dt, 0) / n for dt, n in tot.items() if n}


def _label_trade(f: pd.DataFrame, i: int) -> float | None:
    """Net % of the simulated MR trade started at signal bar i, or None. Entry = next open
    (bar i+1); exit via the shared canonical replay (settle_lock T+2, T+MAX_HOLD time-stop)."""
    n = len(f)
    if i + 1 >= n:
        return None
    entry = float(f.at[i + 1, "open"])
    atrv = float(f.at[i, "atr"])
    kijun = float(f.at[i, "kijun"]) if pd.notna(f.at[i, "kijun"]) else float(f.at[i, "ema21"])
    close = float(f.at[i, "close"])
    if not (np.isfinite(entry) and entry > 0):
        return None
    stop = entry - STOP_ATR * atrv
    target = max(kijun, close * 1.01)
    # Training labels want a value even at the data edge, so use the (partial) mark
    # regardless of `resolved` — matches the historical labelling behaviour.
    _, exit_px, _, _ = simulate_mr_exit(f, i + 1, stop, target, MAX_HOLD, settle_lock=T2_LOCK)
    return (exit_px - entry) / entry * 100 - COST


def train_and_save(prices_dir: Path = Path("data/raw/prices_hist"),
                   artifact_path: Path = ARTIFACT_PATH) -> dict:
    import lightgbm as lgb
    from sklearn.isotonic import IsotonicRegression
    from sklearn.metrics import roc_auc_score

    regime, idx_ret20 = _context(prices_dir)
    files = {p.stem: p for p in prices_dir.glob("*.csv") if p.stem != "VNINDEX"}
    frames = {s: _prep(pd.read_csv(p)) for s, p in files.items()}
    breadth = _breadth_map(frames)

    rows = []
    for sym, f in frames.items():
        dstr = f["date"].astype(str)
        for i in range(90, len(f) - 1):
            sd = dstr.iloc[i]
            fr = feature_row(f, i, regime.get(sd, 1.0), breadth.get(sd, 0.5), idx_ret20.get(sd))
            if fr is None or not is_candidate(fr):
                continue
            net = _label_trade(f, i)
            if net is None:
                continue
            rec = {k: fr[k] for k in FEATURES}
            rec["date"] = sd; rec["win"] = int(net > 0)
            rows.append(rec)
    df = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    if len(df) < 1000:
        return {"status": "insufficient_data", "rows": len(df)}

    # final model on all-but-tail; isotonic calibrated on the tail (time-ordered)
    cut = int(len(df) * 0.85)
    tr, cal = df.iloc[:cut], df.iloc[cut:]
    model = lgb.LGBMClassifier(n_estimators=300, max_depth=4, learning_rate=0.03,
                               subsample=0.8, colsample_bytree=0.8, min_child_samples=40,
                               reg_lambda=1.0, random_state=42, verbose=-1)
    model.fit(tr[FEATURES], tr["win"])
    raw_cal = model.predict_proba(cal[FEATURES])[:, 1]
    iso = IsotonicRegression(out_of_bounds="clip").fit(raw_cal, cal["win"])
    auc = float(roc_auc_score(cal["win"], raw_cal)) if cal["win"].nunique() > 1 else float("nan")

    # refit on ALL data for deployment (keep the tail-fit isotonic map)
    model.fit(df[FEATURES], df["win"])

    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    with artifact_path.open("wb") as fh:
        pickle.dump({"model": model, "iso": iso, "features": FEATURES,
                     "trained_at": datetime.now(timezone.utc).isoformat(),
                     "n_candidates": len(df), "base_win_rate": float(df["win"].mean()),
                     "calib_auc": auc}, fh)
    return {"status": "ok", "rows": len(df), "calib_auc": auc,
            "base_win_rate": float(df["win"].mean()), "artifact": str(artifact_path)}


class WinProbModel:
    _cache = None

    def __init__(self, art):
        self.model = art["model"]; self.iso = art["iso"]; self.features = art["features"]
        self.meta = {k: art.get(k) for k in ("trained_at", "n_candidates", "base_win_rate", "calib_auc")}

    @classmethod
    def load(cls, path: Path = ARTIFACT_PATH):
        if cls._cache is not None:
            return cls._cache
        if not path.exists():
            return None
        with path.open("rb") as fh:
            cls._cache = cls(pickle.load(fh))
        return cls._cache

    def predict(self, fr: dict) -> float:
        x = pd.DataFrame([{k: fr.get(k, 0.0) for k in self.features}]).replace(
            [np.inf, -np.inf], np.nan).fillna(0.0)
        raw = float(self.model.predict_proba(x)[:, 1][0])
        return float(self.iso.transform([raw])[0])
