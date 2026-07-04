"""Information Coefficient (IC) of features vs forward returns on VN30.

Kills the "arbitrary scorecard weights" ambiguity with data: for each feature,
how well does it actually rank tomorrow's winners?

IC = cross-sectional Spearman(feature across symbols on date d, forward return
across symbols on date d), averaged over dates. |IC|>0.03-0.05 with a stable
sign and t-stat>2 is a usable predictor; ~0 means noise (drop it / don't weight it).

Reports IC for horizons T+2 / T+5 / T+10, over the FULL sample and the 2022 crash.
"""
from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from stock_agent.features.indicators import ema, sma, rsi, macd, bollinger, atr, adx

DATA = Path(r"D:\Chungkhoan\data\raw\prices_hist")
HORIZONS = [2, 5, 10]


def build_features(df: pd.DataFrame, idx_ret: dict) -> pd.DataFrame:
    d = df.sort_values("date").reset_index(drop=True).copy()
    d["date"] = pd.to_datetime(d["date"]).dt.date
    c, h, l, v = d["close"], d["high"], d["low"], d["volume"]
    f = pd.DataFrame({"date": d["date"]})
    e9, e21, e50 = ema(c, 9), ema(c, 21), ema(c, 50)
    bb = bollinger(c, 20, 2.0)
    ad = adx(d, 14)
    f["rsi14"] = rsi(c, 14)
    f["rsi2"] = rsi(c, 2)
    f["macd_hist_norm"] = macd(c)["macd_hist"] / c
    f["ema9_21_gap"] = e9 / e21 - 1
    f["px_vs_ema21"] = c / e21 - 1
    f["px_vs_ema50"] = c / e50 - 1
    f["adx14"] = ad["adx14"]
    f["di_diff"] = ad["plus_di14"] - ad["minus_di14"]
    f["vol_ratio"] = v / v.rolling(20, min_periods=20).mean()
    f["breakout_20"] = c / h.rolling(20, min_periods=20).max().shift(1) - 1
    f["bb_pos"] = (c - bb["bb_mid"]) / (bb["bb_upper"] - bb["bb_lower"]).replace(0, np.nan)
    f["dist_below_bblo"] = (bb["bb_lower"] - c) / c            # >0 = below lower band (dip)
    f["atr_pct"] = atr(d, 14) / c
    f["ret_1d"] = c.pct_change(1)
    f["ret_5d"] = c.pct_change(5)
    f["ret_20d"] = c.pct_change(20)
    # relative strength vs index (20d)
    idx20 = pd.Series(d["date"].map(lambda x: idx_ret.get(x, np.nan)))
    f["rs_20d"] = f["ret_20d"] - idx20.values
    # forward returns (close-to-close)
    for hh in HORIZONS:
        f[f"fwd_{hh}"] = c.shift(-hh) / c - 1
    return f


FEATURES = ["rsi14", "rsi2", "macd_hist_norm", "ema9_21_gap", "px_vs_ema21", "px_vs_ema50",
            "adx14", "di_diff", "vol_ratio", "breakout_20", "bb_pos", "dist_below_bblo",
            "atr_pct", "ret_1d", "ret_5d", "ret_20d", "rs_20d"]


def rank_ic(panel: pd.DataFrame, feat: str, fwd: str, start, end):
    sub = panel[(panel["date"] >= start) & (panel["date"] <= end)]
    ics = []
    for dt, g in sub.groupby("date"):
        gg = g[[feat, fwd]].dropna()
        if len(gg) >= 8 and gg[feat].nunique() > 3:
            ics.append(gg[feat].rank().corr(gg[fwd].rank()))
    ics = np.array([x for x in ics if np.isfinite(x)])
    if len(ics) < 20:
        return None
    mean = ics.mean()
    t = mean / (ics.std(ddof=1) / np.sqrt(len(ics))) if ics.std() > 0 else 0.0
    return mean, t, (ics > 0).mean() * 100, len(ics)


def main():
    files = {p.stem: p for p in DATA.glob("*.csv")}
    idx = pd.read_csv(files["VNINDEX"]); idx["date"] = pd.to_datetime(idx["date"]).dt.date
    idx = idx.sort_values("date").reset_index(drop=True)
    idx["ret20"] = idx["close"].pct_change(20)
    idx_ret = dict(zip(idx["date"], idx["ret20"]))

    frames = []
    for sym, p in files.items():
        if sym == "VNINDEX":
            continue
        f = build_features(pd.read_csv(p), idx_ret)
        f["symbol"] = sym
        frames.append(f)
    panel = pd.concat(frames, ignore_index=True)
    print(f"Panel: {len(panel)} rows, {panel['symbol'].nunique()} symbols, "
          f"{panel['date'].min()} -> {panel['date'].max()}\n")

    windows = {
        "FULL 2020-now": (date(2020, 1, 1), date(2026, 7, 1)),
        "2022 crash (Apr-Nov)": (date(2022, 4, 1), date(2022, 11, 30)),
    }
    for wname, (s, e) in windows.items():
        print("=" * 92)
        print(f"WINDOW: {wname}   (Rank IC; |IC|>0.03 & |t|>2 = usable; ~0 = noise)")
        print("-" * 92)
        print(f"{'feature':16s} | " + " | ".join(f"T+{h:<2d} IC   t   pos%" for h in HORIZONS))
        # sort by |IC| at T+5
        order = []
        for feat in FEATURES:
            r5 = rank_ic(panel, feat, "fwd_5", s, e)
            order.append((abs(r5[0]) if r5 else -1, feat))
        for _, feat in sorted(order, reverse=True):
            cells = []
            for h in HORIZONS:
                r = rank_ic(panel, feat, f"fwd_{h}", s, e)
                if r is None:
                    cells.append("   n/a         ")
                else:
                    ic, t, pos, n = r
                    cells.append(f"{ic:+.3f} {t:+5.1f} {pos:4.0f}")
            print(f"{feat:16s} | " + " | ".join(cells))
        print()


if __name__ == "__main__":
    main()
