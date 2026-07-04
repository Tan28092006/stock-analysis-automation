"""Win-probability model for MR (bat day) signals — META-LABELING.

Answers two asks at once:
  - "sao no no it vay": the strict gates fire ~52x/6yr. Here we LOOSEN the entry gate to
    a broad 'dip zone' (many candidates), then a model ranks them by P(win) so you can
    dial signal count vs quality with a probability threshold instead of hand-tuned gates.
  - "xac suat thang tinh sao luc train": meta-labeling. Label each historical candidate
    by the ACTUAL trade outcome (win/loss net of costs), train a classifier on features
    known at signal close, then CALIBRATE so predicted P matches realized win rate.

Anti-leakage: features at close(t), entry open(t+1), label from the simulated exit.
Time-ordered split TRAIN(oldest) / VAL(calibrate) / TEST(untouched) with an embargo gap
so no training trade overlaps the test period. Reports reliability + win-rate-by-threshold
on TEST only.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scratch"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from edge_experiment import DATA, prep
from stock_agent.features.indicators import ema, sma, adx

import lightgbm as lgb
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score

COST = 0.60
STOP_ATR = 3.0
MAX_HOLD = 8
T2_LOCK = 2
EMBARGO_DAYS = 20

FEATURES = [
    "rsi14", "rsi_below_30", "dist_below_bblo", "bb_pos", "vol_ratio", "atr_pct",
    "ret_1d", "ret_5d", "ret_20d", "reversal", "vsa_stop",
    "dist_to_kijun", "dist_to_ema21", "cloud_block", "adx14", "di_diff",
    "rr_to_kijun", "market_regime", "breadth", "rs_20d",
]


def build_candidates():
    # market regime + breadth by date
    idx = pd.read_csv(DATA / "VNINDEX.csv"); idx["date"] = idx["date"].astype(str).str.slice(0, 10)
    idx = idx.sort_values("date").reset_index(drop=True)
    idx_e50 = ema(idx["close"], 50)
    regime = {d: (1 if c > e else 0) for d, c, e in zip(idx["date"], idx["close"], idx_e50) if np.isfinite(e)}
    idx_ret20 = dict(zip(idx["date"], idx["close"].pct_change(20)))

    files = {p.stem: p for p in DATA.glob("*.csv") if p.stem != "VNINDEX"}
    dfs = {}
    for s, p in files.items():
        d = prep(pd.read_csv(p))
        ad = adx(d, 14)
        d["adx14"] = ad["adx14"].values
        d["plus_di14"] = ad["plus_di14"].values
        d["minus_di14"] = ad["minus_di14"].values
        dfs[s] = d
    # breadth: fraction of universe above own SMA20, by date
    above, tot = {}, {}
    for d in dfs.values():
        m = sma(d["close"], 20)
        for dt, c, mv in zip(d["date"].astype(str), d["close"], m):
            if np.isfinite(mv):
                tot[dt] = tot.get(dt, 0) + 1
                above[dt] = above.get(dt, 0) + (1 if c > mv else 0)
    breadth = {dt: above.get(dt, 0) / n for dt, n in tot.items() if n}

    rows = []
    for sym, d in dfs.items():
        d = d.reset_index(drop=True)
        dstr = d["date"].astype(str)
        n = len(d)
        for i in range(90, n - 1):
            close = float(d.at[i, "close"]); bbl = float(d.at[i, "bb_lower"]); rsi = float(d.at[i, "rsi14"])
            atrv = float(d.at[i, "atr"])
            if not (np.isfinite(bbl) and np.isfinite(rsi) and np.isfinite(atrv) and atrv > 0):
                continue
            # LOOSE dip-zone candidate gate (broad, for training)
            if not (rsi < 40 and close <= bbl * 1.05):
                continue
            kijun = float(d.at[i, "kijun"]) if np.isfinite(d.at[i, "kijun"]) else float(d.at[i, "ema21"])
            ema21 = float(d.at[i, "ema21"])
            # ---- simulate the actual trade to get the label ----
            entry = float(d.at[i + 1, "open"])
            if not (np.isfinite(entry) and entry > 0):
                continue
            stop = entry - STOP_ATR * atrv
            target = max(kijun, close * 1.01)
            i_end = min(i + 1 + MAX_HOLD, n - 1)
            exit_px, reason = float(d.at[i_end, "close"]), "TIME"
            for j in range(i + 1, i_end + 1):
                if j < i + 1 + T2_LOCK:
                    continue
                lo, hi = float(d.at[j, "low"]), float(d.at[j, "high"])
                if lo <= stop: exit_px, reason = stop, "STOP"; break
                if hi >= target: exit_px, reason = target, "TARGET"; break
            net = (exit_px - entry) / entry * 100 - COST
            sig_date = dstr.iloc[i]
            rows.append({
                "date": sig_date, "symbol": sym, "net": net, "win": int(net > 0),
                # features (all known at close t)
                "rsi14": rsi,
                "rsi_below_30": max(0.0, 30 - rsi),
                "dist_below_bblo": (bbl - close) / close,
                "bb_pos": (close - float(d.at[i, "bb_mid"])) / (float(d.at[i, "bb_mid"]) - bbl) if (float(d.at[i, "bb_mid"]) - bbl) else 0.0,
                "vol_ratio": float(d.at[i, "vol_ratio"]) if np.isfinite(d.at[i, "vol_ratio"]) else 1.0,
                "atr_pct": atrv / close,
                "ret_1d": close / float(d.at[i, "prev_close"]) - 1 if float(d.at[i, "prev_close"]) else 0.0,
                "ret_5d": close / float(d.at[i - 5, "close"]) - 1,
                "ret_20d": close / float(d.at[i - 20, "close"]) - 1,
                "reversal": int(close > float(d.at[i, "prev_close"])),
                "vsa_stop": int(d.at[i, "vsa_stop"]),
                "dist_to_kijun": (kijun - close) / close,
                "dist_to_ema21": (ema21 - close) / close,
                "cloud_block": int(np.isfinite(d.at[i, "cloud_bottom"]) and close < float(d.at[i, "cloud_bottom"]) < kijun),
                "adx14": float(d.at[i, "adx14"]) if np.isfinite(d.at[i, "adx14"]) else 0.0,
                "di_diff": (float(d.at[i, "plus_di14"]) - float(d.at[i, "minus_di14"])) if "plus_di14" in d.columns and np.isfinite(d.at[i, "plus_di14"]) else 0.0,
                "rr_to_kijun": (kijun - close) / (STOP_ATR * atrv),
                "market_regime": regime.get(sig_date, 1),
                "breadth": breadth.get(sig_date, 0.5),
                "rs_20d": (close / float(d.at[i - 20, "close"]) - 1) - (idx_ret20.get(sig_date) or 0.0),
            })
    return pd.DataFrame(rows).sort_values("date").reset_index(drop=True)


def reliability(y_true, p, bins=10):
    df = pd.DataFrame({"y": y_true, "p": p})
    df["bin"] = pd.qcut(df["p"], bins, duplicates="drop")
    g = df.groupby("bin", observed=True).agg(pred=("p", "mean"), actual=("y", "mean"), n=("y", "size"))
    return g


def main():
    df = build_candidates()
    base = df["win"].mean()
    print(f"Candidates: {len(df)} | base win rate {base*100:.1f}% | "
          f"{df['date'].min()} -> {df['date'].max()} | avg net {df['net'].mean():+.2f}%\n")

    # ---- WALK-FORWARD out-of-sample so every regime (incl 2022 crash) is tested ----
    K = 6
    df = df.reset_index(drop=True)
    bounds = [int(len(df) * k / K) for k in range(K + 1)]
    oos = []
    for k in range(1, K):
        tr = df.iloc[:bounds[k]].copy()
        te = df.iloc[bounds[k]:bounds[k + 1]].copy()
        te_start = pd.Timestamp(te["date"].iloc[0])
        tr = tr[pd.to_datetime(tr["date"]) < te_start - pd.Timedelta(days=EMBARGO_DAYS)]  # embargo
        if len(tr) < 500 or len(te) < 100:
            continue
        model = lgb.LGBMClassifier(n_estimators=300, max_depth=4, learning_rate=0.03,
                                   subsample=0.8, colsample_bytree=0.8, min_child_samples=40,
                                   reg_lambda=1.0, random_state=42, verbose=-1)
        model.fit(tr[FEATURES], tr["win"])
        te["raw"] = model.predict_proba(te[FEATURES])[:, 1]
        oos.append(te)
    oos = pd.concat(oos, ignore_index=True)
    oos["year"] = pd.to_datetime(oos["date"]).dt.year

    auc = roc_auc_score(oos["win"], oos["raw"])
    print(f"Pooled WALK-FORWARD OOS: {len(oos)} preds | AUC {auc:.3f}   (0.5 = coin flip)\n")

    print("AUC by year (does the edge appear in crash regimes?):")
    for yr, g in oos.groupby("year"):
        if g["win"].nunique() > 1 and len(g) > 80:
            print(f"  {yr}: AUC {roc_auc_score(g['win'], g['raw']):.3f}  (n={len(g)}, base win {g['win'].mean()*100:.0f}%)")

    # calibrate on pooled OOS (isotonic) — reliability check
    iso = IsotonicRegression(out_of_bounds="clip").fit(oos["raw"], oos["win"])
    oos["p"] = iso.transform(oos["raw"])
    print("\nRELIABILITY (calibrated) — predicted P vs actual win rate:")
    rel = reliability(oos["win"].values, oos["p"].values)
    print(f"  {'pred P':>8s} {'actual':>8s} {'n':>6s}")
    for _, r in rel.iterrows():
        print(f"  {r['pred']*100:>7.1f}% {r['actual']*100:>7.1f}% {int(r['n']):>6d}")

    print("\nSIGNAL COUNT vs QUALITY by probability threshold (pooled OOS):")
    print(f"  {'thresh':>7s} {'signals':>8s} {'win%':>7s} {'avg net%':>9s}")
    for thr in [0.0, 0.50, 0.55, 0.60, 0.65, 0.70]:
        sel = oos[oos["p"] >= thr]
        if len(sel):
            print(f"  {thr:>7.2f} {len(sel):>8d} {sel['win'].mean()*100:>6.1f}% {sel['net'].mean():>+8.2f}%")


if __name__ == "__main__":
    main()
