"""Ex-ante P(win) for the recent MR signals — what the model WOULD have shown at the time.

For each strict BUY_SETUP in the last 120 days, train the win-prob model ONLY on candidates
dated before that signal (minus 20-session embargo), calibrate, and predict — a leakage-free
"probability at the moment of the suggestion". Then compare to the ACTUAL outcome.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.isotonic import IsotonicRegression

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from stock_agent.config import load_json
from stock_agent.features.signal_engine import prepare_signal_frame, score_precomputed_at
from stock_agent.features import win_probability as wp

DATA = Path(r"D:\Chungkhoan\data\raw\prices_hist")
RULES = load_json(Path(r"D:\Chungkhoan\configs\rules_mr.json"))
VN30 = {"ACB","BID","BSR","CTG","FPT","GAS","GVR","HDB","HPG","LPB","MBB","MSN","MWG",
        "PLX","SAB","SHB","SSB","SSI","STB","TCB","TPB","VCB","VHM","VIB","VIC","VJC",
        "VNM","VPB","VPL","VRE"}
WIN_START = "2026-03-03"


def rules_for(sym):
    if sym in VN30 and RULES.get("vn30_mean_reversion"):
        r = dict(RULES); r["mean_reversion"] = {**RULES["mean_reversion"], **RULES["vn30_mean_reversion"]}
        return r
    return RULES


def main():
    regime, idx_ret20 = wp._context(DATA)
    files = {p.stem: p for p in DATA.glob("*.csv") if p.stem != "VNINDEX"}
    prepped = {s: wp._prep(pd.read_csv(p)) for s, p in files.items()}
    breadth = wp._breadth_map(prepped)

    # 1) full candidate table (features + label + date) for training
    rows = []
    for sym, f in prepped.items():
        ds = f["date"].astype(str)
        for i in range(90, len(f) - 1):
            sd = ds.iloc[i]
            fr = wp.feature_row(f, i, regime.get(sd, 1.0), breadth.get(sd, 0.5), idx_ret20.get(sd))
            if fr is None or not wp.is_candidate(fr):
                continue
            net = wp._label_trade(f, i)
            if net is None:
                continue
            rec = {k: fr[k] for k in wp.FEATURES}
            rec.update(date=sd, win=int(net > 0))
            rows.append(rec)
    cand = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)

    # 2) the actual strict BUY_SETUP signals in the window (production scorer)
    sigs = []
    for sym, p in files.items():
        df = pd.read_csv(p); df["date"] = df["date"].astype(str).str.slice(0, 10)
        if len(df) < 120:
            continue
        feats = prepare_signal_frame(df.sort_values("date").reset_index(drop=True), rules_for(sym))
        m = (feats["date"] >= WIN_START) & (feats.index >= 90)
        for i in np.where(m.fillna(False))[0]:
            if score_precomputed_at(sym, feats, int(i), rules_for(sym)).decision == "BUY_SETUP":
                d = str(feats["date"].iloc[int(i)])
                fr = wp.feature_row(feats, int(i), regime.get(d, 1.0), breadth.get(d, 0.5), idx_ret20.get(d))
                net = wp._label_trade(feats, int(i))
                if fr is not None and net is not None:
                    sigs.append({"date": d, "symbol": sym, "vn30": sym in VN30,
                                 "row": {k: fr[k] for k in wp.FEATURES}, "net": net})
    sigs.sort(key=lambda x: x["date"])

    # 3) per signal date: train on candidates strictly before (date - 20 sessions embargo)
    def train_before(cutoff):
        tr = cand[pd.to_datetime(cand["date"]) < pd.Timestamp(cutoff) - pd.Timedelta(days=28)]
        if len(tr) < 500 or tr["win"].nunique() < 2:
            return None
        cut = int(len(tr) * 0.85)
        a, b = tr.iloc[:cut], tr.iloc[cut:]
        mdl = lgb.LGBMClassifier(n_estimators=300, max_depth=4, learning_rate=0.03, subsample=0.8,
                                 colsample_bytree=0.8, min_child_samples=40, reg_lambda=1.0,
                                 random_state=42, verbose=-1).fit(a[wp.FEATURES], a["win"])
        iso = IsotonicRegression(out_of_bounds="clip").fit(mdl.predict_proba(b[wp.FEATURES])[:, 1], b["win"])
        return mdl, iso

    cache = {}
    print(f"Ex-ante P(win) cho {len(sigs)} keo (model chi train tren du lieu TRUOC ngay do):\n")
    print(f"{'ngay':12s} {'ma':6s} {'VN30':5s} {'P(win) ex-ante':>14s}  {'ket qua thuc':>13s}")
    print("-" * 60)
    hits = []
    for s in sigs:
        if s["date"] not in cache:
            cache[s["date"]] = train_before(s["date"])
        mi = cache[s["date"]]
        if mi is None:
            p = None
        else:
            mdl, iso = mi
            raw = mdl.predict_proba(pd.DataFrame([s["row"]])[wp.FEATURES])[:, 1][0]
            p = float(iso.transform([raw])[0])
        res = f"{s['net']:+.2f}% ({'THANG' if s['net'] > 0 else 'thua'})"
        ps = f"{p*100:.0f}%" if p is not None else "n/a"
        print(f"{s['date']:12s} {s['symbol']:6s} {'VN30' if s['vn30'] else '':5s} {ps:>14s}  {res:>13s}")
        if p is not None:
            hits.append((p, s["net"] > 0))
    if hits:
        hi = [w for p, w in hits if p >= 0.55]; lo = [w for p, w in hits if p < 0.55]
        print(f"\nP>=55%: {sum(hi)}/{len(hi)} thang" + (f" ({np.mean(hi)*100:.0f}%)" if hi else ""))
        print(f"P<55% : {sum(lo)}/{len(lo)} thang" + (f" ({np.mean(lo)*100:.0f}%)" if lo else ""))


if __name__ == "__main__":
    main()
