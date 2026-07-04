"""When did the MR bottom-fisher fire in 2026, and did it cluster near the real bottom?

Leakage-safe: uses the PRODUCTION scorer score_precomputed_at at each historical bar i,
which reads only rows <= i (row i and i-1); entry would be bar i+1 open. All indicators
are causal. NO win_prob shown (the model was trained on 2026 = in-sample).

Marks VN30 members and each signal's distance (trading sessions) from the 2026 bottom.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from stock_agent.config import load_json
from stock_agent.features.signal_engine import prepare_signal_frame, score_precomputed_at

DATA = Path(r"D:\Chungkhoan\data\raw\prices_hist")
RULES = load_json(Path(r"D:\Chungkhoan\configs\rules_mr.json"))
VN30 = {"ACB","BID","BSR","CTG","FPT","GAS","GVR","HDB","HPG","LPB","MBB","MSN","MWG",
        "PLX","SAB","SHB","SSB","SSI","STB","TCB","TPB","VCB","VHM","VIB","VIC","VJC",
        "VNM","VPB","VPL","VRE"}


def main():
    # 2026 index bottom (by intraday low)
    idx = pd.read_csv(DATA / "VNINDEX.csv"); idx["date"] = idx["date"].astype(str).str.slice(0, 10)
    idx = idx[idx["date"] >= "2026-01-01"].sort_values("date").reset_index(drop=True)
    bottom_row = idx.loc[idx["low"].idxmin()]
    bottom_date = str(bottom_row["date"])
    idx_dates = list(idx["date"])
    bi = idx_dates.index(bottom_date)
    print(f"VNINDEX 2026 bottom: {bottom_date} @ {bottom_row['low']:.1f}\n")

    sig = []
    for p in sorted(DATA.glob("*.csv")):
        if p.stem == "VNINDEX":
            continue
        df = pd.read_csv(p); df["date"] = df["date"].astype(str).str.slice(0, 10)
        if len(df) < 120:
            continue
        # VN30-specific looser gate (rsi<35, band x1.02); strict for the rest
        srules = dict(RULES)
        if p.stem in VN30 and RULES.get("vn30_mean_reversion"):
            srules["mean_reversion"] = {**RULES.get("mean_reversion", {}), **RULES["vn30_mean_reversion"]}
        feats = prepare_signal_frame(df.sort_values("date").reset_index(drop=True), srules)
        rsi = feats["rsi14"]; pre = (rsi < 40) & (feats["close"] <= feats["bb_lower"] * 1.04) & \
              (feats["date"] >= "2026-01-01") & (feats.index >= 90)
        for i in np.where(pre.fillna(False))[0]:
            s = score_precomputed_at(p.stem, feats, int(i), srules)
            if s.decision == "BUY_SETUP":
                d = str(feats["date"].iloc[int(i)])
                # sessions from bottom (negative = before bottom, positive = after)
                dist = (idx_dates.index(d) - bi) if d in idx_dates else None
                sig.append((d, p.stem, p.stem in VN30, dist, s.latest_close))

    sig.sort()
    print(f"Tin hieu bat day 2026: {len(sig)} (VN30: {sum(1 for x in sig if x[2])})\n")
    print(f"{'ngay':12s} {'ma':6s} {'VN30':5s} {'so phien tu day':>16s}  gia")
    print("-" * 56)
    for d, sym, is30, dist, close in sig:
        tag = "VN30" if is30 else ""
        ds = f"{dist:+d}" if dist is not None else "?"
        print(f"{d:12s} {sym:6s} {tag:5s} {ds:>16s}  {close:,.0f}")

    # by month
    print("\nTheo thang:")
    bym = {}
    for d, sym, is30, dist, close in sig:
        bym.setdefault(d[:7], []).append(sym)
    for m in sorted(bym):
        syms = bym[m]; n30 = sum(1 for s in syms if s in VN30)
        print(f"  {m}: {len(syms):2d} tin hieu ({n30} VN30) -> {', '.join(syms)}")

    near = [x for x in sig if x[3] is not None and abs(x[3]) <= 10]
    print(f"\nTrong +/-10 phien quanh day ({bottom_date}): {len(near)} tin hieu, "
          f"{sum(1 for x in near if x[2])} la VN30")


if __name__ == "__main__":
    main()
