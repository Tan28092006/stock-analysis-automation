"""How many stocks fired around the historic 2026-03-09 crash?

Per trading day in the crash window, count across VN100 (VN30-looser gate, strict rest):
  BUY_SETUP = reversal confirmed -> actionable bottom-fish
  WATCH     = oversold AT the lower band but no reversal yet = capitulation in progress
The floor day itself shows many WATCH (falling knife); BUYs fire on the bounce. Causal.
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
WIN_START, WIN_END = "2026-03-02", "2026-03-20"


def rules_for(sym):
    if sym in VN30 and RULES.get("vn30_mean_reversion"):
        r = dict(RULES); r["mean_reversion"] = {**RULES["mean_reversion"], **RULES["vn30_mean_reversion"]}
        return r
    return RULES


def main():
    idx = pd.read_csv(DATA / "VNINDEX.csv"); idx["date"] = idx["date"].astype(str).str.slice(0, 10)
    idx = idx.sort_values("date").reset_index(drop=True)
    idx["chg"] = idx["close"].diff()
    print("VNINDEX quanh 09/03:")
    w = idx[(idx["date"] >= WIN_START) & (idx["date"] <= "2026-03-25")]
    for _, r in w.iterrows():
        star = "  <<< SAP" if r["chg"] < -40 else ""
        print(f"  {r['date']}  close {r['close']:7.1f}  ({r['chg']:+6.1f}){star}")

    # per-day BUY / WATCH counts across VN100
    buy = {}; watch = {}; buy_syms = {}
    for p in sorted(DATA.glob("*.csv")):
        if p.stem == "VNINDEX":
            continue
        df = pd.read_csv(p); df["date"] = df["date"].astype(str).str.slice(0, 10)
        if len(df) < 120:
            continue
        feats = prepare_signal_frame(df.sort_values("date").reset_index(drop=True), rules_for(p.stem))
        mask = (feats["date"] >= WIN_START) & (feats["date"] <= WIN_END) & (feats.index >= 90)
        for i in np.where(mask.fillna(False))[0]:
            d = str(feats["date"].iloc[int(i)])
            s = score_precomputed_at(p.stem, feats, int(i), rules_for(p.stem))
            if s.decision == "BUY_SETUP":
                buy[d] = buy.get(d, 0) + 1; buy_syms.setdefault(d, []).append(p.stem + ("*" if p.stem in VN30 else ""))
            elif s.decision == "WATCH":
                watch[d] = watch.get(d, 0) + 1

    print("\nTin hieu theo phien (VN100, * = VN30):")
    print(f"{'ngay':12s} {'BUY':>4s} {'WATCH':>6s}  cac ma BUY")
    print("-" * 70)
    for d in sorted(set(list(buy) + list(watch))):
        print(f"{d:12s} {buy.get(d,0):>4d} {watch.get(d,0):>6d}  {', '.join(buy_syms.get(d, []))}")


if __name__ == "__main__":
    main()
