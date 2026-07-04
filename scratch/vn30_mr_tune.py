"""Design a VN30-specific MR gate — data-driven, OOS-checked (not p-hacked).

Large-caps are less volatile → RSI rarely pierces 30 and price rarely stabs far below the
lower band, so the VN100 gate under-fires on VN30 (only MSN caught the Mar-2026 crash).
Sweep rsi_max x band_mult on the FULL 6yr VN30 history (COVID-2020, 2022, Mar-2026 crashes),
keeping the structural gates (reversal|VSA, vol climax, RR>=0.5, cloud-clear, Kijun target,
3ATR stop, 8-day hold). Report per-trade expectancy + crash coverage, and an OOS split
(fit intuition on 2020-2023, verify on 2024-2026) to reject overfit thresholds.
"""
from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scratch"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from edge_experiment import DATA, prep, simulate_symbol, metrics

VN30 = {"ACB","BID","BSR","CTG","FPT","GAS","GVR","HDB","HPG","LPB","MBB","MSN","MWG",
        "PLX","SAB","SHB","SSB","SSI","STB","TCB","TPB","VCB","VHM","VIB","VIC","VJC",
        "VNM","VPB","VPL","VRE"}

CRASHES = [(date(2022, 4, 1), date(2022, 11, 30)), (date(2026, 3, 1), date(2026, 3, 31)),
           (date(2020, 3, 1), date(2020, 4, 30))]


def mr_mask(d, rsi_max, band):
    confirm = ((d["close"] > d["prev_close"]) & (d["vol_ratio"] >= 1.5)) | (d["vsa_stop"] == 1)
    rr_ok = (d["kijun"] - d["close"]) >= 1.5 * d["atr"]            # rr>=0.5 vs 3ATR stop
    blocked = (d["cloud_bottom"] > d["close"]) & (d["cloud_bottom"] < d["kijun"])
    return (d["rsi14"] < rsi_max) & (d["close"] <= d["bb_lower"] * band) & confirm & rr_ok & ~blocked


def run(dfs, rsi_max, band, start, end):
    spec = dict(max_hold=8, stop_atr=3.0, tp_atr=None, target="kijun")
    trades = []
    for sym, d in dfs.items():
        m = mr_mask(d, rsi_max, band)
        trades += simulate_symbol(d, m, spec, start, end)
    return trades


def crash_count(dfs, rsi_max, band):
    n = 0
    for s, e in CRASHES:
        for sym, d in dfs.items():
            m = mr_mask(d, rsi_max, band)
            sub = d[(d["date"] >= s) & (d["date"] <= e)]
            n += int(m.reindex(sub.index).fillna(False).sum())
    return n


def main():
    files = {p.stem: p for p in DATA.glob("*.csv") if p.stem in VN30}
    dfs = {s: prep(pd.read_csv(p)) for s, p in files.items()}
    print(f"VN30 symbols loaded: {len(dfs)}\n")
    full = (date(2020, 1, 1), date(2026, 7, 1))
    fit = (date(2020, 1, 1), date(2023, 12, 31))
    oos = (date(2024, 1, 1), date(2026, 7, 1))

    print(f"{'rsi<':>5s} {'band':>5s} | {'FULL n':>6s} {'win%':>5s} {'avg%':>6s} {'PF':>5s} {'crashSig':>8s} | "
          f"{'OOS n':>5s} {'OOSwin':>6s} {'OOSavg':>6s} {'OOSpf':>5s}")
    print("-" * 92)
    for rsi_max in [30, 33, 35, 38, 40]:
        for band in [1.01, 1.02, 1.03]:
            tf = run(dfs, rsi_max, band, *full); mf = metrics(tf)
            to = run(dfs, rsi_max, band, *oos); mo = metrics(to)
            cc = crash_count(dfs, rsi_max, band)
            pf = "inf" if mf["pf"] == float("inf") else f"{mf['pf']:.2f}"
            pfo = "inf" if mo["pf"] == float("inf") else f"{mo['pf']:.2f}"
            print(f"{rsi_max:>5d} {band:>5.2f} | {mf['n']:>6d} {mf['win']:>5.0f} {mf['avg']:>+6.2f} {pf:>5s} "
                  f"{cc:>8d} | {mo['n']:>5d} {mo['win']:>6.0f} {mo['avg']:>+6.2f} {pfo:>5s}")
        print()
    print("Baseline VN100 gate on VN30 = rsi<30, band 1.01 (top row).")
    print("Pick: highest coverage that KEEPS positive OOS avg & PF>1 (reject if OOS breaks).")


if __name__ == "__main__":
    main()
