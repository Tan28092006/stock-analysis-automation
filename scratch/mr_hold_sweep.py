"""Does a longer hold (T+15) beat the current 8-session MR hold? Test, don't assume.

Hold is counted in TRADING SESSIONS (bar index) so non-trading days are skipped by
construction. Two sweeps on the MR-hybrid gate (VN100), reported over FULL / 2022 crash /
OOS 2024-26:
  1. max_hold in {8,10,12,15,20,30} with the Kijun target (isolate the hold effect).
  2. at max_hold=15, vary the target (Kijun / SMA20 / EMA50 / +2R / +3R) — a longer hold
     only pays if there is a further target to reach.
Also prints the exit-reason mix (target vs stop vs time) to explain WHY.
"""
from __future__ import annotations

import sys
from collections import Counter
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scratch"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from edge_experiment import DATA, prep, simulate_symbol, metrics, strat_mr_hybrid

PERIODS = {
    "FULL 2020-now": (date(2020, 1, 1), date(2026, 7, 1)),
    "2022 crash": (date(2022, 4, 1), date(2022, 11, 30)),
    "OOS 2024-26": (date(2024, 1, 1), date(2026, 7, 1)),
}


def run(dfs, spec, start, end):
    trades = []
    for sym, d in dfs.items():
        m, _ = strat_mr_hybrid(d)
        trades += simulate_symbol(d, m, spec, start, end)
    return trades


def line(name, trades):
    mk = metrics(trades)
    reasons = Counter(t["reason"] for t in trades)
    pf = "inf" if mk["pf"] == float("inf") else f"{mk['pf']:.2f}"
    mix = " ".join(f"{k}:{v}" for k, v in sorted(reasons.items()))
    return (f"{name:>14s} | n={mk['n']:>3d} win={mk['win']:>4.0f}% avg={mk['avg']:>+5.2f}% "
            f"PF={pf:>5s} port={mk['total']:>+6.1f}% DD={mk['dd']:>5.1f} | {mix}")


def main():
    files = {p.stem: p for p in DATA.glob("*.csv") if p.stem != "VNINDEX"}
    dfs = {s: prep(pd.read_csv(p)) for s, p in files.items()}
    print(f"{len(dfs)} symbols | MR-hybrid gate | stop 3xATR | hold in TRADING SESSIONS\n")

    print("=== SWEEP 1: max_hold (target = Kijun) ===")
    for pname, (s, e) in PERIODS.items():
        print(f"\n-- {pname} --")
        for hold in [8, 10, 12, 15, 20, 30]:
            spec = dict(max_hold=hold, stop_atr=3.0, tp_atr=None, target="kijun")
            print(line(f"hold={hold}", run(dfs, spec, s, e)))

    print("\n\n=== SWEEP 2: target at max_hold=15 ===")
    targets = [("kijun", dict(max_hold=15, stop_atr=3.0, tp_atr=None, target="kijun")),
               ("sma20", dict(max_hold=15, stop_atr=3.0, tp_atr=None, target="sma20")),
               ("ema50", dict(max_hold=15, stop_atr=3.0, tp_atr=None, target="ema50")),
               ("+2R",   dict(max_hold=15, stop_atr=3.0, tp_atr=6.0, target=None)),  # 2R vs 3ATR
               ("+3R",   dict(max_hold=15, stop_atr=3.0, tp_atr=9.0, target=None))]
    for pname, (s, e) in PERIODS.items():
        print(f"\n-- {pname} --")
        for tname, spec in targets:
            print(line(f"tgt={tname}", run(dfs, spec, s, e)))


if __name__ == "__main__":
    main()
