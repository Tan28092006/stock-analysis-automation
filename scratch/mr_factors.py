"""Can a better MR (bottom-fishing) ENTRY lift profit? Rigorous test, same discipline as
the momentum factor test. Same exit for all (stop 3xATR, target Kijun, hold<=15, T+2,
cost 0.6%); vary only the entry gate. VN100 (breadth), per-period + crash + FULL.

  baseline : RSI14<30 & at BB-lower & (reversal+climax | VSA) & RR>=0.5 & cloud-clear
  rsi2     : swap RSI14<30 -> RSI2<10 (Connors — strongest short-term reversal per our IC)
  rsi14_25 : deeper oversold RSI14<25
  combo    : RSI14<35 AND RSI2<10 (both must agree)
  connors  : RSI2<10 & close>SMA200 (classic mean-reversion IN uptrend, no band/cloud)
  deepband : RSI14<30 & close<=BB-lower*0.99 (stab deeper below the band)
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
from edge_experiment import DATA, prep, simulate_symbol, metrics

SPEC = dict(max_hold=15, stop_atr=3.0, tp_atr=None, target="kijun")
PERIODS = {"2022crash": (date(2022, 4, 1), date(2022, 11, 30)), "2024": (date(2024, 1, 1), date(2024, 12, 31)),
           "2025": (date(2025, 1, 1), date(2025, 12, 31)), "2026": (date(2026, 1, 1), date(2026, 7, 3)),
           "FULL": (date(2020, 1, 1), date(2026, 7, 3))}


def _confirm(d):
    return ((d["close"] > d["prev_close"]) & (d["vol_ratio"] >= 1.5)) | (d["vsa_stop"] == 1)

def _rr(d):
    return (d["kijun"] - d["close"]) >= 1.5 * d["atr"]

def _cloud_ok(d):
    blocked = (d["cloud_bottom"] > d["close"]) & (d["cloud_bottom"] < d["kijun"])
    return ~blocked

def m_baseline(d):
    return (d["rsi14"] < 30) & (d["close"] <= d["bb_lower"] * 1.01) & _confirm(d) & _rr(d) & _cloud_ok(d)

def m_rsi2(d):
    return (d["rsi2"] < 10) & (d["close"] <= d["bb_lower"] * 1.01) & _confirm(d) & _rr(d) & _cloud_ok(d)

def m_rsi14_25(d):
    return (d["rsi14"] < 25) & (d["close"] <= d["bb_lower"] * 1.01) & _confirm(d) & _rr(d) & _cloud_ok(d)

def m_combo(d):
    return (d["rsi14"] < 35) & (d["rsi2"] < 10) & (d["close"] <= d["bb_lower"] * 1.01) & _confirm(d) & _rr(d) & _cloud_ok(d)

def m_connors(d):
    return (d["rsi2"] < 10) & (d["close"] > d["sma200"]) & (d["close"] > d["prev_close"])

def m_deepband(d):
    return (d["rsi14"] < 30) & (d["close"] <= d["bb_lower"] * 0.99) & _confirm(d) & _rr(d) & _cloud_ok(d)

VARIANTS = {"baseline": m_baseline, "rsi2": m_rsi2, "rsi14_25": m_rsi14_25,
            "combo": m_combo, "connors": m_connors, "deepband": m_deepband}


def main():
    files = {p.stem: p for p in DATA.glob("*.csv") if p.stem != "VNINDEX"}
    dfs = {s: prep(pd.read_csv(p)) for s, p in files.items()}
    print(f"{len(dfs)} symbols · MR entry variants · exit=stop3ATR/Kijun/hold15 · cost 0.6%\n")
    print(f"{'variant':10s} {'period':10s} | {'n':>4s} {'win%':>5s} {'avg%':>6s} {'PF':>5s} {'port%':>7s} {'DD%':>6s}")
    print("-" * 66)
    for vname, fn in VARIANTS.items():
        for pn, (s, e) in PERIODS.items():
            trades = []
            for sym, d in dfs.items():
                m = fn(d)
                trades += simulate_symbol(d, m, SPEC, s, e)
            mk = metrics(trades)
            pf = "inf" if mk["pf"] == float("inf") else f"{mk['pf']:.2f}"
            print(f"{vname:10s} {pn:10s} | {mk['n']:>4d} {mk['win']:>5.0f} {mk['avg']:>+6.2f} {pf:>5s} "
                  f"{mk['total']:>+7.1f} {mk['dd']:>6.1f}")
        print()


if __name__ == "__main__":
    main()
