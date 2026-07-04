"""Test a BEAR-MARKET filter on quant-momentum: suppress momentum (to cash) during a
sustained downtrend, aiming to fix 2022 without whipsawing the bull years.

The kill-switch (daily EMA200) failed = whipsaw. Try SMOOTHER bear signals that only fire
in a sustained bear, not a fast dip:
  ema200      : index < EMA200 (the failed kill-switch, for reference)
  panic60_50  : > 50% of last 60 sessions had index < EMA50
  panic60_65  : > 65% of last 60 sessions had index < EMA50 (needs a deep, sustained bear)
When bear -> flatten to cash + no new entries; resume when it clears. Causal, next-open.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scratch"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from quant_momentum import load, market_vol, metric, rl, LOT, INIT, VN30
from stock_agent.features.indicators import ema

N, LB, SKIP, TV, COST, BUF, REBAL = 10, 252, 21, 0.20, 0.4, 2, 21
DATA = Path(r"D:\Chungkhoan\data\raw\prices_hist")


def bear_maps(idx):
    d = idx["date"].astype(str).to_numpy()
    c = idx["close"]
    e200 = ema(c, 200).to_numpy(); e50 = ema(c, 50).to_numpy(); cv = c.to_numpy()
    below50 = pd.Series((cv < e50).astype(float))
    frac60 = below50.rolling(60, min_periods=30).mean().to_numpy()
    m = {"none": {}, "ema200": {}, "panic60_50": {}, "panic60_65": {}}
    for i, dd in enumerate(d):
        m["none"][dd] = False
        m["ema200"][dd] = bool(np.isfinite(e200[i]) and cv[i] < e200[i])
        f = frac60[i]
        m["panic60_50"][dd] = bool(np.isfinite(f) and f > 0.50)
        m["panic60_65"][dd] = bool(np.isfinite(f) and f > 0.65)
    return m


def simulate(frames, maps, calendar, mkt_vol, bear):
    bc = sc = COST / 2 / 100
    cash = INIT; hold = {}; pending = {}; nav = []
    for di, d in enumerate(calendar):
        if pending:
            opens = {s: float(frames[s].at[maps[s][d], "open"]) for s in pending if d in maps[s]}
            for s in sorted(pending, key=lambda s: 0 if pending[s] < hold.get(s, {}).get("qty", 0) * opens.get(s, 0) else 1):
                if s not in opens: continue
                op = opens[s]; cur = hold.get(s, {}).get("qty", 0)
                des = rl(pending[s] / op) if op > 0 else 0
                if des < cur:
                    cash += (cur - des) * op * (1 - sc)
                    if des >= LOT: hold[s]["qty"] = des
                    else: hold.pop(s, None)
                elif des > cur:
                    c = (des - cur) * op * (1 + bc)
                    if c <= cash: cash -= c; hold[s] = {"qty": des}
            pending = {}
        px = {s: float(frames[s].at[maps[s][d], "c"]) for s in frames if d in maps[s]}
        if bear.get(d, False):
            if hold: pending = {s: 0.0 for s in hold}       # bear: flatten to cash
        elif di % REBAL == 0:
            scores, vols = {}, {}
            for s, f in frames.items():
                if d not in maps[s]: continue
                j = maps[s][d]
                if j < LB + 1: continue
                cn, ct = float(f.at[j - SKIP, "c"]), float(f.at[j - LB, "c"])
                v = f.at[j, "vol60"]
                if ct > 0 and cn > 0 and np.isfinite(v): scores[s] = cn / ct - 1; vols[s] = float(v)
            if scores:
                ranked = sorted(scores, key=lambda s: -scores[s])
                keep = [s for s in hold if s in set(ranked[:N * BUF])]
                target = (keep + [s for s in ranked[:N] if s not in keep])[:N]
                inv = {s: 1 / max(vols.get(s, .3), .05) for s in target}; ws = sum(inv.values()) or 1
                expo = min(1.0, TV / max(mkt_vol.get(d, .2), 1e-6))
                navc = cash + sum(hold[s]["qty"] * px.get(s, 0) for s in hold)
                tgt = {s: navc * expo * inv[s] / ws for s in target}
                pending = {s: tgt.get(s, 0.0) for s in set(hold) | set(target)}
        nav.append((d, cash + sum(hold[s]["qty"] * px.get(s, 0) for s in hold)))
    return pd.DataFrame(nav, columns=["date", "nav"])


def main():
    idx = pd.read_csv(DATA / "VNINDEX.csv"); idx["date"] = idx["date"].astype(str).str.slice(0, 10)
    idx = idx.sort_values("date").reset_index(drop=True)
    mvol = market_vol(idx); bm = bear_maps(idx)
    periods = {"2022 crash": ("2022-01-01", "2022-12-31"), "2024": ("2024-01-01", "2024-12-31"),
               "2025": ("2025-01-01", "2025-12-31"), "2026": ("2026-01-01", "2026-07-03"),
               "FULL 20-now": ("2020-06-01", "2026-07-03"), "OOS 24-26": ("2024-01-01", "2026-07-03")}
    frames, maps = load(VN30)
    print("VN30 quant-momentum · bear-filter comparison (TOTAL% per period, next-open, cost 0.4%)\n")
    print(f"{'filter':12s} | " + " | ".join(f"{k:>10s}" for k in periods))
    print("-" * (14 + 13 * len(periods)))
    for fname in ["none", "ema200", "panic60_50", "panic60_65"]:
        cells = []
        for pn, (s, e) in periods.items():
            cal = [d for d in idx["date"] if s <= d <= e]
            m = metric(simulate(frames, maps, cal, mvol, bm[fname]))
            cells.append(f"{m[0]:>+10.0f}")
        print(f"{fname:12s} | " + " | ".join(cells))
    print("\nVNINDEX:      | " + " | ".join(
        f"{(idx[(idx['date']>=s)&(idx['date']<=e)]['close'].iloc[-1]/idx[(idx['date']>=s)&(idx['date']<=e)]['close'].iloc[0]-1)*100:>+10.0f}"
        for pn, (s, e) in periods.items()))


if __name__ == "__main__":
    main()
