"""Can a BETTER stock-selection factor lift profit without market timing? Rigorous test.

The bear-filter (timing) failed = whipsaw. Instead diversify the SELECTION factor to cut
momentum-crash structurally. Variants (same engine: inv-vol weight, vol-target, buffer,
next-open fills, monthly rebal, cost 0.4%), ranked cross-sectionally each month:
  mom        : 12-1 return (baseline)
  excess     : 12-1 return - VNINDEX 12-1 (relative strength)
  mom_lowvol : z(mom) - z(vol)  (favor strong AND low-vol -> defensive momentum)
  defensive  : rank by mom, but ONLY among the lower-half-vol names
  high52w    : close / 252-day high (52-week-high proximity, George-Hwang)
Reported per year (2022/24/25/26) + FULL + OOS + Sharpe/DD. VN30 (trustworthy).
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

N, LB, SKIP, TV, COST, BUF, REBAL = 10, 252, 21, 0.20, 0.4, 2, 21
DATA = Path(r"D:\Chungkhoan\data\raw\prices_hist")


def zscore(d: dict) -> dict:
    v = np.array(list(d.values())); mu, sd = v.mean(), v.std()
    if sd == 0: return {k: 0.0 for k in d}
    return {k: (x - mu) / sd for k, x in d.items()}


def simulate(frames, maps, calendar, mkt_vol, mkt_mom, mode):
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
        if di % REBAL == 0:
            mom, vol, hi52 = {}, {}, {}
            for s, f in frames.items():
                if d not in maps[s]: continue
                j = maps[s][d]
                if j < LB + 1: continue
                cn, ct = float(f.at[j - SKIP, "c"]), float(f.at[j - LB, "c"])
                v = f.at[j, "vol60"]
                if ct > 0 and cn > 0 and np.isfinite(v):
                    mom[s] = cn / ct - 1; vol[s] = float(v)
                    hh = float(np.max(f["c"].to_numpy()[j - LB:j + 1]))
                    hi52[s] = float(f.at[j, "c"]) / hh if hh > 0 else 0
            if mom:
                if mode == "mom":
                    score = mom
                elif mode == "excess":
                    mm = mkt_mom.get(d, 0.0); score = {s: mom[s] - mm for s in mom}
                elif mode == "mom_lowvol":
                    zm, zv = zscore(mom), zscore(vol); score = {s: zm[s] - zv[s] for s in mom}
                elif mode == "defensive":
                    med = np.median(list(vol.values())); score = {s: mom[s] for s in mom if vol[s] <= med}
                elif mode == "high52w":
                    score = hi52
                else:
                    score = mom
                if not score: nav.append((d, cash + sum(hold[s]["qty"] * px.get(s, 0) for s in hold))); continue
                ranked = sorted(score, key=lambda s: -score[s])
                keep = [s for s in hold if s in set(ranked[:N * BUF])]
                target = (keep + [s for s in ranked[:N] if s not in keep])[:N]
                inv = {s: 1 / max(vol.get(s, .3), .05) for s in target}; ws = sum(inv.values()) or 1
                expo = min(1.0, TV / max(mkt_vol.get(d, .2), 1e-6))
                navc = cash + sum(hold[s]["qty"] * px.get(s, 0) for s in hold)
                tgt = {s: navc * expo * inv[s] / ws for s in target}
                pending = {s: tgt.get(s, 0.0) for s in set(hold) | set(target)}
        nav.append((d, cash + sum(hold[s]["qty"] * px.get(s, 0) for s in hold)))
    return pd.DataFrame(nav, columns=["date", "nav"])


def main():
    idx = pd.read_csv(DATA / "VNINDEX.csv"); idx["date"] = idx["date"].astype(str).str.slice(0, 10)
    idx = idx.sort_values("date").reset_index(drop=True)
    mvol = market_vol(idx)
    ic = idx["close"].to_numpy()
    mkt_mom = {idx["date"].iloc[i]: (ic[i - SKIP] / ic[i - LB] - 1)
               for i in range(LB + 1, len(idx))}
    frames, maps = load(VN30)
    periods = {"2022": ("2022-01-01", "2022-12-31"), "2024": ("2024-01-01", "2024-12-31"),
               "2025": ("2025-01-01", "2025-12-31"), "2026": ("2026-01-01", "2026-07-03"),
               "FULL": ("2020-06-01", "2026-07-03"), "OOS": ("2024-01-01", "2026-07-03")}
    print("VN30 selection-factor test (TOTAL% per period; FULL also Sharpe/DD)\n")
    print(f"{'factor':11s} | " + " | ".join(f"{k:>6s}" for k in periods) + " | FULL Sh/DD")
    print("-" * 88)
    for mode in ["mom", "excess", "mom_lowvol", "defensive", "high52w"]:
        cells, shdd = [], ""
        for pn, (s, e) in periods.items():
            cal = [d for d in idx["date"] if s <= d <= e]
            m = metric(simulate(frames, maps, cal, mvol, mkt_mom, mode))
            cells.append(f"{m[0]:>+6.0f}")
            if pn == "FULL": shdd = f"{m[3]:.2f}/{m[2]:.0f}"
        print(f"{mode:11s} | " + " | ".join(cells) + f" | {shdd}")
    print("\nVNINDEX:    | " + " | ".join(
        f"{(idx[(idx['date']>=s)&(idx['date']<=e)]['close'].iloc[-1]/idx[(idx['date']>=s)&(idx['date']<=e)]['close'].iloc[0]-1)*100:>+6.0f}"
        for pn, (s, e) in periods.items()))


if __name__ == "__main__":
    main()
