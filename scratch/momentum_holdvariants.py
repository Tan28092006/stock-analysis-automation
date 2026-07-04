"""Does the rotation CHURN hurt? Compare low-churn variants, next-open fills, cost 0.4%.

  rotation : rebalance every 10d, SELL names that drop out of top-N (churn).
  hold     : buy top-N momentum, hold each until it breaks EMA50 / regime exits;
             only top up empty slots at rebalance (minimal churn — let winners run).
  basket   : hold ALL names above EMA50 when RISK_ON (no selection), equal weight.
vs VNINDEX B&H and VN30 equal-weight B&H.  VN30 (trustworthy) + VN100.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scratch"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from edge_experiment import DATA, prep
from stock_agent.features.indicators import ema, sma

LOT = 100; INIT = 1_000_000_000.0; BREADTH_MIN = 0.40; REBAL = 10; LB = 63
VN30 = {"ACB","BID","BSR","CTG","FPT","GAS","GVR","HDB","HPG","LPB","MBB","MSN","MWG",
        "PLX","SAB","SHB","SSB","SSI","STB","TCB","TPB","VCB","VHM","VIB","VIC","VJC",
        "VNM","VPB","VPL","VRE"}


def rl(x): return int(math.floor(max(0, x) / LOT) * LOT)


def breadth_map(frames):
    above, tot = {}, {}
    for f in frames.values():
        m = sma(f["close"], 20)
        for d, c, mv in zip(f["date"].astype(str), f["close"], m):
            if np.isfinite(mv):
                tot[d] = tot.get(d, 0) + 1; above[d] = above.get(d, 0) + (1 if c > mv else 0)
    return {d: above.get(d, 0) / n for d, n in tot.items() if n}


def regime_map(idx, breadth):
    e50 = ema(idx["close"], 50)
    return {d: (c > e and breadth.get(d, 0) >= BREADTH_MIN)
            for d, c, e in zip(idx["date"].astype(str), idx["close"], e50) if np.isfinite(e)}


def simulate(frames, maps, e50s, regime, calendar, n_hold, mode, cost=0.4):
    bc = sc = cost / 2 / 100
    cash = INIT; hold = {}; pb = []; ps = []; nav = []
    for di, d in enumerate(calendar):
        for s in ps:
            if s in hold and d in maps[s]:
                cash += hold[s]["qty"] * float(frames[s].at[maps[s][d], "open"]) * (1 - sc); del hold[s]
        for s, bud in pb:
            if s not in hold and d in maps[s]:
                op = float(frames[s].at[maps[s][d], "open"]); q = rl(min(bud, cash / (1 + bc)) / op)
                if q >= LOT: cash -= q * op * (1 + bc); hold[s] = {"qty": q}
        pb, ps = [], []
        px = {s: float(frames[s].at[maps[s][d], "close"]) for s in frames if d in maps[s]}
        on = regime.get(d, False)
        for s in list(hold):
            if s in px and (not on or px[s] < float(e50s[s].iloc[maps[s][d]])):
                ps.append(s)
        if on and di % REBAL == 0:
            scores = {}
            for s, f in frames.items():
                if d not in maps[s]: continue
                j = maps[s][d]
                if j < LB + 1: continue
                c0 = float(f.at[j - LB, "close"])
                if px[s] > float(e50s[s].iloc[j]) and c0 > 0: scores[s] = px[s] / c0 - 1
            ranked = sorted(scores, key=lambda s: -scores[s])
            navc = cash + sum(hold[s]["qty"] * px.get(s, 0) for s in hold)
            if mode == "basket":
                keep = set(ranked)                      # everyone above EMA50
                for s in list(hold):
                    if s not in keep and s not in ps: ps.append(s)
                eff = [s for s in hold if s not in ps]
                cap = min(len(ranked), 30)
                slots = cap - len(eff)
                for s in ranked:
                    if slots <= 0: break
                    if s not in hold and s not in [b[0] for b in pb]:
                        pb.append((s, navc / cap)); slots -= 1
            else:
                target = set(ranked[:n_hold])
                if mode == "rotation":
                    for s in list(hold):
                        if s not in target and s not in ps: ps.append(s)
                eff = [s for s in hold if s not in ps]
                slots = n_hold - len(eff)
                src = ranked[:n_hold] if mode == "rotation" else ranked
                for s in src:
                    if slots <= 0: break
                    if s not in hold and s not in [b[0] for b in pb]:
                        pb.append((s, navc / n_hold)); slots -= 1
        nav.append((d, cash + sum(hold[s]["qty"] * px.get(s, 0) for s in hold)))
    return pd.DataFrame(nav, columns=["date", "nav"])


def metric(nav):
    s = nav.set_index("date")["nav"]; yrs = len(s) / 252
    cagr = (s.iloc[-1] / INIT) ** (1 / yrs) - 1 if yrs > 0 else 0
    dd = ((s - s.cummax()) / s.cummax()).min(); r = s.pct_change().dropna()
    sh = r.mean() / r.std() * math.sqrt(252) if r.std() > 0 else 0
    return (s.iloc[-1] / INIT - 1) * 100, cagr * 100, dd * 100, sh


def main():
    idx = pd.read_csv(DATA / "VNINDEX.csv"); idx["date"] = idx["date"].astype(str).str.slice(0, 10)
    idx = idx.sort_values("date").reset_index(drop=True)
    periods = {"FULL": ("2020-06-01", "2026-07-03"), "OOS 24-26": ("2024-01-01", "2026-07-03"),
               "H2-2025": ("2025-07-01", "2025-12-31")}
    for uname, uni in [("VN30", VN30), ("VN100", None)]:
        files = {p.stem: p for p in DATA.glob("*.csv") if p.stem != "VNINDEX" and (uni is None or p.stem in uni)}
        frames = {s: prep(pd.read_csv(p)) for s, p in files.items()}
        for f in frames.values(): f["date"] = f["date"].astype(str)
        maps = {s: {dd: i for i, dd in enumerate(f["date"])} for s, f in frames.items()}
        e50s = {s: ema(f["close"], 50) for s, f in frames.items()}
        regime = regime_map(idx, breadth_map(frames))
        print(f"\n########## {uname} (next-open, cost 0.4%) ##########")
        print(f"{'period':10s} {'variant':9s} | {'TOTAL%':>7s} {'CAGR':>6s} {'DD':>6s} {'Sharpe':>6s} | {'VNIDX':>6s} {'eqwB&H':>7s}")
        for pn, (s, e) in periods.items():
            cal = [d for d in idx["date"] if s <= d <= e]
            w = idx[(idx["date"] >= s) & (idx["date"] <= e)]
            bench = (w["close"].iloc[-1] / w["close"].iloc[0] - 1) * 100
            eqw = np.mean([ (f[(f["date"]>=s)&(f["date"]<=e)]["close"].iloc[-1] /
                             f[(f["date"]>=s)&(f["date"]<=e)]["close"].iloc[0] - 1)*100
                            for f in frames.values() if len(f[(f["date"]>=s)&(f["date"]<=e)])>1 ])
            for mode in ["rotation", "hold", "basket"]:
                m = metric(simulate(frames, maps, e50s, regime, cal, 5, mode))
                print(f"{pn:10s} {mode:9s} | {m[0]:>+7.0f} {m[1]:>6.1f} {m[2]:>6.1f} {m[3]:>6.2f} | {bench:>+6.0f} {eqw:>+7.0f}")
            print()


if __name__ == "__main__":
    main()
