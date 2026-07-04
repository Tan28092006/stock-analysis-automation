"""Rigorous momentum-rotation backtest: is the edge real or a lucky param set?

Fixes the optimistic close-fill (execute at NEXT OPEN), then:
  1. PARAM SWEEP: N_HOLD x REBAL x LOOKBACK — is the result stable or a fluke?
  2. COST SENSITIVITY: does rotation churn survive higher round-trip costs?
Reported on VN30 (stable membership = trustworthy) and VN100 (survivorship-inflated),
FULL 2020-now + OOS 2024-26, vs VNINDEX B&H.
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

LOT = 100; INIT = 1_000_000_000.0; BREADTH_MIN = 0.40
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
    e50 = ema(idx["close"], 50); out = {}
    for d, c, e in zip(idx["date"].astype(str), idx["close"], e50):
        if np.isfinite(e):
            out[d] = (c > e and breadth.get(d, 0) >= BREADTH_MIN)
    return out


def simulate(frames, maps, e50s, regime, calendar, n_hold, rebal, lookback, cost_rt):
    """Next-open execution: decide on close[d], trade at open[d+1]. cost_rt = round-trip %."""
    buy_c = cost_rt / 2 / 100; sell_c = cost_rt / 2 / 100
    cash = INIT; hold = {}; pending_buys = []; pending_sells = []; nav = []
    for di, d in enumerate(calendar):
        # 1) execute pending orders at TODAY's open
        for s in pending_sells:
            if s in hold and d in maps[s]:
                op = float(frames[s].at[maps[s][d], "open"])
                cash += hold[s]["qty"] * op * (1 - sell_c); del hold[s]
        for s, budget in pending_buys:
            if s not in hold and d in maps[s]:
                op = float(frames[s].at[maps[s][d], "open"])
                qty = rl(min(budget, cash / (1 + buy_c)) / op)
                if qty >= LOT:
                    cash -= qty * op * (1 + buy_c); hold[s] = {"qty": qty}
        pending_buys, pending_sells = [], []

        px = {s: float(frames[s].at[maps[s][d], "close"]) for s in frames if d in maps[s]}
        on = regime.get(d, False)
        # 2) decide on TODAY's close -> queue for next open
        for s in list(hold):
            if s not in px: continue
            if (not on) or px[s] < float(e50s[s].iloc[maps[s][d]]):
                pending_sells.append(s)
        if on and di % rebal == 0:
            scores = {}
            for s, f in frames.items():
                if d not in maps[s]: continue
                j = maps[s][d]
                if j < lookback + 1: continue
                c0 = float(f.at[j - lookback, "close"])
                if px[s] > float(e50s[s].iloc[j]) and c0 > 0: scores[s] = px[s] / c0 - 1
            target = set(sorted(scores, key=lambda s: -scores[s])[:n_hold])
            for s in list(hold):
                if s not in target and s not in pending_sells: pending_sells.append(s)
            navc = cash + sum(hold[s]["qty"] * px.get(s, 0) for s in hold)
            for s in sorted(target, key=lambda s: -scores[s]):
                if s not in hold: pending_buys.append((s, navc / n_hold))
        nav.append((d, cash + sum(hold[s]["qty"] * px.get(s, 0) for s in hold)))
    return pd.DataFrame(nav, columns=["date", "nav"])


def metric(nav):
    s = nav.set_index("date")["nav"]; yrs = len(s) / 252
    cagr = (s.iloc[-1] / INIT) ** (1 / yrs) - 1 if yrs > 0 else 0
    dd = ((s - s.cummax()) / s.cummax()).min(); r = s.pct_change().dropna()
    sh = r.mean() / r.std() * math.sqrt(252) if r.std() > 0 else 0
    return (s.iloc[-1] / INIT - 1) * 100, cagr * 100, dd * 100, sh


def load(uni):
    files = {p.stem: p for p in DATA.glob("*.csv") if p.stem != "VNINDEX" and (uni is None or p.stem in uni)}
    frames = {s: prep(pd.read_csv(p)) for s, p in files.items()}
    for f in frames.values(): f["date"] = f["date"].astype(str)
    maps = {s: {d: i for i, d in enumerate(f["date"])} for s, f in frames.items()}
    e50s = {s: ema(f["close"], 50) for s, f in frames.items()}
    return frames, maps, e50s


def main():
    idx = pd.read_csv(DATA / "VNINDEX.csv"); idx["date"] = idx["date"].astype(str).str.slice(0, 10)
    idx = idx.sort_values("date").reset_index(drop=True)
    periods = {"FULL": ("2020-06-01", "2026-07-01"), "OOS 2024-26": ("2024-01-01", "2026-07-01")}
    for uname, uni in [("VN30", VN30), ("VN100", None)]:
        frames, maps, e50s = load(uni)
        regime = regime_map(idx, breadth_map(frames))
        cals = {k: [d for d in idx["date"] if v[0] <= d <= v[1]] for k, v in periods.items()}
        benches = {k: (idx[(idx["date"]>=v[0])&(idx["date"]<=v[1])]["close"].iloc[-1] /
                       idx[(idx["date"]>=v[0])&(idx["date"]<=v[1])]["close"].iloc[0] - 1)*100
                   for k, v in periods.items()}
        print(f"\n########## {uname} (next-open fills) ##########  B&H FULL {benches['FULL']:+.0f}% / OOS {benches['OOS 2024-26']:+.0f}%")
        print("\n=== 1) PARAM SWEEP (cost 0.4% rt) — TOTAL% [FULL | OOS], DD FULL ===")
        print(f"{'N':>3s}{'rebal':>6s}{'lookbk':>7s} | {'FULL%':>7s} {'ddFULL':>7s} {'ShFULL':>6s} | {'OOS%':>7s} {'ShOOS':>6s}")
        for n_hold in [3, 5, 8]:
            for rebal in [5, 10, 20]:
                for lb in [63, 126]:
                    tf = metric(simulate(frames, maps, e50s, regime, cals["FULL"], n_hold, rebal, lb, 0.4))
                    to = metric(simulate(frames, maps, e50s, regime, cals["OOS 2024-26"], n_hold, rebal, lb, 0.4))
                    print(f"{n_hold:>3d}{rebal:>6d}{lb:>7d} | {tf[0]:>+7.0f} {tf[2]:>7.1f} {tf[3]:>6.2f} | {to[0]:>+7.0f} {to[3]:>6.2f}")
        print("\n=== 2) COST SENSITIVITY (N=5, rebal=10, lookbk=63) ===")
        print(f"{'cost_rt%':>8s} | {'FULL%':>7s} {'CAGR':>6s} {'DD':>6s} {'Sharpe':>6s} | {'OOS%':>7s}")
        for cost in [0.2, 0.4, 0.6, 0.8, 1.0]:
            tf = metric(simulate(frames, maps, e50s, regime, cals["FULL"], 5, 10, 63, cost))
            to = metric(simulate(frames, maps, e50s, regime, cals["OOS 2024-26"], 5, 10, 63, cost))
            print(f"{cost:>8.1f} | {tf[0]:>+7.0f} {tf[1]:>6.1f} {tf[2]:>6.1f} {tf[3]:>6.2f} | {to[0]:>+7.0f}")


if __name__ == "__main__":
    main()
