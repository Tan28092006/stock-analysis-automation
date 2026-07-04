"""Test a crash KILL-SWITCH on the quant-momentum CORE: go fully to cash on a hard
risk-off signal, to cap the momentum-crash drawdown (2022 was -30%).

Variants: none / EMA200 (VNINDEX<EMA200) / breadth<25% / both.
Question: how much DD does it save, and what does it cost in return? Reuses the exact
quant_momentum engine (12-1, inv-vol, vol-target, buffered, next-open, cost 0.4%).
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scratch"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from quant_momentum import load, market_vol, metric, bench, rl, LOT, INIT, VN30
from stock_agent.features.indicators import ema, sma


def breadth_map(frames):
    above, tot = {}, {}
    for f in frames.values():
        m = sma(f["close"], 20)
        for d, c, mv in zip(f["date"].astype(str), f["close"], m):
            if np.isfinite(mv):
                tot[d] = tot.get(d, 0) + 1; above[d] = above.get(d, 0) + (1 if c > mv else 0)
    return {d: above.get(d, 0) / n for d, n in tot.items() if n}


def kill_map(idx, breadth, mode):
    e200 = ema(idx["close"], 200)
    out = {}
    for d, c, e in zip(idx["date"].astype(str), idx["close"], e200):
        below = np.isfinite(e) and c < e
        brk = breadth.get(d, 1.0) < 0.25
        out[d] = {"none": False, "ema200": below, "breadth": brk, "both": below or brk}[mode]
    return out


def simulate_ks(frames, maps, calendar, mkt_vol, kmap, N=15, lb_long=252, lb_skip=21,
                target_vol=0.20, cost=0.4, buffer_mult=2, rebal=21):
    bc = sc = cost / 2 / 100
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
                    if des >= LOT: hold[s] = {"qty": des}
                    else: hold.pop(s, None)
                elif des > cur:
                    c = (des - cur) * op * (1 + bc)
                    if c <= cash: cash -= c; hold[s] = {"qty": des}
            pending = {}
        px = {s: float(frames[s].at[maps[s][d], "c"]) for s in frames if d in maps[s]}
        if kmap.get(d, False):
            if hold:                       # KILL: flatten everything next open
                pending = {s: 0.0 for s in hold}
        elif di % rebal == 0:
            scores, vols = {}, {}
            for s, f in frames.items():
                if d not in maps[s]: continue
                j = maps[s][d]
                if j < lb_long + 1: continue
                c_now, c_then = float(f.at[j - lb_skip, "c"]), float(f.at[j - lb_long, "c"])
                v = f.at[j, "vol60"]
                if c_then > 0 and c_now > 0 and np.isfinite(v):
                    scores[s] = c_now / c_then - 1; vols[s] = float(v)
            if scores:
                ranked = sorted(scores, key=lambda s: -scores[s])
                top2N = set(ranked[:N * buffer_mult])
                keep = [s for s in hold if s in top2N]
                target = (keep + [s for s in ranked[:N] if s not in keep])[:N]
                invv = {s: 1.0 / max(vols.get(s, 0.3), 0.05) for s in target}
                wsum = sum(invv.values()) or 1.0
                expo = min(1.0, target_vol / max(mkt_vol.get(d, 0.2), 1e-6))
                navc = cash + sum(hold[s]["qty"] * px.get(s, 0) for s in hold)
                tgt = {s: navc * expo * invv[s] / wsum for s in target}
                pending = {s: tgt.get(s, 0.0) for s in set(hold) | set(target)}
        nav.append((d, cash + sum(hold[s]["qty"] * px.get(s, 0) for s in hold)))
    return pd.DataFrame(nav, columns=["date", "nav"])


def main():
    idx = pd.read_csv(Path(r"D:\Chungkhoan\data\raw\prices_hist\VNINDEX.csv"))
    idx["date"] = idx["date"].astype(str).str.slice(0, 10); idx = idx.sort_values("date").reset_index(drop=True)
    mvol = market_vol(idx)
    periods = {"FULL 20-now": ("2020-06-01", "2026-07-03"), "OOS 24-26": ("2024-01-01", "2026-07-03"),
               "H2-2025 bull": ("2025-07-01", "2025-12-31"), "2022 crash": ("2022-04-01", "2022-11-30")}
    for uname, uni in [("VN30", VN30), ("VN100", None)]:
        frames, maps = load(uni)
        breadth = breadth_map(frames)
        print(f"\n############### {uname} — kill-switch test ###############")
        print(f"{'period':13s} {'kill':8s} | {'TOTAL%':>7s} {'DD':>6s} {'Sharpe':>6s} {'Calmar':>6s} | {'eqwB&H':>7s}")
        for pn, (s, e) in periods.items():
            cal = [d for d in idx["date"] if s <= d <= e]
            _, eqw = bench(idx, frames, s, e)
            for mode in ["none", "ema200", "breadth", "both"]:
                km = kill_map(idx, breadth, mode)
                m = metric(simulate_ks(frames, maps, cal, mvol, km))
                print(f"{pn:13s} {mode:8s} | {m[0]:>+7.0f} {m[2]:>6.1f} {m[3]:>6.2f} {m[4]:>6.2f} | {eqw:>+7.0f}")
            print()


if __name__ == "__main__":
    main()
