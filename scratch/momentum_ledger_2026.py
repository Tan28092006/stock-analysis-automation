"""What did quant-momentum actually hold/trade YTD 2026? Monthly-rebalance ledger.

Runs the production quant-momentum (12-1, inv-vol, vol-target, buffered, next-open fills,
cost 0.4%) over 2026-01-01 -> now on VN100, logging each completed round-trip (entry->full
exit) + current holdings, vs VNINDEX YTD.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scratch"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from quant_momentum import load, market_vol, rl, LOT, INIT, VN30

N, LB, SKIP, TV, COST, BUF, REBAL = 10, 252, 21, 0.20, 0.4, 2, 21
DATA = Path(r"D:\Chungkhoan\data\raw\prices_hist")
import argparse
from stock_agent.features.indicators import ema, sma
_ap = argparse.ArgumentParser()
_ap.add_argument("--start", default="2026-01-01"); _ap.add_argument("--end", default="2026-07-03")
_a = _ap.parse_args()
START, END = _a.start, _a.end


def regime_breakdown(idx, frames, cal):
    e50 = ema(idx["close"], 50); e_by = dict(zip(idx["date"].astype(str), e50))
    c_by = dict(zip(idx["date"].astype(str), idx["close"]))
    above, tot = {}, {}
    for f in frames.values():
        m = sma(f["close"], 20)
        for d, c, mv in zip(f["date"].astype(str), f["close"], m):
            if np.isfinite(mv):
                tot[d] = tot.get(d, 0) + 1; above[d] = above.get(d, 0) + (1 if c > mv else 0)
    breadth = {d: above.get(d, 0) / n for d, n in tot.items() if n}
    cnt = {"RISK_ON": 0, "GRIND": 0, "PANIC": 0}
    for d in cal:
        e = e_by.get(d); c = c_by.get(d)
        if e is None or not np.isfinite(e): continue
        cnt["RISK_ON" if (c > e and breadth.get(d, 0) >= 0.40) else ("GRIND" if c > e else "PANIC")] += 1
    return cnt


def main():
    idx = pd.read_csv(DATA / "VNINDEX.csv"); idx["date"] = idx["date"].astype(str).str.slice(0, 10)
    idx = idx.sort_values("date").reset_index(drop=True)
    mvol = market_vol(idx)
    frames, maps = load(None)
    cal = [d for d in idx["date"] if START <= d <= END]
    bc = sc = COST / 2 / 100

    cash = INIT; hold = {}; pending = {}; trades = []
    for di, d in enumerate(cal):
        if pending:
            opens = {s: float(frames[s].at[maps[s][d], "open"]) for s in pending if d in maps[s]}
            for s in sorted(pending, key=lambda s: 0 if pending[s] < hold.get(s, {}).get("qty", 0) * opens.get(s, 0) else 1):
                if s not in opens: continue
                op = opens[s]; cur = hold.get(s, {}).get("qty", 0)
                des = rl(pending[s] / op) if op > 0 else 0
                if des < cur:
                    cash += (cur - des) * op * (1 - sc)
                    if des >= LOT:
                        hold[s]["qty"] = des
                    else:
                        h = hold.pop(s)                       # full exit -> record round trip
                        net = (op / h["avg"] - 1) * 100 - COST
                        trades.append((h["entry_date"], s, s in VN30, d, round(h["avg"], 0), round(op, 0), round(net, 2)))
                elif des > cur:
                    c = (des - cur) * op * (1 + bc)
                    if c <= cash:
                        cash -= c
                        if s in hold:
                            tot = hold[s]["qty"] * hold[s]["avg"] + (des - cur) * op
                            hold[s] = {"qty": des, "avg": tot / des, "entry_date": hold[s]["entry_date"]}
                        else:
                            hold[s] = {"qty": des, "avg": op, "entry_date": d}
            pending = {}
        px = {s: float(frames[s].at[maps[s][d], "c"]) for s in frames if d in maps[s]}
        if di % REBAL == 0:
            sc_, vol = {}, {}
            for s, f in frames.items():
                if d not in maps[s]: continue
                j = maps[s][d]
                if j < LB + 1: continue
                cn, ct = float(f.at[j - SKIP, "c"]), float(f.at[j - LB, "c"])
                v = f.at[j, "vol60"]
                if ct > 0 and cn > 0 and np.isfinite(v): sc_[s] = cn / ct - 1; vol[s] = float(v)
            if sc_:
                ranked = sorted(sc_, key=lambda s: -sc_[s])
                keep = [s for s in hold if s in set(ranked[:N * BUF])]
                target = (keep + [s for s in ranked[:N] if s not in keep])[:N]
                inv = {s: 1 / max(vol.get(s, .3), .05) for s in target}; ws = sum(inv.values()) or 1
                expo = min(1.0, TV / max(mvol.get(d, .2), 1e-6))
                navc = cash + sum(hold[s]["qty"] * px.get(s, 0) for s in hold)
                tgt = {s: navc * expo * inv[s] / ws for s in target}
                pending = {s: tgt.get(s, 0.0) for s in set(hold) | set(target)}

    bench = (idx[idx["date"].isin(cal)]["close"].iloc[-1] / idx[idx["date"].isin(cal)]["close"].iloc[0] - 1) * 100
    nav = cash + sum(hold[s]["qty"] * float(frames[s].at[maps[s][END if END in maps[s] else cal[-1]], "c"]) for s in hold if (END in maps[s] or cal[-1] in maps[s]))

    rb = regime_breakdown(idx, frames, cal)
    n = sum(rb.values()) or 1
    print(f"Quant momentum ledger {START} -> {END}  | VNINDEX {bench:+.1f}%")
    print(f"Xu huong he PHAT HIEN: RISK_ON {rb['RISK_ON']}p ({rb['RISK_ON']*100//n}%) · "
          f"PANIC {rb['PANIC']}p ({rb['PANIC']*100//n}%) · GRIND {rb['GRIND']}p ({rb['GRIND']*100//n}%)\n")
    print("--- KEO DA DONG (round trips) ---")
    print(f"{'vao':11s} {'ma':6s} {'VN30':5s} {'ra':11s} {'gia_vao':>8s} {'gia_ra':>8s} {'net%':>7s}")
    for t in sorted(trades):
        print(f"{t[0]:11s} {t[1]:6s} {'VN30' if t[2] else '':5s} {t[3]:11s} {t[4]:>8,.0f} {t[5]:>8,.0f} {t[6]:>+7.2f}")
    if trades:
        nets = [t[6] for t in trades]
        print(f"\n{len(nets)} keo dong | win {sum(1 for x in nets if x>0)}/{len(nets)} | trung binh {np.mean(nets):+.2f}%/keo")
    print("\n--- DANG GIU (open) ---")
    print(f"{'ma':6s} {'VN30':5s} {'vao':11s} {'gia_vao':>8s} {'gia_nay':>8s} {'uPnL%':>7s} {'w%':>6s}")
    for s, h in sorted(hold.items(), key=lambda kv: kv[1]["entry_date"]):
        cur = float(frames[s].at[maps[s][cal[-1]], "c"]) if cal[-1] in maps[s] else h["avg"]
        up = (cur / h["avg"] - 1) * 100
        w = h["qty"] * cur / nav * 100
        print(f"{s:6s} {'VN30' if s in VN30 else '':5s} {h['entry_date']:11s} {h['avg']:>8,.0f} {cur:>8,.0f} {up:>+7.2f} {w:>5.1f}%")
    print(f"\nNAV cuoi: {(nav/INIT-1)*100:+.1f}%  vs VNINDEX {bench:+.1f}%")


if __name__ == "__main__":
    main()
