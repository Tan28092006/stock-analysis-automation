"""Quant-grade momentum, rigorously backtested. Techniques a quant uses (vs an analyst):
  - 12-1 momentum: rank by return from t-252 to t-21 (skip last month; 1-mo reversal is
    negative, per our IC).
  - inverse-vol (risk-parity) weights, not equal weight.
  - vol-TARGETING exposure: scale total exposure by target_vol / market_vol -> de-risk in
    high-vol regimes (Barroso-Santa Clara momentum-crash control).
  - buffering: hold a name until it drops out of the top 2N (cuts turnover/cost).
  - breadth: VN100 universe (30 names of VN30 is too few; IR = IC x sqrt(breadth)).

QC: next-open fills, full buy/sell/trim rebalancing, monthly. Param sweep + cost
sensitivity + per-regime + OOS. Benchmark = universe equal-weight B&H (the honest bar),
plus VNINDEX. Survivorship caveat stands for VN100 (today's list back-applied).
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

LOT = 100; INIT = 1_000_000_000.0
VN30 = {"ACB","BID","BSR","CTG","FPT","GAS","GVR","HDB","HPG","LPB","MBB","MSN","MWG",
        "PLX","SAB","SHB","SSB","SSI","STB","TCB","TPB","VCB","VHM","VIB","VIC","VJC",
        "VNM","VPB","VPL","VRE"}


def rl(x): return int(math.floor(max(0, x) / LOT) * LOT)


def load(uni):
    files = {p.stem: p for p in DATA.glob("*.csv") if p.stem != "VNINDEX" and (uni is None or p.stem in uni)}
    frames = {}
    for s, p in files.items():
        f = prep(pd.read_csv(p)); f["date"] = f["date"].astype(str)
        r = f["close"].pct_change()
        f["vol60"] = r.rolling(60, min_periods=30).std() * math.sqrt(252)
        f["c"] = f["close"].to_numpy()
        frames[s] = f.reset_index(drop=True)
    maps = {s: {d: i for i, d in enumerate(f["date"])} for s, f in frames.items()}
    return frames, maps


def market_vol(idx):
    r = idx["close"].pct_change()
    v = r.rolling(20, min_periods=10).std() * math.sqrt(252)
    return dict(zip(idx["date"].astype(str), v))


def simulate(frames, maps, calendar, mkt_vol, N, lb_long, lb_skip, target_vol, cost,
             buffer_mult=2, rebal=21):
    bc = sc = cost / 2 / 100
    cash = INIT; hold = {}; pending = {}; nav = []
    for di, d in enumerate(calendar):
        # execute pending target-values at TODAY's open (sells first to free cash)
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
                    add = des - cur; c = add * op * (1 + bc)
                    if c <= cash: cash -= c; hold[s] = {"qty": des}
            pending = {}
        px = {s: float(frames[s].at[maps[s][d], "c"]) for s in frames if d in maps[s]}
        if di % rebal == 0:
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
                add = [s for s in ranked[:N] if s not in keep]
                target = (keep + add)[:N]
                invv = {s: 1.0 / max(vols.get(s, 0.3), 0.05) for s in target}
                wsum = sum(invv.values()) or 1.0
                expo = min(1.0, target_vol / max(mkt_vol.get(d, 0.2), 1e-6))
                navc = cash + sum(hold[s]["qty"] * px.get(s, 0) for s in hold)
                tgt = {s: navc * expo * invv[s] / wsum for s in target}
                pending = {s: tgt.get(s, 0.0) for s in set(hold) | set(target)}
        nav.append((d, cash + sum(hold[s]["qty"] * px.get(s, 0) for s in hold)))
    return pd.DataFrame(nav, columns=["date", "nav"])


def metric(nav):
    s = nav.set_index("date")["nav"]; yrs = len(s) / 252
    cagr = (s.iloc[-1] / INIT) ** (1 / yrs) - 1 if yrs > 0 else 0
    dd = ((s - s.cummax()) / s.cummax()).min(); r = s.pct_change().dropna()
    sh = r.mean() / r.std() * math.sqrt(252) if r.std() > 0 else 0
    return (s.iloc[-1] / INIT - 1) * 100, cagr * 100, dd * 100, sh, (cagr / abs(dd) if dd < 0 else 0)


def bench(idx, frames, s, e):
    w = idx[(idx["date"] >= s) & (idx["date"] <= e)]
    vni = (w["close"].iloc[-1] / w["close"].iloc[0] - 1) * 100
    rs = [(f[(f["date"] >= s) & (f["date"] <= e)]["c"].iloc[-1] / f[(f["date"] >= s) & (f["date"] <= e)]["c"].iloc[0] - 1) * 100
          for f in frames.values() if len(f[(f["date"] >= s) & (f["date"] <= e)]) > 1]
    return vni, float(np.mean(rs))


def main():
    idx = pd.read_csv(DATA / "VNINDEX.csv"); idx["date"] = idx["date"].astype(str).str.slice(0, 10)
    idx = idx.sort_values("date").reset_index(drop=True)
    mvol = market_vol(idx)
    periods = {"FULL 20-now": ("2020-06-01", "2026-07-03"), "OOS 24-26": ("2024-01-01", "2026-07-03"),
               "H2-2025 bull": ("2025-07-01", "2025-12-31"), "2022 crash": ("2022-04-01", "2022-11-30")}
    for uname, uni in [("VN100", None), ("VN30", VN30)]:
        frames, maps = load(uni)
        print(f"\n############### {uname} — quant momentum (12-1, inv-vol, vol-target 20%, buffered) ###############")
        print(f"{'period':13s} | {'TOTAL%':>7s} {'CAGR':>6s} {'DD':>6s} {'Sharpe':>6s} {'Calmar':>6s} | {'VNIDX':>6s} {'eqwB&H':>7s}")
        for pn, (s, e) in periods.items():
            cal = [d for d in idx["date"] if s <= d <= e]
            m = metric(simulate(frames, maps, cal, mvol, N=15, lb_long=252, lb_skip=21, target_vol=0.20, cost=0.4))
            vni, eqw = bench(idx, frames, s, e)
            print(f"{pn:13s} | {m[0]:>+7.0f} {m[1]:>6.1f} {m[2]:>6.1f} {m[3]:>6.2f} {m[4]:>6.2f} | {vni:>+6.0f} {eqw:>+7.0f}")

        cal_f = [d for d in idx["date"] if "2020-06-01" <= d <= "2026-07-03"]
        cal_o = [d for d in idx["date"] if "2024-01-01" <= d <= "2026-07-03"]
        print(f"\n  PARAM SWEEP (FULL total% / OOS total% / FULL Sharpe / FULL DD):")
        print(f"  {'N':>3s}{'lb':>5s}{'skip':>5s}{'tgtV':>6s} | {'FULL%':>6s} {'OOS%':>6s} {'ShF':>5s} {'ddF':>6s}")
        for N in [10, 15, 20]:
            for lb, sk in [(252, 21), (126, 21)]:
                for tv in [0.15, 0.20, 0.30]:
                    mf = metric(simulate(frames, maps, cal_f, mvol, N, lb, sk, tv, 0.4))
                    mo = metric(simulate(frames, maps, cal_o, mvol, N, lb, sk, tv, 0.4))
                    print(f"  {N:>3d}{lb:>5d}{sk:>5d}{tv:>6.2f} | {mf[0]:>+6.0f} {mo[0]:>+6.0f} {mf[3]:>5.2f} {mf[2]:>6.1f}")
        print(f"\n  COST SENSITIVITY (N=15, 12-1, tgtV=0.20):")
        print(f"  {'cost%':>6s} | {'FULL%':>6s} {'CAGR':>5s} {'DD':>6s} {'Sharpe':>6s} | {'OOS%':>6s}")
        for c in [0.2, 0.4, 0.6, 0.8]:
            mf = metric(simulate(frames, maps, cal_f, mvol, 15, 252, 21, 0.20, c))
            mo = metric(simulate(frames, maps, cal_o, mvol, 15, 252, 21, 0.20, c))
            print(f"  {c:>6.1f} | {mf[0]:>+6.0f} {mf[1]:>5.1f} {mf[2]:>6.1f} {mf[3]:>6.2f} | {mo[0]:>+6.0f}")


if __name__ == "__main__":
    main()
