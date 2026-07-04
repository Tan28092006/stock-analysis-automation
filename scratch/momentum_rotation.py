"""Bull-catcher: relative-strength MOMENTUM ROTATION (the right tool to ride uptrends).

Hold the top-N momentum stocks while the market is RISK_ON; exit a name the moment it
breaks its EMA50 (trend over); go fully to cash in PANIC/GRIND. Few trades, long holds,
ride the whole move — the opposite of the churny pullback engine.

Tested on H2-2025 (the steep pre-Christmas bull the MR engine missed) + full 2020-now +
OOS, VN30 and VN100, vs VNINDEX B&H. Costs 0.15% buy / 0.25% sell, lot 100.
"""
from __future__ import annotations

import math
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scratch"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from edge_experiment import DATA, prep
from stock_agent.features.indicators import ema, sma

LOT = 100
INIT = 1_000_000_000.0
N_HOLD = 5
REBAL = 10          # trading sessions
MOM_LOOKBACK = 63   # ~3 months
BUY_COST = 0.0015
SELL_COST = 0.0025
BREADTH_MIN = 0.40

VN30 = {"ACB","BID","BSR","CTG","FPT","GAS","GVR","HDB","HPG","LPB","MBB","MSN","MWG",
        "PLX","SAB","SHB","SSB","SSI","STB","TCB","TPB","VCB","VHM","VIB","VIC","VJC",
        "VNM","VPB","VPL","VRE"}


def round_lot(x):
    return int(math.floor(max(0, x) / LOT) * LOT)


def regime_map(idx, breadth):
    e50 = ema(idx["close"], 50)
    out = {}
    for d, c, e in zip(idx["date"].astype(str), idx["close"], e50):
        if not np.isfinite(e):
            continue
        out[d] = "RISK_ON" if (c > e and breadth.get(d, 0) >= BREADTH_MIN) else ("GRIND" if c > e else "PANIC")
    return out


def breadth_map(frames):
    above, tot = {}, {}
    for f in frames.values():
        m = sma(f["close"], 20)
        for d, c, mv in zip(f["date"].astype(str), f["close"], m):
            if np.isfinite(mv):
                tot[d] = tot.get(d, 0) + 1
                above[d] = above.get(d, 0) + (1 if c > mv else 0)
    return {d: above.get(d, 0) / n for d, n in tot.items() if n}


def simulate(frames, maps, regime, calendar):
    cash = INIT
    hold = {}   # sym -> {qty, entry}
    nav_series = []
    ntrades = 0
    for di, d in enumerate(calendar):
        px = {s: float(frames[s].at[maps[s][d], "close"]) for s in frames if d in maps[s]}
        e50 = {s: float(ema(frames[s]["close"], 50).iloc[maps[s][d]]) if d in maps[s] else None for s in hold}

        # daily trend-exit + risk-off exit
        reg = regime.get(d, "PANIC")
        for s in list(hold):
            if s not in px:
                continue
            broke = e50.get(s) is not None and np.isfinite(e50[s]) and px[s] < e50[s]
            if reg != "RISK_ON" or broke:
                cash += hold[s]["qty"] * px[s] * (1 - SELL_COST)
                del hold[s]; ntrades += 1

        # rebalance
        if di % REBAL == 0 and reg == "RISK_ON":
            # momentum ranks among names above EMA50 with enough history
            scores = {}
            for s, f in frames.items():
                if d not in maps[s]:
                    continue
                j = maps[s][d]
                if j < MOM_LOOKBACK + 1:
                    continue
                c = float(f.at[j, "close"]); c0 = float(f.at[j - MOM_LOOKBACK, "close"])
                e = float(ema(f["close"], 50).iloc[j])
                if c > e and c0 > 0:
                    scores[s] = c / c0 - 1
            target = set(sorted(scores, key=lambda s: -scores[s])[:N_HOLD])
            # sell holds not in target
            for s in list(hold):
                if s not in target and s in px:
                    cash += hold[s]["qty"] * px[s] * (1 - SELL_COST)
                    del hold[s]; ntrades += 1
            # buy target not held (equal weight of current NAV)
            nav_now = cash + sum(hold[s]["qty"] * px.get(s, hold[s]["entry"]) for s in hold)
            for s in sorted(target, key=lambda s: -scores[s]):
                if s in hold or s not in px:
                    continue
                budget = nav_now / N_HOLD
                qty = round_lot(min(budget, cash / (1 + BUY_COST)) / px[s])
                if qty >= LOT:
                    cash -= qty * px[s] * (1 + BUY_COST)
                    hold[s] = {"qty": qty, "entry": px[s]}; ntrades += 1

        nav = cash + sum(hold[s]["qty"] * px.get(s, hold[s]["entry"]) for s in hold)
        nav_series.append((d, nav))
    return pd.DataFrame(nav_series, columns=["date", "nav"]), ntrades


def metrics(nav):
    s = nav.set_index("date")["nav"]; yrs = len(s) / 252
    cagr = (s.iloc[-1] / INIT) ** (1 / yrs) - 1 if yrs > 0 else 0
    dd = ((s - s.cummax()) / s.cummax()).min()
    r = s.pct_change().dropna(); sharpe = r.mean() / r.std() * math.sqrt(252) if r.std() > 0 else 0
    return dict(total=(s.iloc[-1] / INIT - 1) * 100, cagr=cagr * 100, maxdd=dd * 100,
                sharpe=sharpe, calmar=cagr / abs(dd) if dd < 0 else 0)


def main():
    idx = pd.read_csv(DATA / "VNINDEX.csv"); idx["date"] = idx["date"].astype(str).str.slice(0, 10)
    idx = idx.sort_values("date").reset_index(drop=True)
    periods = {
        "H2-2025 bull": ("2025-07-01", "2025-12-31"),
        "YTD-2026": ("2026-01-01", "2026-07-01"),
        "FULL 2020-now": ("2020-06-01", "2026-07-01"),
        "OOS 2024-26": ("2024-01-01", "2026-07-01"),
    }
    for uname, uni in [("VN30", VN30), ("VN100", None)]:
        files = {p.stem: p for p in DATA.glob("*.csv") if p.stem != "VNINDEX" and (uni is None or p.stem in uni)}
        frames = {s: prep(pd.read_csv(p)) for s, p in files.items()}
        for f in frames.values():
            f["date"] = f["date"].astype(str)
        maps = {s: {d: i for i, d in enumerate(f["date"])} for s, f in frames.items()}
        breadth = breadth_map(frames)
        regime = regime_map(idx, breadth)
        print(f"\n===== Universe: {uname} ({len(frames)} symbols), top{N_HOLD} momentum, rebal {REBAL}d =====")
        print(f"{'period':16s} {'trades':>6s} | {'TOTAL%':>8s} {'CAGR%':>7s} {'maxDD%':>7s} {'Sharpe':>7s} | {'VNINDEX%':>9s}")
        print("-" * 74)
        for pname, (s, e) in periods.items():
            cal = [d for d in idx["date"] if s <= d <= e]
            nav, nt = simulate(frames, maps, regime, cal)
            m = metrics(nav)
            w = idx[(idx["date"] >= s) & (idx["date"] <= e)]
            bench = (w["close"].iloc[-1] / w["close"].iloc[0] - 1) * 100
            print(f"{pname:16s} {nt:>6d} | {m['total']:>+8.1f} {m['cagr']:>+7.1f} {m['maxdd']:>7.1f} "
                  f"{m['sharpe']:>7.2f} | {bench:>+9.1f}")


if __name__ == "__main__":
    main()
