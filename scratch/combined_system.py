"""THE full system: CORE momentum-rotation (bull) + SATELLITE MR (crash), regime-switched.

  RISK_ON -> hold top-N momentum (CORE), exit a name on EMA50 break.
  PANIC   -> CORE is in cash; deploy it to MR bottom-fishing signals (risk-sized).
  GRIND   -> cash.

Compares COMBINED vs rotation-only vs MR-only vs VNINDEX across H2-2025 / YTD-2026 / FULL /
OOS, for VN30 and VN100. Costs 0.15%/0.25%, lot 100. Rotation fills at close (optimistic);
MR fills next open. Survivorship caveat applies to VN100 (today's list back-applied).
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

LOT = 100; INIT = 1_000_000_000.0
N_HOLD = 5; REBAL = 10; MOM_LB = 63
BUY_COST = 0.0015; SELL_COST = 0.0025; BREADTH_MIN = 0.40
MR_STOP_ATR = 3.0; MR_HOLD = 15; MR_RISK = 0.015; MR_MAXPOS = 4; T2 = 2
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
            out[d] = "RISK_ON" if (c > e and breadth.get(d, 0) >= BREADTH_MIN) else ("GRIND" if c > e else "PANIC")
    return out


def mr_orders(frames, maps, regime, uni):
    """Strict-hybrid MR entries (VN30 looser gate) whose signal day is PANIC."""
    by_entry = {}
    for sym, d in frames.items():
        rsi_max, band = (35, 1.02) if sym in VN30 else (30, 1.01)
        n = len(d)
        for i in range(90, n - 1):
            sd = str(d.at[i, "date"])
            if regime.get(sd) != "PANIC":
                continue
            atrv = float(d.at[i, "atr"]); close = float(d.at[i, "close"]); bbl = float(d.at[i, "bb_lower"])
            kij = float(d.at[i, "kijun"]) if np.isfinite(d.at[i, "kijun"]) else float(d.at[i, "ema21"])
            if not (np.isfinite(atrv) and atrv > 0 and np.isfinite(bbl)):
                continue
            confirm = (close > float(d.at[i, "prev_close"]) and float(d.at[i, "vol_ratio"]) >= 1.5) or int(d.at[i, "vsa_stop"]) == 1
            cb = float(d.at[i, "cloud_bottom"]) if np.isfinite(d.at[i, "cloud_bottom"]) else 0.0
            if (float(d.at[i, "rsi14"]) < rsi_max and close <= bbl * band and confirm
                    and (kij - close) >= 1.5 * atrv and not (cb and close < cb < kij)):
                entry = float(d.at[i + 1, "open"])
                if np.isfinite(entry) and entry > 0:
                    by_entry.setdefault(str(d.at[i + 1, "date"]), []).append(
                        dict(symbol=sym, entry_idx=i + 1, entry=entry, stop=entry - MR_STOP_ATR * atrv,
                             target=max(kij, close * 1.01)))
    return by_entry


def simulate(frames, maps, regime, calendar, mr_by_entry, use_core=True, use_mr=True):
    cash = INIT; core = {}; mr = []; nav_series = []
    e50c = {s: ema(f["close"], 50) for s, f in frames.items()}
    for di, d in enumerate(calendar):
        px = {s: float(frames[s].at[maps[s][d], "close"]) for s in frames if d in maps[s]}
        reg = regime.get(d, "PANIC")
        # ---- CORE exits (regime off or EMA50 break) ----
        for s in list(core):
            if s not in px: continue
            broke = px[s] < float(e50c[s].iloc[maps[s][d]])
            if reg != "RISK_ON" or broke:
                cash += core[s]["qty"] * px[s] * (1 - SELL_COST); del core[s]
        # ---- MR exits (stop/target/time, T+2 lock) ----
        for pos in list(mr):
            if d not in maps[pos["symbol"]]: continue
            j = maps[pos["symbol"]][d]; held = j - pos["entry_idx"]
            if held < T2: continue
            f = frames[pos["symbol"]]; hi, lo, cl = float(f.at[j, "high"]), float(f.at[j, "low"]), float(f.at[j, "close"])
            ex = None
            if lo <= pos["stop"]: ex = pos["stop"]
            elif hi >= pos["target"]: ex = pos["target"]
            elif held >= MR_HOLD: ex = cl
            if ex is not None:
                cash += pos["qty"] * ex * (1 - SELL_COST); mr.remove(pos)
        nav = cash + sum(core[s]["qty"] * px.get(s, core[s]["entry"]) for s in core) \
                   + sum(p["qty"] * px.get(p["symbol"], p["entry"]) for p in mr)
        # ---- CORE rebalance (RISK_ON) ----
        if use_core and reg == "RISK_ON" and di % REBAL == 0:
            scores = {}
            for s, f in frames.items():
                if d not in maps[s]: continue
                j = maps[s][d]
                if j < MOM_LB + 1: continue
                c0 = float(f.at[j - MOM_LB, "close"])
                if px[s] > float(e50c[s].iloc[j]) and c0 > 0: scores[s] = px[s] / c0 - 1
            target = set(sorted(scores, key=lambda s: -scores[s])[:N_HOLD])
            for s in list(core):
                if s not in target and s in px:
                    cash += core[s]["qty"] * px[s] * (1 - SELL_COST); del core[s]
            navc = cash + sum(core[s]["qty"] * px.get(s, core[s]["entry"]) for s in core) \
                        + sum(p["qty"] * px.get(p["symbol"], p["entry"]) for p in mr)
            for s in sorted(target, key=lambda s: -scores[s]):
                if s in core or s not in px: continue
                qty = rl(min(navc / N_HOLD, cash / (1 + BUY_COST)) / px[s])
                if qty >= LOT:
                    cash -= qty * px[s] * (1 + BUY_COST); core[s] = {"qty": qty, "entry": px[s]}
        # ---- MR entries (PANIC) ----
        if use_mr and reg == "PANIC":
            for o in mr_by_entry.get(d, []):
                if len(mr) >= MR_MAXPOS: break
                psr = o["entry"] - o["stop"]
                if psr <= 0: continue
                qty = rl(min(nav * MR_RISK / psr, nav * 0.25 / o["entry"], cash / (1 + BUY_COST) / o["entry"] * o["entry"]) / o["entry"]) \
                      if False else rl(min(nav * MR_RISK / psr, nav * 0.25 / o["entry"], cash / ((1 + BUY_COST) * o["entry"])))
                if qty < LOT: continue
                cash -= qty * o["entry"] * (1 + BUY_COST)
                mr.append(dict(symbol=o["symbol"], entry_idx=o["entry_idx"], entry=o["entry"],
                               qty=qty, stop=o["stop"], target=o["target"]))
        nav_series.append((d, cash + sum(core[s]["qty"] * px.get(s, core[s]["entry"]) for s in core)
                              + sum(p["qty"] * px.get(p["symbol"], p["entry"]) for p in mr)))
    return pd.DataFrame(nav_series, columns=["date", "nav"])


def metrics(nav):
    s = nav.set_index("date")["nav"]; yrs = len(s) / 252
    cagr = (s.iloc[-1] / INIT) ** (1 / yrs) - 1 if yrs > 0 else 0
    dd = ((s - s.cummax()) / s.cummax()).min(); r = s.pct_change().dropna()
    sh = r.mean() / r.std() * math.sqrt(252) if r.std() > 0 else 0
    return dict(total=(s.iloc[-1] / INIT - 1) * 100, cagr=cagr * 100, maxdd=dd * 100, sharpe=sh,
                calmar=cagr / abs(dd) if dd < 0 else 0)


def main():
    idx = pd.read_csv(DATA / "VNINDEX.csv"); idx["date"] = idx["date"].astype(str).str.slice(0, 10)
    idx = idx.sort_values("date").reset_index(drop=True)
    periods = {"2024 full": ("2024-01-01", "2024-12-31"), "2025 full": ("2025-01-01", "2025-12-31")}
    for uname, uni in [("VN30", VN30), ("VN100", None)]:
        files = {p.stem: p for p in DATA.glob("*.csv") if p.stem != "VNINDEX" and (uni is None or p.stem in uni)}
        frames = {s: prep(pd.read_csv(p)) for s, p in files.items()}
        for f in frames.values(): f["date"] = f["date"].astype(str)
        maps = {s: {d: i for i, d in enumerate(f["date"])} for s, f in frames.items()}
        regime = regime_map(idx, breadth_map(frames))
        mrbe = mr_orders(frames, maps, regime, uni)
        print(f"\n===== {uname} =====")
        print(f"{'period':15s} {'variant':11s} | {'TOTAL%':>8s} {'CAGR%':>7s} {'maxDD%':>7s} {'Sharpe':>6s} {'Calmar':>6s} | {'VNIDX%':>7s}")
        print("-" * 86)
        for pname, (s, e) in periods.items():
            cal = [d for d in idx["date"] if s <= d <= e]
            w = idx[(idx["date"] >= s) & (idx["date"] <= e)]
            bench = (w["close"].iloc[-1] / w["close"].iloc[0] - 1) * 100
            for vname, uc, um in [("COMBINED", True, True), ("core-only", True, False), ("MR-only", False, True)]:
                m = metrics(simulate(frames, maps, regime, cal, mrbe, use_core=uc, use_mr=um))
                print(f"{pname:15s} {vname:11s} | {m['total']:>+8.1f} {m['cagr']:>+7.1f} {m['maxdd']:>7.1f} "
                      f"{m['sharpe']:>6.2f} {m['calmar']:>6.2f} | {bench:>+7.1f}")
            print()


if __name__ == "__main__":
    main()
