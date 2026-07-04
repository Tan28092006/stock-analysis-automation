"""Dual-engine, regime-switched PORTFOLIO backtest.

  RISK_ON  (VNINDEX>EMA50 & breadth>=40%) -> MOMENTUM engine (buy pullbacks, ride trend)
  PANIC    (VNINDEX<EMA50)                 -> MEAN-REVERSION engine (bottom-fish)
  GRIND    (VNINDEX>EMA50 & breadth<40%)   -> CASH (both engines lose here)

Rule gates only (no ML) — leakage-safe causal indicators. Sizing: risk 1.5%/NAV, max 4
positions, 25% weight cap, 60% exposure, lot 100, buy 0.15% / sell 0.25%. Fixed exits
(validated: partial/ATR-trail add nothing). Compares DUAL vs MR-only vs MOM-only vs B&H.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scratch"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from edge_experiment import prep, DATA
from stock_agent.features.indicators import ema, sma

LOT = 100
INIT = 1_000_000_000.0
RISK_PCT = 0.015
MAX_POS = 4
MAX_WEIGHT = 0.25
MAX_EXPOSURE = 0.60
BUY_COST = 0.0015
SELL_COST = 0.0025
T2 = 2
BREADTH_MIN = 0.40


def round_lot(x):
    return int(math.floor(max(0, x) / LOT) * LOT)


def regime_map(idx, breadth):
    e50 = ema(idx["close"], 50)
    out = {}
    for d, c, e in zip(idx["date"].astype(str), idx["close"], e50):
        if not np.isfinite(e):
            continue
        if c > e:
            out[d] = "RISK_ON" if breadth.get(d, 0) >= BREADTH_MIN else "GRIND"
        else:
            out[d] = "PANIC"
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


def build_orders(frames, regime):
    orders = []
    for sym, d in frames.items():
        d = d.reset_index(drop=True)
        n = len(d)
        ds = d["date"].astype(str)
        for i in range(90, n - 1):
            sd = ds.iloc[i]
            reg = regime.get(sd)
            if reg not in ("RISK_ON", "PANIC"):
                continue
            close = float(d.at[i, "close"]); atrv = float(d.at[i, "atr"])
            if not (np.isfinite(atrv) and atrv > 0):
                continue
            entry = float(d.at[i + 1, "open"])
            if not (np.isfinite(entry) and entry > 0):
                continue
            eng = None
            if reg == "PANIC":
                # MR strict-hybrid
                bbl = float(d.at[i, "bb_lower"]); rsi = float(d.at[i, "rsi14"])
                kij = float(d.at[i, "kijun"]) if np.isfinite(d.at[i, "kijun"]) else float(d.at[i, "ema21"])
                cb = float(d.at[i, "cloud_bottom"]) if np.isfinite(d.at[i, "cloud_bottom"]) else 0.0
                confirm = (close > float(d.at[i, "prev_close"]) and float(d.at[i, "vol_ratio"]) >= 1.5) \
                          or int(d.at[i, "vsa_stop"]) == 1
                rr = (kij - close) / (3.0 * atrv)
                blocked = bool(cb) and close < cb < kij
                if rsi < 30 and close <= bbl * 1.01 and confirm and rr >= 0.5 and not blocked:
                    eng = ("MR", entry - 3.0 * atrv, max(kij, close * 1.01), None, 8)
            else:  # RISK_ON -> momentum pullback (let winners run: trail on EMA50, long hold)
                e9 = float(d.at[i, "ema9"]); e21 = float(d.at[i, "ema21"]); e50 = float(d.at[i, "ema50"])
                up = close > e50 and e9 > e21 > e50           # stacked uptrend
                pb = 42 <= float(d.at[i, "rsi14"]) <= 56 and close > float(d.at[i, "prev_close"]) \
                     and close <= e21 * 1.02
                strong = float(d.at[i, "adx14"]) >= 22
                if up and pb and strong:
                    eng = ("MOM", entry - 2.5 * atrv, None, "ema50", 40)
            if eng:
                orders.append(dict(symbol=sym, engine=eng[0], entry_idx=i + 1,
                                   entry_date=str(d.at[i + 1, "date"]), entry=entry,
                                   stop_init=eng[1], target=eng[2], trail_ema=eng[3],
                                   max_hold=eng[4], atr=atrv))
    return pd.DataFrame(orders).sort_values("entry_date").reset_index(drop=True)


def simulate(orders, frames, maps, calendar):
    by_entry = {}
    for _, o in orders.iterrows():
        by_entry.setdefault(o["entry_date"], []).append(o)
    cash = INIT
    open_pos, closed, nav_series = [], [], []
    for d in calendar:
        for pos in list(open_pos):
            f = frames[pos["symbol"]]; im = maps[pos["symbol"]]
            if d not in im:
                continue
            j = im[d]; held = j - pos["entry_idx"]
            if held < T2:
                continue
            hi, lo, cl = float(f.at[j, "high"]), float(f.at[j, "low"]), float(f.at[j, "close"])
            exit_px = reason = None
            if lo <= pos["stop"]:
                exit_px, reason = pos["stop"], "STOP"
            elif pos["target"] is not None and hi >= pos["target"]:
                exit_px, reason = pos["target"], "TARGET"
            elif pos["trail_ema"] is not None and np.isfinite(f.at[j, pos["trail_ema"]]) \
                    and cl < float(f.at[j, pos["trail_ema"]]):
                exit_px, reason = cl, "TRAIL"
            elif held >= pos["max_hold"]:
                exit_px, reason = cl, "TIME"
            if exit_px is not None:
                cash += pos["qty"] * exit_px * (1 - SELL_COST)
                pnl = pos["qty"] * (exit_px - pos["entry"]) - pos["qty"] * exit_px * SELL_COST \
                      - pos["qty"] * pos["entry"] * BUY_COST
                closed.append(dict(symbol=pos["symbol"], engine=pos["engine"], exit_date=d,
                                   reason=reason, ret_pct=pnl / pos["cost0"] * 100, held=held))
                open_pos.remove(pos)
        nav = cash + sum(p["qty"] * float(frames[p["symbol"]].at[maps[p["symbol"]][d], "close"])
                         if d in maps[p["symbol"]] else p["qty"] * p["entry"] for p in open_pos)
        for o in by_entry.get(d, []):
            if len(open_pos) >= MAX_POS:
                break
            psr = o["entry"] - o["stop_init"]
            if psr <= 0:
                continue
            qty = min(round_lot(nav * RISK_PCT / psr), round_lot(nav * MAX_WEIGHT / o["entry"]))
            cur = sum(p["qty"] * p["entry"] for p in open_pos)
            room = min(nav * MAX_EXPOSURE - cur, cash / (1 + BUY_COST))
            qty = min(qty, round_lot(room / o["entry"]))
            if qty < LOT:
                continue
            cost0 = qty * o["entry"] * (1 + BUY_COST)
            cash -= cost0
            open_pos.append(dict(symbol=o["symbol"], engine=o["engine"], entry_idx=o["entry_idx"],
                                 entry=o["entry"], qty=qty, stop=o["stop_init"], target=o["target"],
                                 trail_ema=o["trail_ema"], max_hold=o["max_hold"], cost0=cost0))
        nav_series.append((d, cash + sum(
            p["qty"] * float(frames[p["symbol"]].at[maps[p["symbol"]][d], "close"])
            if d in maps[p["symbol"]] else p["qty"] * p["entry"] for p in open_pos)))
    return pd.DataFrame(nav_series, columns=["date", "nav"]), pd.DataFrame(closed)


def metrics(nav, closed):
    s = nav.set_index("date")["nav"]
    yrs = len(s) / 252
    cagr = (s.iloc[-1] / INIT) ** (1 / yrs) - 1 if yrs > 0 else 0
    dd = ((s - s.cummax()) / s.cummax()).min()
    r = s.pct_change().dropna()
    sharpe = r.mean() / r.std() * math.sqrt(252) if r.std() > 0 else 0
    n = len(closed); win = (closed["ret_pct"] > 0).mean() * 100 if n else 0
    return dict(total=(s.iloc[-1]/INIT-1)*100, cagr=cagr*100, maxdd=dd*100,
                sharpe=sharpe, calmar=cagr/abs(dd) if dd < 0 else 0, trades=n, win=win)


def main():
    files = {p.stem: p for p in DATA.glob("*.csv") if p.stem != "VNINDEX"}
    frames = {s: prep(pd.read_csv(p)) for s, p in files.items()}
    for f in frames.values():
        f["date"] = f["date"].astype(str)
    maps = {s: {d: i for i, d in enumerate(f["date"])} for s, f in frames.items()}
    idx = pd.read_csv(DATA / "VNINDEX.csv"); idx["date"] = idx["date"].astype(str).str.slice(0, 10)
    idx = idx.sort_values("date").reset_index(drop=True)
    breadth = breadth_map(frames)
    regime = regime_map(idx, breadth)
    calendar = [d for d in idx["date"] if d >= "2020-06-01"]
    bench = (idx[idx["date"].isin(calendar)]["close"].iloc[-1] /
             idx[idx["date"].isin(calendar)]["close"].iloc[0] - 1) * 100

    from collections import Counter
    rc = Counter(regime.get(d) for d in calendar)
    print(f"Window {calendar[0]} -> {calendar[-1]} | VNINDEX B&H {bench:+.1f}%")
    print(f"Regime days: RISK_ON {rc['RISK_ON']} | PANIC {rc['PANIC']} | GRIND {rc['GRIND']}\n")

    allo = build_orders(frames, regime)
    variants = {
        "MR only":  allo[allo["engine"] == "MR"],
        "MOM only": allo[allo["engine"] == "MOM"],
        "DUAL":     allo,
    }
    print(f"{'strategy':>9s} {'trades':>7s} {'win%':>6s} | {'TOTAL%':>9s} {'CAGR%':>7s} {'maxDD%':>7s} {'Sharpe':>7s} {'Calmar':>7s}")
    print("-" * 78)
    for name, sub in variants.items():
        nav, closed = simulate(sub, frames, maps, calendar)
        if closed.empty:
            print(f"{name:>9s}   no trades"); continue
        m = metrics(nav, closed)
        print(f"{name:>9s} {m['trades']:>7d} {m['win']:>5.1f} | {m['total']:>+9.1f} {m['cagr']:>+7.1f} "
              f"{m['maxdd']:>7.1f} {m['sharpe']:>7.2f} {m['calmar']:>7.2f}")
        if name == "DUAL":
            eng = closed.groupby("engine")["ret_pct"].agg(["count", "mean"])
            print(f"           engines: " + " | ".join(f"{e}: {int(r['count'])} tr, {r['mean']:+.2f}%/tr"
                                                        for e, r in eng.iterrows()))
    print(f"\nVNINDEX buy&hold: {bench:+.1f}%")


if __name__ == "__main__":
    main()
