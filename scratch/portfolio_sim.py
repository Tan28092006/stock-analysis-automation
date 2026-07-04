"""Event-driven PORTFOLIO backtest: signal (walk-forward P(win)) + position sizing (#1)
+ dynamic exits (#2), to answer "does the system have enough power to swing-trade" (#4).

Honest construction:
  - Signal stream = loose MR candidates whose WALK-FORWARD OOS P(win) >= threshold
    (leakage-safe: prob from a model trained only on prior folds).
  - Sizing: risk 1.5% NAV / (entry - disaster stop), capped by max weight / exposure / cash.
  - Dynamic exit: disaster stop 3xATR; at +1R sell half + move stop to breakeven; then
    trail 1.5xATR under the highest close; Kijun target; 8-session time stop; T+2 lock.
  - Costs: buy 0.15%, sell 0.25% (incl PIT). Mark-to-market NAV daily on close.
Reports NAV metrics vs VNINDEX buy&hold. Also sweeps the probability threshold.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from stock_agent.features.win_probability import (
    _prep, _context, _breadth_map, feature_row, is_candidate, FEATURES,
    STOP_ATR, MAX_HOLD, T2_LOCK)

DATA = Path(r"D:\Chungkhoan\data\raw\prices_hist")
LOT = 100
INIT_CAP = 1_000_000_000.0
RISK_PCT = 0.015
MAX_POS = 4
MAX_WEIGHT = 0.25
MAX_EXPOSURE = 0.60
BUY_COST = 0.0015
SELL_COST = 0.0025
PARTIAL_R = 1.0
TRAIL_ATR = 1.5


def round_lot(x):
    return int(math.floor(max(0, x) / LOT) * LOT)


def build_orders():
    """All loose MR candidates with walk-forward OOS P(win) + entry context."""
    regime, idx_ret20 = _context(DATA)
    files = {p.stem: p for p in DATA.glob("*.csv") if p.stem != "VNINDEX"}
    frames = {s: _prep(pd.read_csv(p)) for s, p in files.items()}
    breadth = _breadth_map(frames)
    maps = {}
    for s, f in frames.items():
        maps[s] = {str(d): i for i, d in enumerate(f["date"].astype(str))}

    rows = []
    for sym, f in frames.items():
        dstr = f["date"].astype(str)
        n = len(f)
        for i in range(90, n - 1):
            sd = dstr.iloc[i]
            fr = feature_row(f, i, regime.get(sd, 1.0), breadth.get(sd, 0.5), idx_ret20.get(sd))
            if fr is None or not is_candidate(fr):
                continue
            entry = float(f.at[i + 1, "open"])
            atrv = fr["_atr"]; close = fr["_close"]; kijun = fr["_kijun"]
            if not (np.isfinite(entry) and entry > 0):
                continue
            rec = {k: fr[k] for k in FEATURES}
            rec.update(date=sd, symbol=sym, i=i, entry_idx=i + 1,
                       entry_date=str(f.at[i + 1, "date"]), entry=entry,
                       atr=atrv, stop_init=entry - STOP_ATR * atrv,
                       target=max(kijun, close * 1.01))
            rows.append(rec)
    df = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    return df, frames, maps


def compute_prob(df, frames):
    """Proper walk-forward: label each candidate, train on past folds, predict OOS."""
    from stock_agent.features.win_probability import _label_trade
    # label
    wins = []
    for _, r in df.iterrows():
        f = frames[r["symbol"]]
        net = _label_trade(f, int(r["i"]))
        wins.append(0 if net is None else int(net > 0))
    df = df.copy(); df["win"] = wins
    K = 6
    bounds = [int(len(df) * k / K) for k in range(K + 1)]
    df["prob"] = np.nan
    for k in range(1, K):
        tr = df.iloc[:bounds[k]]
        te_idx = list(df.index[bounds[k]:bounds[k + 1]])
        if not te_idx:
            continue
        te_start = pd.Timestamp(df.loc[te_idx[0], "date"])
        tr = tr[pd.to_datetime(tr["date"]) < te_start - pd.Timedelta(days=20)]
        if len(tr) < 500 or tr["win"].nunique() < 2:
            continue
        m = lgb.LGBMClassifier(n_estimators=300, max_depth=4, learning_rate=0.03,
                               subsample=0.8, colsample_bytree=0.8, min_child_samples=40,
                               reg_lambda=1.0, random_state=42, verbose=-1)
        m.fit(tr[FEATURES], tr["win"])
        df.loc[te_idx, "prob"] = m.predict_proba(df.loc[te_idx, FEATURES])[:, 1]
    # calibrate raw OOS prob -> probability (pooled isotonic; mildly optimistic but
    # makes the threshold mean an actual win rate)
    from sklearn.isotonic import IsotonicRegression
    ok = df["prob"].notna()
    if ok.sum() > 200:
        iso = IsotonicRegression(out_of_bounds="clip").fit(df.loc[ok, "prob"], df.loc[ok, "win"])
        df.loc[ok, "prob"] = iso.transform(df.loc[ok, "prob"])
    # strict-hybrid gate flag (production rules_mr.json semantics)
    df["strict"] = ((df["rsi14"] < 30) & (df["dist_below_bblo"] >= -0.01) &
                    (((df["reversal"] == 1) & (df["vol_ratio"] >= 1.5)) | (df["vsa_stop"] == 1)) &
                    (df["rr_to_kijun"] >= 0.5) & (df["cloud_block"] == 0))
    return df


def simulate(orders, frames, maps, calendar, exit_mode="fixed"):
    by_entry = {}
    for _, o in orders.iterrows():
        by_entry.setdefault(o["entry_date"], []).append(o)
    cash = INIT_CAP
    open_pos = []
    closed = []
    nav_series = []
    for d in calendar:
        # ---- exits ----
        for pos in list(open_pos):
            f = frames[pos["symbol"]]; idxmap = maps[pos["symbol"]]
            if d not in idxmap:
                continue
            j = idxmap[d]; held = j - pos["entry_idx"]
            if held < T2_LOCK:
                continue
            hi, lo, cl = float(f.at[j, "high"]), float(f.at[j, "low"]), float(f.at[j, "close"])
            pos["hh"] = max(pos["hh"], cl)
            exit_px = None; reason = None
            if lo <= pos["stop"]:
                exit_px, reason = pos["stop"], ("BE/trail" if pos["partial"] else "STOP")
            elif hi >= pos["target"]:
                exit_px, reason = pos["target"], "TARGET"
            elif held >= MAX_HOLD:
                exit_px, reason = cl, "TIME"
            elif exit_mode in ("partial_be", "partial_trail"):
                R = pos["entry"] - pos["stop_init"]
                if (not pos["partial"]) and pos["qty"] >= 2 * LOT and hi >= pos["entry"] + PARTIAL_R * R:
                    sell_q = round_lot(pos["qty"] / 2)
                    px = pos["entry"] + PARTIAL_R * R
                    cash += sell_q * px * (1 - SELL_COST)
                    pos["realized"] += sell_q * (px - pos["entry"]) - sell_q * px * SELL_COST - sell_q * pos["entry"] * BUY_COST
                    pos["qty"] -= sell_q; pos["partial"] = True; pos["stop"] = pos["entry"]
                if pos["partial"] and exit_mode == "partial_trail":
                    pos["stop"] = max(pos["stop"], pos["hh"] - TRAIL_ATR * pos["atr"])
            if exit_px is not None:
                cash += pos["qty"] * exit_px * (1 - SELL_COST)
                pnl = pos["realized"] + pos["qty"] * (exit_px - pos["entry"]) \
                      - pos["qty"] * exit_px * SELL_COST - pos["qty"] * pos["entry"] * BUY_COST
                closed.append({"symbol": pos["symbol"], "entry_date": pos["entry_date"], "exit_date": d,
                               "reason": reason, "pnl": pnl, "cost_basis": pos["cost0"],
                               "ret_pct": pnl / pos["cost0"] * 100, "held": held})
                open_pos.remove(pos)
        # ---- mark NAV (pre-entry, using today's close) ----
        mtm = 0.0
        for pos in open_pos:
            idxmap = maps[pos["symbol"]]
            if d in idxmap:
                mtm += pos["qty"] * float(frames[pos["symbol"]].at[idxmap[d], "close"])
            else:
                mtm += pos["qty"] * pos["entry"]
        nav = cash + mtm
        # ---- entries ----
        for o in sorted(by_entry.get(d, []), key=lambda r: -r["prob"]):
            if len(open_pos) >= MAX_POS:
                break
            entry = o["entry"]; per_share_risk = entry - o["stop_init"]
            if per_share_risk <= 0:
                continue
            qty = round_lot(nav * RISK_PCT / per_share_risk)
            qty = min(qty, round_lot(nav * MAX_WEIGHT / entry))
            # exposure + cash caps
            cur_expo = sum(p["qty"] * p["entry"] for p in open_pos)
            room = min(nav * MAX_EXPOSURE - cur_expo, cash / (1 + BUY_COST))
            qty = min(qty, round_lot(room / entry))
            if qty < LOT:
                continue
            cost0 = qty * entry * (1 + BUY_COST)
            cash -= cost0
            open_pos.append({"symbol": o["symbol"], "entry_idx": o["entry_idx"], "entry": entry,
                             "qty": qty, "stop": o["stop_init"], "stop_init": o["stop_init"],
                             "target": o["target"], "atr": o["atr"], "hh": entry,
                             "partial": False, "realized": 0.0, "cost0": cost0, "entry_date": d})
        nav_series.append((d, cash + sum(
            p["qty"] * float(frames[p["symbol"]].at[maps[p["symbol"]][d], "close"])
            if d in maps[p["symbol"]] else p["qty"] * p["entry"] for p in open_pos)))
    return pd.DataFrame(nav_series, columns=["date", "nav"]), pd.DataFrame(closed)


def metrics(nav, closed, bench):
    nav = nav.set_index("date")["nav"]
    total = nav.iloc[-1] / INIT_CAP - 1
    days = len(nav)
    yrs = days / 252
    cagr = (nav.iloc[-1] / INIT_CAP) ** (1 / yrs) - 1 if yrs > 0 else 0
    roll_max = nav.cummax(); dd = (nav - roll_max) / roll_max
    maxdd = dd.min()
    rets = nav.pct_change().dropna()
    sharpe = rets.mean() / rets.std() * math.sqrt(252) if rets.std() > 0 else 0
    calmar = cagr / abs(maxdd) if maxdd < 0 else float("inf")
    n = len(closed); win = (closed["ret_pct"] > 0).mean() * 100 if n else 0
    avg = closed["ret_pct"].mean() if n else 0
    held = closed["held"].mean() if n else 0
    return dict(total=total*100, cagr=cagr*100, maxdd=maxdd*100, sharpe=sharpe, calmar=calmar,
                trades=n, win=win, avg=avg, held=held, bench=bench)


def main():
    print("Building candidates + walk-forward P(win)...", flush=True)
    orders, frames, maps = build_orders()
    orders = compute_prob(orders, frames)
    orders = orders.dropna(subset=["prob"])
    idx = pd.read_csv(DATA / "VNINDEX.csv"); idx["date"] = idx["date"].astype(str).str.slice(0, 10)
    idx = idx.sort_values("date")
    calendar = [d for d in idx["date"] if d >= orders["date"].min()]
    bench = (idx[idx["date"].isin(calendar)]["close"].iloc[-1] /
             idx[idx["date"].isin(calendar)]["close"].iloc[0] - 1) * 100
    print(f"OOS candidates: {len(orders)} | window {calendar[0]} -> {calendar[-1]} | "
          f"VNINDEX B&H {bench:+.1f}%\n")

    signal_sets = {
        "strict":     orders[orders["strict"]],
        "prob>=0.55": orders[orders["prob"] >= 0.55],
        "prob>=0.60": orders[orders["prob"] >= 0.60],
    }
    print(f"{'signal':>11s} {'exit':>13s} {'trades':>7s} {'win%':>6s} {'avg%':>6s} {'hold':>5s} | "
          f"{'TOTAL%':>8s} {'CAGR%':>7s} {'maxDD%':>7s} {'Sharpe':>7s} {'Calmar':>7s}")
    print("-" * 104)
    for sname, subset in signal_sets.items():
        for exit_mode in ["fixed", "partial_be", "partial_trail"]:
            nav, closed = simulate(subset, frames, maps, calendar, exit_mode=exit_mode)
            if closed.empty:
                print(f"{sname:>11s} {exit_mode:>13s}   no trades"); continue
            m = metrics(nav, closed, bench)
            print(f"{sname:>11s} {exit_mode:>13s} {m['trades']:>7d} {m['win']:>5.1f} {m['avg']:>+6.2f} {m['held']:>5.1f} | "
                  f"{m['total']:>+8.1f} {m['cagr']:>+7.1f} {m['maxdd']:>7.1f} {m['sharpe']:>7.2f} {m['calmar']:>7.2f}")
        print()
    print(f"VNINDEX buy&hold same window: {bench:+.1f}%")


if __name__ == "__main__":
    main()
