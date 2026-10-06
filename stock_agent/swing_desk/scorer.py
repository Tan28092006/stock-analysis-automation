"""Outcome scoring for frozen recommendations (rules fixed in configs/swing_desk/prospective_v1.json).

Fill: limit fills if L <= limit at min(O, limit); ATO fills at O; otherwise `no_fill` (counted).
Lock: entry session i and i+1 have no exits. Session i+2 is close-only: if the stop was touched
during i..i+1 or C <= stop -> exit at C ("stop_locked"); elif C >= target -> exit at C.
From i+3: O <= stop -> O (gap); elif L <= stop -> stop; elif O >= target -> O; elif H >= target ->
target; stop before target on the same bar. A stop exit on a limit-down session (close-to-close
<= -6.5%) is assumed unfilled and carried to the next session's open. Time exit at the close of
i + max_hold_sessions. Rows whose path is not complete in the data stay `pending` (no look-ahead).
"""
from __future__ import annotations

from collections import defaultdict
from datetime import date
import math
import statistics

import pandas as pd

BUY_COST, SELL_COST, STRESS = 0.0015, 0.0025, 0.006
LIMIT_PROXY = 6.5
MIN_RESOLVED, MIN_SESSIONS = 90, 40


def _dates(df: pd.DataFrame) -> list[str]:
    return [str(d)[:10] for d in df["date"]]


def score_row(row: dict, df: pd.DataFrame | None) -> dict:
    out = {k: row.get(k) for k in ("id", "source", "ticker", "target_session", "p_win_stated")}
    if row.get("action") != "BUY":
        return {**out, "status": "not_scored"}
    if df is None or df.empty:
        return {**out, "status": "pending", "reason": "no price data"}
    days = _dates(df)
    if row["target_session"] not in days:
        if row["target_session"] > days[-1]:
            return {**out, "status": "pending"}
        return {**out, "status": "error", "reason": "target session missing from data"}
    i = days.index(row["target_session"])
    o, h, lo, c = (df[k].astype(float).tolist() for k in ("open", "high", "low", "close"))
    entry = row["entry"]
    if entry["type"] == "ato":
        fill = o[i]
    elif lo[i] <= entry["limit_price"]:
        fill = min(o[i], entry["limit_price"])
    else:
        return {**out, "status": "no_fill"}
    stop, target, hold = float(row["stop"]), float(row["target"]), int(row["max_hold_sessions"])
    if fill <= stop:
        return {**out, "status": "error", "reason": "fill at or below stop"}
    exit_idx = exit_price = reason = None
    carry = 0
    pending_stop = False
    for k in range(i + 2, i + hold + 1):
        if k >= len(days):
            return {**out, "status": "pending", "fill": fill}
        limit_down = (c[k] / c[k - 1] - 1) * 100 <= -LIMIT_PROXY
        if pending_stop:
            if limit_down:
                carry += 1
                continue
            exit_idx, exit_price, reason = k, o[k], "stop_gap"
            break
        if k == i + 2:
            if min(lo[i], lo[i + 1]) <= stop or c[k] <= stop:
                if limit_down:
                    carry, pending_stop = carry + 1, True
                    continue
                exit_idx, exit_price, reason = k, c[k], "stop_locked"
                break
            if c[k] >= target:
                exit_idx, exit_price, reason = k, c[k], "target_locked"
                break
        else:
            if o[k] <= stop or lo[k] <= stop:
                if limit_down:
                    carry, pending_stop = carry + 1, True
                    continue
                exit_idx, exit_price, reason = (k, o[k], "stop_gap") if o[k] <= stop else (k, stop, "stop")
                break
            if o[k] >= target or h[k] >= target:
                exit_idx, exit_price, reason = (k, o[k], "target") if o[k] >= target else (k, target, "target")
                break
        if k == i + hold:
            if limit_down:
                carry, pending_stop = carry + 1, True
                continue
            exit_idx, exit_price, reason = k, c[k], "time"
    if exit_idx is None:
        # carried past the horizon by limit-down sessions: sell at the first open that is not limit-down
        k = i + hold + 1
        while k < len(days) and (c[k] / c[k - 1] - 1) * 100 <= -LIMIT_PROXY:
            carry, k = carry + 1, k + 1
        if k >= len(days):
            return {**out, "status": "pending", "fill": fill}
        exit_idx, exit_price, reason = k, o[k], "stop_gap" if pending_stop else "time_carry"
    risk_pct = (fill - stop) / fill * 100
    net = (exit_price * (1 - SELL_COST) / (fill * (1 + BUY_COST)) - 1) * 100
    mae = (min(lo[i:exit_idx + 1]) / fill - 1) * 100
    mfe = (max(h[i:exit_idx + 1]) / fill - 1) * 100
    return {**out, "status": "resolved", "fill": fill, "exit_price": exit_price, "exit_reason": reason,
            "exit_date": days[exit_idx], "holding_sessions": exit_idx - i, "limit_down_carry": carry,
            "net_pct": net, "net_pct_stress": (exit_price / fill - 1) * 100 - STRESS * 100,
            "R": net / risk_pct, "risk_pct": risk_pct, "mae_pct": mae, "mfe_pct": mfe,
            "mae_R": mae / risk_pct, "mfe_R": mfe / risk_pct,
            "stop_hit": reason.startswith("stop"), "target_hit": reason.startswith("target"),
            "win": net > 0}


def _mean(values):
    values = [v for v in values if v is not None]
    return statistics.mean(values) if values else None


def _median(values):
    values = [v for v in values if v is not None]
    return statistics.median(values) if values else None


def summarize(scored: list[dict], *, activity_floor: int = 4) -> dict:
    arms = defaultdict(list)
    for r in scored:
        arms[r["source"]].append(r)
    out = {"arms": {}, "paired": {}, "activity": {}}
    for arm, rows in sorted(arms.items()):
        res = [r for r in rows if r["status"] == "resolved"]
        nf = [r for r in rows if r["status"] == "no_fill"]
        wins = [r["win"] for r in res]
        base = sum(wins) / len(wins) if wins else None
        prob = [r for r in res if isinstance(r.get("p_win_stated"), (int, float))]
        brier = _mean([(r["p_win_stated"] - r["win"]) ** 2 for r in prob]) if prob else None
        ref = _mean([(base - r["win"]) ** 2 for r in prob]) if prob and base is not None else None
        out["arms"][arm] = {
            "rows": len(rows), "resolved": len(res), "pending": sum(r["status"] == "pending" for r in rows),
            "no_fill": len(nf), "no_fill_rate": len(nf) / (len(res) + len(nf)) * 100 if res or nf else None,
            "errors": sum(r["status"] == "error" for r in rows),
            "win_rate": base * 100 if base is not None else None,
            "mean_net_pct": _mean([r["net_pct"] for r in res]),
            "mean_net_pct_stress": _mean([r["net_pct_stress"] for r in res]),
            "mean_R": _mean([r["R"] for r in res]),
            "median_mae_pct": _median([r.get("mae_pct") for r in res]),
            "median_mfe_pct": _median([r.get("mfe_pct") for r in res]),
            "brier": brier, "brier_reference": ref,
            "brier_skill": (1 - brier / ref) if brier is not None and ref else None,
            "sessions": len({r["target_session"] for r in res}),
        }
        months = defaultdict(lambda: {"ideas": 0, "sessions": set()})
        for r in rows:
            m = months[r["target_session"][:7]]
            m["ideas"] += 1
            m["sessions"].add(r["target_session"])
        out["activity"][arm] = {m: {"ideas": v["ideas"], "distinct_sessions": len(v["sessions"]),
                                    "floor": activity_floor, "floor_met": v["ideas"] >= activity_floor}
                                for m, v in sorted(months.items())}

    def per_session(arm):
        groups = defaultdict(list)
        for r in arms.get(arm, []):
            if r["status"] == "resolved":
                groups[r["target_session"]].append(r["net_pct"])
        return {s: statistics.mean(v) for s, v in groups.items()}

    for a, b in (("desk", "baseline"), ("desk", "control"), ("single", "baseline"), ("desk", "single")):
        pa, pb = per_session(a), per_session(b)
        common = sorted(set(pa) & set(pb))
        if common:
            diffs = [pa[s] - pb[s] for s in common]
            out["paired"][f"{a}_minus_{b}"] = {"sessions": len(common), "mean_diff_pct": statistics.mean(diffs)}
    d, bl = out["arms"].get("desk", {}), out["arms"].get("baseline", {})
    enough = (d.get("resolved", 0) >= MIN_RESOLVED and bl.get("resolved", 0) >= MIN_RESOLVED
              and d.get("sessions", 0) >= MIN_SESSIONS)
    out["conclusion"] = ("NOT PROVEN: sample below the registered floor (>= 90 resolved per arm D and B, >= 40 sessions, "
                         "both regimes); describe only" if not enough else
                         "Floor reached: run the registered weekly block bootstrap before any claim")
    return out
