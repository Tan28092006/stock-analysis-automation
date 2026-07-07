"""Forward-test ledger for the LIVE MR + momentum recommendations.

The daily scan caches are single-snapshot (overwritten each run), so without this the
day's picks vanish and a multi-month paper-trade can't be scored later. This module:

  log_recommendations(mr_payload, momentum_payload)
      -> append every actionable pick to data/pipeline/forward_test.jsonl (append-only,
         idempotent per (engine, symbol, signal_date, kind)) capturing entry / stop /
         target / max_hold / regime / win_prob / rules_hash AT RECOMMENDATION TIME.

  score()
      -> replay each pick against data/raw/prices_hist (the SAME store the signals were
         generated on — not the divergent Yahoo `prices` dir the legacy resolver used),
         resolving MR trades by their OWN stop/target/max_hold (T+2 lock, next-open fill)
         and momentum/watch picks by a fixed forward-return horizon. Returns a summary
         with win-rate, avg net, and by-regime / by-P(win)-bucket breakdowns so the live
         calibration of P(win) can be checked against reality.

Honest scope: MR buys/prob_buys are the scoreable trades (they carry a real risk plan).
Momentum picks and MR watches are logged with a forward-return proxy only (momentum is a
portfolio rotation, watches have no risk plan) — labelled as informational.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

LEDGER_PATH = Path("data/pipeline/forward_test.jsonl")
PRICES_DIR = Path("data/raw/prices_hist")

COST_PCT = 0.60          # round-trip, matches win_probability/backtest labelling convention
T2_LOCK = 2
MOM_HORIZON = 21         # ~1 rebalance month; forward-return window for momentum/watch


def _ledger_keys() -> set:
    keys = set()
    if not LEDGER_PATH.exists():
        return keys
    with LEDGER_PATH.open(encoding="utf-8") as fh:
        for line in fh:
            try:
                r = json.loads(line)
                keys.add((r["engine"], r["symbol"], r["signal_date"], r.get("kind", "")))
            except Exception:
                continue
    return keys


def _append(rows: list[dict]) -> int:
    if not rows:
        return 0
    LEDGER_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LEDGER_PATH.open("a", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    return len(rows)


def log_recommendations(mr_payload: dict | None, momentum_payload: dict | None) -> dict:
    """Append today's actionable picks to the forward-test ledger (idempotent)."""
    seen = _ledger_keys()
    new: list[dict] = []

    if mr_payload:
        regime = (mr_payload.get("market") or {}).get("state")
        rules_hash = mr_payload.get("rules_hash")
        sig_date = mr_payload.get("data_date")
        buckets = [("buy", mr_payload.get("buys", [])),
                   ("prob_buy", mr_payload.get("prob_buys", [])),
                   ("watch", mr_payload.get("watches", []))]
        for kind, items in buckets:
            for p in items:
                sd = p.get("date") or sig_date
                key = ("mr", p["symbol"], sd, kind)
                if key in seen:
                    continue
                seen.add(key)
                new.append({
                    "logged_date": sig_date, "engine": "mr", "kind": kind,
                    "symbol": p["symbol"], "signal_date": sd,
                    "entry_reference": p.get("entry_reference"), "stop_loss": p.get("stop_loss"),
                    "take_profit": p.get("take_profit"), "max_hold_days": p.get("max_hold_days"),
                    "reward_risk": p.get("reward_risk"), "win_prob": p.get("win_prob"),
                    "close": p.get("close"), "rsi14": p.get("rsi14"), "gate": p.get("gate"),
                    "regime": regime, "rules_hash": rules_hash,
                })

    if momentum_payload:
        regime = (momentum_payload.get("market") or {}).get("state")
        sig_date = momentum_payload.get("data_date")
        for p in momentum_payload.get("picks", []):
            key = ("momentum", p["symbol"], sig_date, "pick")
            if key in seen:
                continue
            seen.add(key)
            new.append({
                "logged_date": sig_date, "engine": "momentum", "kind": "pick",
                "symbol": p["symbol"], "signal_date": sig_date,
                "entry_reference": p.get("close"), "weight_pct": p.get("weight_pct"),
                "qty": p.get("qty"), "momentum_12_1_pct": p.get("momentum_12_1_pct"),
                "vn30": p.get("vn30"), "regime": regime,
                "exposure_pct": momentum_payload.get("exposure_pct"),
                "variant": momentum_payload.get("variant", "all"),
            })

    n = _append(new)
    return {"appended": n, "ledger_total": len(seen)}


# ---------------------------------------------------------------- scoring
def _load_frame(symbol: str) -> pd.DataFrame | None:
    p = PRICES_DIR / f"{symbol.upper()}.csv"
    if not p.exists():
        return None
    df = pd.read_csv(p)
    df["date"] = df["date"].astype(str).str.slice(0, 10)
    return df.sort_values("date").reset_index(drop=True)


def _idx_of(frame: pd.DataFrame, dstr: str) -> int | None:
    hit = np.where(frame["date"].to_numpy() == dstr)[0]
    return int(hit[0]) if len(hit) else None


def _replay_mr(frame: pd.DataFrame, i: int, stop: float, target: float, max_hold: int):
    """Next-open fill at i+1, T+2 lock, first of stop/target/time. Returns
    (status, net_pct, exit_reason) — status 'pending' if not enough future bars yet."""
    n = len(frame)
    if i + 1 >= n:
        return "pending", None, None
    entry = float(frame.at[i + 1, "open"])
    if not (np.isfinite(entry) and entry > 0):
        return "pending", None, None
    end = min(i + 1 + max_hold, n - 1)
    reason, exit_px = None, None
    for j in range(i + 1, end + 1):
        k = j - (i + 1)
        if k < T2_LOCK:
            continue
        lo, hi = float(frame.at[j, "low"]), float(frame.at[j, "high"])
        if stop and lo <= stop:
            reason, exit_px = "STOP", stop; break
        if target and hi >= target:
            reason, exit_px = "TARGET", target; break
        if k >= max_hold:
            reason, exit_px = "TIME", float(frame.at[j, "close"]); break
    if exit_px is None:
        # no exit yet AND we haven't reached max_hold in available data -> still open
        if (n - 1) - (i + 1) < max_hold:
            return "pending", None, None
        exit_px, reason = float(frame.at[end, "close"]), "TIME"
    net = (exit_px - entry) / entry * 100 - COST_PCT
    return "resolved", round(net, 2), reason


def _fwd_return(frame: pd.DataFrame, i: int, horizon: int):
    """Forward return from bar i's close to i+horizon close (informational)."""
    n = len(frame)
    if i + horizon >= n:
        if i + 1 >= n:
            return "pending", None
        # partial: use latest available if at least a few bars in
        if (n - 1) - i < 3:
            return "pending", None
        j = n - 1
    else:
        j = i + horizon
    c0, c1 = float(frame.at[i, "close"]), float(frame.at[j, "close"])
    if c0 <= 0:
        return "pending", None
    return "resolved", round((c1 / c0 - 1) * 100, 2)


def _read_ledger() -> list[dict]:
    if not LEDGER_PATH.exists():
        return []
    out = []
    with LEDGER_PATH.open(encoding="utf-8") as fh:
        for line in fh:
            try:
                out.append(json.loads(line))
            except Exception:
                continue
    return out


def score() -> dict:
    """Resolve every ledger row against prices_hist; return a scoreable summary."""
    rows = _read_ledger()
    frames: dict[str, pd.DataFrame] = {}

    def frame_for(sym):
        if sym not in frames:
            frames[sym] = _load_frame(sym)
        return frames[sym]

    mr_trades, mom_fwd, watch_fwd = [], [], []
    pending = 0
    for r in rows:
        f = frame_for(r["symbol"])
        if f is None:
            continue
        i = _idx_of(f, r["signal_date"])
        if i is None:
            continue
        if r["engine"] == "mr" and r["kind"] in ("buy", "prob_buy"):
            stop = r.get("stop_loss") or 0.0
            target = r.get("take_profit") or 0.0
            mh = int(r.get("max_hold_days") or 15)
            st, net, reason = _replay_mr(f, i, stop, target, mh)
            if st == "pending":
                pending += 1
            else:
                mr_trades.append({**r, "net": net, "exit_reason": reason, "win": int(net > 0)})
        elif r["engine"] == "momentum":
            st, fwd = _fwd_return(f, i, MOM_HORIZON)
            if st == "pending":
                pending += 1
            else:
                mom_fwd.append({**r, "fwd_return": fwd, "win": int(fwd > 0)})
        elif r["engine"] == "mr" and r["kind"] == "watch":
            mh = int(r.get("max_hold_days") or 15)
            st, fwd = _fwd_return(f, i, mh)
            if st == "pending":
                pending += 1
            else:
                watch_fwd.append({**r, "fwd_return": fwd, "win": int(fwd > 0)})

    def agg(items, val="net"):
        if not items:
            return {"n": 0}
        v = np.array([x[val] for x in items], dtype=float)
        return {"n": len(v), "win_rate": round(float((v > 0).mean()) * 100, 1),
                "avg": round(float(v.mean()), 2), "median": round(float(np.median(v)), 2),
                "best": round(float(v.max()), 2), "worst": round(float(v.min()), 2)}

    # P(win) live-calibration check on resolved MR trades that carried a win_prob
    pw = [t for t in mr_trades if t.get("win_prob") is not None]
    calib = []
    if pw:
        dfp = pd.DataFrame(pw)
        dfp["bucket"] = pd.cut(dfp["win_prob"], [0, 0.50, 0.55, 0.60, 1.01],
                               labels=["<50%", "50-55%", "55-60%", ">=60%"], right=False)
        for b, g in dfp.groupby("bucket", observed=True):
            calib.append({"bucket": str(b), "n": len(g),
                          "pred_win": round(float(g["win_prob"].mean()) * 100, 1),
                          "actual_win": round(float(g["win"].mean()) * 100, 1),
                          "avg_net": round(float(g["net"].mean()), 2)})

    return {
        "ledger_rows": len(rows), "pending": pending,
        "mr_trades": agg(mr_trades, "net"),
        "mr_by_regime": {reg: agg([t for t in mr_trades if t.get("regime") == reg], "net")
                         for reg in sorted({t.get("regime") for t in mr_trades if t.get("regime")})},
        "pwin_calibration": calib,
        "momentum_fwd21": agg(mom_fwd, "fwd_return"),
        "watch_fwd": agg(watch_fwd, "fwd_return"),
    }


if __name__ == "__main__":
    print(json.dumps(score(), ensure_ascii=False, indent=2))
