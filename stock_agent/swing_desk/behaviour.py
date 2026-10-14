"""Behavioural red flags B1-B9 and cooling-off (pure).

B1-B7 + COOL are evaluated at check time; B8/B9 are review-only. Flags that need the
journal are UNVERIFIED without `portfolio.txt` (counts as WARN). Window semantics (user
decision 2026-10-06, counted in trading sessions):
  B3  revenge: a realized loss closed in the trade session or the previous one -> hard.
  B3P two consecutive realized losses -> hard for one extra session (loss session + 2).
  COOL a behavioural BLOCK recorded for this ticker in this trade session holds until the next.
Rationale and sources: docs/audits/2026-10-06-swing-desk-signal-research.md §5.
"""
from __future__ import annotations

from datetime import date
import statistics

import pandas as pd

from .calendar_events import session_distance
from .gate import FAIL, NA, PASS, UNVERIFIED, _ok
from .sizing import effective_loss_pct

BEHAVIOUR_HARD = {"B1", "B3", "B3P", "B4", "COOL"}


def _rule(rule_id, status, severity, reason=""):
    return {"id": rule_id, "status": status, "severity": severity, "reason": reason, "kind": "behaviour"}


def _days_ago(d: str, trade_day: date) -> int:
    return session_distance(date.fromisoformat(d), trade_day)


def evaluate_behaviour(*, ticker: str, ind: dict, journal, trade_day: date, cfg: dict, plan: dict,
                       price: float | None = None, reason: str | None = None, setups: list[str] = (),
                       checks: list[dict] = ()) -> list[dict]:
    b = cfg["behaviour"]
    lim = cfg["gate"]["limit_proxy_pct"]
    close = ind["close"]
    rules = []

    # B1 FOMO / ceiling (hard)
    why = []
    if _ok(ind.get("ret_1d")) and ind["ret_1d"] >= b["b1_ret_1d_pct"]:
        why.append(f"ret_1d {ind['ret_1d']:+.1f}%")
    if ind.get("limit_up_t") or ind.get("limit_up_t1"):
        why.append("limit-up proxy at t or t-1")
    ceiling = close * (1 + 7.0 / 100)
    if price is not None and price >= ceiling * (1 - b["b1_ceiling_buffer_pct"] / 100):
        why.append(f"price within {b['b1_ceiling_buffer_pct']}% of ceiling {ceiling:,.0f}")
    rules.append(_rule("B1", FAIL, "hard", "FOMO: " + "; ".join(why)) if why else _rule("B1", PASS, "hard"))

    # B2 gap chase (soft)
    if price is None:
        rules.append(_rule("B2", NA, "soft", "no --price given"))
    else:
        move = (price / close - 1) * 100
        limit = b["b2_setup_gap_pct"] if setups else b["b2_gap_pct"]
        rules.append(_rule("B2", FAIL, "soft", f"price {move:+.1f}% vs reference (limit +{limit}%)")
                     if move >= limit else _rule("B2", PASS, "soft"))

    # B6 hype (soft)
    hype = []
    if reason and reason.lower() in b["b6_hype_reasons"] and not setups:
        hype.append(f"reason '{reason}' without S-setup")
    if _ok(ind.get("vol_ratio_20"), ind.get("ret_1d")) and ind["vol_ratio_20"] >= b["b6_vol_ratio"] \
            and ind["ret_1d"] >= b["b6_ret_1d_pct"]:
        hype.append(f"volume x{ind['vol_ratio_20']:.1f} with ret_1d {ind['ret_1d']:+.1f}%")
    rules.append(_rule("B6", FAIL, "soft", "hype: " + "; ".join(hype)) if hype else _rule("B6", PASS, "soft"))

    # COOL: behavioural BLOCK earlier in this session for this ticker (hard until next session)
    if b.get("cooling_off_enabled", True):
        prior = [c for c in checks if c.get("ticker") == ticker and c.get("trade_session") == str(trade_day)
                 and c.get("behaviour_block")]
        rules.append(_rule("COOL", FAIL, "hard", "behavioural BLOCK earlier this session: re-check from next session ATO")
                     if prior else _rule("COOL", PASS, "hard"))

    if journal is None:
        for rid, sev in (("B3", "hard"), ("B4", "hard"), ("B5", "soft"), ("B7", "soft")):
            rules.append(_rule(rid, UNVERIFIED, sev, "no portfolio.txt journal"))
        return rules

    closed = sorted(journal.closed, key=lambda t: (t["exit_date"], t["line"]))
    closed = [t for t in closed if t["exit_date"] <= str(trade_day)]
    recent_losses = [t for t in closed if t["net_pct"] < 0 and _days_ago(t["exit_date"], trade_day) <= b["b3_loss_window_sessions"]]
    rules.append(_rule("B3", FAIL, "hard", f"revenge: realized loss {recent_losses[-1]['ticker']} on {recent_losses[-1]['exit_date']}")
                 if recent_losses else _rule("B3", PASS, "hard"))
    k = b["b3_consecutive_losses"]
    last = closed[-k:]
    if len(last) == k and all(t["net_pct"] < 0 for t in last) \
            and _days_ago(last[-1]["exit_date"], trade_day) <= b["b3_consecutive_pause_sessions"]:
        rules.append(_rule("B3P", FAIL, "hard", f"{k} consecutive losses (last {last[-1]['exit_date']}): one-session pause"))
    else:
        rules.append(_rule("B3P", PASS, "hard"))

    # B4 averaging down (hard unless a scale-in plan preceded the first lot)
    pos = journal.positions.get(ticker)
    entry = price or close
    if pos and entry < pos["avg_cost"]:
        if pos.get("scalein_plan"):
            rules.append(_rule("B4", PASS, "hard", "below average cost but scale-in plan journaled before first lot"))
        else:
            rules.append(_rule("B4", FAIL, "hard", f"averaging down: {entry:,.0f} < average cost {pos['avg_cost']:,.0f}"))
    else:
        rules.append(_rule("B4", PASS, "hard"))

    # B5 size creep (soft; > cap handled by G9 hard)
    eff = effective_loss_pct(cfg) / 100
    buys = [e for e in journal.buys() if e.date <= str(trade_day)][-b["b5_lookback_entries"]:]
    req = plan.get("requested")
    value = req["value"] if req else plan["position_value"]
    if req and req["weight_pct"] > cfg["money"]["max_weight_pct"]:
        rules.append(_rule("B5", FAIL, "soft", f"weight {req['weight_pct']:.1f}% > {cfg['money']['max_weight_pct']}%"))
    elif len(buys) < b["b5_min_entries"]:
        rules.append(_rule("B5", NA, "soft", f"fewer than {b['b5_min_entries']} prior entries"))
    else:
        median = statistics.median(e.qty * e.price * eff for e in buys)
        ratio = value * eff / median if median else float("inf")
        rules.append(_rule("B5", FAIL, "soft", f"risk {ratio:.2f}x median of last {len(buys)} entries")
                     if ratio > b["b5_size_creep_ratio"] else _rule("B5", PASS, "soft"))

    # B7 overtrading (soft): this entry included
    entries = [e for e in journal.buys() if e.date <= str(trade_day)]
    n5 = sum(1 for e in entries if _days_ago(e.date, trade_day) < 5) + 1
    n21 = sum(1 for e in entries if _days_ago(e.date, trade_day) < 21) + 1
    if n5 > b["b7_max_entries_5"] or n21 > b["b7_max_entries_21"]:
        rules.append(_rule("B7", FAIL, "soft", f"overtrading: {n5} entries in 5 sessions, {n21} in 21 (incl. this)"))
    else:
        rules.append(_rule("B7", PASS, "soft"))
    return rules


def behaviour_block(rules: list[dict]) -> bool:
    return any(r["kind"] == "behaviour" and r["status"] == FAIL and r["severity"] == "hard" for r in rules)


def review_flags(journal, frames: dict[str, pd.DataFrame], cfg: dict) -> dict:
    """B8 stop not honoured, B9 disposition effect (review only)."""
    stop_pct = cfg["stops"]["user_stop_pct"] / 100
    b8 = []
    for t in journal.closed:
        df = frames.get(t["ticker"])
        if df is None:
            continue
        days = [str(d)[:10] for d in df["date"]]
        closes = dict(zip(days, df["close"].astype(float)))
        stop = t["entry_avg"] * (1 - stop_pct)
        sellable = [d for d in days if t["entry_date"] < d < t["exit_date"]
                    and _days_ago(t["entry_date"], date.fromisoformat(d)) >= 2]
        breach = next((d for d in sellable if closes[d] <= stop), None)
        if breach and _days_ago(breach, date.fromisoformat(t["exit_date"])) > 1:
            b8.append({"ticker": t["ticker"], "breach_close": breach, "exit_date": t["exit_date"], "line": t["line"]})
    b = cfg["behaviour"]
    winners = [t["holding_sessions"] for t in journal.closed if t["net_pct"] > 0]
    losers = [t["holding_sessions"] for t in journal.closed if t["net_pct"] <= 0]
    flag = None
    detail = f"needs >= {b['b9_min_closed_trades']} closed trades"
    if len(journal.closed) >= b["b9_min_closed_trades"] and winners and losers:
        mw, ml = statistics.median(winners), statistics.median(losers)
        flag = ml > b["b9_ratio"] * mw
        detail = f"median holding losers {ml} vs winners {mw} sessions"
    return {"B8": {"count": len(b8), "items": b8},
            "B9": {"flag": flag, "detail": detail, "n_closed": len(journal.closed)}}
