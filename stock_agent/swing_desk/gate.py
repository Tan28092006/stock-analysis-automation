"""BASIC GATE G0-G11 and setup detection (pure, deterministic).

hard FAIL -> BLOCK; soft FAIL -> WARN; missing input -> UNVERIFIED, which counts as WARN and
never as PASS. Rule ids and rationale: docs/audits/2026-10-06-swing-desk-signal-research.md §4.
"""
from __future__ import annotations

from datetime import date
import math

from ..data.exchange_calendar import add_trading_days
from .calendar_events import vn30f_expiry
from .sizing import effective_loss_pct

PASS, FAIL, UNVERIFIED, NA = "PASS", "FAIL", "UNVERIFIED", "NA"


def rule(rule_id: str, status: str, severity: str, reason: str = "", kind: str = "gate") -> dict:
    return {"id": rule_id, "status": status, "severity": severity, "reason": reason, "kind": kind}


def verdict(rules: list[dict]) -> str:
    if any(r["status"] == FAIL and r["severity"] == "hard" for r in rules):
        return "BLOCK"
    if any(r["status"] == FAIL or r["status"] == UNVERIFIED for r in rules):
        return "WARN"
    return "PASS"


def _ok(*values) -> bool:
    return all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) for v in values)


def detect_setups(ind: dict, cfg: dict, s2: bool = False) -> list[str]:
    setups = []
    if _ok(ind.get("rsi2"), ind.get("sma50")) and ind["rsi2"] < cfg["setups"]["s1_rsi2_max"] \
            and ind["close"] > ind["sma50"]:
        setups.append("S1")
    if s2:
        setups.append("S2")
    return setups


def reward_risk(ind: dict, entry: float, cfg: dict, setups: list[str]) -> tuple[float, float]:
    """(target, reward/risk) with the user 3% stop and the primary round-trip cost."""
    target = ind.get("sma20") if setups else ind.get("high_20")
    if not _ok(target, entry):
        return float("nan"), float("nan")
    cost = entry * cfg["gate"]["g8_cost_pct"] / 100
    stop = entry * (1 - cfg["stops"]["user_stop_pct"] / 100)
    return target, (target - entry - cost) / (entry - stop + cost)


def _result(rules, setups, out_of_scope=False, extra=None):
    out = {"verdict": verdict(rules), "rules": rules,
           "failed": [r["id"] for r in rules if r["status"] == FAIL],
           "unverified": [r["id"] for r in rules if r["status"] == UNVERIFIED],
           "setups": setups, "out_of_scope": out_of_scope}
    out.update(extra or {})
    return out


def _g9(ticker: str, plan: dict, portfolio: dict, cfg: dict) -> dict:
    m, g = cfg["money"], cfg["gate"]
    req = plan.get("requested")
    value = req["value"] if req else plan["position_value"]
    nav = plan["nav"]
    eff = effective_loss_pct(cfg) / 100
    problems = []
    if req and req["over_budget"]:
        problems.append(f"effective loss {req['vnd_at_risk_effective']:,.0f} > budget {plan['risk_budget_vnd']:,.0f} "
                        f"(compliant size {plan['qty']:,} shares)")
    if value / nav * 100 > m["max_weight_pct"] + 1e-9:
        problems.append(f"weight {value / nav * 100:.1f}% > {m['max_weight_pct']}%")
    if not portfolio or "positions" not in portfolio:
        if problems:
            return rule("G9", FAIL, "hard", "; ".join(problems))
        return rule("G9", UNVERIFIED, "hard", "no portfolio.txt: open positions, open risk and sector counts unknown")
    positions = {t: p for t, p in portfolio["positions"].items() if p.get("qty")}
    holding = {t for t in positions if t != ticker}
    if len(holding) + 1 > m["max_positions"]:
        problems.append(f"positions {len(holding) + 1} > {m['max_positions']}")
    open_value = sum(p["qty"] * p["avg_cost"] for p in positions.values())
    open_risk = open_value * eff + value * eff
    if open_risk > nav * m["max_open_risk_pct"] / 100 + 1e-6:
        problems.append(f"open risk {open_risk / nav * 100:.2f}% NAV > {m['max_open_risk_pct']}%")
    if open_value + value > nav * m["max_exposure_pct"] / 100 + 1e-6:
        problems.append(f"exposure {(open_value + value) / nav * 100:.1f}% > {m['max_exposure_pct']}%")
    for group, limit in (("bank", g["max_bank_positions"]), ("vingroup", g["max_vingroup_positions"])):
        members = set(cfg["sectors"][group])
        if ticker in members:
            count = len((holding | {ticker}) & members)
            if count > limit:
                problems.append(f"{group} positions {count} > {limit}")
    if problems:
        return rule("G9", FAIL, "hard", "; ".join(problems))
    return rule("G9", PASS, "hard", f"order {value / nav * 100:.1f}% NAV, open risk {open_risk / nav * 100:.2f}% NAV")


def _g10(ticker: str, trade_day: date, events, cfg: dict) -> dict:
    window = cfg["gate"]["g10_event_window_sessions"]
    expiry = vn30f_expiry(trade_day.year, trade_day.month)
    if trade_day == expiry:
        return rule("G10", FAIL, "soft", f"{trade_day} is VN30F expiry day (settlement = trimmed 30-min average)")
    if not events:
        return rule("G10", UNVERIFIED, "soft", "no event calendar: ex-rights/dividend/AGM/earnings unknown")
    end = add_trading_days(trade_day, window)
    hits = [e for e in events.get("events", []) if e.get("ticker") == ticker
            and str(trade_day) <= str(e.get("date")) <= str(end)]
    if hits:
        desc = ", ".join(f"{e.get('type', 'event')} {e['date']}" for e in hits)
        return rule("G10", FAIL, "soft", f"event within {window} sessions: {desc}")
    checked = (events.get("checked") or {}).get(ticker)
    if not checked or str(checked) < str(trade_day):
        return rule("G10", UNVERIFIED, "soft", "event calendar not checked for this ticker/session")
    return rule("G10", PASS, "soft", f"no known event to {end}")


def evaluate_gate(*, ticker: str, ind: dict | None, market: dict, freshness: dict, universe: list[str],
                  cfg: dict, trade_day: date, plan: dict | None, price: float | None = None,
                  s2: bool = False, portfolio: dict | None = None, events: dict | None = None) -> dict:
    g = cfg["gate"]
    ticker = ticker.upper()
    if not freshness.get("ok"):
        return _result([rule("G0", FAIL, "hard", freshness.get("reason") or "snapshot not fresh/verified")], [])
    rules = [rule("G0", PASS, "hard", freshness.get("reason", ""))]
    if ticker not in universe:
        rules.append(rule("G1", FAIL, "hard", f"OUT_OF_SCOPE: {ticker} not in current VN30 universe"))
        return _result(rules, [], out_of_scope=True)
    rules.append(rule("G1", PASS, "hard"))
    if ind is None:
        rules.append(rule("G3", UNVERIFIED, "hard", "no indicator data for ticker"))
        return _result(rules, [])
    setups = detect_setups(ind, cfg, s2)

    state = market.get("state")
    if state == "PANIC_BEAR":
        rules.append(rule("G2", FAIL, "hard", "VNINDEX close < EMA50 (PANIC_BEAR)"))
    elif state == "GRIND":
        rules.append(rule("G2", FAIL, "soft", "GRIND: VNINDEX > EMA50 but breadth < 40%"))
    elif state == "RISK_ON":
        rules.append(rule("G2", PASS, "hard"))
    else:
        rules.append(rule("G2", UNVERIFIED, "hard", f"market regime unknown ({state})"))

    if not _ok(ind.get("sma50")):
        rules.append(rule("G3", UNVERIFIED, "hard", "SMA50 unavailable"))
    elif ind["close"] > ind["sma50"]:
        rules.append(rule("G3", PASS, "hard"))
    else:
        rules.append(rule("G3", FAIL, "hard", f"close {ind['close']:,.0f} <= SMA50 {ind['sma50']:,.0f}"))

    r1, r3, rsi2 = ind.get("ret_1d"), ind.get("ret_3d"), ind.get("rsi2")
    if not _ok(r1):
        rules.append(rule("G4", UNVERIFIED, "hard", "ret_1d unavailable"))
    elif r1 >= g["limit_proxy_pct"]:
        rules.append(rule("G4", FAIL, "hard", f"ret_1d {r1:+.1f}% >= +{g['limit_proxy_pct']}% (limit-up proxy)"))
    else:
        soft = []
        if r1 >= g["g4_soft_ret_1d_pct"]:
            soft.append(f"ret_1d {r1:+.1f}%")
        if _ok(r3) and r3 >= g["g4_soft_ret_3d_pct"]:
            soft.append(f"ret_3d {r3:+.1f}%")
        if _ok(rsi2) and rsi2 >= g["g4_soft_rsi2"]:
            soft.append(f"RSI2 {rsi2:.0f}")
        rules.append(rule("G4", FAIL, "soft", "chasing: " + ", ".join(soft)) if soft else rule("G4", PASS, "hard"))

    if price is None:
        rules.append(rule("G5", NA, "hard", "no --price given"))
    else:
        move = (price / ind["close"] - 1) * 100
        if move >= g["g5_hard_pct"]:
            rules.append(rule("G5", FAIL, "hard", f"price {move:+.1f}% vs last close"))
        elif move >= g["g5_soft_pct"]:
            rules.append(rule("G5", FAIL, "soft", f"price {move:+.1f}% vs last close"))
        else:
            rules.append(rule("G5", PASS, "hard", f"price {move:+.1f}% vs last close"))

    atr_pct, mae80 = ind.get("atr_pct"), ind.get("locked_mae_p80")
    if not _ok(atr_pct):
        rules.append(rule("G6", UNVERIFIED, "hard", "ATR unavailable"))
    elif atr_pct > g["g6_atr_hard_pct"]:
        rules.append(rule("G6", FAIL, "hard", f"ATR {atr_pct:.1f}% > {g['g6_atr_hard_pct']}%: 3% stop is inside noise"))
    elif atr_pct > g["g6_atr_soft_pct"] or (_ok(mae80) and mae80 < g["g6_locked_mae_p80_pct"]):
        why = [f"ATR {atr_pct:.1f}%"] + ([f"locked MAE p80 {mae80:.1f}%"] if _ok(mae80) else [])
        rules.append(rule("G6", FAIL, "soft", "stop vs noise: " + ", ".join(why)))
    else:
        rules.append(rule("G6", PASS, "hard", f"ATR {atr_pct:.1f}%"))

    adv = ind.get("avg_value_20_bn")
    if not _ok(adv):
        rules.append(rule("G7", UNVERIFIED, "soft", "20-session traded value unavailable"))
    else:
        order = (plan.get("requested") or {}).get("value", plan["position_value"]) if plan else 0.0
        share = order / (adv * 1e9) * 100 if adv else float("inf")
        if adv < g["g7_min_avg_value_bn"] or share > g["g7_max_order_share_pct"]:
            rules.append(rule("G7", FAIL, "soft", f"avg value {adv:.0f} bn, order {share:.1f}% of it"))
        else:
            rules.append(rule("G7", PASS, "soft"))

    entry = price or ind["close"]
    target, rr = reward_risk(ind, entry, cfg, setups)
    if not _ok(rr):
        rules.append(rule("G8", UNVERIFIED, "soft", "target unavailable"))
    elif rr < g["g8_min_reward_risk"]:
        rules.append(rule("G8", FAIL, "soft", f"R:R {rr:.2f} < {g['g8_min_reward_risk']} (target {target:,.0f})"))
    else:
        rules.append(rule("G8", PASS, "soft", f"R:R {rr:.2f} (target {target:,.0f})"))

    rules.append(_g9(ticker, plan, portfolio, cfg) if plan else rule("G9", UNVERIFIED, "hard", "no size plan"))
    rules.append(_g10(ticker, trade_day, events, cfg))
    rules.append(rule("G11", PASS, "soft", "+".join(setups)) if setups
                 else rule("G11", FAIL, "soft", "no S1/S2 setup: buying without a signal"))
    return _result(rules, setups, extra={"target": target, "reward_risk": rr})
