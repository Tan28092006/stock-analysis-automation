"""Bottom-fish mode rules (pure): fall classification, BF-G gate, BF-B flags, profile plan.

Rule ids, sources and rationale: docs/audits/2026-10-06-bottom-fishing-research.md (BF-doc §5-§9)
and the builder spec T12 §4-§8. Every rule is an advisory filter with no prospective evidence;
type (b) falls have literature evidence of continued drift. Fail-closed: in this mode missing data
for BF-G0/G2/G5/G6/G8/G10 is a hard FAIL (BLOCK), never UNVERIFIED.
"""
from __future__ import annotations

from datetime import date
import math

from ..data.exchange_calendar import add_trading_days
from .bf_indicators import episode_id
from .calendar_events import SELL_FROM, earliest_sell, session_distance
from .gate import FAIL, NA, PASS, UNVERIFIED

CAVEAT = "Chỉ tư vấn; bắt đáy chưa có bằng chứng edge tiến cứu; loại (b) có bằng chứng trôi giảm."
P3_NOTE = "no evidence of edge; measurement only"
FAIL_CLOSED = ("BF-G0", "BF-G2", "BF-G5", "BF-G6", "BF-G8", "BF-G10")
FC = "UNVERIFIED -> fail-closed"


def _rule(rule_id, status, severity, reason="", kind="gate"):
    return {"id": rule_id, "status": status, "severity": severity, "reason": reason, "kind": kind}


def _ok(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def _d(text) -> date:
    return date.fromisoformat(str(text)[:10])


def _age(day: str, as_of: str) -> int:
    """Sessions from `day` to `as_of` (0 when the same session or later)."""
    return max(0, session_distance(_d(day), _d(as_of)))


# ------------------------------------------------------------------ facts

def news_status(news: dict | None, data_as_of: str | None) -> tuple[bool, str]:
    if not news:
        return False, "news not checked (no bf-news record)"
    if not data_as_of or str(news.get("checked_session", "")) < str(data_as_of):
        return False, f"news check {news.get('checked_session')} older than data {data_as_of}"
    if news.get("class") == "none_found" and len(news.get("searched") or []) < 2:
        return False, "class none_found needs >= 2 searched sources"
    return True, f"news checked {news.get('checked_session')} ({news.get('class')})"


def _weight(item: dict, cfg: dict) -> float:
    return float(cfg["evidence_weights"].get(str(item.get("evidence_type", "")).upper(), 0.0))


# News whose subject is the market, a sector or a corporate group (e.g. Vingroup contagion) or the
# ticker's own price action is evidence of a market-wide / technical fall, not a firm-specific
# fundamental shock: it never makes a fall type (b) and never fires BF-G6 (amendment 2026-10-08).
MARKET_SCOPE_TYPES = frozenset({"market", "market_group", "market_macro", "market_flows", "market/flows",
                                "group", "group_contagion", "group_price", "sector", "foreign_flow",
                                "technical", "macro", "policy_macro"})


def fact_negative_items(news: dict | None, data_as_of: str, cfg: dict, *, window: bool = True) -> list[dict]:
    c = cfg["classification"]
    out = []
    for it in (news or {}).get("items", []):
        if window and _age(it["date"], data_as_of) > c["news_lookback_sessions"]:
            continue
        if str(it.get("type", "")).strip().lower() in MARKET_SCOPE_TYPES and it.get("type") not in c["fact_negative_types"]:
            continue
        strong = _weight(it, cfg) >= c["fact_min_weight"] and int(it.get("impact", 0)) <= c["fact_max_impact"]
        if strong or it.get("type") in c["fact_negative_types"]:
            out.append(it)
    return out


def in_fall_episode(fields: dict, cfg: dict) -> bool:
    """A current fall episode exists: drawdown from the 60-session high <= -15%, or a floor close inside
    the episode. Without it the ticker is not a bottom-fish candidate (spec §4 else-branch: route to swing)."""
    dd = fields.get("drawdown_60_pct")
    start, last_floor = fields.get("episode_start_date"), fields.get("last_floor_date")
    if not fields.get("episode_sessions"):
        return False
    return (_ok(dd) and dd <= cfg["episode"]["min_drawdown_pct"]) or bool(last_floor and start and last_floor >= start)


def classify_fall(fields: dict | None, market: dict, news: dict | None, *, data_as_of: str | None,
                  cfg: dict) -> dict:
    """T12 §4. UNCLASSIFIED is treated as type b by the gate (BLOCK for P1/P2)."""
    c = cfg["classification"]
    sha = (news or {}).get("source_sha")

    def out(kind, reason, evidence=()):
        return {"fall_type": kind, "reason": reason, "evidence": list(evidence), "news_sha": sha}

    if fields is None:
        return out("UNCLASSIFIED", "no verified price data for the ticker")
    if fields.get("beta_60") is None or fields.get("idio_ret_10d") is None:
        return out("UNCLASSIFIED", f"beta_60 unavailable ({fields.get('index_status', 'index frame missing')})")
    if not (market or {}).get("state"):
        return out("UNCLASSIFIED", "market summary unavailable (main VN30 snapshot not fresh)")
    ok, why = news_status(news, data_as_of)
    if not ok:
        return out("UNCLASSIFIED", why)
    facts = fact_negative_items(news, data_as_of, cfg)
    if facts:
        return out("b", "negative FACT item(s) within 20 sessions",
                   [f"{i['date']} {i['evidence_type']} {i['impact']:+d} {i['type']} {i['url']}" for i in facts])
    idio = fields["idio_ret_10d"]
    panic = market.get("state") == "PANIC_BEAR"
    breadth = market.get("breadth_pct")
    is_market = (_ok(fields.get("idx_ret_5d")) and fields["idx_ret_5d"] <= c["market_idx_ret_5d_pct"]) or \
        (panic and _ok(breadth) and breadth < c["market_breadth_max_pct"])
    i5 = fields.get("idx_ret_5d")
    metrics = [f"idx_ret_5d {i5:.1f}%" if _ok(i5) else "idx_ret_5d n/a", f"idio_ret_10d {idio:.1f}%",
               f"beta_60 {fields['beta_60']:.2f}",
               f"regime {market.get('state')} breadth {breadth}"]
    if not in_fall_episode(fields, cfg):
        return out("NOT_A_BOTTOMFISH", "no fall episode (drawdown_60 > -15% and no floor close in the episode): "
                                       "use swing mode", metrics)
    if is_market and idio > c["idio_market_max_pct"]:
        return out("a", "market-wide panic with small idiosyncratic part", metrics)
    if idio <= c["idio_fundamental_pct"]:
        return out("b", "large idiosyncratic fall with unknown cause (conservative)", metrics)
    if idio <= c["idio_market_max_pct"] and news.get("class") == "none_found":
        if panic and cfg["regime"].get("type_c_requires_not_panic", True):
            return out("a", "no-news flush during PANIC_BEAR -> type a (BF-G9)", metrics)
        return out("c", "idiosyncratic flush, cited search found no negative FACT", metrics)
    return out("NOT_A_BOTTOMFISH", "no qualifying fall: use swing mode", metrics)


# ------------------------------------------------------------------ plan

def bf_plan(entry: float, nav: float, *, profile: str, cfg: dict, fall_type: str | None, fields: dict,
            trade_day: date, lot: int = 1, amount: float | None = None, tranche: int | None = None) -> dict | None:
    """T12 §8: size on the effective loss (stop + one extra floor + costs); P0 has no plan."""
    profile = profile.upper()
    if profile == "P0":
        return None
    if entry <= 0 or nav <= 0:
        raise ValueError("entry and NAV must be positive")
    p = cfg["profiles"][profile]
    rt = cfg["costs"]["round_trip_pct"]
    eff = (p["assumed_loss_pct"] + rt) if "assumed_loss_pct" in p else \
        (p["catastrophe_stop_pct"] + p["floor_gap_allowance_pct"] + rt)
    mult = {}
    if fall_type == "c" and "type_c_risk_multiplier" in p:
        mult["type_c"] = p["type_c_risk_multiplier"]
    ratio = fields.get("index_atr_ratio") if fields else None
    if _ok(ratio) and ratio >= cfg["regime"]["high_vol_atr_ratio"]:
        mult["high_vol"] = cfg["regime"]["high_vol_risk_multiplier"]
    m = math.prod(mult.values()) if mult else 1.0
    risk = nav * p["risk_pct_nav"] / 100 * m
    by_risk = risk / (eff / 100)
    cap = nav * p.get("max_weight_pct", 100.0) / 100
    value_max = min(by_risk, cap)
    qty = int(math.floor(value_max / entry / lot) * lot)
    tranches = int(p.get("max_tranches", 1))
    tranche_qty = int(math.floor(qty / tranches / lot) * lot) if tranches > 1 else qty
    order_qty = tranche_qty if tranches > 1 else qty
    value = order_qty * entry
    panic_low = (fields or {}).get("panic_low")
    inv_mult = p.get("invalidation_mult")
    invalidation = panic_low * inv_mult if inv_mult is not None and _ok(panic_low) else None
    sessions = p.get("time_stop_sessions", p.get("time_review_sessions"))
    out = {
        "profile": profile, "label": p.get("label"), "entry_ref": entry, "nav": nav,
        "risk_pct_nav": p["risk_pct_nav"], "multipliers": mult, "risk_multiplier": m, "risk_vnd": risk,
        "effective_loss_pct": eff, "value_max": value_max, "capped_by": "max_weight" if cap < by_risk else None,
        "qty": qty, "max_tranches": tranches, "tranche_qty": tranche_qty if tranches > 1 else None,
        "tranche": tranche or (1 if tranches > 1 else None), "order_qty": order_qty, "position_value": value,
        "weight_pct": value / nav * 100, "vnd_at_risk_effective": value * eff / 100,
        "invalidation": invalidation, "invalidation_rule": p.get("invalidation", "new negative FACT / adverse terms"),
        "catastrophe_stop": entry * (1 - p["catastrophe_stop_pct"] / 100) if "catastrophe_stop_pct" in p else None,
        "time_stop_sessions": sessions, "time_stop_session": str(add_trading_days(trade_day, int(sessions))),
        "earliest_sell": f"{earliest_sell(trade_day)} {SELL_FROM}",
        "entry_limit": round(entry * p.get("entry_limit_mult", 1.005), 2),
        "scenarios": {"three_more_floors_pct": (1 - 0.93 ** 3) * 100,
                      "three_more_floors_vnd": value * (1 - 0.93 ** 3) + value * rt / 100,
                      "effective_loss_vnd": value * eff / 100},
    }
    if "edge_note" in p:
        out["edge_note"] = p["edge_note"]
    if amount is not None:
        rq = int(math.floor(amount / entry / lot) * lot)
        rv = rq * entry
        out["requested"] = {"amount": amount, "qty": rq, "value": rv, "weight_pct": rv / nav * 100,
                            "risk_effective_vnd": rv * eff / 100, "risk_effective_pct_nav": rv * eff / nav,
                            "over_budget": rv * eff / 100 > risk + 1e-6}
    return out


# ------------------------------------------------------------------ journal helpers

def bf_buys(journal, ticker: str | None = None) -> list:
    return [e for e in journal.buys() if e.reason == "bottomfish" and (ticker is None or e.ticker == ticker)]


def bf_positions(journal) -> dict:
    return {t: p for t, p in journal.positions.items() if p.get("reason") == "bottomfish" and p.get("qty")}


def _position_eff(journal, ticker: str, cfg: dict) -> float:
    """Effective loss % for an open BF position: its journaled profile, else the most conservative."""
    rt = cfg["costs"]["round_trip_pct"]
    effs = {}
    for name, p in cfg["profiles"].items():
        if "assumed_loss_pct" in p:
            effs[name] = p["assumed_loss_pct"] + rt
        elif "catastrophe_stop_pct" in p:
            effs[name] = p["catastrophe_stop_pct"] + p["floor_gap_allowance_pct"] + rt
    profiles = [e.keys.get("profile", "").upper() for e in bf_buys(journal, ticker)]
    known = [effs[p] for p in profiles if p in effs]
    return known[0] if known else max(effs.values())


def _episode_buys(journal, ticker: str, ep_id: str | None, ep_start: str | None) -> list:
    out = []
    for e in bf_buys(journal, ticker):
        key = e.keys.get("episode", "").upper()
        if (ep_id and key == ep_id) or (not key and ep_start and e.date >= ep_start):
            out.append(e)
    return out


def _episode_key(e) -> str:
    return e.keys.get("episode", "").upper() or f"{e.ticker}-{e.date}"


def thesis_for(journal, ticker: str, data_as_of: str, profile: str, tranche: int, first_lot_date: str | None):
    """THESIS rows that can satisfy BF-G12 (newest first)."""
    rows = journal.theses(ticker)
    if profile == "P2" and tranche >= 2 and first_lot_date:
        rows = [e for e in rows if e.date <= first_lot_date and e.scalein]
    else:
        rows = [e for e in rows if e.date >= data_as_of]
    return sorted(rows, key=lambda e: (e.date, e.line), reverse=True)


REQUIRED_THESIS = ("bf_type", "profile", "invalid", "time_stop", "max_attempts")


# ------------------------------------------------------------------ gate

def evaluate_bf_gate(*, ticker: str, profile: str, fields: dict | None, market: dict, fall: dict,
                     snapshot: dict, in_scope: bool, held: bool, journal, news: dict | None, events: dict | None,
                     plan: dict | None, price: float | None, tranche: int | None, trade_day: date,
                     data_as_of: str | None, cfg: dict, swing_effective_loss_pct: float = 4.6,
                     universe_label: str = "VN30") -> list[dict]:
    ticker, profile = ticker.upper(), profile.upper()
    pcfg = cfg["profiles"][profile]
    as_of = data_as_of or str(trade_day)
    rules = []

    # BF-G0 verified snapshot for the ticker
    rules.append(_rule("BF-G0", PASS, "hard", snapshot.get("reason", "")) if snapshot.get("ok") and fields is not None
                 else _rule("BF-G0", FAIL, "hard", f"{FC}: {snapshot.get('reason') or 'no verified data for ticker'}"))

    # BF-G1 scope for new entries
    if in_scope:
        rules.append(_rule("BF-G1", PASS, "hard", f"{universe_label} / extra_universe"))
    else:
        note = "; held position: check/review only (HOLD/SELL/AVOID), never BUY" if held else ""
        rules.append(_rule("BF-G1", FAIL, "hard", f"OUT_OF_SCOPE for new BF entries: {ticker} not in {universe_label} "
                                                  f"or extra_universe{note}"))

    # BF-G2 fall type vs profile
    ft = fall.get("fall_type")
    if profile == "P0":
        rules.append(_rule("BF-G2", FAIL, "hard", "profile P0: no bottom-fishing (default recommendation)"))
    elif ft == "UNCLASSIFIED":
        rules.append(_rule("BF-G2", FAIL, "hard", f"{FC}: fall UNCLASSIFIED ({fall.get('reason')}), treated as b"))
    elif ft == "NOT_A_BOTTOMFISH":
        rules.append(_rule("BF-G2", FAIL, "hard", "no qualifying fall: use swing mode"))
    elif ft not in pcfg["allowed_types"]:
        rules.append(_rule("BF-G2", FAIL, "hard", f"type {ft} not allowed for {profile} (allowed {pcfg['allowed_types']})"))
    else:
        rules.append(_rule("BF-G2", PASS, "hard", f"type {ft}: {fall.get('reason')}"))

    # BF-G3 never on a floor; BF-G4 stabilisation
    if fields is None:
        rules.append(_rule("BF-G3", FAIL, "hard", f"{FC}: no price data"))
        rules.append(_rule("BF-G4", FAIL, "hard", f"{FC}: no price data"))
    else:
        why = []
        if fields.get("floor_t"):
            why.append(f"close at floor proxy (ret_1d {fields.get('ret_1d', float('nan')):+.2f}%)")
        if fields.get("locked_floor_t") or fields.get("locked_floor_t1"):
            why.append("locked floor (O=H=L=C) at t or t-1")
        buf = cfg["limit"]["floor_price_buffer_pct"]
        if price is not None and _ok(fields.get("floor_price_next")) and price <= fields["floor_price_next"] * (1 + buf / 100):
            why.append(f"price {price:,.0f} within {buf}% of floor {fields['floor_price_next']:,.0f}")
        rules.append(_rule("BF-G3", FAIL, "hard", "; ".join(why)) if why else _rule("BF-G3", PASS, "hard"))
        st = fields.get("stab_state")
        desc = (f"stab_state {st}, sessions_since_floor {fields.get('sessions_since_floor')}, "
                f"low {fields.get('low')} vs prev {fields.get('prev_low')}, panic_low {fields.get('panic_low')}")
        rules.append(_rule("BF-G4", PASS, "hard", desc) if st == "STAB2" else _rule("BF-G4", FAIL, "hard", "not stabilised: " + desc))

    rules.append(_g5(ticker, profile, events, trade_day, as_of, cfg))
    rules.append(_g6(profile, news, trade_day, as_of, cfg))

    # BF-G7 liquidity (soft)
    adv = (fields or {}).get("avg_value_20_bn")
    if not _ok(adv):
        rules.append(_rule("BF-G7", FAIL, "soft", "20-session traded value unavailable"))
    else:
        order = ((plan or {}).get("requested") or {}).get("value", (plan or {}).get("position_value", 0.0))
        share = order / (adv * 1e9) * 100 if adv else float("inf")
        liq = cfg["liquidity"]
        if adv < liq["min_avg_value_20_bn"] or share > liq["max_order_share_pct"]:
            rules.append(_rule("BF-G7", FAIL, "soft", f"avg value {adv:.0f} bn, order {share:.2f}% of it"))
        else:
            rules.append(_rule("BF-G7", PASS, "soft", f"avg value {adv:.0f} bn"))

    rules.append(_g8(news, as_of, cfg))

    # BF-G9 context (information only)
    notes = []
    if plan and plan.get("multipliers"):
        notes.append("risk x " + " x ".join(f"{v} ({k})" for k, v in plan["multipliers"].items()))
    if "PANIC_BEAR" in str(fall.get("reason", "")):
        notes.append("type c during PANIC_BEAR reclassified as a")
    rules.append(_rule("BF-G9", PASS, "info", "; ".join(notes) or "no context adjustment"))

    rules.append(_g10(ticker, profile, journal, plan, cfg, swing_effective_loss_pct))
    rules.append(_g11(ticker, profile, journal, fields, tranche, trade_day, cfg))
    rules.append(_g12(ticker, profile, journal, fields, tranche, as_of))
    return rules


def _g5(ticker, profile, events, trade_day, as_of, cfg):
    ev = cfg["events"]
    if not events:
        return _rule("BF-G5", FAIL, "hard", f"{FC}: no event calendar")
    checked = (events.get("checked") or {}).get(ticker)
    if not checked or str(checked) < str(trade_day):
        return _rule("BF-G5", FAIL, "hard", f"{FC}: event calendar not checked for {ticker} / {trade_day}")
    resolved = (events.get("resolved") or {}).get(ticker, [])
    done = {(r.get("date"), r.get("type")) for r in resolved}
    end = str(add_trading_days(trade_day, ev["window_sessions"]))
    problems = []
    for e in (x for x in events.get("events", []) if x.get("ticker") == ticker):
        key = (e.get("date"), e.get("type"))
        dilution = e.get("type") in ev["dilution_types"] or e.get("pending_dilution")
        if dilution and key not in done and (e.get("pending_dilution") or str(e.get("date")) >= str(trade_day)):
            problems.append(f"unresolved dilution event {e.get('type')} {e.get('date')} (record the result with events --resolved)")
        elif not dilution and str(trade_day) <= str(e.get("date")) <= end:
            problems.append(f"{e.get('type')} {e.get('date')} within {ev['window_sessions']} sessions")
    if profile == "P3" and not [r for r in resolved if str(r.get("date", "")) <= as_of]:
        problems.append("P3 needs the event resolved first (EGM result + terms recorded with events --resolved)")
    if problems:
        return _rule("BF-G5", FAIL, "hard", "; ".join(problems))
    return _rule("BF-G5", PASS, "hard", f"no blocking event to {end}")


def _g6(profile, news, trade_day, as_of, cfg):
    ok, why = news_status(news, as_of)
    if not ok:
        return _rule("BF-G6", FAIL, "hard", f"{FC}: {why}")
    insider = [i for i in news.get("insider_sell", []) if str(i.get("start", "")) <= str(trade_day) <= str(i.get("end", ""))]
    if profile == "P3":
        allf = fact_negative_items(news, as_of, cfg, window=False)
        if allf:
            last = max(i["date"] for i in allf)
            need = cfg["profiles"]["P3"]["min_sessions_since_last_fact_negative"]
            if _age(last, as_of) < need:
                return _rule("BF-G6", FAIL, "hard", f"last negative FACT {last} is {_age(last, as_of)} sessions old (< {need})")
    else:
        facts = fact_negative_items(news, as_of, cfg)
        if facts:
            return _rule("BF-G6", FAIL, "hard", "negative FACT within 20 sessions: " +
                         ", ".join(f"{i['date']} {i['type']}" for i in facts))
    if insider:
        return _rule("BF-G6", FAIL, "soft", "active insider/related-party sell registration: " +
                     ", ".join(f"{i['start']}..{i['end']} {i.get('shares')}" for i in insider))
    return _rule("BF-G6", PASS, "hard", why)


def _g8(news, as_of, cfg):
    margin = (news or {}).get("margin")
    if not margin:
        return _rule("BF-G8", FAIL, "hard", f"{FC}: margin / warning status not checked")
    look = cfg["margin"]["lookback_sessions"]
    checked = margin.get("checked")
    if not checked or _age(checked, as_of) > look:
        return _rule("BF-G8", FAIL, "hard", f"{FC}: margin status checked {checked}, older than {look} sessions")
    problems = []
    if margin.get("hose_ineligible"):
        problems.append("on the HOSE margin-ineligible list")
    if str(margin.get("status", "normal")).lower() not in {"normal", ""}:
        problems.append(f"status {margin['status']}")
    cut = margin.get("broker_cut")
    if cut and _age(cut, as_of) <= look:
        problems.append(f"broker margin cut {cut} within {look} sessions")
    for it in (news or {}).get("items", []):
        if it.get("type") in {"margin_cut", "warning", "control", "suspension"} and _age(it["date"], as_of) <= look:
            problems.append(f"{it['type']} {it['date']}")
    if problems:
        return _rule("BF-G8", FAIL, "hard", "; ".join(problems))
    return _rule("BF-G8", PASS, "hard", f"margin checked {checked}")


def _g10(ticker, profile, journal, plan, cfg, swing_eff):
    if journal is None:
        return _rule("BF-G10", FAIL, "hard", f"{FC}: bottom-fishing needs swing_desk/portfolio.txt (journal)")
    if plan is None:
        return _rule("BF-G10", FAIL, "hard", f"{FC}: no plan" + (" (profile P0)" if profile == "P0" else ""))
    b = cfg["budget"]
    nav = plan["nav"]
    req = plan.get("requested")
    value = req["value"] if req else plan["position_value"]
    eff = plan["effective_loss_pct"] / 100
    problems = []
    if req and req["over_budget"]:
        problems.append(f"effective loss {req['risk_effective_vnd']:,.0f} > budget {plan['risk_vnd']:,.0f} "
                        f"(compliant size {plan['order_qty']:,} shares)")
    maxw = cfg["profiles"][profile].get("max_weight_pct")
    if maxw is not None and value / nav * 100 > maxw + 1e-9:
        problems.append(f"weight {value / nav * 100:.1f}% > {maxw}%")
    bfpos = {t: p for t, p in bf_positions(journal).items() if t != ticker}
    if len(bfpos) + 1 > b["max_bf_positions"]:
        problems.append(f"BF positions {len(bfpos) + 1} > {b['max_bf_positions']}")
    own = journal.positions.get(ticker, {}).get("cost_value", 0.0) if ticker in bf_positions(journal) else 0.0
    bf_value = sum(p["cost_value"] for p in bfpos.values()) + own + value
    if bf_value > nav * b["max_bf_exposure_pct"] / 100 + 1e-6:
        problems.append(f"BF exposure {bf_value / nav * 100:.1f}% > {b['max_bf_exposure_pct']}%")
    open_risk = value * eff
    for t, p in journal.positions.items():
        if not p.get("qty"):
            continue
        e = _position_eff(journal, t, cfg) / 100 if p.get("reason") == "bottomfish" else swing_eff / 100
        open_risk += p["cost_value"] * e
    if open_risk > nav * b["max_total_open_risk_pct"] / 100 + 1e-6:
        problems.append(f"total open risk {open_risk / nav * 100:.2f}% NAV > {b['max_total_open_risk_pct']}%")
    if problems:
        return _rule("BF-G10", FAIL, "hard", "; ".join(problems))
    return _rule("BF-G10", PASS, "hard", f"order {value / nav * 100:.1f}% NAV, open risk {open_risk / nav * 100:.2f}% NAV")


def _g11(ticker, profile, journal, fields, tranche, trade_day, cfg):
    if journal is None:
        return _rule("BF-G11", FAIL, "hard", f"{FC}: no journal")
    if profile == "P0":
        return _rule("BF-G11", NA, "hard", "profile P0")
    pc = cfg["profiles"][profile]
    a = cfg["attempts"]
    ep_id = episode_id(ticker, fields) if fields else None
    ep_buys = _episode_buys(journal, ticker, ep_id, (fields or {}).get("episode_start_date"))
    problems = []
    if profile == "P2":
        n = pc["max_tranches"]
        k = tranche or len(ep_buys) + 1
        if len(ep_buys) >= n or k > n:
            problems.append(f"tranche {k} with {len(ep_buys)} BF buys in episode {ep_id} (declared max {n})")
    elif len(ep_buys) >= pc["max_attempts_per_episode"]:
        problems.append(f"{len(ep_buys)} BF attempt(s) already in episode {ep_id} (max {pc['max_attempts_per_episode']})")
    for t in journal.closed:
        if t["ticker"] == ticker and t.get("reason") == "bottomfish" and t["net_pct"] < 0 \
                and 0 <= session_distance(_d(t["exit_date"]), trade_day) <= a["cooldown_sessions_after_bf_loss"]:
            problems.append(f"cooldown: BF loss on {ticker} closed {t['exit_date']} "
                            f"(< {a['cooldown_sessions_after_bf_loss']} sessions)")
    month = str(trade_day)[:7]
    month_eps = {_episode_key(e) for e in bf_buys(journal) if e.date[:7] == month}
    if not ep_buys and len(month_eps) >= a["max_new_bf_episodes_per_month"]:
        problems.append(f"{len(month_eps)} new BF episodes already this month (max {a['max_new_bf_episodes_per_month']})")
    if problems:
        return _rule("BF-G11", FAIL, "hard", "; ".join(problems))
    return _rule("BF-G11", PASS, "hard", f"attempt {len(ep_buys) + 1} in episode {ep_id}")


def _g12(ticker, profile, journal, fields, tranche, as_of):
    if journal is None:
        return _rule("BF-G12", FAIL, "hard", "no journal: write a THESIS line first")
    if profile == "P0":
        return _rule("BF-G12", NA, "hard", "profile P0")
    ep_id = episode_id(ticker, fields) if fields else None
    ep_buys = _episode_buys(journal, ticker, ep_id, (fields or {}).get("episode_start_date"))
    first = min((e.date for e in ep_buys), default=None)
    k = tranche or len(ep_buys) + 1
    rows = thesis_for(journal, ticker, as_of, profile, k, first)
    if not rows:
        where = f"dated <= first tranche {first} with plan=scalein" if profile == "P2" and k >= 2 and first else f"dated >= {as_of}"
        return _rule("BF-G12", FAIL, "hard", f"no THESIS for {ticker} {where} (bf_type=, profile=, invalid=, "
                                             "time_stop=, max_attempts=, checklist=done)")
    best = None
    for e in rows:
        keys = e.keys
        missing = [x for x in REQUIRED_THESIS if not keys.get(x)]
        if keys.get("checklist", "").lower() != "done":
            missing.append("checklist=done")
        if keys.get("profile", "").upper() not in ("", profile):
            missing.append(f"profile={profile} (THESIS says {keys.get('profile')})")
        if profile == "P2" and not e.scalein:
            missing.append("plan=scalein (P2 plan must be written before the first tranche)")
        if not missing:
            return _rule("BF-G12", PASS, "hard", f"THESIS line {e.line} ({e.date})")
        best = best or (e, missing)
    return _rule("BF-G12", FAIL, "hard", f"THESIS line {best[0].line} missing: {', '.join(best[1])}")


# ------------------------------------------------------------------ behaviour

def evaluate_bf_behaviour(*, ticker: str, fields: dict | None, journal, plan: dict | None, price: float | None,
                          why: str | None, reason: str | None, fall_type: str | None, profile: str,
                          tranche: int | None, structural_ok: bool, trade_day: date, cfg: dict,
                          close_lookup=None) -> list[dict]:
    """BF-B1..BF-B8 (T12 §6). B2/B3 soft by user decision 2026-10-06; B9 is checklist-only."""
    b = cfg["behaviour"]
    ticker, profile = ticker.upper(), profile.upper()
    why = (why or "").lower() or None
    f = fields or {}
    out = []

    def add(rule_id, failed, severity, text, unverified=False):
        status = UNVERIFIED if unverified else (FAIL if failed else PASS)
        out.append(_rule(rule_id, status, severity, text if (failed or unverified) else "", kind="behaviour"))

    # BF-B1 knife on a floor day (hard)
    w = []
    if fields is None:
        w.append("no price data (fail-closed)")
    if f.get("floor_t"):
        w.append("floor close today")
    if f.get("locked_floor_t") or f.get("locked_floor_t1"):
        w.append("locked floor at t or t-1")
    if price is not None and _ok(f.get("floor_price_next")) and \
            price <= f["floor_price_next"] * (1 + cfg["limit"]["floor_price_buffer_pct"] / 100):
        w.append(f"price {price:,.0f} at/near floor {f['floor_price_next']:,.0f}")
    add("BF-B1", bool(w), "hard", "catching a falling knife: " + "; ".join(w))

    # BF-B2 anchoring on the old high (soft by user decision)
    dd = f.get("drawdown_250_pct")
    target = None
    if journal is not None:
        rows = [e for e in journal.theses(ticker) if "target" in e.keys]
        if rows:
            try:
                target = float(rows[-1].keys["target"].replace(",", "").replace("_", ""))
            except ValueError:
                target = None
    anchored = _ok(dd) and dd <= b["b2_drawdown_250_pct"] and (
        (why in b["b2_reasons"]) or (target is not None and _ok(f.get("high_60")) and target >= f["high_60"]))
    add("BF-B2", anchored, "soft", f"anchoring: drawdown_250 {dd:.0f}%" + (f", why={why}" if why else "") +
        (f", THESIS target {target:,.0f} >= high_60" if target is not None and _ok(f.get('high_60')) and target >= f['high_60'] else "")
        if _ok(dd) else "")

    # BF-B3 "it's cheap now" (soft by user decision)
    cheap = why in b["b3_reasons"] or (_ok(dd) and dd <= b["b3_drawdown_250_pct"] and fall_type != "a")
    add("BF-B3", cheap, "soft", "'cheap now': " + (f"why={why}" if why in b["b3_reasons"] else
                                                     f"drawdown_250 {dd if not _ok(dd) else round(dd)}% with type {fall_type}"))

    # BF-B8 attention / crowd (soft)
    crowd = [x for x in ((reason or "").lower(), why or "") if x in b["b8_reasons"]]
    add("BF-B8", bool(crowd), "soft", f"attention/crowd reason: {', '.join(crowd)}")

    if journal is None:
        for rid, sev in (("BF-B4", "hard"), ("BF-B5", "hard"), ("BF-B6", "soft"), ("BF-B7", "soft")):
            add(rid, False, sev, "no portfolio.txt journal", unverified=True)
        return out

    # BF-B4 averaging down (hard unless a pre-written P2 scale-in tranche that passes BF-G3/G4)
    pos = journal.positions.get(ticker)
    entry = price or (plan or {}).get("entry_ref") or f.get("close")
    if pos and entry and entry < pos["avg_cost"]:
        n = cfg["profiles"].get("P2", {}).get("max_tranches", 3)
        k = tranche or 0
        if profile == "P2" and pos.get("scalein_plan") and structural_ok and 1 <= k <= n:
            add("BF-B4", False, "hard", "")
        else:
            add("BF-B4", True, "hard", f"averaging down: {entry:,.0f} < average cost {pos['avg_cost']:,.0f}"
                + ("" if profile == "P2" else " (only a P2 plan=scalein tranche passing BF-G3/G4 is exempt)")
                + ("" if structural_ok else "; tranche fails BF-G3/G4") + (f"; tranche {k} > {n}" if k > n else ""))
    else:
        add("BF-B4", False, "hard", "")

    # BF-B5 revenge on the same ticker (hard)
    cool = cfg["attempts"]["cooldown_sessions_after_bf_loss"]
    losses = [t for t in journal.closed if t["ticker"] == ticker and t.get("reason") == "bottomfish" and t["net_pct"] < 0
              and 0 <= session_distance(_d(t["exit_date"]), trade_day) <= cool]
    add("BF-B5", bool(losses), "hard", f"BF loss on {ticker} closed {losses[-1]['exit_date'] if losses else ''} "
        f"within {cool} sessions")

    # BF-B6 paper loss -> more risk (soft)
    paper = []
    for t, p in journal.positions.items():
        close = close_lookup(t) if close_lookup else None
        if close and p.get("avg_cost") and (close / p["avg_cost"] - 1) * 100 <= b["b6_paper_loss_pct"]:
            paper.append(f"{t} {(close / p['avg_cost'] - 1) * 100:.1f}%")
    add("BF-B6", bool(paper), "soft", "open position(s) with paper loss: " + ", ".join(paper))

    # BF-B7 size up after a BF win (soft)
    buys = [e for e in bf_buys(journal) if e.date <= str(trade_day)]
    closed_bf = sorted([t for t in journal.closed if t.get("reason") == "bottomfish"], key=lambda t: (t["exit_date"], t["line"]))
    value = ((plan or {}).get("requested") or {}).get("value", (plan or {}).get("position_value"))
    if buys and closed_bf and closed_bf[-1]["net_pct"] > 0 and value:
        prev = buys[-1].qty * buys[-1].price
        ratio = value / prev if prev else float("inf")
        add("BF-B7", ratio > b["b7_size_ratio_after_win"], "soft",
            f"size x{ratio:.2f} of the previous BF entry right after a BF win")
    else:
        out.append(_rule("BF-B7", NA, "soft", "no previous winning BF trade", kind="behaviour"))
    return out


def held_facts(journal, ticker: str, plan_trade_day: date) -> dict | None:
    """Review facts for a held position (never a sell/hold instruction)."""
    if journal is None or ticker not in journal.positions:
        return None
    p = journal.positions[ticker]
    rows = [e for e in journal.theses(ticker)]
    keys = rows[-1].keys if rows else {}
    return {"qty": p["qty"], "avg_cost": p["avg_cost"], "first_date": p["first_date"], "last_date": p["last_date"],
            "earliest_sell": f"{earliest_sell(_d(p['last_date']))} {SELL_FROM}",
            "sellable": earliest_sell(_d(p["last_date"])) <= plan_trade_day,
            "invalid": keys.get("invalid"), "time_stop": keys.get("time_stop"),
            "thesis_line": rows[-1].line if rows else None,
            "note": "Desk không khuyến nghị bán hay giữ: ghi kế hoạch (invalid=, time_stop=) và tự quyết."}
