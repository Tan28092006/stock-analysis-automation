"""Bottom-fish pure rules (T12 spec §4-§8): fall classification, sizing, BF-G gate, BF-B flags."""
from datetime import date

import pytest

TDAY = date(2026, 9, 22)
AS_OF = "2026-09-21"
THESIS = ("2026-09-21|THESIS|AAA|||||||T1|bf_type=a; profile=P1; invalid=close<18000; "
          "time_stop=2026-10-06; max_attempts=1; checklist=done\n")
NAV = "2026-09-01|NAV||11200000||||||\n"


def bf():
    from stock_agent.swing_desk.config import load_bf_config
    return load_bf_config()


def fields(**over):
    base = {"date": AS_OF, "close": 20000.0, "ret_1d": 1.0, "floor_t": False, "locked_floor_t": False,
            "locked_floor_t1": False, "floor_streak": 0, "locked_floor_streak": 0, "sessions_since_floor": 3,
            "last_floor_date": "2026-09-16", "floor_price_next": 18600.0, "floor_price_today": 18000.0,
            "higher_low_t": True, "low": 19800.0, "prev_low": 19500.0, "high_60": 30000.0,
            "drawdown_60_pct": -33.0, "high_250": 31000.0, "drawdown_250_pct": -35.0,
            "episode_start_date": "2026-09-10", "episode_sessions": 8, "panic_low": 18500.0,
            "new_episode_low_2": False, "stab_state": "STAB2", "ret_10d": -12.0, "idx_ret_5d": -6.0,
            "idx_ret_10d": -9.0, "beta_60": 1.0, "idio_ret_10d": -3.0, "index_atr_ratio": 1.0,
            "avg_value_20_bn": 100.0, "index_status": "OK"}
    base.update(over)
    return base


MARKET = {"state": "PANIC_BEAR", "regime": "PANIC_BEAR", "breadth_pct": 20.0}
CALM = {"state": "RISK_ON", "regime": "RISK_ON", "breadth_pct": 60.0}


def news(cls="opinion_only", items=(), margin=True, insider=(), checked="2026-09-22", searched=()):
    rec = {"checked_session": checked, "class": cls, "items": list(items), "insider_sell": list(insider),
           "searched": list(searched) or ["https://www.hsx.vn/a", "https://cafef.vn/b"], "source_sha": "abc"}
    if margin:
        rec["margin"] = {"hose_ineligible": False, "broker_cut": None, "status": "normal", "checked": AS_OF,
                         "urls": ["https://www.hsx.vn/margin"]}
    return rec


def item(day, ev="FACT_PRIMARY", impact=-2, kind="provision"):
    return {"date": day, "evidence_type": ev, "impact": impact, "type": kind, "url": "https://x.vn/a", "quote": "q"}


def events(checked="2026-09-22", evs=(), resolved=None):
    return {"checked": {"AAA": checked}, "events": [{"ticker": "AAA", **e} for e in evs],
            "resolved": resolved or {}}


def journal(text=NAV + THESIS):
    from stock_agent.swing_desk.journal import parse_journal
    j = parse_journal(text, price_lookup=lambda t, d: 20000.0)
    assert j.errors == [], j.errors
    return j


# ------------------------------------------------------------------ classification

def classify(f, market=MARKET, n=None, **kw):
    from stock_agent.swing_desk.bottomfish import classify_fall
    return classify_fall(f, market, n, data_as_of=AS_OF, cfg=bf(), **kw)


def test_classify_a_market_panic():
    assert classify(fields(), n=news())["fall_type"] == "a"


def test_classify_b_via_fact_item():
    r = classify(fields(), n=news(items=[item("2026-09-15")]))
    assert r["fall_type"] == "b" and r["evidence"]


def test_classify_b_via_margin_cut_item_even_with_low_weight():
    r = classify(fields(), n=news(items=[item("2026-09-15", ev="OPINION", impact=0, kind="margin_cut")]))
    assert r["fall_type"] == "b"


def test_old_fact_item_outside_20_sessions_does_not_count():
    assert classify(fields(), n=news(items=[item("2026-07-01")]))["fall_type"] == "a"


def test_classify_b_via_large_idio_fall_with_none_found():
    r = classify(fields(idio_ret_10d=-18.0, idx_ret_5d=0.0), market=CALM, n=news(cls="none_found"))
    assert r["fall_type"] == "b" and "unknown cause" in r["reason"]


def test_classify_c_tech_flush_and_panic_reclassified_as_a():
    f = fields(idio_ret_10d=-10.0, idx_ret_5d=-1.0)
    assert classify(f, market=CALM, n=news(cls="none_found"))["fall_type"] == "c"
    r = classify(f, market={"state": "PANIC_BEAR", "breadth_pct": 45.0}, n=news(cls="none_found"))
    assert r["fall_type"] == "a" and "PANIC_BEAR" in r["reason"]


def test_unclassified_without_news_or_beta_and_not_a_bottomfish():
    assert classify(fields(), n=None)["fall_type"] == "UNCLASSIFIED"
    assert classify(fields(), n=news(checked="2026-09-01"))["fall_type"] == "UNCLASSIFIED"
    r = classify(fields(beta_60=None, idio_ret_10d=None), n=news())
    assert r["fall_type"] == "UNCLASSIFIED" and "beta" in r["reason"]
    assert classify(None, n=news())["fall_type"] == "UNCLASSIFIED"
    calm = classify(fields(idio_ret_10d=-2.0, idx_ret_5d=0.0), market=CALM, n=news())
    assert calm["fall_type"] == "NOT_A_BOTTOMFISH"


def test_none_found_with_fewer_than_two_searched_sources_is_unclassified():
    r = classify(fields(idio_ret_10d=-10.0, idx_ret_5d=0.0), market=CALM,
                 n=news(cls="none_found", searched=["https://www.hsx.vn/a"]))
    assert r["fall_type"] == "UNCLASSIFIED" and "searched" in r["reason"]


# ------------------------------------------------------------------ sizing

def plan(profile, *, nav=11_200_000.0, entry=25_000.0, fall="a", f=None, cfg=None, **kw):
    from stock_agent.swing_desk.bottomfish import bf_plan
    return bf_plan(entry, nav, profile=profile, cfg=cfg or bf(), fall_type=fall, fields=f or fields(),
                   trade_day=TDAY, lot=1, **kw)


def spec_cfg(risk=0.5):
    c = bf()
    for p in ("P1", "P3"):
        c["profiles"][p]["risk_pct_nav"] = risk
    return c


@pytest.mark.parametrize("profile,fall,risk,eff,value,qty", [
    ("P1", "a", 56_000, 19.6, 285_714.29, 11),
    ("P1", "c", 28_000, 19.6, 142_857.14, 5),
    ("P2", "a", 112_000, 32.6, 343_558.28, 13),
    ("P3", "b", 56_000, 50.6, 110_671.94, 4),
])
def test_spec_worked_numbers_at_half_percent(profile, fall, risk, eff, value, qty):
    p = plan(profile, fall=fall, cfg=spec_cfg(0.5))
    assert p["risk_vnd"] == pytest.approx(risk)
    assert p["effective_loss_pct"] == pytest.approx(eff)
    assert p["value_max"] == pytest.approx(value, abs=0.01)
    assert p["qty"] == qty
    if profile == "P2":
        assert p["tranche_qty"] == 4 and p["max_tranches"] == 3


@pytest.mark.parametrize("profile,fall,risk,qty", [
    ("P1", "a", 112_000, 22), ("P1", "c", 56_000, 11), ("P2", "a", 112_000, 13), ("P3", "b", 112_000, 8)])
def test_user_decision_one_percent_nav(profile, fall, risk, qty):
    p = plan(profile, fall=fall)
    assert p["risk_vnd"] == pytest.approx(risk) and p["qty"] == qty


def test_multipliers_compose_and_weight_cap_binds():
    p = plan("P1", fall="c", f=fields(index_atr_ratio=2.5))
    assert p["risk_multiplier"] == pytest.approx(0.25) and p["risk_vnd"] == pytest.approx(28_000) and p["qty"] == 5
    c = bf()
    c["profiles"]["P1"]["max_weight_pct"] = 3.0
    capped = plan("P1", cfg=c)
    assert capped["capped_by"] == "max_weight" and capped["qty"] == 13          # 336,000 / 25,000


def test_requested_amount_over_budget_and_scenarios():
    p = plan("P1", amount=1_000_000)
    req = p["requested"]
    assert req["qty"] == 40 and req["value"] == 1_000_000
    assert req["risk_effective_vnd"] == pytest.approx(196_000) and req["over_budget"]
    s = p["scenarios"]
    assert s["three_more_floors_vnd"] == pytest.approx(p["position_value"] * (1 - 0.93 ** 3) + p["position_value"] * 0.006)
    assert s["effective_loss_vnd"] == pytest.approx(p["position_value"] * 0.196)
    assert p["catastrophe_stop"] == pytest.approx(25_000 * 0.88)
    assert p["invalidation"] == 18_500.0 and p["time_stop_session"] == "2026-10-06"
    assert p["earliest_sell"].startswith("2026-09-24")
    assert plan("P2")["invalidation"] == pytest.approx(18_500 * 0.95)
    p3 = plan("P3", fall="b")
    assert p3["invalidation"] is None and p3["catastrophe_stop"] is None and p3["edge_note"]


def test_p0_has_no_plan():
    assert plan("P0") is None


# ------------------------------------------------------------------ gate

def gate(profile="P1", *, f="default", market=MARKET, fall="a", n="default", ev="default", j="default",
         price=None, amount=None, tranche=None, in_scope=True, snapshot=None):
    from stock_agent.swing_desk.bottomfish import bf_plan, evaluate_bf_gate
    f = fields() if f == "default" else f
    n = news() if n == "default" else n
    ev = events() if ev == "default" else ev
    j = journal() if j == "default" else j
    nav = (j.nav() if j else None) or 11_200_000.0
    p = bf_plan(price or 20000.0, nav, profile=profile, cfg=bf(), fall_type=fall, fields=f or fields(),
                trade_day=TDAY, lot=1, amount=amount, tranche=tranche) if profile != "P0" else None
    rules = evaluate_bf_gate(ticker="AAA", profile=profile, fields=f, market=market,
                             fall={"fall_type": fall, "reason": "test"}, snapshot=snapshot or {"ok": True, "reason": "ok"},
                             in_scope=in_scope, held=False, journal=j, news=n, events=ev, plan=p, price=price,
                             tranche=tranche, trade_day=TDAY, data_as_of=AS_OF, cfg=bf())
    return {r["id"]: r for r in rules}


def failed(rules, sev=None):
    return {k for k, r in rules.items() if r["status"] == "FAIL" and (sev is None or r["severity"] == sev)}


def test_clean_context_passes_every_rule():
    from stock_agent.swing_desk.gate import verdict
    r = gate()
    assert failed(r) == set(), {k: v["reason"] for k, v in r.items() if v["status"] != "PASS"}
    assert set(r) >= {f"BF-G{i}" for i in range(13)}
    assert verdict(list(r.values())) == "PASS"


@pytest.mark.parametrize("kwargs,rule,sev", [
    (dict(snapshot={"ok": False, "reason": "stale"}), "BF-G0", "hard"),
    (dict(in_scope=False), "BF-G1", "hard"),
    (dict(fall="b"), "BF-G2", "hard"),
    (dict(fall="UNCLASSIFIED"), "BF-G2", "hard"),
    (dict(fall="NOT_A_BOTTOMFISH"), "BF-G2", "hard"),
    (dict(f=fields(floor_t=True, ret_1d=-6.9, stab_state="FALLING")), "BF-G3", "hard"),
    (dict(f=fields(locked_floor_t1=True)), "BF-G3", "hard"),
    (dict(price=18700.0), "BF-G3", "hard"),
    (dict(f=fields(stab_state="WEAK")), "BF-G4", "hard"),
    (dict(ev=events(evs=[{"date": "2026-11-20", "type": "egm_capital"}])), "BF-G5", "hard"),
    (dict(ev=events(evs=[{"date": "2026-10-01", "type": "ex_dividend"}])), "BF-G5", "hard"),
    (dict(n=news(items=[item("2026-09-18")])), "BF-G6", "hard"),
    (dict(n=news(insider=[{"start": "2026-09-01", "end": "2026-10-30", "shares": 9e6, "url": "https://x"}])), "BF-G6", "soft"),
    (dict(f=fields(avg_value_20_bn=10.0)), "BF-G7", "soft"),
    (dict(n={**news(), "margin": {"hose_ineligible": False, "broker_cut": "2026-09-15", "status": "normal",
                                  "checked": AS_OF, "urls": ["https://x"]}}), "BF-G8", "hard"),
    (dict(n={**news(), "margin": {"hose_ineligible": True, "broker_cut": None, "status": "normal",
                                  "checked": AS_OF, "urls": ["https://x"]}}), "BF-G8", "hard"),
    (dict(amount=2_000_000), "BF-G10", "hard"),
    (dict(j=journal(NAV + THESIS + "2026-09-15|BUY|AAA||21000|5|bottomfish|PASS|calm||bf_type=a; profile=P1; "
                                  "episode=AAA-20260910; attempt=1\n")), "BF-G11", "hard"),
    (dict(j=journal(NAV)), "BF-G12", "hard"),
])
def test_each_rule_fails_with_expected_severity(kwargs, rule, sev):
    r = gate(**kwargs)
    assert r[rule]["status"] == "FAIL" and r[rule]["severity"] == sev, r[rule]


@pytest.mark.parametrize("kwargs,rule", [
    (dict(snapshot={"ok": False, "reason": "no verified snapshot"}, f=None), "BF-G0"),
    (dict(fall="UNCLASSIFIED"), "BF-G2"),
    (dict(ev=None), "BF-G5"),
    (dict(ev=events(checked="2026-09-19")), "BF-G5"),
    (dict(n=None), "BF-G6"),
    (dict(n=news(margin=False)), "BF-G8"),
    (dict(n={**news(), "margin": {"hose_ineligible": False, "broker_cut": None, "status": "normal",
                                  "checked": "2026-07-01", "urls": ["https://x"]}}), "BF-G8"),
    (dict(j=None), "BF-G10"),
])
def test_fail_closed_rules_are_hard_fail_never_unverified(kwargs, rule):
    r = gate(**kwargs)
    assert r[rule]["status"] == "FAIL" and r[rule]["severity"] == "hard"
    assert not [x for x in r.values() if x["status"] == "UNVERIFIED" and x["id"] in
                {"BF-G0", "BF-G2", "BF-G5", "BF-G6", "BF-G8", "BF-G10"}]


def test_missing_fields_fail_closed_for_price_rules():
    r = gate(f=None)
    assert {"BF-G3", "BF-G4"} <= failed(r, "hard")


def test_p0_always_blocks():
    r = gate("P0")
    assert r["BF-G2"]["status"] == "FAIL" and "P0" in r["BF-G2"]["reason"]


def test_p3_blocks_until_events_resolved_and_ten_sessions_after_last_fact():
    fact = [item("2026-09-14")]                    # 5 sessions before as_of
    egm = [{"date": "2026-09-15", "type": "egm_capital", "pending_dilution": True}]
    r = gate("P3", fall="b", n=news(items=fact), ev=events(evs=egm))
    assert {"BF-G5", "BF-G6"} <= failed(r, "hard")
    resolved = {"AAA": [{"date": "2026-09-15", "type": "egm_capital", "source": "https://ir", "recorded": "2026-09-18"}]}
    r = gate("P3", fall="b", n=news(items=fact), ev=events(evs=egm, resolved=resolved))
    assert "BF-G5" not in failed(r) and "BF-G6" in failed(r)
    old = [item("2026-09-04")]                     # 11 sessions before as_of
    p3j = journal(NAV + THESIS.replace("bf_type=a; profile=P1", "bf_type=b; profile=P3"))
    r = gate("P3", fall="b", n=news(items=old), ev=events(evs=egm, resolved=resolved), j=p3j)
    assert failed(r) == set(), {k: v["reason"] for k, v in r.items() if v["status"] == "FAIL"}


def test_p2_needs_scalein_plan_and_counts_tranches():
    thesis = THESIS.replace("profile=P1", "profile=P2").replace("max_attempts=1", "max_attempts=3; plan=scalein")
    r = gate("P2", j=journal(NAV + thesis))
    assert failed(r) == set(), {k: v["reason"] for k, v in r.items() if v["status"] == "FAIL"}
    r = gate("P2", j=journal(NAV + THESIS.replace("profile=P1", "profile=P2")))
    assert "BF-G12" in failed(r) and "scalein" in r["BF-G12"]["reason"]
    buys = "".join(f"2026-09-1{k}|BUY|AAA||21000|4|bottomfish|PASS|calm||bf_type=a; profile=P2; episode=AAA-20260910; "
                   f"tranche={k - 4}/3\n" for k in (5, 6, 7))
    r = gate("P2", j=journal(NAV + thesis.replace("2026-09-21", "2026-09-14") + buys), tranche=4)
    assert "BF-G11" in failed(r)


def test_bf_loss_cooldown_and_monthly_episode_cap():
    loss = ("2026-09-08|BUY|AAA||21000|5|bottomfish|PASS|calm||bf_type=a; profile=P1; episode=AAA-20260901\n"
            "2026-09-11|SELL|AAA||19000|5|stop||fear||\n")
    r = gate(j=journal(NAV + THESIS + loss))
    assert "BF-G11" in failed(r) and "cooldown" in r["BF-G11"]["reason"]
    two = ("2026-09-03|BUY|BBB||21000|5|bottomfish|PASS|calm||episode=BBB-20260901\n"
           "2026-09-08|BUY|VCB||21000|5|bottomfish|PASS|calm||episode=VCB-20260905\n")
    r = gate(j=journal(NAV + THESIS + two))
    assert "BF-G11" in failed(r) and "month" in r["BF-G11"]["reason"]


def test_budget_counts_bf_positions_and_total_open_risk():
    held = ("2026-09-03|BUY|BBB||21000|5|bottomfish|PASS|calm||episode=BBB-20260901\n"
            "2026-09-04|BUY|VCB||21000|5|bottomfish|PASS|calm||episode=VCB-20260901\n")
    r = gate(j=journal(NAV + THESIS + held))
    assert "BF-G10" in failed(r) and "BF positions" in r["BF-G10"]["reason"]
    swing = "2026-09-03|BUY|HPG||25000|100|S1|PASS|calm||\n2026-09-04|BUY|FPT||25000|200|S1|PASS|calm||\n"
    r = gate(j=journal(NAV + THESIS + swing))
    assert "BF-G10" in failed(r) and "open risk" in r["BF-G10"]["reason"]


# ------------------------------------------------------------------ behaviour

def flags(*, f=None, j="default", price=None, why=None, reason="bottomfish", fall="a", profile="P1",
          amount=None, tranche=None, g3g4_pass=True, closes=None):
    from stock_agent.swing_desk.bottomfish import bf_plan, evaluate_bf_behaviour
    f = f or fields()
    j = journal() if j == "default" else j
    p = bf_plan(price or 20000.0, 11_200_000.0, profile=profile, cfg=bf(), fall_type=fall, fields=f,
                trade_day=TDAY, lot=1, amount=amount, tranche=tranche)
    rules = evaluate_bf_behaviour(ticker="AAA", fields=f, journal=j, plan=p, price=price, why=why, reason=reason,
                                  fall_type=fall, profile=profile, tranche=tranche, structural_ok=g3g4_pass,
                                  trade_day=TDAY, cfg=bf(), close_lookup=(closes or {}).get)
    return {r["id"]: r for r in rules}


def test_behaviour_clean_passes():
    r = flags()
    assert failed(r) == set() and all(x["kind"] == "behaviour" for x in r.values())
    assert set(r) >= {f"BF-B{i}" for i in range(1, 9)}


@pytest.mark.parametrize("kwargs,rule,sev", [
    (dict(f=fields(floor_t=True)), "BF-B1", "hard"),
    (dict(f=fields(locked_floor_t1=True)), "BF-B1", "hard"),
    (dict(price=18700.0), "BF-B1", "hard"),
    (dict(f=fields(drawdown_250_pct=-45.0), why="anchor"), "BF-B2", "soft"),
    (dict(f=fields(drawdown_250_pct=-45.0, high_60=21000.0),
          j=journal(NAV + THESIS.replace("checklist=done", "target=22000; checklist=done"))), "BF-B2", "soft"),
    (dict(why="cheap"), "BF-B3", "soft"),
    (dict(f=fields(drawdown_250_pct=-55.0), fall="c"), "BF-B3", "soft"),
    (dict(j=journal(NAV + THESIS + "2026-09-15|BUY|AAA||25000|5|bottomfish|PASS|calm||episode=AAA-20260910\n")),
     "BF-B4", "hard"),
    (dict(j=journal(NAV + THESIS + "2026-09-08|BUY|AAA||21000|5|bottomfish|PASS|calm||\n"
                                   "2026-09-11|SELL|AAA||19000|5|stop||fear||\n")), "BF-B5", "hard"),
    (dict(j=journal(NAV + THESIS + "2026-09-08|BUY|HPG||30000|5|S1|PASS|calm||\n"), closes={"HPG": 26000.0}),
     "BF-B6", "soft"),
    (dict(j=journal(NAV + THESIS + "2026-09-03|BUY|BBB||20000|10|bottomfish|PASS|calm||\n"
                                   "2026-09-08|SELL|BBB||22000|10|target||calm||\n"), amount=500_000), "BF-B7", "soft"),
    (dict(reason="forum"), "BF-B8", "soft"),
    (dict(why="kol"), "BF-B8", "soft"),
])
def test_each_flag(kwargs, rule, sev):
    r = flags(**kwargs)
    assert r[rule]["status"] == "FAIL" and r[rule]["severity"] == sev, r[rule]


def test_b2_b3_are_soft_by_user_decision():
    r = flags(f=fields(drawdown_250_pct=-60.0), why="cheap", fall="b")
    assert r["BF-B2"]["severity"] == r["BF-B3"]["severity"] == "soft"


def test_b4_scalein_exemption_needs_tranche_to_pass_g3_g4():
    text = (NAV + THESIS.replace("profile=P1", "profile=P2").replace("2026-09-21", "2026-09-14")
            .replace("checklist=done", "plan=scalein; checklist=done")
            + "2026-09-15|BUY|AAA||25000|4|bottomfish|PASS|calm||profile=P2; episode=AAA-20260910; tranche=1/3\n")
    ok = flags(j=journal(text), profile="P2", tranche=2)
    assert ok["BF-B4"]["status"] == "PASS"
    bad = flags(j=journal(text), profile="P2", tranche=2, g3g4_pass=False)
    assert bad["BF-B4"]["status"] == "FAIL"
    over = flags(j=journal(text), profile="P2", tranche=4)
    assert over["BF-B4"]["status"] == "FAIL"


def _sessions_before(n):
    from datetime import timedelta
    from stock_agent.swing_desk.calendar_events import session_distance
    d = TDAY
    while session_distance(d, TDAY) < n:
        d -= timedelta(days=1)
    from stock_agent.data.exchange_calendar import is_trading_day
    while not is_trading_day(d):
        d -= timedelta(days=1)
    assert session_distance(d, TDAY) == n
    return str(d)


def test_b5_cooldown_counted_in_sessions():
    def text(sell):
        return (NAV + THESIS + "2026-07-01|BUY|AAA||21000|5|bottomfish|PASS|calm||\n"
                f"{sell}|SELL|AAA||19000|5|stop||fear||\n")
    assert flags(j=journal(text(_sessions_before(21))))["BF-B5"]["status"] == "PASS"
    assert flags(j=journal(text(_sessions_before(20))))["BF-B5"]["status"] == "FAIL"
