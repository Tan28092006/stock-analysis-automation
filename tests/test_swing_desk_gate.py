from datetime import date

import pytest

from stock_agent.swing_desk.config import load_config

CFG = load_config()
TRADE_DAY = date(2026, 9, 22)          # Tuesday; not a VN30F expiry
UNIVERSE = ["HPG", "VCB", "ACB", "BID", "FPT"]


def good_ind(**over):
    ind = {"date": "2026-09-21", "close": 27350.0, "prev_close": 27000.0, "ret_1d": 1.3, "ret_3d": -2.0,
           "rsi2": 3.0, "sma5": 27600.0, "sma20": 29000.0, "sma50": 25000.0, "atr_pct": 2.5,
           "locked_mae_p80": -2.5, "avg_value_20_bn": 300.0, "vol_ratio_20": 1.0, "high_20": 30000.0,
           "limit_up_t": False, "limit_up_t1": False, "limit_down_t": False}
    ind.update(over)
    return ind


def run(ind=None, *, market="RISK_ON", fresh=True, ticker="HPG", price=None, amount=None, s2=False,
        portfolio=None, events="checked"):
    from stock_agent.swing_desk import gate, sizing
    ind = ind or good_ind()
    entry = price or ind["close"]
    nav = 1_000_000_000
    plan = sizing.plan(entry, nav, CFG, amount=amount)
    ev = {"checked": {ticker: str(TRADE_DAY)}, "events": []} if events == "checked" else events
    return gate.evaluate_gate(ticker=ticker, ind=ind, market={"state": market},
                              freshness={"ok": fresh, "reason": "stale snapshot" if not fresh else ""},
                              universe=UNIVERSE, cfg=CFG, trade_day=TRADE_DAY, plan=plan, price=price,
                              s2=s2, portfolio=portfolio if portfolio is not None else {"positions": {}, "nav": nav},
                              events=ev)


def status(result, rule_id):
    return next(r for r in result["rules"] if r["id"] == rule_id)


def test_clean_setup_passes_structural_gate():
    res = run()
    assert res["verdict"] == "PASS", res["failed"]
    assert res["setups"] == ["S1"]


def test_g0_stale_snapshot_blocks_everything_and_evaluates_nothing_else():
    res = run(fresh=False)
    assert res["verdict"] == "BLOCK" and res["failed"] == ["G0"]
    assert [r["id"] for r in res["rules"]] == ["G0"]


def test_g1_out_of_scope():
    res = run(ticker="XYZ")
    assert res["verdict"] == "BLOCK" and res["failed"] == ["G1"] and res["out_of_scope"]


@pytest.mark.parametrize("market,verdict,state", [("PANIC_BEAR", "BLOCK", "FAIL"), ("GRIND", "WARN", "FAIL"),
                                                   ("UNKNOWN", "WARN", "UNVERIFIED")])
def test_g2_regime(market, verdict, state):
    res = run(market=market)
    assert res["verdict"] == verdict and status(res, "G2")["status"] == state


def test_g3_below_sma50_blocks():
    res = run(good_ind(sma50=28000.0))
    assert res["verdict"] == "BLOCK" and "G3" in res["failed"]


def test_g3_missing_sma50_is_unverified_never_pass():
    res = run(good_ind(sma50=float("nan")))
    assert res["verdict"] == "WARN" and status(res, "G3")["status"] == "UNVERIFIED"


@pytest.mark.parametrize("over,verdict", [({"ret_1d": 6.6}, "BLOCK"), ({"ret_1d": 4.2}, "WARN"),
                                          ({"ret_3d": 8.5}, "WARN"), ({"rsi2": 92.0}, "WARN")])
def test_g4_chasing(over, verdict):
    ind = good_ind(**over)
    res = run(ind)
    assert "G4" in res["failed"]
    assert status(res, "G4")["severity"] == ("hard" if verdict == "BLOCK" else "soft")


@pytest.mark.parametrize("mult,severity", [(1.051, "hard"), (1.035, "soft")])
def test_g5_intraday_chase(mult, severity):
    res = run(price=27350.0 * mult)
    assert "G5" in res["failed"] and status(res, "G5")["severity"] == severity


def test_g5_not_evaluated_without_price():
    assert status(run(), "G5")["status"] == "NA"


@pytest.mark.parametrize("over,severity", [({"atr_pct": 4.5}, "hard"), ({"atr_pct": 3.5}, "soft"),
                                           ({"locked_mae_p80": -3.4}, "soft")])
def test_g6_stop_vs_noise(over, severity):
    res = run(good_ind(**over))
    assert "G6" in res["failed"] and status(res, "G6")["severity"] == severity


def test_g7_liquidity_soft():
    res = run(good_ind(avg_value_20_bn=15.0))
    assert res["verdict"] == "WARN" and "G7" in res["failed"]
    res = run(good_ind(avg_value_20_bn=30.0), amount=200_000_000)   # 200m > 5% of 30bn? no (1.5bn)
    assert "G7" not in res["failed"]


def test_g8_reward_risk_soft():
    res = run(good_ind(sma20=28500.0))
    assert res["verdict"] == "WARN" and "G8" in res["failed"]


def test_g9_risk_budget_hard_on_oversized_amount():
    res = run(amount=300_000_000)
    assert res["verdict"] == "BLOCK" and "G9" in res["failed"]


def test_g9_position_count_and_bank_concentration():
    positions = {s: {"qty": 1000, "avg_cost": 20000.0} for s in ("ACB", "BID")}
    res = run(ticker="VCB", portfolio={"positions": positions, "nav": 1e9})
    assert "G9" in res["failed"] and "bank" in status(res, "G9")["reason"]
    many = {s: {"qty": 1000, "avg_cost": 20000.0} for s in ("ACB", "FPT", "MWG", "VNM")}
    res = run(portfolio={"positions": many, "nav": 1e9})
    assert "G9" in res["failed"] and "positions" in status(res, "G9")["reason"]


def test_g9_without_journal_is_unverified():
    from stock_agent.swing_desk import gate
    res = run(portfolio={})
    assert status(res, "G9")["status"] == "UNVERIFIED" and res["verdict"] == "WARN"


def test_g10_events_unknown_known_and_expiry():
    res = run(events=None)
    assert status(res, "G10")["status"] == "UNVERIFIED" and res["verdict"] == "WARN"
    ev = {"checked": {"HPG": str(TRADE_DAY)}, "events": [{"ticker": "HPG", "date": "2026-09-24", "type": "ex_dividend"}]}
    res = run(events=ev)
    assert "G10" in res["failed"] and status(res, "G10")["severity"] == "soft"
    from stock_agent.swing_desk import gate, sizing
    expiry = date(2026, 10, 15)
    plan = sizing.plan(27350, 1e9, CFG)
    res = gate.evaluate_gate(ticker="HPG", ind=good_ind(), market={"state": "RISK_ON"}, freshness={"ok": True},
                             universe=UNIVERSE, cfg=CFG, trade_day=expiry, plan=plan, price=None, s2=False,
                             portfolio={"positions": {}, "nav": 1e9},
                             events={"checked": {"HPG": str(expiry)}, "events": []})
    assert "G10" in res["failed"] and "VN30F" in status(res, "G10")["reason"]


def test_g11_no_setup_soft_and_s2_counts_as_setup():
    res = run(good_ind(rsi2=40.0))
    assert res["verdict"] == "WARN" and "G11" in res["failed"]
    res = run(good_ind(rsi2=40.0), s2=True)
    assert "G11" not in res["failed"] and res["setups"] == ["S2"]


def test_unverified_never_yields_pass():
    from stock_agent.swing_desk.gate import verdict
    assert verdict([{"id": "G3", "status": "UNVERIFIED", "severity": "hard"}]) == "WARN"
    assert verdict([{"id": "G3", "status": "PASS", "severity": "hard"}]) == "PASS"
    assert verdict([{"id": "G3", "status": "FAIL", "severity": "hard"},
                    {"id": "G7", "status": "UNVERIFIED", "severity": "soft"}]) == "BLOCK"
