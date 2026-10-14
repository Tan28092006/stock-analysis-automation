from datetime import date

import pytest

from stock_agent.swing_desk.config import load_config
from stock_agent.swing_desk.journal import parse_journal
from tests.test_swing_desk_gate import good_ind

CFG = load_config()
TRADE_DAY = date(2026, 9, 22)   # Tue; previous sessions: 21 (Mon), 18 (Fri), 17 (Thu)


def flags(journal_text=None, *, ind=None, price=None, amount=None, reason=None, setups=("S1",),
          ticker="HPG", checks=None, review=False, bars=None):
    from stock_agent.swing_desk import behaviour, sizing
    ind = ind or good_ind()
    journal = parse_journal(journal_text) if journal_text is not None else None
    assert journal is None or journal.errors == []
    plan = sizing.plan(price or ind["close"], 1e9, CFG, amount=amount)
    return behaviour.evaluate_behaviour(ticker=ticker, ind=ind, journal=journal, trade_day=TRADE_DAY,
                                        cfg=CFG, plan=plan, price=price, reason=reason,
                                        setups=list(setups), checks=checks or [])


def get(rules, rule_id):
    return next(r for r in rules if r["id"] == rule_id)


def test_no_journal_makes_journal_flags_unverified():
    rules = flags(None)
    for rid in ("B3", "B4", "B5", "B7"):
        assert get(rules, rid)["status"] == "UNVERIFIED"
    assert get(rules, "B1")["status"] == "PASS"


@pytest.mark.parametrize("over,price", [({"ret_1d": 5.2}, None), ({"limit_up_t1": True}, None),
                                        ({}, 27350 * 1.07 * 0.995)])
def test_b1_fomo_ceiling_is_hard(over, price):
    rules = flags("", ind=good_ind(**over), price=price)
    assert get(rules, "B1")["status"] == "FAIL" and get(rules, "B1")["severity"] == "hard"


def test_b2_gap_chase_soft():
    rules = flags("", price=27350 * 1.031)
    assert get(rules, "B2")["status"] == "FAIL" and get(rules, "B2")["severity"] == "soft"
    rules = flags("", price=27350 * 1.016)                       # with a setup: 1.5% threshold
    assert get(rules, "B2")["status"] == "FAIL"
    rules = flags("", price=27350 * 1.016, setups=())
    assert get(rules, "B2")["status"] == "PASS"


def test_b3_revenge_counts_sessions_not_calendar_days():
    loss_fri = "2026-09-16|BUY|FPT||10000|1000|S1|PASS|calm||\n2026-09-18|SELL|FPT||9500|1000|stop||fear||\n"
    loss_mon = loss_fri.replace("2026-09-18|SELL", "2026-09-21|SELL")
    # Monday loss -> Tuesday trade: previous session -> hard BLOCK
    assert get(flags(loss_mon), "B3")["status"] == "FAIL"
    assert get(flags(loss_mon), "B3")["severity"] == "hard"
    # Friday loss -> Tuesday trade: 2 sessions ago (Mon in between) -> outside 1-session window
    assert get(flags(loss_fri), "B3")["status"] == "PASS"
    win = loss_mon.replace("9500", "10500")
    assert get(flags(win), "B3")["status"] == "PASS"


def test_b3_two_consecutive_losses_pause_one_extra_session():
    text = ("2026-09-14|BUY|FPT||10000|1000|S1|PASS|calm||\n2026-09-16|SELL|FPT||9500|1000|stop||fear||\n"
            "2026-09-16|BUY|MWG||10000|1000|S1|PASS|calm||\n2026-09-18|SELL|MWG||9600|1000|stop||fear||\n")
    rules = flags(text)                       # Fri second loss, Tue trade = 2 sessions later
    pause = get(rules, "B3P")
    assert pause["status"] == "FAIL" and pause["severity"] == "hard"
    assert get(rules, "B3")["status"] == "PASS"


def test_b4_averaging_down_hard_unless_scalein_plan():
    text = "2026-09-15|BUY|HPG||28000|1000|S1|PASS|calm||\n"
    assert get(flags(text), "B4")["status"] == "FAIL"      # close 27350 < avg 28000
    assert get(flags(text), "B4")["severity"] == "hard"
    planned = text.replace("calm||", "calm||plan=scalein")
    assert get(flags(planned), "B4")["status"] == "PASS"
    late_plan = text + "2026-09-21|THESIS|HPG|||||||T9|plan=scalein\n"
    assert get(flags(late_plan), "B4")["status"] == "FAIL"   # plan written after the first lot
    above = text.replace("28000", "26000")
    assert get(flags(above), "B4")["status"] == "PASS"


def test_b5_size_creep():
    rows = "".join(f"2026-09-{d:02d}|BUY|FPT||10000|1000|S1|PASS|calm||\n2026-09-{d:02d}|SELL|FPT||10000|1000|time||calm||\n"
                   for d in (14, 15, 16))
    # median prior entry value 10m; request 100m -> 10x
    rules = flags(rows, amount=100_000_000)
    assert get(rules, "B5")["status"] == "FAIL" and get(rules, "B5")["severity"] == "soft"
    assert get(flags(rows, amount=12_000_000), "B5")["status"] == "PASS"
    assert get(flags("".join(rows.splitlines(True)[:2]), amount=100_000_000), "B5")["status"] in {"PASS", "NA"}


def test_b6_hype_reason_without_setup_or_volume_spike():
    assert get(flags("", reason="forum", setups=()), "B6")["status"] == "FAIL"
    assert get(flags("", reason="forum"), "B6")["status"] == "PASS"
    spike = good_ind(vol_ratio_20=3.5, ret_1d=4.2)
    assert get(flags("", ind=spike), "B6")["status"] == "FAIL"


def test_b7_overtrading_windows():
    rows = "".join(f"2026-09-{d:02d}|BUY|X{d}||10000|100|S1|PASS|calm||\n" for d in (17, 18, 21))
    assert get(flags(rows), "B7")["status"] == "FAIL"                      # 3 in 5 sessions + this = 4
    assert get(flags(rows.split("\n", 1)[1]), "B7")["status"] == "PASS"


def test_cooling_off_behavioural_block_holds_until_next_session():
    checks = [{"ticker": "HPG", "trade_session": str(TRADE_DAY), "verdict": "BLOCK", "behaviour_block": True}]
    cool = get(flags("", checks=checks), "COOL")
    assert cool["status"] == "FAIL" and cool["severity"] == "hard"
    old = [{"ticker": "HPG", "trade_session": "2026-09-21", "verdict": "BLOCK", "behaviour_block": True}]
    assert get(flags("", checks=old), "COOL")["status"] == "PASS"


def test_b8_stop_not_honoured_review_flag():
    from stock_agent.swing_desk.behaviour import review_flags
    from tests.swing_desk_support import frame
    import numpy as np
    closes = np.full(30, 10000.0)
    closes[-6] = 9600.0            # close <= 3% stop at a sellable session
    df = frame(closes, end=date(2026, 9, 21))
    days = [str(d) for d in df.date]
    buy, sell = days[-10], days[-2]
    j = parse_journal(f"{buy}|BUY|FPT||10000|1000|S1|PASS|calm||\n{sell}|SELL|FPT||10000|1000|time||calm||\n")
    out = review_flags(j, {"FPT": df}, CFG)
    assert out["B8"]["count"] == 1
    timely = parse_journal(f"{buy}|BUY|FPT||10000|1000|S1|PASS|calm||\n{days[-5]}|SELL|FPT||9700|1000|stop||calm||\n")
    assert review_flags(timely, {"FPT": df}, CFG)["B8"]["count"] == 0


def test_b9_disposition_needs_ten_closed_trades():
    from stock_agent.swing_desk.behaviour import review_flags
    from stock_agent.data.exchange_calendar import add_trading_days
    rows = []
    start = date(2026, 6, 1)
    for i in range(10):
        b = add_trading_days(start, i * 12)
        losing = i % 2 == 0
        s = add_trading_days(b, 8 if losing else 2)
        rows.append(f"{b}|BUY|FPT||10000|100|S1|PASS|calm||\n{s}|SELL|FPT||{9500 if losing else 10500}|100|time||calm||\n")
    j = parse_journal("".join(rows))
    assert review_flags(j, {}, CFG)["B9"]["flag"] is True
    j9 = parse_journal("".join(rows[:9]))
    assert review_flags(j9, {}, CFG)["B9"]["flag"] is None
