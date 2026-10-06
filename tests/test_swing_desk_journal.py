from datetime import date
import os

import pytest

EXAMPLE = """\
# date      |action|ticker|amount_vnd |price  |qty |reason |gate |emotion|thesis_id|note
2026-10-07  |BUY   |HPG   |50000000   |27350  |    |S1     |PASS |calm   |T001     |rsi2 3.1
2026-10-07  |THESIS|HPG   |           |       |    |       |     |       |T001     |S1 dip; stop 26530; target 28800
2026-10-12  |SELL  |HPG   |           |28100  |1828|target |     |calm   |T001     |
2026-10-08  |BUY   |VIC   |80000000   |       |    |forum  |NONE |fomo   |         |group tip
"""


def test_example_parses_sorted_with_estimated_price():
    from stock_agent.swing_desk.journal import parse_journal
    j = parse_journal(EXAMPLE, price_lookup=lambda t, d: 40000.0 if t == "VIC" else None)
    assert j.errors == []
    assert [e.action for e in j.events] == ["BUY", "THESIS", "BUY", "SELL"]
    hpg = j.events[0]
    assert hpg.qty == 1828 and hpg.price == 27350  # odd lots: floor(50e6 / 27350) and not hpg.price_estimated
    vic = j.events[2]
    assert vic.price == 40000.0 and vic.price_estimated and vic.qty == 2000
    assert j.positions["VIC"]["qty"] == 2000 and "HPG" not in j.positions
    trade = j.closed[0]
    assert trade["ticker"] == "HPG" and trade["qty"] == 1828
    assert trade["gate"] == "PASS" and trade["emotion"] == "calm"
    assert trade["holding_sessions"] == 3
    gross = 28100 / 27350 - 1
    assert trade["gross_pct"] == pytest.approx(gross * 100)
    assert trade["net_pct"] == pytest.approx(((28100 * (1 - .0025)) / (27350 * (1 + .0015)) - 1) * 100)


@pytest.mark.parametrize("line,needle", [
    ("2026-13-01|BUY|HPG|1000000|10000||S1|PASS|calm||", "date"),
    ("2026-10-01|HOLD|HPG|1000000|10000||S1|PASS|calm||", "action"),
    ("2026-10-01|BUY|HPG", "fields"),
    ("2026-10-01|BUY|HPG|||||PASS|calm||", "amount"),
    ("2026-10-01|BUY|HPG|1000000|10000||S1|MAYBE|calm||", "gate"),
    ("2026-10-01|BUY|HPG|1000000|10000||S1|PASS|angry||", "emotion"),
    ("2026-10-01|BUY|HPG|1000000|10000||hunch|PASS|calm||", "reason"),
    ("2026-10-01|BUY|h!|1000000|10000||S1|PASS|calm||", "ticker"),
    ("2026-10-01|BUY|HPG|abc|10000||S1|PASS|calm||", "number"),
])
def test_invalid_lines_report_line_numbers_and_are_skipped(line, needle):
    from stock_agent.swing_desk.journal import parse_journal
    j = parse_journal("# header\n\n" + line + "\n")
    assert len(j.errors) == 1 and j.errors[0]["line"] == 3
    assert needle in j.errors[0]["error"].lower()
    assert j.events == []


def test_fifo_partial_sells_and_oversell():
    from stock_agent.swing_desk.journal import parse_journal
    text = ("2026-09-03|BUY|FPT||10000|1000|S1|PASS|calm||\n"
            "2026-09-04|BUY|FPT||12000|1000|S1|PASS|calm||\n"
            "2026-09-09|SELL|FPT||13000|1500|target||calm||\n")
    j = parse_journal(text)
    assert j.errors == []
    t = j.closed[0]
    assert t["entry_avg"] == pytest.approx((1000 * 10000 + 500 * 12000) / 1500)
    assert t["entry_date"] == "2026-09-03"
    pos = j.positions["FPT"]
    assert pos["qty"] == 500 and pos["avg_cost"] == pytest.approx(12000)
    bad = parse_journal(text + "2026-09-10|SELL|FPT||13000|900|target||calm||\n")
    assert bad.errors and bad.errors[0]["line"] == 4 and "exceeds" in bad.errors[0]["error"]
    all_out = parse_journal(text + "2026-09-10|SELL|FPT||13000||target||calm||\n")
    assert "FPT" not in all_out.positions and len(all_out.closed) == 2


def test_cash_and_nav_statements():
    from stock_agent.swing_desk.journal import parse_journal
    text = ("2026-09-01|NAV||800000000||||||\n"
            "2026-09-02|CASH||200000000||||||deposit\n"
            "2026-09-03|BUY|FPT||10000|1000|S1|PASS|calm||\n"
            "2026-09-08|SELL|FPT||11000|1000|target||calm||\n"
            "2026-09-09|CASH||-50000000||||||withdraw\n")
    j = parse_journal(text)
    assert j.errors == []
    pnl = 1000 * 11000 * (1 - .0025) - 1000 * 10000 * (1 + .0015)
    assert j.nav() == pytest.approx(800_000_000 + 200_000_000 + pnl - 50_000_000)
    assert parse_journal("2026-09-03|BUY|FPT||10000|1000|S1|PASS|calm||\n").nav() is None


def test_gate_autofill_from_checks_log_same_ticker_and_session():
    from stock_agent.swing_desk.journal import parse_journal
    checks = [{"ticker": "FPT", "trade_session": "2026-09-03", "verdict": "WARN"},
              {"ticker": "FPT", "trade_session": "2026-09-04", "verdict": "BLOCK"},
              {"ticker": "HPG", "trade_session": "2026-09-03", "verdict": "PASS"}]
    j = parse_journal("2026-09-03|BUY|FPT||10000|1000|S1||calm||\n"
                      "2026-09-03|BUY|VNM||10000|1000|S1||calm||\n", checks=checks)
    fpt, vnm = j.events
    assert fpt.gate == "WARN" and fpt.gate_source == "checks_log"
    assert vnm.gate == "NONE" and vnm.gate_source == "default"


def test_scalein_plan_is_recorded_on_first_lot():
    from stock_agent.swing_desk.journal import parse_journal
    j = parse_journal("2026-09-03|BUY|FPT||10000|1000|S1|PASS|calm||plan=scalein\n")
    assert j.positions["FPT"]["scalein_plan"] is True


def test_loader_never_writes_the_journal(tmp_path):
    from stock_agent.swing_desk.journal import load_journal
    path = tmp_path / "portfolio.txt"
    path.write_text(EXAMPLE + "garbage line\n", encoding="utf-8")
    before = (path.read_bytes(), os.stat(path).st_mtime_ns)
    j = load_journal(path, price_lookup=lambda t, d: 40000.0)
    assert j.errors and j.errors[0]["line"] == 6
    assert (path.read_bytes(), os.stat(path).st_mtime_ns) == before
    assert load_journal(tmp_path / "missing.txt") is None
