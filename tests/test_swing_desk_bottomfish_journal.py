"""Journal: reason `bottomfish`, structured note keys, malformed keys reported with line numbers."""
import os

import pytest

GOOD = (
    "2026-11-03|THESIS|HPG|||||||T020|bf_type=a; profile=P1; invalid=close<24100; time_stop=2026-11-17; "
    "max_attempts=1; target=26500; checklist=done\n"
    "2026-11-03|BUY|HPG|280000|25000||bottomfish|PASS|calm|T020|bf_type=a; profile=P1; episode=HPG-20261028; "
    "attempt=1; why=other; tranche=1/1\n")


def test_bottomfish_reason_and_keys_parse():
    from stock_agent.swing_desk.journal import parse_journal
    j = parse_journal(GOOD)
    assert j.errors == []
    buy = j.buys()[0]
    assert buy.reason == "bottomfish" and buy.qty == 11
    assert buy.keys == {"bf_type": "a", "profile": "P1", "episode": "HPG-20261028", "attempt": "1",
                        "why": "other", "tranche": "1/1"}
    th = j.theses("HPG")[0].keys
    assert th["invalid"] == "close<24100" and th["time_stop"] == "2026-11-17" and th["checklist"] == "done"
    assert j.positions["HPG"]["reason"] == "bottomfish"


@pytest.mark.parametrize("note,needle", [
    ("bf_type=d", "bf_type"), ("profile=P7", "profile"), ("episode=HPG-2026", "episode"),
    ("attempt=0", "attempt"), ("tranche=3/2", "tranche"), ("tranche=2", "tranche"),
])
def test_malformed_buy_keys_are_reported_with_line_numbers(note, needle):
    from stock_agent.swing_desk.journal import parse_journal
    j = parse_journal("# header\n2026-11-03|BUY|HPG|280000|25000||bottomfish|PASS|calm||" + note + "\n")
    assert len(j.errors) == 1 and j.errors[0]["line"] == 2 and needle in j.errors[0]["error"]


@pytest.mark.parametrize("note,needle", [
    ("bf_type=a; time_stop=17/11/2026", "time_stop"), ("bf_type=a; max_attempts=x", "max_attempts"),
    ("bf_type=a; checklist=yes", "checklist"), ("bf_type=a; target=abc", "target"),
])
def test_malformed_thesis_keys(note, needle):
    from stock_agent.swing_desk.journal import parse_journal
    j = parse_journal("2026-11-03|THESIS|HPG|||||||T1|" + note + "\n")
    assert len(j.errors) == 1 and j.errors[0]["line"] == 1 and needle in j.errors[0]["error"]


def test_swing_notes_with_equals_are_not_validated():
    from stock_agent.swing_desk.journal import parse_journal
    j = parse_journal("2026-11-03|BUY|HPG|280000|25000||S1|PASS|calm||profile=whatever; attempt=x\n"
                      "2026-11-03|THESIS|HPG|||||||T1|S1 dip; stop 24250; target=26000\n")
    assert j.errors == []


def test_parser_never_writes_the_file(tmp_path):
    from stock_agent.swing_desk.journal import load_journal
    path = tmp_path / "portfolio.txt"
    path.write_text(GOOD + "2026-11-03|BUY|HPG|1|1||bottomfish|||| bf_type=z\n", encoding="utf-8")
    before = (path.read_bytes(), os.stat(path).st_mtime_ns)
    j = load_journal(path)
    assert len(j.errors) == 1
    assert (path.read_bytes(), os.stat(path).st_mtime_ns) == before
