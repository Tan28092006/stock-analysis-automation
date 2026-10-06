"""Both skill copies carry the bottom-fish mode; checklist prompt exists with 10 numbered items."""
from pathlib import Path
import re

from tests.test_swing_desk_skill import CLAUDE, CODEX, ROOT, frontmatter

CAVEAT = "Chỉ tư vấn; bắt đáy chưa có bằng chứng edge tiến cứu; loại (b) có bằng chứng trôi giảm."


def test_mode4_and_caveat_in_both_copies():
    a, body_a = frontmatter(CLAUDE)
    b, body_b = frontmatter(CODEX)
    assert a["description"] == b["description"] and "(4) a bottom-fish check after sharp falls" in a["description"]
    for body in (body_a, body_b):
        assert "## Mode 4: bottom-fish check" in body and CAVEAT in body
        assert "no evidence of edge; measurement only" in body and "BOTTOMFISH_CHECKLIST.md" in body
    assert "`refresh --held` makes VCI calls" in body_b


def test_checklist_has_ten_numbered_items_and_matches_the_desk_caveat():
    text = (ROOT / "swing_desk/prompts/BOTTOMFISH_CHECKLIST.md").read_text(encoding="utf-8")
    numbers = [int(m) for m in re.findall(r"^(\d+)\. ", text, re.M)]
    assert numbers == list(range(1, 11))
    from stock_agent.swing_desk.bottomfish import CAVEAT as code_caveat
    assert code_caveat == CAVEAT and CAVEAT in text


def test_example_journal_has_a_bottomfish_buy_and_thesis():
    from stock_agent.swing_desk.journal import parse_journal
    text = (ROOT / "swing_desk/portfolio.example.txt").read_text(encoding="utf-8")
    j = parse_journal(text, price_lookup=lambda t, d: 40000.0)
    assert j.errors == []
    assert [e.ticker for e in j.buys() if e.reason == "bottomfish"] == ["HPG"]
    assert j.theses("HPG")[-1].keys["checklist"] == "done"
