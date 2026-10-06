"""Review gains a bottom-fish section (BF-B10) only when reason=bottomfish rows exist."""
from tests.swing_desk_bf_support import panic_desk
from tests.swing_desk_support import NOW_EOD


def test_bf_b10_flags_position_held_past_time_stop(tmp_path, monkeypatch):
    from stock_agent.swing_desk import review
    paths = panic_desk(tmp_path, monkeypatch, thesis=False)
    with paths.portfolio.open("a", encoding="utf-8") as f:
        f.write("2026-09-03|THESIS|AAA|||||||T9|bf_type=a; profile=P1; invalid=close<1000; time_stop=2026-09-11; "
                "max_attempts=1; checklist=done\n"
                "2026-09-03|BUY|AAA||20000|5|bottomfish|PASS|calm|T9|bf_type=a; profile=P1; episode=AAA-20260903; attempt=1\n")
    rep = review.build_review(paths, now=NOW_EOD)
    bf = rep["bottomfish"]
    assert bf["episodes"] == [{"ticker": "AAA", "episode": "AAA-20260903", "attempts": 1, "first": "2026-09-03",
                               "tranches": []}]
    row = bf["open"][0]
    assert row["ticker"] == "AAA" and row["b10"] and "time stop" in row["b10"][0]
    assert row["floor_lock_days"] == 2 and bf["b10_count"] == 1
    assert "## Bắt đáy (BF)" in rep["markdown"] and "BF-B10" in rep["markdown"]


def test_bf_b10_below_invalidation(tmp_path, monkeypatch):
    from stock_agent.swing_desk import review
    paths = panic_desk(tmp_path, monkeypatch, thesis=False)
    with paths.portfolio.open("a", encoding="utf-8") as f:
        f.write("2026-09-03|THESIS|AAA|||||||T9|bf_type=a; profile=P1; invalid=close<99999; time_stop=2026-12-31; "
                "max_attempts=1; checklist=done\n"
                "2026-09-03|BUY|AAA||20000|5|bottomfish|PASS|calm|T9|episode=AAA-20260903\n")
    row = review.build_review(paths, now=NOW_EOD)["bottomfish"]["open"][0]
    assert any("invalidation" in x for x in row["b10"])


def test_swing_only_journal_has_no_bf_section(tmp_path, monkeypatch):
    from stock_agent.swing_desk import review
    paths = panic_desk(tmp_path, monkeypatch)
    with paths.portfolio.open("a", encoding="utf-8") as f:
        f.write("2026-09-03|BUY|BBB||20000|5|S1|PASS|calm||\n")
    rep = review.build_review(paths, now=NOW_EOD)
    assert "bottomfish" not in rep and "Bắt đáy" not in rep["markdown"]
