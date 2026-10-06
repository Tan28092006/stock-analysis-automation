"""Agent-supplied, cited bottom-fish facts (bf-news) and event resolution (T12 spec §7)."""
from datetime import datetime

import pytest

from stock_agent.data.exchange_calendar import VN_TIMEZONE
from stock_agent.swing_desk.config import DeskPaths

NOW = datetime(2026, 10, 6, 20, 0, tzinfo=VN_TIMEZONE)
ITEM = "2026-09-24|FACT_PRIMARY|-2|provision|https://cdn.pnj.io/608.pdf|trích lập 7.071 tỷ"


def desk(tmp_path):
    return DeskPaths(repo=tmp_path, data_root=tmp_path, desk=tmp_path / "swing_desk")


@pytest.mark.parametrize("text,needle", [
    ("2026-09-24|FACT_PRIMARY|-2|provision||no url", "url"),
    ("2026-09-24|FACT_PRIMARY|-2|provision|ftp://x|q", "url"),
    ("2026-09-24|FACT|-2|provision|https://x.vn|q", "evidence_type"),
    ("2026-09-24|FACT_PRIMARY|-3|provision|https://x.vn|q", "impact"),
    ("2026-09-24|FACT_PRIMARY|x|provision|https://x.vn|q", "impact"),
    ("24/09/2026|FACT_PRIMARY|-2|provision|https://x.vn|q", "date"),
    ("2026-09-24|FACT_PRIMARY|-2|provision|https://x.vn|" + "q" * 201, "quote"),
    ("2026-09-24|FACT_PRIMARY|-2", "fields"),
])
def test_item_validation(text, needle):
    from stock_agent.swing_desk.bf_facts import parse_item
    with pytest.raises(ValueError, match=needle):
        parse_item(text)


def test_item_parses_with_weight():
    from stock_agent.swing_desk.bf_facts import parse_item
    it = parse_item(ITEM)
    assert it == {"date": "2026-09-24", "evidence_type": "FACT_PRIMARY", "impact": -2, "type": "provision",
                  "url": "https://cdn.pnj.io/608.pdf", "quote": "trích lập 7.071 tỷ"}


def test_margin_and_insider_parse():
    from stock_agent.swing_desk.bf_facts import parse_insider, parse_margin
    m = parse_margin("hose_ineligible=no|broker_cut=2026-07-22|checked=2026-10-07|https://a.vn/1|https://b.vn/2")
    assert m == {"hose_ineligible": False, "broker_cut": "2026-07-22", "status": "normal", "checked": "2026-10-07",
                 "urls": ["https://a.vn/1", "https://b.vn/2"]}
    with pytest.raises(ValueError, match="url"):
        parse_margin("hose_ineligible=no|checked=2026-10-07")
    i = parse_insider("2026-10-01..2026-10-30|9000000|https://x.vn/a")
    assert i == {"start": "2026-10-01", "end": "2026-10-30", "shares": 9000000.0, "url": "https://x.vn/a"}


def test_none_found_needs_two_searched_sources_one_primary(tmp_path):
    from stock_agent.swing_desk import bf_facts
    paths = desk(tmp_path)
    with pytest.raises(ValueError, match="searched"):
        bf_facts.mark_bf_news(paths, "HPG", "2026-10-07", news_class="none_found", searched=["https://www.hsx.vn/a"], now=NOW)
    with pytest.raises(ValueError, match="primary"):
        bf_facts.mark_bf_news(paths, "HPG", "2026-10-07", news_class="none_found",
                              searched=["https://cafef.vn/a", "https://vneconomy.vn/b"], now=NOW)
    rec = bf_facts.mark_bf_news(paths, "HPG", "2026-10-07", news_class="none_found",
                                searched=["https://www.hsx.vn/a", "https://cafef.vn/b"], now=NOW)
    assert rec["class"] == "none_found"
    rec = bf_facts.mark_bf_news(paths, "HPG", "2026-10-07", news_class="none_found",
                                searched=["https://www.hoaphat.com.vn/ir", "https://hpg.com.vn/x", "https://cafef.vn/b"],
                                now=NOW)
    assert len(rec["searched"]) == 3      # company domain counts as primary
    with pytest.raises(ValueError, match="class"):
        bf_facts.mark_bf_news(paths, "HPG", "2026-10-07", news_class="unknown", now=NOW)


def test_sha_recorded_and_remark_replaces_only_that_ticker_session(tmp_path):
    from stock_agent.swing_desk import bf_facts
    paths = desk(tmp_path)
    a = bf_facts.mark_bf_news(paths, "PNJ", "2026-10-07", news_class="fact_negative",
                              items=[bf_facts.parse_item(ITEM)], now=NOW)
    bf_facts.mark_bf_news(paths, "HPG", "2026-10-07", news_class="opinion_only", now=NOW)
    assert len(a["source_sha"]) == 64 and a["written_at"].startswith("2026-10-06T13:00")
    b = bf_facts.mark_bf_news(paths, "PNJ", "2026-10-07", news_class="opinion_only", now=NOW)
    store = bf_facts.load_bf_news(paths)
    assert store["PNJ"]["2026-10-07"]["class"] == "opinion_only" and b["source_sha"] != a["source_sha"]
    assert store["HPG"]["2026-10-07"]["class"] == "opinion_only"
    bf_facts.mark_bf_news(paths, "PNJ", "2026-10-08", news_class="fact_negative", items=[bf_facts.parse_item(ITEM)], now=NOW)
    store = bf_facts.load_bf_news(paths)
    assert set(store["PNJ"]) == {"2026-10-07", "2026-10-08"}
    assert bf_facts.latest_news(store, "PNJ")["checked_session"] == "2026-10-08"
    assert bf_facts.latest_news(store, "VCB") is None
    with pytest.raises(ValueError, match="url"):
        bf_facts.mark_bf_news(paths, "PNJ", "2026-10-08", news_class="fact_negative",
                              items=[{"date": "2026-09-24", "evidence_type": "FACT_PRIMARY", "impact": -2,
                                      "type": "provision", "url": "", "quote": ""}], now=NOW)


def test_events_pending_dilution_and_resolved_round_trip(tmp_path):
    from stock_agent.swing_desk import bf_facts, service
    paths = desk(tmp_path)
    service.mark_events(paths, "PNJ", "2026-10-07", [{"date": "2026-10-21", "type": "egm_capital"},
                                                    {"date": "2026-10-21", "type": "private_placement"}],
                        "https://cdn.pnj.io/622.pdf", pending_dilution=True)
    data = service.load_events(paths)
    assert all(e["pending_dilution"] for e in data["events"]) and data["checked"]["PNJ"] == "2026-10-07"
    bf_facts.mark_resolved(paths, "PNJ", "2026-10-22", [{"date": "2026-10-21", "type": "egm_capital"}],
                           "https://cdn.pnj.io/result.pdf")
    data = service.load_events(paths)
    assert data["resolved"]["PNJ"] == [{"date": "2026-10-21", "type": "egm_capital", "source": "https://cdn.pnj.io/result.pdf",
                                        "recorded_session": "2026-10-22"}]
    assert len(data["events"]) == 2           # resolution never deletes the calendar
    with pytest.raises(ValueError, match="source"):
        bf_facts.mark_resolved(paths, "PNJ", "2026-10-22", [{"date": "2026-10-21", "type": "egm_capital"}], "")


def test_swing_mark_events_output_unchanged_without_flag(tmp_path):
    from stock_agent.swing_desk import service
    paths = desk(tmp_path)
    data = service.mark_events(paths, "AAA", "2026-09-22", [{"date": "2026-09-25", "type": "ex_dividend"}], "src")
    assert data == {"checked": {"AAA": "2026-09-22"},
                    "events": [{"ticker": "AAA", "source": "src", "date": "2026-09-25", "type": "ex_dividend"}]}
