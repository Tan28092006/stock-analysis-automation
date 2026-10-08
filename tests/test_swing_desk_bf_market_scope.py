"""Market/sector/group-scope news is evidence of a market-wide fall (type a), not a firm-specific
fundamental shock (type b): it must not classify a fall as (b) nor fire BF-G6."""
from stock_agent.swing_desk import bottomfish
from stock_agent.swing_desk.config import load_bf_config

CFG = load_bf_config()


def news(*items):
    return {"items": list(items), "class": "none_found"}


def item(kind, impact=-1, ev="FACT_REPORTED", day="2026-10-01"):
    return {"date": day, "evidence_type": ev, "impact": impact, "type": kind, "url": "https://x.vn/a", "quote": "q"}


def test_market_scope_items_are_not_firm_facts():
    got = bottomfish.fact_negative_items(news(item("market_group"), item("group_contagion"), item("foreign_flow"),
                                              item("technical"), item("market"), item("sector")),
                                         "2026-10-07", CFG)
    assert got == []


def test_firm_items_still_count():
    got = bottomfish.fact_negative_items(news(item("market_group"), item("provision", impact=-2, ev="FACT_PRIMARY")),
                                         "2026-10-07", CFG)
    assert [i["type"] for i in got] == ["provision"]


def test_configured_negative_types_count_even_with_weak_impact():
    got = bottomfish.fact_negative_items(news(item("legal", impact=0, ev="OPINION")), "2026-10-07", CFG)
    assert [i["type"] for i in got] == ["legal"]
