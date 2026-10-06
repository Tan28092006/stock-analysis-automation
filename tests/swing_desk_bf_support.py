"""Offline bottom-fish fixtures (imported, not collected).

PNJ 2026-09-25..2026-10-06 bars are copied from docs/audits/2026-10-06-bottom-fishing-research.md
§10 (an ILLUSTRATION: development data; no threshold was fitted to it). Earlier PNJ bars and all
index/peer bars are synthetic warm-up.
"""
from __future__ import annotations

from datetime import date, datetime
import json
from pathlib import Path

import numpy as np

from stock_agent.data.exchange_calendar import VN_TIMEZONE
from stock_agent.swing_desk.config import DeskPaths
from tests.swing_desk_support import AS_OF, dates, frame, make_repo, uptrend, write_snapshot

ROOT = Path(__file__).resolve().parents[1]
PNJ_AS_OF = date(2026, 10, 6)
PNJ_NOW = datetime(2026, 10, 6, 17, 0, tzinfo=VN_TIMEZONE)
PNJ_TAIL = [  # date, open, high, low, close, volume  (BF-doc §10)
    ("2026-09-25", 33200, 34400, 32800, 33000, 5.3e6),
    ("2026-09-28", 30700, 30700, 30700, 30700, 2.1e6),
    ("2026-09-29", 28600, 28600, 28600, 28600, 1.2e6),
    ("2026-09-30", 26600, 26600, 26600, 26600, 2.4e6),
    ("2026-10-01", 24750, 24750, 24750, 24750, 1.7e6),
    ("2026-10-02", 23050, 25450, 23050, 23050, 76.1e6),
    ("2026-10-05", 23050, 23300, 21450, 21650, 11.0e6),
    ("2026-10-06", 21600, 22350, 20150, 20150, 16.3e6),
]


def bf_repo(tmp_path, universe=("AAA", "BBB", "VCB")) -> DeskPaths:
    repo = make_repo(tmp_path, list(universe))
    for name in ("bottomfish_v1.json", "bottomfish_prospective_v1.json"):
        (repo / "configs/swing_desk" / name).write_bytes((ROOT / "configs/swing_desk" / name).read_bytes())
    return DeskPaths(repo=repo, data_root=repo, desk=repo / "swing_desk")


def pnj_index(n: int = 300):
    up = uptrend(n - 38, 1200.0, seed=7)
    up = up / up[-1] * 1900.0
    down = np.linspace(1900.0, 1785.0, 31)[1:]
    tail = np.linspace(1785.0, 1759.0, 8)[1:]
    return frame(np.concatenate([up, down, tail]), end=PNJ_AS_OF, volume=5e8)


def pnj_frame(n: int = 300):
    days = dates(n, PNJ_AS_OF)
    assert [str(d) for d in days[-8:]] == [r[0] for r in PNJ_TAIL]
    pos = {str(d): i for i, d in enumerate(days)}
    anchors = [(0, 84000.0), (pos["2026-07-02"], 63000.0), (pos["2026-07-24"], 30750.0), (n - 9, 33300.0)]
    xs, ys = zip(*anchors)
    warm = np.interp(np.arange(n - 8), xs, ys)
    o = np.concatenate([warm, [r[1] for r in PNJ_TAIL]])
    h = np.concatenate([warm * 1.005, [r[2] for r in PNJ_TAIL]])
    lo = np.concatenate([warm * 0.995, [r[3] for r in PNJ_TAIL]])
    c = np.concatenate([warm, [r[4] for r in PNJ_TAIL]])
    v = np.concatenate([np.full(n - 8, 6e6), [r[5] for r in PNJ_TAIL]])
    return frame(c, end=PNJ_AS_OF, opens=o, highs=h, lows=lo, volume=v)


def pnj_desk(tmp_path, monkeypatch, *, margin: bool = False, journal: bool = True) -> DeskPaths:
    """VN30 snapshot (AAA/BBB/VCB) + isolated extra snapshot (PNJ + VNINDEX) as of 2026-10-06."""
    from stock_agent.swing_desk import bf_facts, service
    paths = bf_repo(tmp_path)
    idx = pnj_index()
    main = {"VNINDEX": idx}
    for i, sym in enumerate(("AAA", "BBB", "VCB")):
        main[sym] = frame(uptrend(300, 20000.0 + 1000 * i, seed=11 + i), end=PNJ_AS_OF, volume=3e6)
    write_snapshot(paths.paper_snapshots, main, monkeypatch, as_of=PNJ_AS_OF, name="20261006T100000000000Z")
    write_snapshot(paths.extra_snapshots, {"PNJ": pnj_frame(), "VNINDEX": idx}, monkeypatch, as_of=PNJ_AS_OF,
                   name="20261006T100500000000Z")
    paths.desk.mkdir(parents=True, exist_ok=True)
    if journal:
        paths.portfolio.write_text("2026-10-01|NAV||11200000||||||\n"
                                   "2026-10-06|BUY|PNJ||20150|5|other|NONE|hope||\n", encoding="utf-8")
    service.mark_events(paths, "PNJ", "2026-10-07", [{"date": "2026-10-21", "type": "egm_capital"},
                                                    {"date": "2026-10-21", "type": "private_placement"}],
                        "https://cdn.pnj.io/622-2026.pdf", pending_dilution=True)
    bf_facts.mark_bf_news(
        paths, "PNJ", "2026-10-07", news_class="fact_negative",
        items=[bf_facts.parse_item("2026-09-24|FACT_PRIMARY|-2|provision|https://cdn.pnj.io/608-2026.pdf|trích lập 7.071 tỷ"),
               bf_facts.parse_item("2026-09-29|FACT_PRIMARY|-2|dilution|https://vneconomy.vn/pnj-550-trieu-cp.htm|550 triệu cp")],
        insider_sell=[bf_facts.parse_insider("2026-10-01..2026-10-30|9000000|https://thitruongtaichinhtiente.vn/pnj")],
        margin=(bf_facts.parse_margin("hose_ineligible=no|broker_cut=2026-07-22|checked=2026-10-07|"
                                      "https://vneconomy.vn/margin|https://vnbusiness.vn/margin") if margin else None),
        now=PNJ_NOW)
    return paths


def panic_frames(n: int = 300) -> dict:
    """VN30 market panic as of AS_OF (2026-09-21): VNINDEX -6% in 5 sessions; AAA (beta 2) hit two
    floors at t-4 / t-3, then three non-floor sessions with higher lows (STAB2)."""
    rng = np.random.default_rng(21)
    ir = rng.normal(0.0008, 0.006, n)
    ir[0] = 0.0
    ir[-5:] = [-0.034, -0.033, 0.005, 0.003, 0.002]
    idx = frame(1200.0 * np.cumprod(1 + ir), volume=5e8)
    frames = {"VNINDEX": idx, "AAA": frame(20000.0 * np.cumprod(1 + 2 * ir), volume=3e6)}
    for i, sym in enumerate(("BBB", "VCB")):
        frames[sym] = frame(uptrend(n, 21000.0 + 1000 * i, seed=12 + i), volume=3e6)
    return frames


def panic_desk(tmp_path, monkeypatch, *, thesis: bool = True, news: bool = True) -> DeskPaths:
    from stock_agent.swing_desk import bf_facts, service
    paths = bf_repo(tmp_path)
    write_snapshot(paths.paper_snapshots, panic_frames(), monkeypatch)
    paths.desk.mkdir(parents=True, exist_ok=True)
    text = "2026-09-01|NAV||11200000||||||\n"
    if thesis:
        text += ("2026-09-21|THESIS|AAA|||||||T1|bf_type=a; profile=P1; invalid=close<panic_low; "
                 "time_stop=2026-10-06; max_attempts=1; target=24000; checklist=done\n")
    paths.portfolio.write_text(text, encoding="utf-8")
    service.mark_events(paths, "AAA", "2026-09-22", [], "https://www.hsx.vn/lich-su-kien")
    if news:
        bf_facts.mark_bf_news(paths, "AAA", "2026-09-22", news_class="opinion_only",
                              searched=["https://www.hsx.vn/cbtt-aaa", "https://cafef.vn/aaa"],
                              margin=bf_facts.parse_margin("hose_ineligible=no|broker_cut=none|checked=2026-09-21|"
                                                           "https://www.hsx.vn/margin-list"),
                              now=datetime(2026, 9, 21, 18, 0, tzinfo=VN_TIMEZONE))
    return paths


__all__ = ["AS_OF", "PNJ_AS_OF", "PNJ_NOW", "bf_repo", "panic_desk", "panic_frames", "pnj_desk", "pnj_frame"]
