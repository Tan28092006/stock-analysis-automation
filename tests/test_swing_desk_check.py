from datetime import datetime
import json

import pytest

from stock_agent.data.exchange_calendar import VN_TIMEZONE
from stock_agent.swing_desk.config import DeskPaths
from tests.swing_desk_support import NOW_EOD, frame, make_repo, market_frames, uptrend, with_dip, write_snapshot


def setup_desk(tmp_path, monkeypatch, frames=None):
    repo = make_repo(tmp_path)
    paths = DeskPaths(repo=repo, data_root=repo, desk=repo / "swing_desk")
    frames = frames or market_frames()
    manifest = write_snapshot(paths.paper_snapshots, frames, monkeypatch)
    return paths, manifest


def ready(paths, journal="2026-09-01|NAV||1000000000||||||\n"):
    from stock_agent.swing_desk.service import mark_events
    paths.desk.mkdir(parents=True, exist_ok=True)
    paths.portfolio.write_text(journal, encoding="utf-8")
    mark_events(paths, "AAA", "2026-09-22", [], "test fixture")


def test_clean_dip_is_pass_and_logged(tmp_path, monkeypatch):
    from stock_agent.swing_desk import reclog, service
    paths, _ = setup_desk(tmp_path, monkeypatch)
    ready(paths)
    r = service.check(paths, "AAA", now=NOW_EOD, s2_override=[])
    assert r["verdict"] == "PASS", [(x["id"], x["reason"]) for x in r["rules"] if x["status"] != "PASS"]
    assert r["setups"] == ["S1"] and r["data_as_of"] == "2026-09-21"
    assert r["trade_session"] == "2026-09-22" and r["earliest_sell"] == "2026-09-24 13:00"
    p = r["plan"]
    assert p["qty"] > 0 and p["vnd_at_risk_effective"] <= 10_000_000 + 1
    rows = reclog.read_rows(paths.checks_log)
    assert rows[-1]["verdict"] == "PASS" and reclog.verify_chain(paths.checks_log)["ok"]


def test_stale_snapshot_blocks_with_g0_only(tmp_path, monkeypatch):
    from stock_agent.swing_desk import service
    paths, _ = setup_desk(tmp_path, monkeypatch)
    later = datetime(2026, 9, 22, 17, 0, tzinfo=VN_TIMEZONE)
    r = service.check(paths, "AAA", now=later, s2_override=[])
    assert r["verdict"] == "BLOCK" and r["failed"] == ["G0"] and "stale" in r["rules"][0]["reason"]


def test_tampered_snapshot_blocks(tmp_path, monkeypatch):
    from stock_agent.swing_desk import service
    paths, manifest = setup_desk(tmp_path, monkeypatch)
    csv = manifest.parent / "prices" / "AAA.csv"
    csv.write_text(csv.read_text().replace("\n2026-09-21,", "\n2026-09-21,1"), encoding="utf-8")
    r = service.check(paths, "AAA", now=NOW_EOD, s2_override=[], record=False)
    assert r["verdict"] == "BLOCK" and r["failed"] == ["G0"] and "verification" in r["rules"][0]["reason"]


def test_out_of_scope_ticker(tmp_path, monkeypatch):
    from stock_agent.swing_desk import service
    paths, _ = setup_desk(tmp_path, monkeypatch)
    r = service.check(paths, "XYZ", now=NOW_EOD, s2_override=[], record=False)
    assert r["verdict"] == "BLOCK" and r["out_of_scope"] and r["failed"] == ["G1"]


def test_intraday_check_uses_previous_session_bar(tmp_path, monkeypatch):
    from stock_agent.swing_desk import service
    paths, _ = setup_desk(tmp_path, monkeypatch)
    ready(paths)
    morning = datetime(2026, 9, 22, 10, 0, tzinfo=VN_TIMEZONE)
    r = service.check(paths, "AAA", now=morning, s2_override=[], record=False)
    assert r["data_as_of"] == "2026-09-21" and r["trade_session"] == "2026-09-22"
    assert "EOD data as of 2026-09-21" == r["eod_note"]


def test_behaviour_block_holds_until_next_session(tmp_path, monkeypatch):
    from stock_agent.swing_desk import service
    paths, _ = setup_desk(tmp_path, monkeypatch)
    ready(paths, "2026-09-01|NAV||1000000000||||||\n"
                 "2026-09-17|BUY|BBB||20000|1000|S1|PASS|calm||\n2026-09-21|SELL|BBB||19000|1000|stop||fear||\n")
    first = service.check(paths, "AAA", now=NOW_EOD, s2_override=[])
    assert first["verdict"] == "BLOCK" and "B3" in first["failed"] and first["behaviour_block"]
    assert first["recheck_from"].startswith("2026-09-23")
    again = service.check(paths, "AAA", now=NOW_EOD, s2_override=[])
    assert "COOL" in again["failed"]


def test_warn_requires_thesis_before_ack(tmp_path, monkeypatch):
    from stock_agent.swing_desk import service
    paths, _ = setup_desk(tmp_path, monkeypatch)
    ready(paths)
    r = service.check(paths, "BBB", now=NOW_EOD, s2_override=[], ack=True)     # no setup -> G11 WARN
    assert r["verdict"] == "WARN" and r["thesis_required"] and r["ack_refused"] and not r["acknowledged"]
    with paths.portfolio.open("a", encoding="utf-8") as f:
        f.write("2026-09-21|THESIS|BBB|||||||T1|trend; stop 3%; target high20; invalid if < SMA50\n")
    r = service.check(paths, "BBB", now=NOW_EOD, s2_override=[], ack=True)
    assert r["acknowledged"] and not r["ack_refused"]


def test_render_shows_both_vnd_at_risk_and_t2(tmp_path, monkeypatch):
    from stock_agent.swing_desk import render, service
    paths, _ = setup_desk(tmp_path, monkeypatch)
    ready(paths)
    r = service.check(paths, "AAA", now=NOW_EOD, s2_override=[], price=None, amount=300_000_000, record=False)
    text = render.render_check(r)
    assert "Bán được sớm nhất: chiều 2026-09-24 13:00" in text
    assert "ở stop 3%" in text and "hiệu dụng" in text and "Số tiền bạn nhập" in text
    assert "G9" in text and r["verdict"] == "BLOCK"


def test_s2_reuses_mr_scan(tmp_path, monkeypatch):
    from stock_agent.swing_desk import data as dl
    paths, manifest = setup_desk(tmp_path, monkeypatch)
    symbols, status = dl.s2_symbols(manifest, "2026-09-21")
    assert status == "OK" and isinstance(symbols, list)
