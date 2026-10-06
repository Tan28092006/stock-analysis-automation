"""`check --mode bottomfish` end to end on offline fixtures (no network)."""
import pytest

from tests.swing_desk_bf_support import PNJ_NOW, panic_desk, pnj_desk
from tests.swing_desk_support import NOW_EOD

PNJ_EXPECTED = {"BF-G1", "BF-G2", "BF-G3", "BF-G4", "BF-G5", "BF-G6", "BF-G8", "BF-G12", "BF-B1"}


def test_pnj_2026_10_06_illustration_blocks(tmp_path, monkeypatch):
    """ILLUSTRATION, not a tuning target: PNJ bars 25/09-06/10 from BF-doc §10 with the cited facts
    (two FACT_PRIMARY items, pending EGM dilution vote, no margin check). Thresholds were set from the
    literature before this fixture and were not fitted to it."""
    from stock_agent.swing_desk import render, service
    paths = pnj_desk(tmp_path, monkeypatch)
    r = service.check(paths, "PNJ", now=PNJ_NOW, mode="bottomfish", profile="P1", price=18750.0, amount=500_000,
                      record=False)
    assert r["verdict"] == "BLOCK"
    assert PNJ_EXPECTED <= set(r["failed"]), r["failed"]
    assert r["snapshot"]["source"] == "extra" and r["snapshot"]["ok"] and "BF-G0" not in r["failed"]
    assert r["fall"]["fall_type"] == "b"
    assert r["fields"]["floor_t"] and r["fields"]["floor_price_next"] == 18750.0
    g6 = next(x for x in r["rules"] if x["id"] == "BF-G6")
    assert "2026-09-24" in g6["reason"] and "2026-09-29" in g6["reason"]
    g8 = next(x for x in r["rules"] if x["id"] == "BF-G8")
    assert "fail-closed" in g8["reason"]
    assert r["earliest_sell"] == "2026-10-09 13:00" and r["held"]["qty"] == 5
    text = render.render_bf_check(r)
    assert "PNJ — BLOCK" in text and "Chỉ tư vấn; bắt đáy chưa có bằng chứng edge tiến cứu" in text
    assert "Kịch bản thêm 3 phiên sàn" in text and "P0 — không bắt đáy" in text


def test_pnj_with_margin_checked_passes_g8_and_still_blocks(tmp_path, monkeypatch):
    """ILLUSTRATION variant: margin status recorded (broker cut 2026-07-22, > 20 sessions earlier)."""
    from stock_agent.swing_desk import service
    paths = pnj_desk(tmp_path, monkeypatch, margin=True)
    r = service.check(paths, "PNJ", now=PNJ_NOW, mode="bottomfish", profile="P1", price=18750.0, record=False)
    assert "BF-G8" not in r["failed"] and r["verdict"] == "BLOCK"
    assert (PNJ_EXPECTED - {"BF-G8"}) <= set(r["failed"])


def test_pnj_p3_still_blocks_before_the_egm_and_says_no_edge(tmp_path, monkeypatch):
    from stock_agent.swing_desk import render, service
    paths = pnj_desk(tmp_path, monkeypatch, margin=True)
    r = service.check(paths, "PNJ", now=PNJ_NOW, mode="bottomfish", profile="P3", record=False)
    assert r["verdict"] == "BLOCK" and {"BF-G1", "BF-G5", "BF-G6"} <= set(r["failed"])
    assert "BF-G2" not in r["failed"]                      # type b is P3's type ...
    assert r["edge_note"] == "no evidence of edge; measurement only"
    assert "no evidence of edge; measurement only" in render.render_bf_check(r)


def test_held_outside_vn30_without_extra_snapshot_fails_closed(tmp_path, monkeypatch):
    from stock_agent.swing_desk import service
    paths = pnj_desk(tmp_path, monkeypatch)
    import shutil
    shutil.rmtree(paths.extra_snapshots)
    r = service.check(paths, "PNJ", now=PNJ_NOW, mode="bottomfish", profile="P1", record=False)
    assert r["verdict"] == "BLOCK" and {"BF-G0", "BF-G1", "BF-G2", "BF-G3", "BF-G4"} <= set(r["failed"])
    assert "refresh --held" in next(x for x in r["rules"] if x["id"] == "BF-G0")["reason"]


def test_vn30_panic_fixture_passes_and_logs(tmp_path, monkeypatch):
    from stock_agent.swing_desk import reclog, render, service
    paths = panic_desk(tmp_path, monkeypatch)
    r0 = service.check(paths, "AAA", now=NOW_EOD, mode="bottomfish", profile="P1", record=False)
    price = round(r0["fields"]["close"] * 1.005)
    r = service.check(paths, "AAA", now=NOW_EOD, mode="bottomfish", profile="P1", price=price, reason="bottomfish",
                      emotion="calm")
    assert r["verdict"] == "PASS", [(x["id"], x["reason"]) for x in r["rules"] if x["status"] not in ("PASS", "NA")]
    assert r["fall"]["fall_type"] == "a" and r["fields"]["stab_state"] == "STAB2"
    p = r["plan"]
    assert p["profile"] == "P1" and p["qty"] > 0 and p["risk_pct_nav"] == 1.0 and r["nav_source"] == "journal"
    assert p["vnd_at_risk_effective"] <= 112_000 + 1
    assert r["episode_id"].startswith("AAA-") and r["attempt_no"] == 1
    row = reclog.read_rows(paths.checks_log)[-1]
    assert row["mode"] == "bottomfish" and row["bf_type"] == "a" and row["profile"] == "P1"
    assert row["episode_id"] == r["episode_id"] and row["bf_news_sha"] and row["events_sha"]
    assert reclog.verify_chain(paths.checks_log)["ok"]
    assert "AAA — PASS" in render.render_bf_check(r)


def test_default_profile_is_p0_and_always_blocks(tmp_path, monkeypatch):
    from stock_agent.swing_desk import service
    paths = panic_desk(tmp_path, monkeypatch)
    r = service.check(paths, "AAA", now=NOW_EOD, mode="bottomfish", record=False)
    assert r["profile"] == "P0" and r["verdict"] == "BLOCK" and "BF-G2" in r["failed"] and r["plan"] is None


def test_ack_does_not_clear_g12(tmp_path, monkeypatch):
    from stock_agent.swing_desk import service
    paths = panic_desk(tmp_path, monkeypatch, thesis=False)
    r = service.check(paths, "AAA", now=NOW_EOD, mode="bottomfish", profile="P1", ack=True, record=False)
    assert r["verdict"] == "BLOCK" and "BF-G12" in r["failed"] and r["ack_refused"] and not r["acknowledged"]


def test_missing_news_and_calendar_block(tmp_path, monkeypatch):
    from stock_agent.swing_desk import service
    paths = panic_desk(tmp_path, monkeypatch, news=False)
    paths.events.unlink()
    r = service.check(paths, "AAA", now=NOW_EOD, mode="bottomfish", profile="P1", record=False)
    assert r["verdict"] == "BLOCK" and {"BF-G2", "BF-G5", "BF-G6", "BF-G8"} <= set(r["failed"])
    assert r["fall"]["fall_type"] == "UNCLASSIFIED"


def test_bottomfish_behaviour_block_sets_recheck(tmp_path, monkeypatch):
    from stock_agent.swing_desk import service
    paths = panic_desk(tmp_path, monkeypatch)
    r = service.check(paths, "AAA", now=NOW_EOD, mode="bottomfish", profile="P1", record=False,
                      price=None, why="cheap", reason="forum")
    assert "BF-B3" in r["failed"] and "BF-B8" in r["failed"] and r["verdict"] == "WARN"


def test_unknown_mode_rejected(tmp_path, monkeypatch):
    from stock_agent.swing_desk import service
    paths = panic_desk(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="mode"):
        service.check(paths, "AAA", now=NOW_EOD, mode="yolo", record=False)


def test_cli_bottomfish_check_scan_facts_and_events(tmp_path, monkeypatch, capsys):
    from stock_agent.swing_desk.__main__ import main
    paths = panic_desk(tmp_path, monkeypatch, news=False)
    base = ["--repo", str(paths.repo), "--now", NOW_EOD.isoformat()]
    assert main(base + ["bf-news", "AAA", "--session", "2026-09-22", "--class", "opinion_only",
                        "--margin", "hose_ineligible=no|broker_cut=none|checked=2026-09-21|https://www.hsx.vn/m"]) == 0
    assert main(base + ["bf-news", "AAA", "--session", "2026-09-22", "--class", "none_found",
                        "--searched", "https://cafef.vn/a"]) == 2
    capsys.readouterr()
    assert main(base + ["check", "AAA", "--mode", "bottomfish", "--profile", "P1", "--no-log"]) == 0
    assert "AAA — PASS" in capsys.readouterr().out
    assert main(base + ["check", "AAA", "--mode", "bottomfish", "--no-log"]) == 2      # default P0
    assert main(base + ["scan", "--mode", "bottomfish"]) == 0
    assert "Bảng bắt đáy" in capsys.readouterr().out
    assert main(base + ["events", "AAA", "--session", "2026-09-22", "--event", "2026-10-20:egm_capital",
                        "--pending-dilution", "--source", "https://ir.aaa.vn/egm"]) == 0
    assert main(base + ["check", "AAA", "--mode", "bottomfish", "--profile", "P1", "--no-log"]) == 2
    assert main(base + ["events", "AAA", "--session", "2026-10-21", "--resolved", "2026-10-20:egm_capital",
                        "--source", "https://ir.aaa.vn/egm-result"]) == 0
    from stock_agent.swing_desk.service import load_events
    data = load_events(paths)
    assert data["resolved"]["AAA"][0]["type"] == "egm_capital" and data["events"][0]["pending_dilution"]
