"""Desk services read long windows from the history store and record which source each verdict used."""
from dataclasses import replace
import json

import pytest

from tests.swing_desk_history_support import history_desk, write_store
from tests.swing_desk_support import NOW_EOD, market_frames, write_snapshot

PROV = ("history_source", "history_store_id", "history_store_manifest_sha256", "history_first_date", "history_rows")


@pytest.fixture(autouse=True)
def _fresh_cache():
    from stock_agent.swing_desk import history
    history.clear_cache()


def _ready(paths):
    from stock_agent.swing_desk.service import mark_events
    paths.desk.mkdir(parents=True, exist_ok=True)
    paths.portfolio.write_text("2026-09-01|NAV||1000000000||||||\n", encoding="utf-8")
    mark_events(paths, "AAA", "2026-09-22", [], "test fixture")


def _strip(value):
    """Drop history provenance (the only thing --history may add when the series agree)."""
    if isinstance(value, dict):
        return {k: _strip(v) for k, v in value.items() if not k.startswith("history")}
    if isinstance(value, list):
        return [_strip(v) for v in value]
    return value


# ------------------------------------------------------------------ swing check / state

def test_check_uses_store_for_long_windows_and_logs_the_source(tmp_path, monkeypatch):
    from stock_agent.swing_desk import reclog, render, service
    paths, _, long, _, store = history_desk(tmp_path, monkeypatch)
    _ready(paths)
    r = service.check(paths, "BBB", now=NOW_EOD, s2_override=[])
    h = r["history"]
    assert h["history_source"] == "history_store" and h["history_store_id"] == store.name and h["history_rows"] == 600
    assert r["indicators"]["sma200"] == pytest.approx(round(long["BBB"]["close"], 2).tail(200).mean())
    assert r["indicators"]["n_locked"] == 250 and r["indicators"]["rows"] == 150
    row = reclog.read_rows(paths.checks_log)[-1]
    assert {k: row[k] for k in PROV} == {k: h[k] for k in PROV}
    assert store.name in render.render_check(r)


def test_check_history_off_keeps_snapshot_windows(tmp_path, monkeypatch):
    from stock_agent.swing_desk import service
    paths, _, _, _, _ = history_desk(tmp_path, monkeypatch, history="off")
    _ready(paths)
    r = service.check(paths, "BBB", now=NOW_EOD, s2_override=[], record=False)
    assert r["history"]["history_mode"] == "off" and r["history"]["history_source"] == "snapshot_650d"
    assert r["indicators"]["sma200"] is None and r["indicators"]["n_locked"] == 149


def test_on_and_off_are_identical_when_the_snapshot_covers_every_window(tmp_path, monkeypatch):
    """300-session snapshot inside a 600-session store: every window fits, so only provenance differs."""
    from stock_agent.swing_desk import render, service
    paths, _, _, _, _ = history_desk(tmp_path, monkeypatch, snap_n=300)
    _ready(paths)
    off = replace(paths, history="off")
    for t in ("AAA", "BBB", "VCB"):
        a = service.check(paths, t, now=NOW_EOD, s2_override=[], record=False)
        b = service.check(off, t, now=NOW_EOD, s2_override=[], record=False)
        assert a["history"]["history_source"] == "history_store"
        assert _strip(a) == _strip(b), t
    sa = service.compute_state(paths, NOW_EOD, s2_override=[], with_flows_data=False)
    sb = service.compute_state(off, NOW_EOD, s2_override=[], with_flows_data=False)
    assert _strip(sa["tickers"]) == _strip(sb["tickers"]) and sa["market"] == sb["market"]
    assert "history_store" in service.gate_table(sa) and "history" not in service.gate_table(sb)


def test_state_ticker_json_and_manifest_record_history(tmp_path, monkeypatch):
    from stock_agent.swing_desk import service
    paths, _, _, _, store = history_desk(tmp_path, monkeypatch, fail=("VCB",))
    state = service.compute_state(paths, NOW_EOD, s2_override=[], with_flows_data=False)
    folder = service.write_state(paths, state)
    aaa = json.loads((folder / "tickers" / "AAA.json").read_text(encoding="utf-8"))
    vcb = json.loads((folder / "tickers" / "VCB.json").read_text(encoding="utf-8"))
    assert aaa["history_source"] == "history_store" and aaa["history_store_id"] == store.name
    assert all(aaa[k] is not None for k in PROV) and aaa["history_data_issues"] == []
    assert vcb["history_source"] == "snapshot_650d" and vcb["history_rows"] == 150
    assert any(n["level"] == "WARN" for n in vcb["history_notes"])
    m = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))["history"]
    assert m["history_mode"] == "on" and m["history_source"] == "per_ticker" and m["fallbacks"] == ["VCB"]
    assert m["stores"] == {store.name: aaa["history_store_manifest_sha256"]}
    assert set(m["per_ticker"]["AAA"]) >= set(PROV)


def test_mismatch_is_flagged_in_output_and_state(tmp_path, monkeypatch):
    from stock_agent.swing_desk import render, service

    def edit(sym, body):
        if sym == "AAA":
            body[0]["c"] = [round(v * 0.98, 2) for v in body[0]["c"]]
            body[0]["o"] = [round(v * 0.98, 2) for v in body[0]["o"]]
            body[0]["l"] = [round(v * 0.98, 2) for v in body[0]["l"]]
            body[0]["h"] = [round(v * 0.98, 2) for v in body[0]["h"]]
        return body
    paths, _, _, _, _ = history_desk(tmp_path, monkeypatch, edit=edit)
    _ready(paths)
    r = service.check(paths, "AAA", now=NOW_EOD, s2_override=[], record=False)
    assert r["history"]["history_mismatch"] is True and r["history"]["history_source"] == "snapshot_650d"
    assert r["indicators"]["n_locked"] == 149
    assert "history_mismatch" in render.render_check(r)
    state = service.compute_state(paths, NOW_EOD, s2_override=[], with_flows_data=False)
    assert state["manifest"]["history"]["mismatches"] == ["AAA"]
    assert "history_mismatch" in service.gate_table(state)


def test_baseline_rows_carry_history_source(tmp_path, monkeypatch):
    from stock_agent.swing_desk import baseline, service
    paths, _, _, _, store = history_desk(tmp_path, monkeypatch, snap_n=300)
    state = service.compute_state(paths, NOW_EOD, s2_override=[], with_flows_data=False)
    for t in state["tickers"].values():          # fixture market is GRIND (G2): force the control pool
        t["structural_gate_ex_setup"] = "PASS"
    rows = baseline.candidate_rows(state, "control")
    assert rows and all(r["history_source"] == "history_store" and r["history_store_id"] == store.name for r in rows)


# ------------------------------------------------------------------ bottom-fish

def _bf_paths(tmp_path, monkeypatch, **kw):
    from tests.swing_desk_bf_support import ROOT
    paths, manifest, long, snap, store = history_desk(tmp_path, monkeypatch, **kw)
    for name in ("bottomfish_v1.json", "bottomfish_prospective_v1.json"):
        (paths.repo / "configs/swing_desk" / name).write_bytes((ROOT / "configs/swing_desk" / name).read_bytes())
    return paths, long, store


def test_bottomfish_check_and_scan_use_the_store(tmp_path, monkeypatch):
    from stock_agent.swing_desk import render, service
    from stock_agent.swing_desk.bf_service import scan_rows, scan_table
    paths, long, store = _bf_paths(tmp_path, monkeypatch)
    _ready(paths)
    r = service.check(paths, "BBB", now=NOW_EOD, mode="bottomfish", record=True)
    assert r["history"]["history_source"] == "history_store" and r["fields"]["high_250"] is not None
    assert store.name in render.render_bf_check(r)
    from stock_agent.swing_desk import reclog
    assert reclog.read_rows(paths.checks_log)[-1]["history_source"] == "history_store"
    scan = scan_rows(paths, NOW_EOD)
    assert all(row["history"]["history_source"] == "history_store" for row in scan["rows"])
    assert scan["history"]["history_source"] == "history_store"
    assert "history_store" in scan_table(scan)


def test_bottomfish_history_off_matches_snapshot_only(tmp_path, monkeypatch):
    from stock_agent.swing_desk import service
    paths, _, _ = _bf_paths(tmp_path, monkeypatch, history="off")
    _ready(paths)
    r = service.check(paths, "BBB", now=NOW_EOD, mode="bottomfish", record=False)
    assert r["history"]["history_mode"] == "off" and r["fields"]["high_250"] is None
