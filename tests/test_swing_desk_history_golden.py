"""History-on golden (T16): swing + bottom-fish outputs computed with a small fixture history store.

Fixture: 150-session verified snapshot inside a 600-session store (same provider bytes), so
SMA200, high_250, MAE/MFE (250 entries) and the bottom-fish episode come from the store; plus
one ticker (VCB) whose store series was re-adjusted (history_mismatch -> snapshot, not stitched).
Recorded with SWING_DESK_WRITE_GOLDEN=1. Ignored: code hash, snapshot/store manifest sha256
(manifests embed temporary paths / wall-clock completion time) and the history root path.
"""
import json
import os
from pathlib import Path
import re

import pytest

from tests.swing_desk_history_support import history_desk
from tests.swing_desk_support import NOW_EOD

GOLDEN = Path(__file__).resolve().parent / "fixtures" / "swing_desk_golden_history_v1.json"
VOLATILE = {"desk_code_hash", "snapshot_manifest_sha256", "history_store_manifest_sha256", "history_root",
            "stores", "frames", "manifest"}


def _clean(value):
    if isinstance(value, dict):
        return {k: _clean(v) for k, v in value.items() if k not in VOLATILE}
    if isinstance(value, list):
        return [_clean(v) for v in value]
    return value


def _text(text: str) -> str:
    return re.sub(r"sha [0-9a-f]{12}", "sha <sha>", text)


def _readjust(sym, body):
    if sym == "VCB":
        for k in ("o", "h", "l", "c"):
            body[0][k] = [round(v * 0.98, 2) for v in body[0][k]]
    return body


def collect(tmp_path, monkeypatch) -> dict:
    from tests.swing_desk_bf_support import ROOT
    from stock_agent.swing_desk import history, render, service
    from stock_agent.swing_desk.bf_service import scan_rows, scan_table
    history.clear_cache()
    paths, _, _, _, _ = history_desk(tmp_path, monkeypatch, edit=_readjust)
    for name in ("bottomfish_v1.json", "bottomfish_prospective_v1.json"):
        (paths.repo / "configs/swing_desk" / name).write_bytes((ROOT / "configs/swing_desk" / name).read_bytes())
    paths.desk.mkdir(parents=True, exist_ok=True)
    paths.portfolio.write_text("2026-09-01|NAV||1000000000||||||\n", encoding="utf-8")
    service.mark_events(paths, "AAA", "2026-09-22", [], "test fixture")
    out = {}
    kw = dict(now=NOW_EOD, record=False)
    for t in ("AAA", "BBB", "VCB"):
        r = service.check(paths, t, s2_override=[], **kw)
        out[f"swing_{t}"] = {"json": _clean(r), "text": _text(render.render_check(r))}
        r = service.check(paths, t, mode="bottomfish", profile="P1", **kw)
        out[f"bf_{t}"] = {"json": _clean(r), "text": _text(render.render_bf_check(r))}
    state = service.compute_state(paths, NOW_EOD, s2_override=[], with_flows_data=False)
    out["state_tickers"] = _clean(state["tickers"])
    out["state_history"] = _clean(state["manifest"]["history"])
    out["gate_table"] = _text(service.gate_table(state))
    scan = scan_rows(paths, NOW_EOD)
    out["bf_scan_rows"] = _clean(scan["rows"])
    out["bf_scan_table"] = _text(scan_table(scan))
    return json.loads(json.dumps(out, ensure_ascii=False, sort_keys=True, default=str))


def test_history_on_outputs_match_golden(tmp_path, monkeypatch):
    got = collect(tmp_path, monkeypatch)
    if os.environ.get("SWING_DESK_WRITE_GOLDEN") == "1":
        GOLDEN.parent.mkdir(parents=True, exist_ok=True)
        GOLDEN.write_text(json.dumps(got, ensure_ascii=False, indent=1, sort_keys=True) + "\n", encoding="utf-8")
        pytest.skip("golden written")
    want = json.loads(GOLDEN.read_text(encoding="utf-8"))
    assert set(got) == set(want)
    for key in sorted(want):
        assert got[key] == want[key], key


def test_golden_fixture_exercises_store_fallback_and_mismatch():
    want = json.loads(GOLDEN.read_text(encoding="utf-8"))
    sources = {t: want[f"swing_{t}"]["json"]["history"]["history_source"] for t in ("AAA", "BBB", "VCB")}
    assert sources == {"AAA": "history_store", "BBB": "history_store", "VCB": "snapshot_650d"}
    assert want["swing_VCB"]["json"]["history"]["history_mismatch"] is True
    assert want["swing_BBB"]["json"]["indicators"]["n_locked"] == 250
    assert want["swing_VCB"]["json"]["indicators"]["n_locked"] == 149
    assert want["state_history"]["mismatches"] == ["VCB"]
