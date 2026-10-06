"""Legacy dashboard cache read/write failures are logged, and still fail safe."""
from __future__ import annotations

import json
import logging

import pytest

from stock_agent.features import mr_scan, momentum_scan, scan_guard, swing_scan


@pytest.fixture
def isolated(tmp_path, monkeypatch):
    monkeypatch.setattr(scan_guard, "readiness", lambda prices_dir: {"data_ready": True})
    for module in (mr_scan, momentum_scan, swing_scan):
        monkeypatch.setattr(module, "PRICES_DIR", tmp_path / "prices")
        monkeypatch.setattr(module, "scan_input_snapshot", lambda prices_dir: "snap")
    monkeypatch.setattr(mr_scan, "_compute", lambda recent_days, min_win_prob: {
        "buys": [], "data_date": None, "min_win_prob": min_win_prob,
        "rules_hash": mr_scan.compute_rules_hash(mr_scan._load_rules())})
    monkeypatch.setattr(momentum_scan, "_compute", lambda *a, **kw: {"picks": [], "data_date": None})
    monkeypatch.setattr(swing_scan, "_compute", lambda *a, **kw: {"picks": [], "data_date": None})
    return tmp_path


CASES = [(mr_scan, lambda: mr_scan.mr_scan()), (momentum_scan, lambda: momentum_scan.momentum_scan()),
         (swing_scan, lambda: swing_scan.swing_scan())]


@pytest.mark.parametrize("module,scan", CASES, ids=["mr", "momentum", "swing"])
def test_unreadable_and_unwritable_cache_is_logged_and_recomputed(isolated, monkeypatch, caplog, module, scan):
    cache = isolated / "cache_is_a_directory"
    cache.mkdir()
    monkeypatch.setattr(module, "CACHE_PATH", cache)
    with caplog.at_level(logging.WARNING, logger=module.__name__):
        payload = scan()
    assert payload["input_snapshot"] == "snap"  # still computed and returned
    messages = [r.getMessage() for r in caplog.records if r.name == module.__name__]
    assert any("cache read failed" in m for m in messages)
    assert any("cache write failed" in m for m in messages)


@pytest.mark.parametrize("module,scan", CASES, ids=["mr", "momentum", "swing"])
def test_corrupt_cache_json_is_logged_then_replaced(isolated, monkeypatch, caplog, module, scan):
    cache = isolated / "cache.json"
    cache.write_text("{not json", encoding="utf-8")
    monkeypatch.setattr(module, "CACHE_PATH", cache)
    with caplog.at_level(logging.WARNING, logger=module.__name__):
        scan()
    assert any("cache read failed" in r.getMessage() for r in caplog.records if r.name == module.__name__)
    assert json.loads(cache.read_text(encoding="utf-8"))["input_snapshot"] == "snap"


def test_position_refresh_failure_on_cache_hit_is_logged(isolated, monkeypatch, caplog):
    from stock_agent.features import position_manager
    cache = isolated / "mr.json"
    monkeypatch.setattr(mr_scan, "CACHE_PATH", cache)
    first = mr_scan.mr_scan()
    assert cache.exists() and first["input_snapshot"] == "snap"

    def broken(*a, **kw):
        raise RuntimeError("positions store unreadable")

    monkeypatch.setattr(position_manager, "check_positions", broken)
    monkeypatch.setattr(mr_scan, "_compute", lambda *a, **kw: pytest.fail("cache hit expected"))
    with caplog.at_level(logging.WARNING, logger=mr_scan.__name__):
        cached = mr_scan.mr_scan()
    assert cached["input_snapshot"] == "snap"
    assert any("position refresh failed" in r.getMessage() for r in caplog.records)


def test_eod_update_swing_warmup_failure_is_logged_not_swallowed(monkeypatch, caplog):
    from stock_agent.pipeline import eod_update

    def broken(force=False):
        raise RuntimeError("swing boom")

    monkeypatch.setattr(swing_scan, "swing_scan", broken)
    monkeypatch.setattr(momentum_scan, "momentum_scan", lambda force=False: {"active": True, "picks": []})
    with caplog.at_level(logging.WARNING, logger=eod_update.__name__):
        result = eod_update.rebuild_momentum_cache()
    assert result["active"] is True and "status" not in result  # momentum still rebuilt
    assert any("swing" in r.getMessage() and "swing boom" in r.getMessage() for r in caplog.records)
