"""Crash/concurrency-safe writers keep their exact legacy byte format."""
from __future__ import annotations

from datetime import date
import json
import os
from pathlib import Path
import sys
import threading
import types

import pandas as pd
import pytest


def _fsync_spy(monkeypatch):
    calls = []
    real = os.fsync

    def spy(fd):
        calls.append(fd)
        return real(fd)

    monkeypatch.setattr(os, "fsync", spy)
    return calls


# ------------------------------------------------------------ snapshot manifest
def _snapshot(tmp_path, monkeypatch, out_name="snap"):
    from stock_agent.data import reconciliation as rec
    from tests.test_reconciliation import payload
    monkeypatch.setattr(rec, "completed_session_date", lambda: date(2026, 9, 21))

    def fetch(symbol, start, as_of):
        raw = json.dumps(payload(symbol)).encode()
        return rec.parse_vci_history(json.loads(raw), symbol, start, as_of), {
            "source": "VCI", "fetched_at": "2026-09-22T06:00:00+00:00", "raw_bytes": raw}

    out = tmp_path / out_name
    return rec, out, lambda: rec.build_market_snapshot(["FPT", "VNINDEX"], out, date(2026, 9, 18),
                                                       date(2026, 9, 21), min_rows=2, fetcher=fetch)


def test_snapshot_manifest_is_fsynced_and_byte_identical_to_legacy_format(tmp_path, monkeypatch):
    rec, out, build = _snapshot(tmp_path, monkeypatch)
    calls = _fsync_spy(monkeypatch)
    manifest = build()
    assert calls, "manifest must be fsynced before replace"
    legacy = tmp_path / "legacy.json"
    legacy.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    assert (out / "manifest.json").read_bytes() == legacy.read_bytes()
    assert not [p for p in out.iterdir() if p.name.endswith(".tmp")]
    assert rec.verify_snapshot(out / "manifest.json")["status"] == "verified"


def test_interrupted_manifest_write_leaves_no_partial_manifest(tmp_path, monkeypatch):
    rec, out, build = _snapshot(tmp_path, monkeypatch)
    real_replace = os.replace

    def crash(src, dst, *a, **kw):
        if Path(dst).name == "manifest.json":
            raise OSError("power loss")
        return real_replace(src, dst, *a, **kw)

    monkeypatch.setattr(os, "replace", crash)
    with pytest.raises(OSError, match="power loss"):
        build()
    assert not (out / "manifest.json").exists()
    assert not [p for p in out.iterdir() if p.name.endswith(".tmp")]


# ------------------------------------------------------------ repository.write_json
def test_write_json_uses_unique_temp_and_fsync_with_same_bytes(tmp_path, monkeypatch):
    from stock_agent.data.repository import write_json
    target = tmp_path / "state.json"
    decoy = tmp_path / "state.json.tmp"  # another writer's fixed temp name in flight
    decoy.mkdir()
    calls = _fsync_spy(monkeypatch)
    payload = {"a": 1.5, "b": float("nan"), "text": "Đã cập nhật"}
    write_json(target, payload)
    assert calls and decoy.is_dir()
    legacy = tmp_path / "legacy.json"
    with legacy.open("w", encoding="utf-8") as stream:
        json.dump({"a": 1.5, "b": None, "text": "Đã cập nhật"}, stream, ensure_ascii=False, indent=2)
    assert target.read_bytes() == legacy.read_bytes()
    assert sorted(p.name for p in tmp_path.iterdir()) == ["legacy.json", "state.json", "state.json.tmp"]


# ------------------------------------------------------------ eod_update prices_hist
def _fake_vnstock(monkeypatch, rows):
    class Quote:
        def __init__(self, symbol):
            self.symbol = symbol

        def history(self, start, end, interval):
            return pd.DataFrame(rows[self.symbol])

    class Vnstock:
        def stock(self, symbol, source):
            return types.SimpleNamespace(quote=Quote(symbol))

    module = types.ModuleType("vnstock")
    module.Vnstock = Vnstock
    monkeypatch.setitem(sys.modules, "vnstock", module)


def test_prices_hist_writer_avoids_shared_tmp_name_and_keeps_csv_format(tmp_path, monkeypatch):
    from stock_agent.pipeline import eod_update
    old = pd.DataFrame({"date": ["2026-09-17", "2026-09-18"], "open": [10.0, 11.0], "high": [12.0, 12.0],
                        "low": [9.0, 10.0], "close": [11.0, 11.5], "volume": [100, 200]})
    for sym in ("AAA", "VNINDEX"):
        old.to_csv(tmp_path / f"{sym}.csv", index=False)
    new = {"time": ["2026-09-21"], "open": [11.0], "high": [12.5], "low": [10.5], "close": [12.0], "volume": [300]}
    _fake_vnstock(monkeypatch, {"AAA": new, "VNINDEX": new})
    monkeypatch.setattr(eod_update, "PRICES_DIR", tmp_path)
    monkeypatch.setattr(eod_update, "_fetch_end_date", lambda: date(2026, 9, 21))
    monkeypatch.setattr(eod_update.time, "sleep", lambda s: None)
    decoy = tmp_path / "AAA.csv.tmp"  # the old shared temp name, held by a concurrent writer
    decoy.mkdir()
    result = eod_update.refresh_prices()
    assert result["updated"] == 2 and result["failed"] == 0
    assert decoy.is_dir() and not [p for p in tmp_path.iterdir() if p.name.startswith(".")]
    expected_new = pd.DataFrame(new).rename(columns={"time": "date"})
    expected_new[["open", "high", "low", "close"]] *= 1000.0
    merged = pd.concat([old, expected_new], ignore_index=True).drop_duplicates(
        subset=["date"], keep="last").sort_values("date")
    merged.to_csv(tmp_path / "expected.csv", index=False)
    assert (tmp_path / "AAA.csv").read_bytes() == (tmp_path / "expected.csv").read_bytes()


# ------------------------------------------------------------ dashboard data-update guard
def test_dashboard_data_update_start_is_atomic(monkeypatch):
    from stock_agent import app
    release, entered = threading.Event(), threading.Event()
    runs = []

    def fake_update():
        runs.append(1)
        entered.set()
        release.wait(5)
        app._finish_data_update()

    monkeypatch.setattr(app, "_do_data_update", fake_update)
    monkeypatch.setitem(app._DATA_UPDATE, "running", False)
    results = []
    barrier = threading.Barrier(8)

    def request():
        barrier.wait()
        results.append(app._start_data_update())

    threads = [threading.Thread(target=request) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(5)
    assert results.count(True) == 1 and results.count(False) == 7
    assert entered.wait(5) and app._DATA_UPDATE["running"] is True
    release.set()
    for _ in range(100):
        if not app._DATA_UPDATE["running"]:
            break
        threading.Event().wait(0.05)
    assert runs == [1] and app._DATA_UPDATE["running"] is False
    assert isinstance(app._DATA_UPDATE_LOCK, type(threading.Lock()))
