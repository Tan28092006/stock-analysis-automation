from datetime import date
import hashlib
import json

import pandas as pd
import pytest

from stock_agent.data import reconciliation as rec


def payload(symbol="FPT", dates=("2026-09-18", "2026-09-21", "2026-09-22")):
    return [{"symbol": symbol, "t": [str(int(pd.Timestamp(d, tz="UTC").timestamp())) for d in dates],
             "o": [100000] * len(dates), "h": [102000] * len(dates),
             "l": [99000] * len(dates), "c": [101000] * len(dates), "v": [1000] * len(dates)}]


def test_parse_keeps_explicit_vnd_and_excludes_intraday():
    frame = rec.parse_vci_history(payload(), "FPT", date(2026, 9, 18), date(2026, 9, 21))
    assert list(frame.date) == [date(2026, 9, 18), date(2026, 9, 21)]
    assert frame.close.tolist() == [101000, 101000]


@pytest.mark.parametrize("kind", ["nonpositive", "inconsistent", "duplicate", "nan", "wrong_symbol", "empty"])
def test_parse_rejects_invalid_source_rows_without_repair(kind):
    data = payload()
    if kind == "nonpositive": data[0]["c"][0] = 0
    if kind == "inconsistent": data[0]["o"][0] = 110000
    if kind == "duplicate": data[0]["t"][1] = data[0]["t"][0]
    if kind == "nan": data[0]["v"][0] = None
    if kind == "wrong_symbol": data[0]["symbol"] = "OTHER"
    if kind == "empty": data = []
    with pytest.raises(ValueError):
        rec.parse_vci_history(data, "FPT", date(2026, 9, 18), date(2026, 9, 21))


def test_snapshot_is_isolated_hashed_complete_and_immutable(tmp_path, monkeypatch):
    monkeypatch.setattr(rec, "completed_session_date", lambda: date(2026, 9, 21))
    def fetch(symbol, start, as_of):
        raw = json.dumps(payload(symbol)).encode()
        return rec.parse_vci_history(json.loads(raw), symbol, start, as_of), {
            "source": "VCI", "fetched_at": "2026-09-22T06:00:00+00:00", "raw_bytes": raw}
    out = tmp_path / "isolated"
    manifest = rec.build_market_snapshot(["FPT", "VNINDEX"], out, date(2026, 9, 18), date(2026, 9, 21), min_rows=2, fetcher=fetch)
    assert manifest["status"] == "verified"
    assert manifest["files"]["VNINDEX"]["unit"] == "index_points"
    for entry in manifest["files"].values():
        assert hashlib.sha256((out / entry["path"]).read_bytes()).hexdigest() == entry["sha256"]
        assert hashlib.sha256((out / entry["raw_path"]).read_bytes()).hexdigest() == entry["raw_sha256"]
    assert "unverified" in manifest["adjustment_basis"]
    assert rec.verify_snapshot(out / "manifest.json")["status"] == "verified"
    with pytest.raises(FileExistsError):
        rec.build_market_snapshot(["FPT"], out, date(2026, 9, 18), date(2026, 9, 21), fetcher=fetch)
    (out / "prices/FPT.csv").write_text("changed")
    with pytest.raises(ValueError, match="hash"):
        rec.verify_snapshot(out / "manifest.json")


def test_missing_or_stale_symbol_blocks_entire_snapshot(tmp_path, monkeypatch):
    monkeypatch.setattr(rec, "completed_session_date", lambda: date(2026, 9, 21))
    def fetch(symbol, start, as_of):
        if symbol == "TCX": raise ValueError("no data")
        raw = json.dumps(payload(symbol, ("2026-09-18",))).encode()
        return rec.parse_vci_history(json.loads(raw), symbol, start, as_of), {
            "source": "VCI", "fetched_at": "2026-09-22T06:00:00+00:00", "raw_bytes": raw}
    manifest = rec.build_market_snapshot(["FPT", "TCX"], tmp_path / "failed", date(2026, 9, 18), date(2026, 9, 21), min_rows=1, fetcher=fetch)
    assert manifest["status"] == "blocked"
    assert len(manifest["failures"]) == 2
    with pytest.raises(ValueError, match="blocked"):
        rec.verify_snapshot(tmp_path / "failed/manifest.json")


def test_future_cutoff_rejected_before_creating_output(tmp_path, monkeypatch):
    monkeypatch.setattr(rec, "completed_session_date", lambda: date(2026, 9, 21))
    with pytest.raises(ValueError, match="completed"):
        rec.build_market_snapshot(["FPT"], tmp_path / "future", date(2026, 9, 18), date(2026, 9, 22))
    assert not (tmp_path / "future").exists()


def test_repair_only_invalid_row_when_neighbors_agree():
    valid = rec.parse_vci_history(payload(dates=("2026-09-17", "2026-09-18", "2026-09-21")), "FPT", date(2026, 9, 17), date(2026, 9, 21))
    old = valid.copy(); old.loc[1, "close"] = 0
    candidate, decisions = rec.reconcile_invalid_rows(old, valid)
    assert candidate.close.tolist() == valid.close.tolist()
    assert old.loc[1, "close"] == 0
    assert decisions[0]["status"] == "staged_verified_neighbors"
    disagree = valid.copy(); disagree.loc[0, "close"] = 100000
    unchanged, decisions = rec.reconcile_invalid_rows(old, disagree)
    assert unchanged.loc[1, "close"] == 0
    assert decisions[0]["status"] == "quarantine"


def test_fetch_is_read_only_bounded_and_uses_explicit_cutoff(monkeypatch):
    monkeypatch.setattr(rec, "completed_session_date", lambda: date(2026, 9, 21))
    calls = []
    class Response:
        content = json.dumps(payload()).encode()
        def raise_for_status(self): pass
    def post(url, **kwargs):
        calls.append((url, kwargs))
        return Response()
    monkeypatch.setattr(rec.requests, "post", post)
    frame, source = rec.fetch_vci_history("fpt", date(2026, 9, 18), date(2026, 9, 22))
    assert frame.date.iloc[-1] == date(2026, 9, 21)
    assert source["raw_sha256"] == hashlib.sha256(source["raw_bytes"]).hexdigest()
    assert calls[0][1]["json"]["symbols"] == ["FPT"]
    assert calls[0][1]["timeout"] == (10, 30)
    assert pd.Timestamp(calls[0][1]["json"]["to"], unit="s", tz="UTC") == pd.Timestamp("2026-09-21T17:00:00Z")
    with pytest.raises(ValueError, match="symbol"):
        rec.fetch_vci_history("../FPT", date(2026, 9, 18), date(2026, 9, 21))
    with pytest.raises(ValueError, match="start"):
        rec.fetch_vci_history("FPT", date(2026, 9, 23), date(2026, 9, 23))


@pytest.mark.parametrize("symbols,minimum", [([], 1), (["FPT", "FPT"], 1), (["../FPT"], 1), (["FPT"], 0)])
def test_snapshot_rejects_bad_contract_without_creating_output(tmp_path, monkeypatch, symbols, minimum):
    monkeypatch.setattr(rec, "completed_session_date", lambda: date(2026, 9, 21))
    with pytest.raises(ValueError):
        rec.build_market_snapshot(symbols, tmp_path / "bad", date(2026, 9, 18), date(2026, 9, 21), min_rows=minimum)
    assert not (tmp_path / "bad").exists()


def test_verify_rejects_path_escape(tmp_path):
    manifest = {"kind": "market_snapshot", "status": "verified", "symbols": ["FPT"],
                "files": {"FPT": {"path": "../outside.csv", "sha256": "anything"}}}
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="escaped"):
        rec.verify_snapshot(path)


def test_repair_quarantines_edges_and_still_invalid_upstream():
    valid = rec.parse_vci_history(payload(dates=("2026-09-17", "2026-09-18", "2026-09-21")), "FPT", date(2026, 9, 17), date(2026, 9, 21))
    old = valid.copy(); old.loc[0, "close"] = 0
    unchanged, decisions = rec.reconcile_invalid_rows(old, valid)
    assert unchanged.loc[0, "close"] == 0
    assert decisions[0]["status"] == "quarantine"
    old = valid.copy(); old.loc[1, "close"] = 0
    unchanged, decisions = rec.reconcile_invalid_rows(old, old)
    assert unchanged.loc[1, "close"] == 0
