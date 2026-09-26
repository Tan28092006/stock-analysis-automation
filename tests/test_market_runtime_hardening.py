from datetime import date, datetime, timezone
import hashlib
import json

import pandas as pd
import pytest
import requests

from stock_agent.data import exchange_calendar as cal, reconciliation as rec
from stock_agent.pipeline import paper_runner as runner


def make_snapshot(root, symbols=("AAA",), end=date(2026, 9, 21)):
    start = date(2025, 2, 3)
    dates = cal.trading_days_between(start, end)
    def fetch(symbol, start, as_of):
        raw = json.dumps([{"symbol": symbol, "t": [int(pd.Timestamp(d, tz="UTC").timestamp()) for d in dates],
            "o": [10000 + i * 10 for i in range(len(dates))],
            "h": [10200 + i * 10 for i in range(len(dates))],
            "l": [9800 + i * 10 for i in range(len(dates))],
            "c": [10000 + i * 10 for i in range(len(dates))], "v": [1000000] * len(dates)}]).encode()
        return rec.parse_vci_history(json.loads(raw), symbol, start, as_of), {
            "raw_bytes": raw, "fetched_at": "2026-09-21T09:30:00+00:00"}
    rec.build_market_snapshot(list(symbols) + ["VNINDEX"], root, start, end, fetcher=fetch)
    path = root / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest.update(fetched_at="2026-09-21T09:30:00+00:00", completed_at="2026-09-21T09:31:00+00:00")
    path.write_text(json.dumps(manifest))
    return path


def test_missing_2025_swap_holiday():
    assert not cal.is_trading_day(date(2025, 5, 2))
    assert cal.next_trading_day(date(2025, 4, 29)) == date(2025, 5, 5)


def test_only_documented_symbol_transfer_gaps_are_exempt():
    from stock_agent.data.exchange_calendar import symbol_trading_days_between
    assert symbol_trading_days_between("BSR", date(2025, 1, 6), date(2025, 1, 17)) == [date(2025, 1, 6), date(2025, 1, 17)]
    assert symbol_trading_days_between("MCH", date(2025, 12, 17), date(2025, 12, 25)) == [date(2025, 12, 17), date(2025, 12, 25)]
    assert len(symbol_trading_days_between("FPT", date(2025, 12, 17), date(2025, 12, 25))) == 7


def test_transient_fetch_retries_bounded_without_retrying_bad_data(monkeypatch):
    monkeypatch.setattr(rec, "completed_session_date", lambda: date(2026, 9, 21))
    sleeps, calls = [], []
    monkeypatch.setattr(rec.time, "sleep", sleeps.append)
    class Response:
        content = json.dumps([{"symbol": "FPT", "t": [int(pd.Timestamp("2026-09-21", tz="UTC").timestamp())],
                             "o": [10], "h": [11], "l": [9], "c": [10], "v": [100]}]).encode()
        def raise_for_status(self): pass
    def post(*a, **kw):
        calls.append(1)
        if len(calls) < 3:
            raise requests.ReadTimeout("transient")
        return Response()
    monkeypatch.setattr(rec.requests, "post", post)
    frame, source = rec.fetch_vci_history("FPT", date(2026, 9, 18), date(2026, 9, 21))
    assert len(calls) == 3 and len(sleeps) == 2 and min(sleeps) >= 3.1
    assert source["attempts"] == 3 and len(frame) == 1
    calls.clear()
    monkeypatch.setattr(rec.requests, "post", lambda *a, **k: (_ for _ in ()).throw(requests.ReadTimeout("down")))
    with pytest.raises(requests.ReadTimeout):
        rec.fetch_vci_history("FPT", date(2026, 9, 18), date(2026, 9, 21))
    assert len(sleeps) == 4


def test_forged_verified_flag_cannot_create_paper_ledger(tmp_path):
    payload = {"session": "2026-09-21", "input_snapshot": "abc", "source_verified": True,
               "data_ready": True, "mode": "paper", "recommendations": [], "rules_hash": "r"}
    with pytest.raises(ValueError, match="manifest"):
        runner.commit_paper_run(payload, tmp_path, now=datetime(2026, 9, 21, 10, tzinfo=timezone.utc))
    assert not list(tmp_path.rglob("paper.json"))


def test_verified_end_to_end_record_and_retry(tmp_path):
    manifest = make_snapshot(tmp_path / "snapshot")
    now = datetime(2026, 9, 21, 10, tzinfo=timezone.utc)
    kwargs = dict(output_dir=tmp_path / "out", manifest_path=manifest, record=True, now=now)
    result = runner.run_paper(manifest.parent / "prices", ["AAA"], **kwargs)
    assert result["status"] == "recorded"
    before = (tmp_path / "out/runs/2026-09-21/paper.json").read_bytes()
    assert runner.run_paper(manifest.parent / "prices", ["AAA"], **kwargs)["status"] == "already_recorded"
    assert (tmp_path / "out/runs/2026-09-21/paper.json").read_bytes() == before
    assert result["snapshot_manifest_sha256"] == hashlib.sha256(manifest.read_bytes()).hexdigest()


def test_future_source_cannot_be_backdated(tmp_path):
    manifest = make_snapshot(tmp_path / "snapshot")
    data = json.loads(manifest.read_text())
    data["files"]["AAA"]["fetched_at"] = "2026-09-22T10:00:00+00:00"
    manifest.write_text(json.dumps(data))
    result = runner.run_paper(manifest.parent / "prices", ["AAA"], manifest_path=manifest,
                              output_dir=tmp_path / "out", record=True,
                              now=datetime(2026, 9, 21, 10, tzinfo=timezone.utc))
    assert not result["source_verified"] and result["status"] == "preview_only"
    assert not list((tmp_path / "out").rglob("paper.json"))
