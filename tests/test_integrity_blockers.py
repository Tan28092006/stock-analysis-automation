"""Regression gates for the 2026-09-22 audit; no network or production writes."""
from datetime import date, datetime, timezone
import json

import pandas as pd
import pytest

from stock_agent.config import load_rules, load_universe
from stock_agent.data import exchange_calendar as calendar
from stock_agent.features import mr_scan as mr
from stock_agent.pipeline import forward_test as ft, label_resolver as lr


def bars(n=100, end="2026-09-22"):
    return pd.DataFrame({"date": pd.bdate_range(end=end, periods=n).strftime("%Y-%m-%d"),
                         "open": 100., "high": 101., "low": 99., "close": 100., "volume": 100000.})


def test_current_vn30_membership():
    u = load_universe()
    assert len(set(u["symbols"])) == 30
    assert {"MCH", "TCX"} <= set(u["symbols"])
    assert not {"PLX", "TPB"} & set(u["symbols"])
    assert u["effective_from"] == "2026-08-03"


@pytest.mark.parametrize("stamp,expected", [
    ("2026-09-22T08:59:59+00:00", "2026-09-21"),
    ("2026-09-22T09:00:00+00:00", "2026-09-22"),
    ("2026-09-27T10:00:00+00:00", "2026-09-25"),
    ("2026-09-02T10:00:00+00:00", "2026-09-01"),
    ("2026-09-21T01:00:00-07:00", "2026-09-18"),
])
def test_completed_session_uses_vietnam_time(stamp, expected):
    assert str(calendar.completed_session_date(datetime.fromisoformat(stamp))) == expected


def test_reject_ambiguous_naive_clock():
    with pytest.raises(ValueError):
        calendar.completed_session_date(datetime(2026, 9, 22, 16))


def test_dashboard_loader_excludes_intraday_and_stale(tmp_path, monkeypatch):
    monkeypatch.setattr(calendar, "completed_session_date", lambda now=None: date(2026, 9, 21), raising=False)
    bars().to_csv(tmp_path / "AAA.csv", index=False)
    bars().to_csv(tmp_path / "VNINDEX.csv", index=False)
    bars(end="2026-09-18").to_csv(tmp_path / "STALE.csv", index=False)
    f = mr._load_frames(tmp_path)
    assert f["AAA"].date.iloc[-1] == "2026-09-21"
    assert "STALE" not in f
    assert mr._market_state(tmp_path, f)["date"] == "2026-09-21"


def test_unfinished_fixed_horizon_is_pending():
    assert ft._fwd_return(bars(6), 0, 21) == ("pending", None)


def test_legacy_without_provenance_is_not_resolved(tmp_path, monkeypatch):
    p = tmp_path / "pending.jsonl"
    rec = {"symbol": "AAA", "signal_date": "2026-07-01", "entry_reference": 12., "status": "pending"}
    original = json.dumps(rec) + "\n"
    p.write_text(original, encoding="utf-8")
    bars(80).to_csv(tmp_path / "AAA.csv", index=False)
    resolved = tmp_path / "resolved.jsonl"
    monkeypatch.setattr(lr, "PENDING_PREDICTIONS_PATH", p)
    monkeypatch.setattr(lr, "RESOLVED_LABELS_PATH", resolved)
    result = lr.resolve_pending_labels(load_rules(), price_dir=tmp_path, today=date(2026, 9, 22))
    assert result["resolved"] == 0
    assert result["quarantined"] == 1
    assert p.read_text(encoding="utf-8") == original
    assert not resolved.exists()


def test_forward_score_excludes_unproven_records(tmp_path, monkeypatch):
    p = tmp_path / "ft.jsonl"
    rec = {"symbol": "AAA", "signal_date": "2026-07-01", "engine": "mr", "kind": "buy",
           "entry_reference": 100., "stop_loss": 95., "take_profit": 110., "max_hold_days": 15}
    p.write_text(json.dumps(rec) + "\n", encoding="utf-8")
    bars(80).to_csv(tmp_path / "AAA.csv", index=False)
    monkeypatch.setattr(ft, "LEDGER_PATH", p)
    monkeypatch.setattr(ft, "PRICES_DIR", tmp_path)
    out = ft.score()
    assert out["mr_trades"]["n"] == 0
    assert out["quarantined"] == 1


def test_future_logger_records_actual_time_and_model(tmp_path, monkeypatch):
    p = tmp_path / "ft.jsonl"
    monkeypatch.setattr(ft, "LEDGER_PATH", p)
    mr_payload = {"data_date": "2026-09-21", "rules_hash": "rules", "input_snapshot": "snapshot",
                  "model": {"model_version": "modelsha", "trained_at": "2026-07-03T06:25:00+00:00"},
                  "buys": [{"symbol": "AAA", "date": "2026-09-21", "close": 100.,
                            "entry_reference": 100., "stop_loss": 95., "take_profit": 110.,
                            "max_hold_days": 15, "win_prob": .6}]}
    before = datetime.now(timezone.utc)
    assert ft.log_recommendations(mr_payload, None)["appended"] == 1
    rec = json.loads(p.read_text(encoding="utf-8"))
    assert datetime.fromisoformat(rec["recorded_at"]) >= before
    assert rec["model_version"] == "modelsha"
    assert rec["input_snapshot"] == "snapshot"
    assert rec["mode"] == "paper"
