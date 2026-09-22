"""Regression gates for the 2026-09-22 audit; no network or production writes."""
from datetime import date, datetime, timezone
import json

import pandas as pd
import pytest

from stock_agent.config import load_rules, load_universe
from stock_agent.data import exchange_calendar as calendar
from stock_agent.features import mr_scan as mr
from stock_agent.pipeline import forward_test as ft, label_resolver as lr
from stock_agent.pipeline.ledger_integrity import audit_rows, provenance_reasons, price_reasons


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
    ("2026-09-02T10:00:00+00:00", "2026-08-28"),
    ("2026-01-02T10:00:00+00:00", "2025-12-31"),
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


def valid_record(**extra):
    return {"schema_version": 2, "symbol": "AAA", "signal_date": "2026-07-01",
            "logged_date": "2026-07-01", "engine": "mr", "kind": "buy",
            "recorded_at": "2026-07-01T10:00:00+00:00", "mode": "paper",
            "data_source": "prices_hist", "input_snapshot": "snapshot", "rules_hash": "r",
            "model_version": "m", "model_trained_at": "2026-06-01T00:00:00+00:00",
            "entry_reference": 100., "stop_loss": 95., "take_profit": 110.,
            "max_hold_days": 15, "win_prob": .6, **extra}


@pytest.mark.parametrize("extra,reason", [
    ({"mode": "demo"}, "non_paper_record"),
    ({"data_source": "prices"}, "unverified_price_source"),
    ({"recorded_at": "2026-07-03T10:00:00+00:00"}, "not_recorded_in_ex_ante_window"),
    ({"recorded_at": "2026-07-01T08:59:59+00:00"}, "not_recorded_in_ex_ante_window"),
    ({"recorded_at": "2026-07-01T10:00:00"}, "unverifiable_timing"),
    ({"model_trained_at": "2026-07-02T10:00:00+00:00"}, "model_not_available_at_prediction"),
    ({"logged_date": "2026-07-03"}, "stale_signal"),
    ({"signal_date": "2026-07-04"}, "non_trading_signal_date"),
])
def test_provenance_failure_modes(extra, reason):
    assert reason in provenance_reasons(valid_record(**extra))


def test_duplicate_and_conflict_accounting():
    f = bars(80)
    r = valid_record()
    good = audit_rows([r, r], lambda sym: f)
    assert good["eligible"] == 1 and good["reason_counts"]["duplicate"] == 1
    conflict = audit_rows([r, {**r, "entry_reference": 999}], lambda sym: f)
    assert conflict["eligible"] == 0
    assert conflict["reason_counts"]["conflicting_duplicate"] == 2
    assert conflict["reason_counts"]["reference_price_mismatch"] == 1


def test_bad_and_missing_reference_prices():
    r = valid_record()
    assert price_reasons(r, None) == ["missing_prices"]
    assert price_reasons(r, bars(3)) == ["missing_or_duplicate_signal_bar"]
    assert price_reasons({**r, "entry_reference": -2}, bars(80)) == ["invalid_reference"]
    assert price_reasons({**r, "entry_reference": "bad"}, bars(80)) == ["invalid_reference"]


def test_mr_artifact_version_tracks_loaded_bytes(tmp_path, monkeypatch):
    import hashlib
    import pickle
    from stock_agent.features import win_probability as wp
    monkeypatch.setattr(wp.WinProbModel, "_cache", None)
    a, b = tmp_path / "a.pkl", tmp_path / "b.pkl"
    a.write_bytes(pickle.dumps({"model": None, "iso": None, "features": [], "trained_at": "a"}))
    b.write_bytes(pickle.dumps({"model": None, "iso": None, "features": [], "trained_at": "b"}))
    loaded = wp.WinProbModel.load(a)
    assert loaded.meta["model_version"] == hashlib.sha256(a.read_bytes()).hexdigest()
    assert wp.WinProbModel.load(b).meta["trained_at"] == "b"
    a.write_bytes(b.read_bytes())
    assert wp.WinProbModel.load(a).meta["trained_at"] == "b"
    assert wp.WinProbModel.load(tmp_path / "absent.pkl") is None


def test_bad_ohlc_is_quarantined_before_replay(tmp_path, monkeypatch):
    p = tmp_path / "ft.jsonl"
    p.write_text(json.dumps(valid_record()) + "\n", encoding="utf-8")
    f = bars(80)
    i = f.index[f.date == "2026-07-02"][0]
    f.loc[i, "low"] = -1
    f.to_csv(tmp_path / "AAA.csv", index=False)
    monkeypatch.setattr(ft, "LEDGER_PATH", p)
    monkeypatch.setattr(ft, "PRICES_DIR", tmp_path)
    out = ft.score()
    assert out["mr_trades"]["n"] == 0
    assert out["quarantined"] == 1


def test_supported_paper_replay_uses_complete_horizon(tmp_path, monkeypatch):
    p = tmp_path / "ft.jsonl"
    p.write_text(json.dumps(valid_record()) + "\n", encoding="utf-8")
    bars(80).to_csv(tmp_path / "AAA.csv", index=False)
    monkeypatch.setattr(ft, "LEDGER_PATH", p)
    monkeypatch.setattr(ft, "PRICES_DIR", tmp_path)
    out = ft.score()
    assert out["quarantined"] == 0
    assert out["mr_trades"]["n"] == 1
    assert out["mr_trades"]["avg"] == -.6


def test_cache_snapshot_changes_on_content_and_cutoff(tmp_path, monkeypatch):
    from stock_agent.data.eod import scan_input_snapshot, completed_bars
    monkeypatch.setattr(calendar, "completed_session_date", lambda now=None: date(2026, 9, 21))
    p = tmp_path / "AAA.csv"
    f = bars()
    f.to_csv(p, index=False)
    first = scan_input_snapshot(tmp_path)
    f.loc[0, "close"] = 101
    f.to_csv(p, index=False)
    assert scan_input_snapshot(tmp_path) != first
    assert completed_bars(f, date(2026, 9, 18)).date.max() == "2026-09-18"
    dates = pd.DataFrame({"date": ["bad", "2026-09-19", "2026-09-21", "2026-09-22"]})
    assert completed_bars(dates).date.tolist() == ["2026-09-21"]


@pytest.mark.parametrize("module_name", ["mr_scan", "momentum_scan", "swing_scan"])
def test_old_dashboard_cache_is_never_reused(tmp_path, monkeypatch, module_name):
    from importlib import import_module
    from stock_agent.config import compute_rules_hash, load_json
    module = import_module(f"stock_agent.features.{module_name}")
    monkeypatch.setattr(module, "PRICES_DIR", tmp_path)
    monkeypatch.setattr(module, "CACHE_PATH", tmp_path / "cache.json")
    bars().to_csv(tmp_path / "VNINDEX.csv", index=False)
    old = {"rules_hash": compute_rules_hash(load_json(module.MR_RULES_PATH)),
           "data_date": "2026-09-22", "top_n": 10, "min_win_prob": .55, "old": True}
    module.CACHE_PATH.write_text(json.dumps(old), encoding="utf-8")
    monkeypatch.setattr(module, "_compute", lambda *args: {"new": True})
    assert getattr(module, module_name)()["new"] is True


def test_malformed_ledger_is_counted_not_silently_dropped(tmp_path, monkeypatch):
    p = tmp_path / "ft.jsonl"
    p.write_text('{"broken":\n', encoding="utf-8")
    monkeypatch.setattr(ft, "LEDGER_PATH", p)
    result = ft.score()
    assert result["ledger_rows"] == 1
    assert result["quarantined"] == 1


def test_yahoo_overreturned_intraday_bar_never_reaches_cache(tmp_path, monkeypatch):
    import sys
    from types import SimpleNamespace
    from stock_agent.data import providers
    f = bars(100)
    raw = f.rename(columns={c: c.title() for c in f.columns}).set_index("Date")
    monkeypatch.setitem(sys.modules, "yfinance", SimpleNamespace(Ticker=lambda *a: SimpleNamespace(history=lambda **k: raw)))
    monkeypatch.setattr(providers, "PRICE_CACHE_DIR", tmp_path)
    monkeypatch.setattr(calendar, "completed_session_date", lambda now=None: date(2026, 9, 21))
    result = providers.YahooProvider().history("AAA", date(2026, 1, 1), date(2026, 9, 22))
    assert str(result.frame.date.max()) == "2026-09-21"
    assert pd.read_csv(tmp_path / "AAA.csv").date.max() == "2026-09-21"


def test_eod_refresh_replaces_existing_tail_but_rejects_intraday(tmp_path, monkeypatch):
    import sys
    from types import SimpleNamespace
    from stock_agent.pipeline import eod_update as job
    monkeypatch.setattr(calendar, "completed_session_date", lambda now=None: date(2026, 9, 21))
    monkeypatch.setattr(job, "_fetch_end_date", lambda: date(2026, 9, 21))
    monkeypatch.setattr(job, "PRICES_DIR", tmp_path)
    monkeypatch.setattr(job.time, "sleep", lambda *a: None)
    old = bars(1, end="2026-09-21")
    old.to_csv(tmp_path / "AAA.csv", index=False)
    old.to_csv(tmp_path / "VNINDEX.csv", index=False)
    new = bars(2)
    new[["open", "high", "low", "close"]] *= 1.01
    calls = []

    def history(**kwargs):
        calls.append(kwargs)
        return new.rename(columns={"date": "time"})

    monkeypatch.setitem(sys.modules, "vnstock", SimpleNamespace(
        Vnstock=lambda: SimpleNamespace(stock=lambda **k: SimpleNamespace(quote=SimpleNamespace(history=history)))))
    result = job.refresh_prices()
    assert result["updated"] == 2
    for symbol in ("AAA", "VNINDEX"):
        written = pd.read_csv(tmp_path / f"{symbol}.csv")
        assert written.date.tolist() == ["2026-09-21"]
        assert written.close.iloc[0] == (101000 if symbol == "AAA" else 101)
    assert all(c["start"] == "2026-09-21" for c in calls)


def test_dashboard_to_paper_ledger_end_to_end(tmp_path, monkeypatch):
    """Real rule scan -> temporary cache -> logger -> replay; no production state."""
    from types import SimpleNamespace
    import numpy as np
    from stock_agent.features import momentum_scan as mom, position_manager as pos
    from stock_agent.features import win_probability as wp
    from stock_agent.pipeline import forward_test as forward
    monkeypatch.setattr(calendar, "completed_session_date", lambda now=None: date(2026, 9, 21))
    f = bars(360)
    c = 100 + np.arange(len(f)) * .1 + np.sin(np.arange(len(f)))
    for col in ("open", "high", "low", "close"):
        f[col] = c + (1 if col == "high" else -1 if col == "low" else 0)
    for symbol in ("AAA", "VNINDEX"):
        f.to_csv(tmp_path / f"{symbol}.csv", index=False)
    for module in (mr, mom):
        monkeypatch.setattr(module, "PRICES_DIR", tmp_path)
        monkeypatch.setattr(module, "CACHE_PATH", tmp_path / f"{module.__name__}.json")
    monkeypatch.setattr(pos, "check_positions", lambda *a: [])
    monkeypatch.setattr(pos, "check_momentum_positions", lambda *a: [])
    model = SimpleNamespace(meta={"model_version": "sha", "trained_at": "2026-07-01T00:00:00+00:00"}, predict=lambda fr: .6)
    monkeypatch.setattr(wp.WinProbModel, "load", lambda: model)
    mr_payload = mr.mr_scan(force=True)
    mom_payload = mom.momentum_scan(force=True)
    assert mr_payload["data_date"] == mom_payload["data_date"] == "2026-09-21"
    assert mr_payload["model"]["model_version"] == "sha"
    assert mom_payload["picks"][0]["symbol"] == "AAA"
    monkeypatch.setattr(forward, "datetime", SimpleNamespace(now=lambda tz: datetime(2026, 9, 21, 10, tzinfo=timezone.utc)))
    monkeypatch.setattr(forward, "LEDGER_PATH", tmp_path / "ledger.jsonl")
    monkeypatch.setattr(forward, "PRICES_DIR", tmp_path)
    assert forward.log_recommendations(mr_payload, mom_payload)["appended"] >= 1
    result = forward.score()
    assert result["quarantined"] == 0
    assert result["pending"] == result["ledger_rows"]


def test_position_alerts_in_eod_payload_exclude_intraday(tmp_path, monkeypatch):
    from stock_agent.features import position_manager as pos
    monkeypatch.setattr(calendar, "completed_session_date", lambda now=None: date(2026, 9, 21))
    monkeypatch.setattr(pos, "PRICES_DIR", tmp_path)
    bars().to_csv(tmp_path / "AAA.csv", index=False)
    assert pos._symbol_frame("AAA").date.iloc[-1] == "2026-09-21"


@pytest.mark.parametrize("values", [[], ["bad", "d0"]])
def test_eod_rejects_empty_or_entirely_invalid_dates(values):
    from stock_agent.data.eod import completed_bars
    assert completed_bars(pd.DataFrame({"date": values})).empty
