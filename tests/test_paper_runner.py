from datetime import date, datetime, timezone
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


def prices(root, symbol="AAA", end=date(2026, 9, 21)):
    from stock_agent.data.exchange_calendar import trading_days_between
    dates = trading_days_between(date(2025, 2, 3), end)[-320:]
    c = 100 + np.arange(len(dates)) * .02 + np.sin(np.arange(len(dates)))
    f = pd.DataFrame({"date": dates, "open": c, "high": c + 2, "low": c - 2, "close": c, "volume": 1e6})
    f.to_csv(root / f"{symbol}.csv", index=False)
    return f


def test_readiness_rejects_missing_stale_and_invalid(tmp_path):
    from stock_agent.pipeline.paper_runner import assess_market_data
    prices(tmp_path, "VNINDEX")
    prices(tmp_path)
    now = datetime(2026, 9, 21, 10, tzinfo=timezone.utc)
    result = assess_market_data(tmp_path, ["AAA", "BBB"], now=now)
    assert not result["data_ready"] and "BBB" in result["issues"]
    frame = prices(tmp_path, "BBB", end=date(2026, 9, 18))
    result = assess_market_data(tmp_path, ["AAA", "BBB"], now=now)
    assert "stale" in str(result["issues"]["BBB"])
    frame = prices(tmp_path, "BBB")
    frame.loc[len(frame)-1, "close"] = 0
    frame.to_csv(tmp_path / "BBB.csv", index=False)
    assert not assess_market_data(tmp_path, ["AAA", "BBB"], now=now)["data_ready"]


def test_readiness_never_backdates_a_forward_run(tmp_path):
    from stock_agent.pipeline.paper_runner import assess_market_data
    prices(tmp_path, "VNINDEX")
    prices(tmp_path)
    before = assess_market_data(tmp_path, ["AAA"], now=datetime(2026, 9, 22, 4, tzinfo=timezone.utc))
    assert before["data_ready"] and not before["recording_window_open"]
    after = assess_market_data(tmp_path, ["AAA"], now=datetime(2026, 9, 21, 10, tzinfo=timezone.utc))
    assert after["recording_window_open"]


def test_isolated_scans_dont_load_legacy_model_or_positions(tmp_path, monkeypatch):
    from stock_agent.features import mr_scan as mr, momentum_scan as mom, win_probability as wp, position_manager as pos
    prices(tmp_path, "VNINDEX")
    prices(tmp_path)
    monkeypatch.setattr(wp.WinProbModel, "load", lambda *a, **k: pytest.fail("legacy model loaded"))
    monkeypatch.setattr(pos, "PositionStore", lambda: pytest.fail("production positions read"))
    a = mr._compute(0, .55, prices_dir=tmp_path, use_model=False, include_positions=False)
    b = mom._compute(5, prices_dir=tmp_path, include_positions=False)
    assert a["model"]["available"] is False
    assert a["data_date"] == b["data_date"] == "2026-09-21"
    assert not a["positions"] and not b["positions"]


def test_paper_commit_is_immutable_idempotent_and_exante(tmp_path):
    from stock_agent.pipeline.paper_runner import commit_paper_run
    now = datetime(2026, 9, 21, 10, tzinfo=timezone.utc)
    from tests.test_market_runtime_hardening import make_snapshot
    from stock_agent.pipeline.paper_runner import run_paper
    manifest = make_snapshot(tmp_path / "snapshot")
    payload = run_paper(manifest.parent / "prices", ["AAA"], manifest_path=manifest, now=now)
    result = commit_paper_run(payload, tmp_path, now=now)
    assert result["status"] == "recorded"
    path = Path(result["path"])
    before = path.read_bytes()
    assert commit_paper_run(payload, tmp_path, now=now)["status"] == "already_recorded"
    assert path.read_bytes() == before
    with pytest.raises(ValueError, match="conflict"):
        commit_paper_run({**payload, "input_snapshot": "different"}, tmp_path, now=now)
    with pytest.raises(ValueError, match="window"):
        commit_paper_run({**payload, "session": "2026-09-18"}, tmp_path, now=now)
    with pytest.raises(ValueError, match="verified"):
        commit_paper_run({**payload, "source_verified": False}, tmp_path, now=now)


def test_local_cache_without_source_manifest_is_preview_only(tmp_path):
    from stock_agent.pipeline.paper_runner import run_paper
    source = tmp_path / "source"
    source.mkdir()
    prices(source, "VNINDEX")
    prices(source)
    result = run_paper(source, ["AAA"], output_dir=tmp_path / "runs", record=True,
                       now=datetime(2026, 9, 21, 10, tzinfo=timezone.utc))
    assert result["status"] == "preview_only"
    assert result["source_verified"] is False
    assert not list((tmp_path / "runs").glob("*/paper.json"))
