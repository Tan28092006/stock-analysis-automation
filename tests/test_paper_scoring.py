from datetime import date, datetime, timezone
import json

import pandas as pd
import pytest

from stock_agent.pipeline import paper_runner as runner
from tests.test_market_runtime_hardening import make_snapshot


def record(tmp_path, track="mr"):
    manifest = make_snapshot(tmp_path / "original")
    now = datetime(2026, 9, 21, 10, tzinfo=timezone.utc)
    payload = runner.run_paper(manifest.parent / "prices", ["AAA"], manifest_path=manifest, now=now)
    close = float(pd.read_csv(manifest.parent / "prices/AAA.csv").close.iloc[-1])
    payload["recommendations"] = [{"track": track, "symbol": "AAA", "date": "2026-09-21", "close": close,
        "entry_reference": close, "stop_loss": close - 500, "take_profit": close + 2000, "max_hold_days": 15}]
    runner.commit_paper_run(payload, tmp_path / "out", now=now)
    return manifest


def test_pending_is_not_a_loss_or_profit_and_original_is_immutable(tmp_path):
    from stock_agent.pipeline.paper_scoring import score_paper_runs
    manifest = record(tmp_path)
    path = tmp_path / "out/runs/2026-09-21/paper.json"
    before = path.read_bytes()
    result = score_paper_runs(tmp_path / "out", manifest.parent / "prices", manifest_path=manifest)
    assert result["pending"] == 1 and result["resolved"] == 0 and not result["errors"]
    assert result["records"][0]["net_return_pct"] is None
    assert path.read_bytes() == before


def test_stop_gap_uses_worse_open_not_optimistic_stop(tmp_path):
    from stock_agent.pipeline.paper_scoring import score_paper_runs
    record(tmp_path)
    def crash(items):
        if items[0]["symbol"] == "AAA":
            for i, stamp in enumerate(items[0]["t"]):
                if str(pd.Timestamp(stamp, unit="s").date()) == "2026-09-24":
                    for key, value in {"o": 8000, "l": 7900, "h": 10000, "c": 9000}.items():
                        items[0][key][i] = value
        return items
    current = make_snapshot(tmp_path / "current", end=date(2026, 9, 25), transform=crash)
    result = score_paper_runs(tmp_path / "out", current.parent / "prices", manifest_path=current)
    assert not result["errors"] and result["resolved"] == 1
    row = result["records"][0]
    assert row["exit_price"] == 8000 and row["reason"] == "stop"
    assert row["net_return_pct"] == pytest.approx((8000 / row["entry_price"] - 1) * 100 - .6)


@pytest.mark.parametrize("defect", ["malformed", "tampered", "source_revision", "wrong_directory"])
def test_scoring_fails_closed_for_corrupt_records_or_revised_history(tmp_path, defect):
    from stock_agent.pipeline.paper_scoring import score_paper_runs
    original = record(tmp_path)
    path = tmp_path / "out/runs/2026-09-21/paper.json"
    if defect == "malformed":
        path.write_text("{")
    elif defect == "tampered":
        row = json.loads(path.read_text())
        row["recommendations"][0]["stop_loss"] = 1
        path.write_text(json.dumps(row))
    def revise(items):
        if defect == "source_revision" and items[0]["symbol"] == "AAA":
            items[0]["c"][-5] += 1
        return items
    current = make_snapshot(tmp_path / "current", end=date(2026, 9, 25), transform=revise)
    prices = original.parent / "prices" if defect == "wrong_directory" else current.parent / "prices"
    if defect == "wrong_directory":
        with pytest.raises(ValueError, match="directory"):
            score_paper_runs(tmp_path / "out", prices, manifest_path=current)
    else:
        result = score_paper_runs(tmp_path / "out", prices, manifest_path=current)
        assert result["errors"] and result["resolved"] == result["pending"] == 0


def test_momentum_is_informational_and_remains_pending(tmp_path):
    from stock_agent.pipeline.paper_scoring import score_paper_runs
    manifest = record(tmp_path, track="momentum")
    result = score_paper_runs(tmp_path / "out", manifest.parent / "prices", manifest_path=manifest)
    assert result["records"][0]["metric"] == "informational_close_return_21_sessions"
    assert result["pending"] == 1


def test_no_records_is_an_empty_observation_set(tmp_path):
    from stock_agent.pipeline.paper_scoring import score_paper_runs
    manifest = make_snapshot(tmp_path / "source")
    result = score_paper_runs(tmp_path / "out", manifest.parent / "prices", manifest_path=manifest)
    assert result["pending"] == result["resolved"] == 0
    assert result["status"] == "ok"
