import json
from pathlib import Path

from scripts import ml_research_r5 as R
from stock_agent.research_ml import evidence as E

PROTO = json.loads(R.PROTOCOL.read_text())


def test_protocol_freezes_user_floors():
    f = PROTO["floors"]
    assert (f["train_dates_min"], f["test_dates_min"], f["events_per_class_min"]) == (120, 40, 50)
    assert f["activity"]["flat_to_long_entries_per_month_min"] == 4
    assert f["activity"]["distinct_entry_days_per_month_min"] == 3


def test_drop_family_removes_only_that_family():
    base = set(R.families(PROTO, "baseline_all_features"))
    dropped = set(R.families(PROTO, "drop_f0_risk"))
    assert base - dropped == {"f0_vol_20", "f0_drawdown_60"}


def test_committed_r5_bundle_is_sealed_and_blocked_not_pass():
    run = Path("docs/audits/2026-10-04-r5-bundle")
    verdict = E.validate_bundle(run)
    assert verdict["status"] == "BLOCKED" and not verdict["problems"]
    assert json.loads((run / "decision.json").read_text())["live_eligible"] is False
