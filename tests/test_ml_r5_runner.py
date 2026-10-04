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


def test_weekly_protocol_registered_with_same_floors_and_independence_rule():
    from scripts import ml_research_r5_weekly as W
    p = json.loads(W.PROTOCOL.read_text())
    assert p["floors"] == PROTO["floors"] and p["parent"] == "ml_research_v1"
    assert "every 5th session" in p["independence_rule"] and p["folds"]["first_train_dates"] == 121


def test_circular_block_bootstrap_detects_signal_and_not_noise():
    import numpy as np
    from scripts.ml_research_r5_weekly import circular_block_ci
    rng = np.random.default_rng(0)
    signal = rng.normal(1.0, 0.5, 80)
    noise = rng.normal(0.0, 1.0, 80)
    assert circular_block_ci(signal, 4, 2000, 1)["ci95"][0] > 0
    assert circular_block_ci(noise, 4, 2000, 1)["p_one_sided_le0"] > 0.05


def test_committed_weekly_bundle_inconclusive_never_pass():
    run = Path("docs/audits/2026-10-04-r5-weekly-bundle")
    assert E.validate_bundle(run)["status"] == "engineering_complete"
    d = json.loads((run / "decision.json").read_text())
    assert d["status"] == "INCONCLUSIVE" and d["live_eligible"] is False
    m = json.loads((run / "metrics.json").read_text())
    assert "control_label_shuffle_within_date" in m["trials"]  # negative control retained
