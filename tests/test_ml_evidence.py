import json

import pytest

from stock_agent.research_ml import evidence as E


def build(tmp_path, **over):
    run = E.init_bundle(tmp_path, "p1", "exp", "run1")
    docs = {"experiment.json": {"hypothesis": "H-X", "sample_floors": {"dates": 10}, "activity_floor": 1},
            "source_manifest.json": {}, "environment.json": {"python": "3.11"}, "dataset_manifest.json": {},
            "splits.json": {"folds": [{"index": 0, "problems": []}]},
            "metrics.json": {"brier": 0.2, "sample_support": {"dates": 20}},
            "decision.json": {"status": "engineering_complete"}}
    docs.update(over)
    for name, payload in docs.items():
        E.write_json(run / name, payload)
    E.write_json(run / "transforms" / "fold0.json", {"a": 1})
    E.write_json(run / "predictions" / "oof.json", {"p": [0.1]})
    E.seal(run)
    return run


def test_complete_bundle_is_engineering_complete_only(tmp_path):
    r = E.validate_bundle(build(tmp_path))
    assert r["status"] == "engineering_complete" and not r["problems"] and not r["blocked"]


def test_missing_required_files_blocked(tmp_path):
    run = build(tmp_path)
    (run / "metrics.json").unlink()
    assert E.validate_bundle(run)["status"] == "BLOCKED"


def test_hash_drift_after_seal_fails(tmp_path):
    run = build(tmp_path)
    (run / "predictions" / "oof.json").write_text('{"p": [0.9]}')
    r = E.validate_bundle(run)
    assert r["status"] == "FAIL" and any("hash_drift" in p for p in r["problems"])


def test_nan_metric_cannot_even_be_written_and_inf_string_blocked(tmp_path):
    with pytest.raises(ValueError):
        E.write_json(tmp_path / "x.json", {"m": float("nan")})
    run = build(tmp_path / "b")
    (run / "metrics.json").write_text('{"brier": Infinity, "sample_support": {"dates": 20}}')
    E.seal(run)
    assert any("nonfinite" in p for p in E.validate_bundle(run)["problems"])


def test_floors_and_activity_floor_unfrozen_blocks(tmp_path):
    run = build(tmp_path, **{"experiment.json": {"hypothesis": "H"}})
    assert {"sample_floors_not_frozen", "activity_floor_not_frozen"} <= set(E.validate_bundle(run)["blocked"])
    low = build(tmp_path / "c", **{"metrics.json": {"sample_support": {"dates": 3}}})
    assert "below_floor:dates" in E.validate_bundle(low)["blocked"]


def test_single_class_fold_blocks(tmp_path):
    run = build(tmp_path, **{"splits.json": {"folds": [{"index": 0, "problems": ["test_single_class"]}]}})
    assert E.validate_bundle(run)["status"] == "BLOCKED"


def test_research_pass_needs_regression_and_reviewer(tmp_path):
    run = build(tmp_path, **{"decision.json": {"status": "research_pass"}})
    probs = E.validate_bundle(run)["problems"]
    assert "research_pass_without_regression_evidence" in probs and "research_pass_without_reviewer" in probs


def test_unsealed_extra_file_and_run_dir_never_overwritten(tmp_path):
    run = build(tmp_path)
    (run / "portfolio" / "late.json").write_text("{}")
    assert any("unsealed" in p for p in E.validate_bundle(run)["problems"])
    with pytest.raises(FileExistsError):
        E.init_bundle(tmp_path, "p1", "exp", "run1")


def test_unsealed_bundle_is_blocked(tmp_path):
    run = build(tmp_path)
    exp = json.loads((run / "experiment.json").read_text())
    exp.pop("artifact_hashes")
    E.write_json(run / "experiment.json", exp)
    assert "bundle_not_sealed" in E.validate_bundle(run)["blocked"]
