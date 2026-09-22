"""A candidate is an immutable experiment, never an implicit production release."""
from pathlib import Path
import hashlib
import json
import pickle

import numpy as np
import pandas as pd
import pytest

from stock_agent.features import win_probability as wp


def release_module():
    from stock_agent.features import model_release
    return model_release


def metrics(**changes):
    return dict(test_auc=.65, test_brier=.19, baseline_brier=.24,
                test_ece=.06, test_rows=150, calibration_rows=150, **changes)


def test_candidate_gates_are_predeclared_and_never_live_approval():
    module = release_module()
    report = module.evaluate_candidate(metrics(), snapshot_verified=True)
    assert report["paper_eligible"]
    assert report["shadow_eligible"]
    assert report["live_approved"] is False
    assert "point_in_time_universe_unverified" in report["limitations"]
    assert report["criteria"] == module.CANDIDATE_GATES


@pytest.mark.parametrize("key,value", [("test_auc", .49), ("test_brier", .30),
                                      ("test_ece", .2), ("test_auc", np.nan),
                                      ("test_rows", 10)])
def test_failed_metric_gate_cannot_be_paper_eligible(key, value):
    values = metrics()
    values[key] = value
    report = release_module().evaluate_candidate(values, snapshot_verified=True)
    assert not report["paper_eligible"]
    assert report["failures"]
    assert not report["live_approved"]


def test_unverified_snapshot_cannot_enable_paper_or_shadow():
    report = release_module().evaluate_candidate(metrics(), snapshot_verified=False)
    assert not report["paper_eligible"]
    assert not report["shadow_eligible"]
    assert "snapshot_not_verified" in report["failures"]


def test_candidate_staging_is_immutable_and_content_addressed(tmp_path):
    module = release_module()
    artifact = {"model": None, "iso": None, "features": wp.FEATURES,
                "training_metadata": {"dataset_sha256": "abc"}, "trained_at": "2025-01-01"}
    target = tmp_path / "candidate.pkl"
    result = module.stage_candidate(artifact, candidate_dir=tmp_path, artifact_path=target)
    saved = pickle.loads(target.read_bytes())
    manifest = json.loads(Path(result["release_manifest"]).read_text(encoding="utf-8"))
    assert saved["release"]["live_approved"] is False
    assert manifest["artifact_sha256"] == hashlib.sha256(target.read_bytes()).hexdigest()
    assert manifest["code_provenance"]["source_sha256"]
    with pytest.raises(FileExistsError):
        module.stage_candidate(artifact, candidate_dir=tmp_path, artifact_path=target)


def test_default_training_target_is_candidate_not_production():
    import inspect
    assert inspect.signature(wp.train_and_save).parameters["artifact_path"].default is None


def test_training_refuses_explicit_production_replacement():
    with pytest.raises(ValueError, match="production"):
        wp.train_and_save(Path("does-not-exist"), wp.ARTIFACT_PATH)


def test_stage_refuses_production_path():
    with pytest.raises(ValueError, match="production"):
        release_module().stage_candidate({}, artifact_path=wp.ARTIFACT_PATH)


def test_shared_features_have_batch_single_parity_and_finite_output():
    data = pd.DataFrame([{key: 1. for key in wp.FEATURES}, {key: 2. for key in wp.FEATURES}])
    data.loc[0, wp.FEATURES[0]] = np.inf
    data.loc[0, wp.FEATURES[1]] = np.nan
    batch = wp._model_features(data, wp.FEATURES)
    single = wp._model_features(data.iloc[:1], wp.FEATURES)
    pd.testing.assert_frame_equal(batch.iloc[:1], single)
    assert np.isfinite(batch.to_numpy()).all()
    with pytest.raises(ValueError, match="feature"):
        wp._model_features(data.drop(columns=wp.FEATURES[0]), wp.FEATURES)


@pytest.mark.parametrize("raw,calibrated", [(np.nan, .5), (.5, np.nan), (.5, 1.1), (-.1, .5)])
def test_invalid_model_outputs_fail_closed(raw, calibrated):
    class Model:
        def predict_proba(self, x):
            return np.array([[1 - raw, raw]])
    class Iso:
        def transform(self, x):
            return [calibrated]
    model = wp.WinProbModel({"model": Model(), "iso": Iso(), "features": wp.FEATURES})
    with pytest.raises(ValueError, match="probability"):
        model.predict({key: 1. for key in wp.FEATURES})


def test_candidate_metrics_record_fixed_train_prior_baseline(monkeypatch):
    import lightgbm
    class Model:
        def fit(self, x, y):
            return self
        def predict_proba(self, x):
            p = np.repeat(.6, len(x))
            return np.column_stack([1-p, p])
    monkeypatch.setattr(lightgbm, "LGBMClassifier", lambda **kw: Model())
    data = pd.DataFrame({"signal_date": np.repeat(pd.bdate_range("2023-01-02", periods=500), 3)})
    data["exit_date"] = data.signal_date + pd.offsets.BDay(3)
    data["symbol"] = ["AAA", "BBB", "CCC"] * 500
    data["win"] = np.arange(len(data)) % 2
    for col in wp.FEATURES:
        data[col] = 1.
    artifact = wp._fit_candidates(data)
    evaluation = artifact["evaluation"]
    assert evaluation["baseline_probability"] == artifact["base_win_rate"]
    assert evaluation["threshold"] == .65
    assert evaluation["test_rows"] > 100
    assert len(artifact["training_metadata"]["evaluation_sha256"]) == 64


def test_candidate_loader_rejects_unverified_or_tampered_release(tmp_path):
    module = release_module()
    result = module.stage_candidate({"model": None, "iso": None, "features": wp.FEATURES}, candidate_dir=tmp_path)
    with pytest.raises(ValueError, match="eligible"):
        module.load_candidate(Path(result["artifact"]), mode="shadow")
    path = Path(result["artifact"])
    path.write_bytes(path.read_bytes() + b"tampered")
    with pytest.raises(ValueError, match="hash"):
        module.load_candidate(path, mode="shadow")


def test_candidate_loader_has_no_live_mode(tmp_path):
    with pytest.raises(ValueError, match="mode"):
        release_module().load_candidate(tmp_path / "none.pkl", mode="live")


def test_verified_candidate_loads_explicitly_without_changing_default(tmp_path, monkeypatch):
    module = release_module()
    snapshot = {"verified": True, "manifest_path": str(tmp_path / "manifest.json"),
                "manifest_sha256": "snapshot", "files": {"AAA": "hash"}}
    monkeypatch.setattr(module, "verify_snapshot", lambda *args: snapshot)
    result = module.stage_candidate({"model": None, "iso": None, "features": wp.FEATURES,
                                     "evaluation": metrics()}, candidate_dir=tmp_path, snapshot=snapshot)
    model = module.load_candidate(Path(result["artifact"]), mode="paper")
    assert model.meta["release"]["paper_eligible"]
    assert model.meta["model_version"] == result["artifact_sha256"]
    assert wp.ARTIFACT_PATH == Path("data/models/win_prob_mr.pkl")


def test_snapshot_verification_delegates_raw_source_equivalence(tmp_path, monkeypatch):
    from stock_agent.data import reconciliation
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text('{}', encoding="utf-8")
    def reject(path):
        raise ValueError("CSV/source mismatch")
    monkeypatch.setattr(reconciliation, "verify_snapshot", reject)
    with pytest.raises(ValueError, match="CSV/source"):
        release_module().verify_snapshot(manifest_path, tmp_path / "prices")


def test_manifest_serialization_failure_does_not_leave_partial_artifact(tmp_path):
    target = tmp_path / "bad.pkl"
    with pytest.raises(ValueError):
        release_module().stage_candidate({"evaluation": {"test_auc": np.nan}}, artifact_path=target)
    assert not target.exists()
