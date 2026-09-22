"""Leakage regressions: clocks, rows and fit provenance, not return targets."""
from datetime import date
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from stock_agent.features.calibration import preprocess_features_robust, _time_split
from stock_agent.features.ensemble_model import EnsembleTrainer, EnsembleConfig
from stock_agent.features.feature_engineering_v2 import add_regime_features
from stock_agent.features import ml_models, win_probability as wp


def panel(days=120, symbols=3):
    dates = np.repeat(pd.bdate_range("2024-01-02", periods=days), symbols)
    n = len(dates)
    return pd.DataFrame({"symbol": [f"S{i % symbols}" for i in range(n)], "signal_date": dates,
                         "exit_date": dates + pd.offsets.BDay(5), "label_resolved": True,
                         "feature_x": np.arange(n, dtype=float), "net_t2_win": np.arange(n) % 2,
                         "net_t2_return_pct": np.where(np.arange(n) % 2, 1., -1.), "decision": "BUY_SETUP"})


def test_preprocess_is_row_independent_and_never_fits_transform():
    frame = pd.DataFrame({"symbol": ["A", "B"], "feature_rsi14": [90., np.nan]})
    bounds = {"feature_rsi14": (0., 100.)}
    batch, _ = preprocess_features_robust(frame, ["feature_rsi14"], bounds)
    single, _ = preprocess_features_robust(frame.iloc[[1]], ["feature_rsi14"], bounds)
    assert batch.iloc[1, 0] == single.iloc[0, 0] == 50.
    with pytest.raises(ValueError, match="bounds"):
        preprocess_features_robust(frame, ["feature_rsi14"], {})


def test_v2_all_regime_columns_are_prefix_invariant():
    ret = np.r_[np.sin(np.arange(100)) * .002, np.sin(np.arange(100)) * .8]
    df = pd.DataFrame({"date": pd.bdate_range("2024-01-01", periods=200), "return_1d": ret})
    full, prefix = add_regime_features(df), add_regime_features(df.iloc[:100])
    pd.testing.assert_frame_equal(full.iloc[:100], prefix)


def test_split_groups_dates_and_purges_label_end():
    train, val, test = _time_split(panel(41, 7))
    assert train.signal_date.max() < val.signal_date.min()
    assert train.exit_date.max() < val.signal_date.min()
    assert val.exit_date.max() < test.signal_date.min()
    assert not set(train.signal_date) & set(val.signal_date)


def test_missing_label_end_is_rejected():
    with pytest.raises(ValueError, match="exit_date"):
        _time_split(panel().drop(columns="exit_date"))


class Spy:
    instances = []

    def __init__(self):
        self.__class__.instances.append(self)
        self.predictions = []

    def fit(self, x, y, **kwargs):
        self.fit_rows = set(x.index)
        self.fitted_x = x.copy()
        return self

    def predict_proba(self, x):
        self.predictions.append(set(x.index))
        self.last_x = x.copy()
        return np.tile([.3, .7], (len(x), 1))


def spy_trainer(monkeypatch):
    Spy.instances = []
    trainer = EnsembleTrainer(EnsembleConfig(min_train_rows=20, n_splits=3))
    for name in ("_build_lgb", "_build_xgb", "_build_ridge", "_build_cat"):
        monkeypatch.setattr(trainer, name, Spy)
    monkeypatch.setattr(trainer, "_ridge_proba", lambda m, x: m.predict_proba(x)[:, 1])
    return trainer


def test_ensemble_evaluation_never_predicts_its_fit_rows(monkeypatch):
    trainer = spy_trainer(monkeypatch)
    assert trainer.train(panel(), ["feature_x"])["status"] == "trained"
    assert any(m.predictions for m in Spy.instances)
    for model in Spy.instances:
        for rows in model.predictions:
            assert not rows & model.fit_rows


def test_ensemble_bounds_do_not_see_holdout(monkeypatch):
    trainer = spy_trainer(monkeypatch)
    data = panel()
    data.loc[data.signal_date >= sorted(data.signal_date.unique())[96], "feature_x"] = 1e9
    trainer.train(data, ["feature_x"])
    assert trainer.winsorize_bounds["feature_x"][1] < 1000


def test_daily_update_routes_through_fresh_purged_training(monkeypatch):
    trainer = spy_trainer(monkeypatch)
    trainer.feature_columns = ["feature_x"]
    trainer.lgb_model = trainer.xgb_model = trainer.cat_model = object()
    calls = []
    monkeypatch.setattr(trainer, "train", lambda *a, **k: calls.append(a) or {"status": "trained"})
    result = trainer.daily_update(panel())
    assert calls and result["mode"] == "purged_retrain"


def test_model_selection_ignores_test_metrics():
    def result(family, val, test):
        return ml_models.TrainingResult(family, "trained", metrics={
            "validation": {"avg_net_return_pct": val}, "test": {"avg_net_return_pct": test}})
    assert ml_models._select_model([result("a", 1, -100), result("b", 0, 100)]) == "a"


def test_mr_partial_label_rejected():
    frame = pd.DataFrame({"date": ["2026-01-05", "2026-01-06"], "open": [100., 100.],
                          "high": [101., 101.], "low": [99., 99.], "close": [100., 100.],
                          "atr": [2., 2.], "kijun": [110., 110.], "ema21": [100., 100.]})
    assert wp._label_trade(frame, 0) is None


def test_mr_label_stop_is_anchored_to_signal_close(monkeypatch):
    frame = pd.DataFrame({"date": ["2026-01-05", "2026-01-06"], "open": [100., 105.],
                          "close": [100., 105.], "atr": [2., 2.], "kijun": [110., 110.]})
    captured = []
    monkeypatch.setattr(wp, "simulate_mr_exit", lambda f, i, stop, *a, **k:
                        captured.append(stop) or (1, 110., "target", True))
    wp._label_trade(frame, 0)
    assert captured == [94.]


def test_ensemble_requires_fitted_bounds_for_inference():
    trainer = EnsembleTrainer()
    trainer.feature_columns = ["feature_x"]
    with pytest.raises(ValueError, match="preprocessing"):
        trainer._prepare_features(pd.DataFrame({"feature_x": [1.]}))


def test_ensemble_metadata_survives_save_load(tmp_path, monkeypatch):
    trainer = spy_trainer(monkeypatch)
    trainer.train(panel(), ["feature_x"])
    path = tmp_path / "model.pkl"
    trainer.save(path)
    loaded = EnsembleTrainer.load(path)
    assert loaded.training_metadata == trainer.training_metadata


def test_legacy_models_store_and_reuse_train_bounds(monkeypatch):
    train, val, test = _time_split(panel())
    val = val.assign(feature_x=1e7)
    test = test.assign(feature_x=1e8)
    model = Spy()
    monkeypatch.setattr(ml_models, "_build_estimator", lambda family: (model, None))
    saved = []
    monkeypatch.setattr(ml_models, "_save_artifact", lambda path, payload: saved.append(payload))
    result = ml_models._train_one_family("spy", train, val, test, ["feature_x"], .5, "2026-09-22")
    assert result.status == "trained"
    assert saved[0]["winsorize_bounds"]["feature_x"][1] < 1000
    assert model.last_x.feature_x.max() < 1000


def test_mr_classifier_is_not_refit_after_calibration(monkeypatch):
    import lightgbm
    from sklearn.isotonic import IsotonicRegression
    data = panel(500)
    for col in wp.FEATURES:
        data[col] = np.arange(len(data)) * .001
    data["win"] = data.net_t2_win
    fit_calls = []
    class Model(Spy):
        def fit(self, x, y, **kwargs):
            fit_calls.append(set(x.index))
            return super().fit(x, y, **kwargs)
    model = Model()
    monkeypatch.setattr(lightgbm, "LGBMClassifier", lambda **kw: model)
    art = wp._fit_candidates(data)
    assert len(fit_calls) == 1
    assert all(not rows & fit_calls[0] for rows in model.predictions)
    assert isinstance(art["iso"], IsotonicRegression)
    assert art["training_metadata"]["calibration_label_end"] < art["training_metadata"]["evaluation_test_start"]
    assert "test_brier" in art and "test_auc" in art


def test_hpo_never_sees_outer_holdout_or_refits_global_bounds(monkeypatch):
    import optuna, lightgbm, xgboost, catboost
    trainer = spy_trainer(monkeypatch)
    trainer.feature_columns = ["feature_x"]
    data = panel()
    cutoff = sorted(data.signal_date.unique())[96]
    data.loc[data.signal_date >= cutoff, "feature_x"] = 1e9
    for module, attr in ((lightgbm, "LGBMClassifier"), (xgboost, "XGBClassifier"), (catboost, "CatBoostClassifier")):
        monkeypatch.setattr(module, attr, lambda **kw: Spy())
    trial = SimpleNamespace(suggest_int=lambda name, low, high: low, suggest_float=lambda name, low, high, **kw: low)
    study = SimpleNamespace(best_params={}, best_value=0., optimize=lambda objective, **kw: objective(trial))
    monkeypatch.setattr(optuna, "create_study", lambda **kw: study)
    trainer.run_hpo(data, n_trials=1)
    heldout = set(data.index[data.signal_date >= cutoff])
    assert Spy.instances
    for model in Spy.instances:
        assert not model.fit_rows & heldout
        assert all(not rows & heldout for rows in model.predictions)
        assert model.fitted_x.feature_x.max() < 1000


def test_invalid_prices_fail_before_training(tmp_path):
    from stock_agent.config import load_rules
    from stock_agent.agents.parallel_engine import build_labeled_dataset_fast
    f = pd.DataFrame({"date": pd.bdate_range("2025-01-01", periods=200),
                      "open": 100., "high": 101., "low": 99., "close": 100., "volume": 1e6})
    f.loc[0, "close"] = 0.
    f.to_csv(tmp_path / "BAD.csv", index=False)
    with pytest.raises(ValueError, match="BAD"):
        build_labeled_dataset_fast(["BAD"], load_rules(), tmp_path)
