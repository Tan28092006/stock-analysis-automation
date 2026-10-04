import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression, Ridge

from stock_agent.research_ml import harness as H
from stock_agent.research_ml.contracts import ContractError

TZ = "Asia/Ho_Chi_Minh"


def dataset(n_days=60, n_sym=6, horizon=5, seed=1):
    rng = np.random.default_rng(seed)
    days = pd.bdate_range("2025-01-02", periods=n_days)
    rows = []
    for i, d in enumerate(days):
        for s in range(n_sym):
            x1, x2 = rng.normal(size=2)
            y = float(x1 + .5 * x2 + rng.normal(scale=1.0) > 0)
            end = days[min(i + horizon, n_days - 1)]
            rows.append(dict(symbol=f"S{s}", signal_date=d, x1=x1, x2=x2 if rng.random() > .1 else np.nan,
                             win=y, label_available_at=end.tz_localize(TZ) + pd.Timedelta(hours=16)))
    return pd.DataFrame(rows)


def test_folds_purge_by_label_availability_and_group_dates():
    df = dataset()
    folds = H.purged_grouped_folds(df, 3)
    sig = pd.to_datetime(df.signal_date)
    for f in folds:
        assert f.conclusive and f.purged_rows > 0
        train_avail = pd.to_datetime(df.label_available_at.iloc[f.train], utc=True)
        assert (train_avail < f.boundary).all()
        assert sig.iloc[f.train].max() < sig.iloc[f.test].min()
        assert not set(sig.iloc[f.train]) & set(sig.iloc[f.test])  # a date never straddles
        assert set(f.class_counts) == {"train", "test"}


def test_folds_require_timestamps_and_flag_single_class():
    df = dataset()
    with pytest.raises(ContractError):
        H.purged_grouped_folds(df.drop(columns="label_available_at"), 3)
    naive = df.assign(label_available_at=pd.to_datetime(df.label_available_at).dt.tz_localize(None))
    with pytest.raises(ContractError, match="naive"):
        H.purged_grouped_folds(naive, 3)
    flat = df.assign(win=1.0)
    assert all("train_single_class" in f.problems for f in H.purged_grouped_folds(flat, 3))


def test_transform_fit_on_train_only_and_missing_mask_before_imputation():
    train = pd.DataFrame({"a": [1., 2, 3, np.nan, 100], "b": [5., 5, 5, 5, 5]})
    tf = H.FoldTransform(winsor=None, scaler="standard").fit(train)
    test = pd.DataFrame({"a": [np.nan, 2.0], "b": [5.0, np.inf]})
    out = tf.transform(test)
    assert out["a__missing"].tolist() == [1.0, 0.0] and "b__missing" not in out
    assert np.isfinite(out.to_numpy()).all()
    changed = tf.transform(test.assign(a=[np.nan, 2.0]))
    pd.testing.assert_frame_equal(out, changed)
    # refit on different data gives different params => no hidden refit at inference
    assert tf.sha256() != H.FoldTransform(winsor=None, scaler="standard").fit(train * 2).sha256()
    assert (out["b"] == 0).all()  # constant column: scale guarded, no division blow-up


def test_transform_schema_mismatch_fails_closed_and_serialization_roundtrip():
    X = pd.DataFrame({"a": [1., 2, 3, 4], "b": [4., 3, 2, 1]})
    tf = H.FoldTransform().fit(X)
    with pytest.raises(ContractError, match="schema"):
        tf.transform(X[["b", "a"]])
    with pytest.raises(ContractError, match="schema"):
        tf.transform(X.assign(c=1.0))
    again = H.FoldTransform.from_json(tf.to_json())
    pd.testing.assert_frame_equal(tf.transform(X), again.transform(X))
    batch = tf.transform(X)
    single = pd.concat([tf.transform(X.iloc[[i]]) for i in range(len(X))])
    pd.testing.assert_frame_equal(batch, single)  # batch/single-row parity


def test_winsor_bounds_come_from_train_not_test():
    train = pd.DataFrame({"a": np.arange(100.0)})
    tf = H.FoldTransform(winsor=(.01, .99), scaler=None).fit(train)
    out = tf.transform(pd.DataFrame({"a": [1e9]}))
    assert out.a.iloc[0] == pytest.approx(np.quantile(np.arange(100.0), .99))


BASES = {"logit": lambda: LogisticRegression(max_iter=200),
         "hgb": lambda: HistGradientBoostingClassifier(max_iter=20, random_state=0)}
TF = lambda: H.FoldTransform()


def test_oof_never_covers_first_block_and_missing_component_rejected():
    df = dataset()
    X, y = df[["x1", "x2"]], df["win"]
    folds = H.purged_grouped_folds(df, 3)
    oof, transforms = H.forward_oof(X, y, folds, BASES, TF)
    first_block = set(range(len(df))) - set(oof.row)
    assert first_block and min(oof.row) > 0  # uncovered rows stay uncovered, no in-sample fill
    assert len(transforms) == 3
    with pytest.raises(ContractError, match="missing base"):
        H.component_matrix(oof[oof.component != "hgb"], ["logit", "hgb"])
    with pytest.raises(ContractError, match="missing base"):
        H.predict_blend(LogisticRegression(), {"logit": np.array([.5])}, ["logit", "hgb"])


def test_meta_uses_only_labels_available_at_cutoff():
    df = dataset()
    X, y = df[["x1", "x2"]], df["win"]
    oof, _ = H.forward_oof(X, y, H.purged_grouped_folds(df, 3), BASES, TF)
    comps = list(BASES)
    late = pd.Timestamp("2030-01-01", tz=TZ)
    _, n_all = H.fit_meta(oof, y, df.label_available_at, late, comps, LogisticRegression)
    mid = pd.to_datetime(df.label_available_at, utc=True).sort_values().iloc[len(df) * 3 // 4]
    _, n_mid = H.fit_meta(oof, y, df.label_available_at, mid, comps, LogisticRegression)
    assert 0 < n_mid < n_all
    with pytest.raises(ContractError):
        H.fit_meta(oof, y, df.label_available_at, pd.Timestamp("2000-01-01", tz=TZ), comps, LogisticRegression)


def test_nested_stack_future_mutation_invariance_and_test_labels_unused():
    df = dataset(n_days=80)
    X, y = df[["x1", "x2"]], df["win"]
    outer = H.purged_grouped_folds(df, 3)
    base = H.nested_stack_eval(X, y, df, outer, bases=BASES, meta_factory=LogisticRegression,
                               transform_factory=TF)
    assert {"stack", "fixed_mean", "base_logit", "base_hgb"} <= set(base.columns) and base.fold.nunique() == 3
    # flipping every label of the LAST outer test block cannot change earlier-fold predictions
    last = outer[-1]
    y2 = y.copy()
    y2.iloc[last.test] = 1 - y2.iloc[last.test]
    mutated = H.nested_stack_eval(X, y2, df, outer, bases=BASES, meta_factory=LogisticRegression,
                                  transform_factory=TF)
    earlier = base.fold < last.index
    np.testing.assert_allclose(base[earlier][["stack", "fixed_mean"]].to_numpy(),
                               mutated[earlier][["stack", "fixed_mean"]].to_numpy())
    # outer test labels never reach the fit of their own fold either
    same_fold = base.fold == last.index
    np.testing.assert_allclose(base.loc[same_fold, "stack"], mutated.loc[same_fold, "stack"])


def test_diversity_flags_identical_models():
    df = dataset()
    X, y = df[["x1", "x2"]], df["win"]
    same = {"a": lambda: LogisticRegression(max_iter=200), "b": lambda: LogisticRegression(max_iter=200)}
    oof, _ = H.forward_oof(X, y, H.purged_grouped_folds(df, 3), same, TF)
    assert H.base_diversity(oof, y)["error_corr"]["a"]["b"] == pytest.approx(1.0)


GOOD = {"contract_version": H.CONTRACT_VERSION, "data_available_at": "2026-01-01T10:00:00+07:00",
        "fit_label_available_max": "2026-01-01T09:00:00+07:00", "fitted_at": "2026-01-02T10:00:00+07:00",
        "selected_at": "2026-01-02T11:00:00+07:00", "calibrated_at": "2026-01-02T12:00:00+07:00",
        "evaluated_at": "2026-01-02T13:00:00+07:00", "label_maturity_policy": "matured-only"}


def test_model_contract_requires_every_timestamp_in_order():
    d = pd.Timestamp("2026-01-05", tz=TZ)
    assert H.validate_model_contract(GOOD, d) == []
    assert any("calibrated_at" in p for p in H.validate_model_contract({k: v for k, v in GOOD.items() if k != "calibrated_at"}, d))
    assert any("order" in p for p in H.validate_model_contract({**GOOD, "evaluated_at": "2026-01-02T11:30:00+07:00"}, d))
    assert any("naive" in p for p in H.validate_model_contract({**GOOD, "fitted_at": "2026-01-02T10:00:00"}, d))
    assert "fit_used_unavailable_labels" in H.validate_model_contract(
        {**GOOD, "fit_label_available_max": "2026-01-03T00:00:00+07:00"}, d)
    assert any("after_decision" in p for p in H.validate_model_contract(GOOD, pd.Timestamp("2026-01-02", tz=TZ)))
    assert "contract_version" in H.validate_model_contract({**GOOD, "contract_version": "v1"}, d)
