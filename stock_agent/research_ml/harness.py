"""R4: purged date folds, fold-fitted transforms, OOF stacking, model timestamp contract.

Everything fit here sees only training-fold rows whose labels were *available*
before the fold boundary. No in-sample prediction ever feeds a meta learner.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from typing import Callable

import numpy as np
import pandas as pd

from .contracts import VN_TZ, ContractError, as_utc

CONTRACT_VERSION = "model-contract-v2"


# ---------------------------------------------------------------- folds
@dataclass
class Fold:
    index: int
    train: np.ndarray            # positional row indices
    test: np.ndarray
    boundary: pd.Timestamp       # first decision time of the test block (UTC)
    purged_rows: int
    train_dates: int
    test_dates: int
    class_counts: dict
    problems: list[str] = field(default_factory=list)

    @property
    def conclusive(self) -> bool:
        return not self.problems


def purged_grouped_folds(frame: pd.DataFrame, n_splits: int, *, min_train_dates: int = 5,
                         min_test_dates: int = 1, label_col: str = "win",
                         first_train_dates: int | None = None,
                         test_block_dates: int | None = None) -> list[Fold]:
    """Expanding date folds; all symbols of a signal date share a fold.

    ``label_available_at`` is mandatory (tz-aware): a train row is kept only if its
    label was known strictly before the test block's first decision time.
    """
    if "label_available_at" not in frame or "signal_date" not in frame:
        raise ContractError("label_available_at and signal_date are mandatory")
    avail = as_utc(frame["label_available_at"], "label_available_at").reset_index(drop=True)
    signal = pd.to_datetime(frame["signal_date"]).dt.normalize().reset_index(drop=True)
    days = np.sort(signal.unique())
    if first_train_dates is None and len(days) < n_splits + 1:
        raise ContractError("not enough independent dates for the requested folds")
    if first_train_dates is not None:
        # Fixed geometry: first_train_dates of history, then consecutive test blocks.
        if not test_block_dates:
            raise ContractError("test_block_dates required with first_train_dates")
        edges = list(range(first_train_dates, len(days), test_block_dates))
        edges = [e for e in edges if e + test_block_dates <= len(days)]
        if not edges:
            raise ContractError("not enough independent dates for fixed fold geometry")
        blocks = [days[:first_train_dates]] + [days[e:e + test_block_dates] for e in edges]
        n_splits = len(edges)
    else:
        blocks = np.array_split(days, n_splits + 1)
    y = frame[label_col].reset_index(drop=True) if label_col in frame else None
    folds = []
    for k in range(1, n_splits + 1):
        test_days, prior_days = blocks[k], np.concatenate(blocks[:k])
        boundary = (pd.Timestamp(test_days[0]).tz_localize(VN_TZ) + pd.Timedelta(hours=16)).tz_convert("UTC")
        in_prior = signal.isin(prior_days)
        keep = in_prior & (avail < boundary)
        test_mask = signal.isin(test_days)
        problems = []
        counts = {}
        if y is not None:
            counts = {"train": y[keep].value_counts().to_dict(), "test": y[test_mask].value_counts().to_dict()}
            if y[keep].nunique() < 2:
                problems.append("train_single_class")
            if y[test_mask].nunique() < 2:
                problems.append("test_single_class")
        if signal[keep].nunique() < min_train_dates:
            problems.append("insufficient_train_dates")
        if len(test_days) < min_test_dates:
            problems.append("insufficient_test_dates")
        folds.append(Fold(k - 1, np.flatnonzero(keep), np.flatnonzero(test_mask), boundary,
                          int((in_prior & ~keep).sum()), int(signal[keep].nunique()), len(test_days),
                          counts, problems))
    return folds


# ---------------------------------------------------------------- transforms
class FoldTransform:
    """Median impute + missing indicators (from the pre-imputation mask) + optional
    winsorise + optional scale. Fit on a training fold; schema mismatch fails closed."""

    def __init__(self, winsor: tuple[float, float] | None = (0.01, 0.99), scaler: str | None = "robust"):
        if scaler not in (None, "standard", "robust"):
            raise ValueError("scaler must be None/standard/robust")
        self.winsor, self.scaler, self.params = winsor, scaler, None

    @staticmethod
    def _clean(X: pd.DataFrame) -> pd.DataFrame:
        out = X.apply(pd.to_numeric, errors="coerce").astype(float)
        return out.mask(~np.isfinite(out))  # +/-inf is missing, never a value

    def fit(self, X: pd.DataFrame) -> "FoldTransform":
        X = self._clean(X)
        med = X.median().fillna(0.0)
        indicators = [c for c in X.columns if X[c].isna().any()]
        Xf = X.fillna(med)
        lo = hi = None
        if self.winsor:
            lo, hi = Xf.quantile(self.winsor[0]), Xf.quantile(self.winsor[1])
            Xf = Xf.clip(lo, hi, axis=1)
        if self.scaler == "standard":
            center, scale = Xf.mean(), Xf.std(ddof=0)
        elif self.scaler == "robust":
            center, scale = Xf.median(), Xf.quantile(.75) - Xf.quantile(.25)
        else:
            center, scale = pd.Series(0.0, index=X.columns), pd.Series(1.0, index=X.columns)
        scale = scale.where(scale.abs() > 1e-12, 1.0)  # constant/near-zero spread columns
        self.params = {"columns": list(X.columns), "median": med.to_dict(), "indicators": indicators,
                       "lo": None if lo is None else lo.to_dict(), "hi": None if hi is None else hi.to_dict(),
                       "center": center.to_dict(), "scale": scale.to_dict(),
                       "winsor": self.winsor, "scaler": self.scaler}
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.params is None:
            raise ContractError("transform not fitted")
        p = self.params
        if list(X.columns) != p["columns"]:
            raise ContractError(f"feature schema mismatch: expected {p['columns']}, got {list(X.columns)}")
        X = self._clean(X)
        mask = X.isna()  # original missing mask BEFORE imputation
        Xf = X.fillna(pd.Series(p["median"]))
        if p["lo"] is not None:
            Xf = Xf.clip(pd.Series(p["lo"]), pd.Series(p["hi"]), axis=1)
        Xf = (Xf - pd.Series(p["center"])) / pd.Series(p["scale"])
        for c in p["indicators"]:
            Xf[f"{c}__missing"] = mask[c].astype(float)
        return Xf

    def to_json(self) -> str:
        return json.dumps(self.params, sort_keys=True)

    def sha256(self) -> str:
        return hashlib.sha256(self.to_json().encode()).hexdigest()

    @classmethod
    def from_json(cls, text: str) -> "FoldTransform":
        p = json.loads(text)
        t = cls(tuple(p["winsor"]) if p["winsor"] else None, p["scaler"])
        t.params = p
        return t


# ---------------------------------------------------------------- OOF + stacking
Factory = Callable[[], object]


def _predict(model, X) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    return np.asarray(model.predict(X), dtype=float)


def forward_oof(X: pd.DataFrame, y: pd.Series, folds: list[Fold], bases: dict[str, Factory],
                transform_factory: Callable[[], FoldTransform]) -> tuple[pd.DataFrame, dict]:
    """Per fold: fit transform + each base on train only; predict the test block.

    Rows in no test block are *uncovered* (reported), never filled in-sample.
    """
    rows, transforms = [], {}
    for f in folds:
        if not f.conclusive:
            continue
        tf = transform_factory().fit(X.iloc[f.train])
        transforms[f.index] = tf
        Xtr, Xte = tf.transform(X.iloc[f.train]), tf.transform(X.iloc[f.test])
        for name, make in bases.items():
            model = make().fit(Xtr, y.iloc[f.train])
            rows.append(pd.DataFrame({"row": f.test, "fold": f.index, "component": name,
                                      "pred": _predict(model, Xte)}))
    oof = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=["row", "fold", "component", "pred"])
    return oof, transforms


def component_matrix(oof: pd.DataFrame, components: list[str]) -> pd.DataFrame:
    wide = oof.pivot_table(index="row", columns="component", values="pred", aggfunc="first")
    absent = [c for c in components if c not in wide]
    if absent:
        raise ContractError(f"missing base components: {absent}")  # no silent 0.5 / reweight
    wide = wide[components]
    if wide.isna().any().any():
        raise ContractError("missing component predictions for some rows")
    return wide


def fit_meta(oof: pd.DataFrame, y: pd.Series, label_available_at: pd.Series, cutoff: pd.Timestamp,
             components: list[str], meta_factory: Factory):
    """Meta learner on OOF rows whose labels were available at ``cutoff``."""
    wide = component_matrix(oof, components)
    avail = as_utc(label_available_at.reset_index(drop=True), "label_available_at")
    ok = avail.iloc[wide.index] <= pd.Timestamp(cutoff).tz_convert("UTC")
    ok.index = wide.index
    wide = wide[ok]
    if wide.empty or y.iloc[wide.index].nunique() < 2:
        raise ContractError("meta training set empty or single-class at cutoff")
    return meta_factory().fit(wide.to_numpy(), y.iloc[wide.index]), len(wide)


def predict_blend(meta, preds: dict[str, np.ndarray], components: list[str]) -> np.ndarray:
    absent = [c for c in components if c not in preds]
    if absent:
        raise ContractError(f"missing base components: {absent}")
    M = np.column_stack([preds[c] for c in components])
    return _predict(meta, M)


def nested_stack_eval(X: pd.DataFrame, y: pd.Series, frame: pd.DataFrame, outer: list[Fold], *,
                      bases: dict[str, Factory], meta_factory: Factory,
                      transform_factory: Callable[[], FoldTransform], inner_splits: int = 2) -> pd.DataFrame:
    """Outer forward test predictions for: each base, fixed mean (control), OOF stack.

    The meta learner sees only inner-fold OOF predictions made inside the outer
    train set; outer test labels never touch any fit.
    """
    comps = list(bases)
    out = []
    for f in outer:
        if not f.conclusive:
            continue
        sub = frame.iloc[f.train].reset_index(drop=True)
        Xs, ys = X.iloc[f.train].reset_index(drop=True), y.iloc[f.train].reset_index(drop=True)
        inner = purged_grouped_folds(sub, inner_splits, min_train_dates=2)
        oof, _ = forward_oof(Xs, ys, inner, bases, transform_factory)
        meta, n_meta = fit_meta(oof, ys, sub["label_available_at"], f.boundary, comps, meta_factory)
        tf = transform_factory().fit(Xs)
        Xtr, Xte = tf.transform(Xs), tf.transform(X.iloc[f.test])
        preds = {n: _predict(make().fit(Xtr, ys), Xte) for n, make in bases.items()}
        block = pd.DataFrame({"row": f.test, "fold": f.index, "y": y.iloc[f.test].to_numpy(),
                              "stack": predict_blend(meta, preds, comps),
                              "fixed_mean": np.mean([preds[c] for c in comps], axis=0), "meta_rows": n_meta,
                              **{f"base_{n}": p for n, p in preds.items()}})
        out.append(block)
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def base_diversity(oof: pd.DataFrame, y: pd.Series) -> dict:
    """Error correlation between bases on OOF rows (identical models show ~1.0)."""
    wide = oof.pivot_table(index="row", columns="component", values="pred", aggfunc="first")
    err = wide.sub(y.iloc[wide.index].to_numpy(), axis=0)
    return {"error_corr": err.corr().round(6).to_dict(), "rows": len(wide)}


# ---------------------------------------------------------------- model timestamps
REQUIRED_TIMESTAMPS = ("data_available_at", "fit_label_available_max", "fitted_at", "selected_at",
                       "calibrated_at", "evaluated_at")


def validate_model_contract(meta: dict, decision_time: pd.Timestamp) -> list[str]:
    """Problems (empty list = contract satisfied; not a profitability claim)."""
    problems = []
    if meta.get("contract_version") != CONTRACT_VERSION:
        problems.append("contract_version")
    stamps = {}
    for key in REQUIRED_TIMESTAMPS:
        try:
            stamps[key] = pd.Timestamp(meta[key])
            if stamps[key].tzinfo is None or pd.isna(stamps[key]):
                raise ValueError
        except (KeyError, TypeError, ValueError):
            problems.append(f"{key}:missing_or_naive")
    if problems:
        return problems
    d = pd.Timestamp(decision_time)
    order = ["data_available_at", "fitted_at", "selected_at", "calibrated_at", "evaluated_at"]
    for a, b in zip(order, order[1:]):
        if stamps[a] > stamps[b]:
            problems.append(f"order:{a}>{b}")
    if stamps["fit_label_available_max"] > stamps["fitted_at"]:
        problems.append("fit_used_unavailable_labels")
    for k, v in stamps.items():
        if v > d:
            problems.append(f"{k}:after_decision_time")
    if not meta.get("label_maturity_policy"):
        problems.append("label_maturity_policy")
    return problems
