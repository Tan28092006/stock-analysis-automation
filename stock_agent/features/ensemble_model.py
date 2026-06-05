"""Stacking Ensemble Model — LightGBM + XGBoost + Ridge → Meta-Learner.

Provides:
- Walk-forward validation (expanding window)
- Incremental daily retraining (warm-start)
- Optuna HPO (weekly)
- Drift detection + auto-disable
- SHAP explanations per prediction

Usage
-----
>>> trainer = EnsembleTrainer.from_config(rules)
>>> trainer.train(dataset)
>>> proba, shap_values = trainer.predict(features_df)
>>> trainer.daily_update(new_rows)
"""

from __future__ import annotations

import json
import logging
import pickle
import warnings as _warnings
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ..constants import MODEL_DIR
from .calibration import selection_metrics

logger = logging.getLogger(__name__)

# Suppress LightGBM/XGBoost verbosity
_warnings.filterwarnings("ignore", category=UserWarning, module="lightgbm")
_warnings.filterwarnings("ignore", category=UserWarning, module="xgboost")

ENSEMBLE_DIR = MODEL_DIR / "ensemble"
ENSEMBLE_REGISTRY_PATH = ENSEMBLE_DIR / "ensemble_registry.json"
DRIFT_LOG_PATH = ENSEMBLE_DIR / "drift_log.jsonl"


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class EnsembleConfig:
    """Configuration for the stacking ensemble."""
    # LightGBM defaults
    lgb_n_estimators: int = 300
    lgb_max_depth: int = 5
    lgb_learning_rate: float = 0.05
    lgb_subsample: float = 0.8
    lgb_colsample_bytree: float = 0.8
    lgb_reg_alpha: float = 0.1
    lgb_reg_lambda: float = 1.0
    lgb_min_child_samples: int = 10

    # XGBoost defaults
    xgb_n_estimators: int = 300
    xgb_max_depth: int = 4
    xgb_learning_rate: float = 0.05
    xgb_subsample: float = 0.8
    xgb_colsample_bytree: float = 0.8
    xgb_reg_alpha: float = 0.1
    xgb_reg_lambda: float = 1.0

    # Ridge defaults
    ridge_alpha: float = 1.0

    # CatBoost defaults
    cat_n_estimators: int = 300
    cat_depth: int = 6
    cat_learning_rate: float = 0.05
    cat_l2_leaf_reg: float = 3.0

    # Meta-learner
    meta_C: float = 1.0

    # Incremental training
    incremental_trees: int = 20  # Trees to add during daily update
    min_train_rows: int = 200
    probability_threshold: float = 0.55

    # Drift detection
    drift_window: int = 20  # Rolling window for drift check
    drift_warning_drop_pct: float = 15.0
    drift_disable_drop_pct: float = 25.0

    # Walk-forward
    n_splits: int = 5  # TimeSeriesSplit folds

    @classmethod
    def from_rules(cls, rules: dict) -> "EnsembleConfig":
        ml = rules.get("ml", {})
        ensemble = ml.get("ensemble", {})
        return cls(
            lgb_n_estimators=ensemble.get("lgb_n_estimators", cls.lgb_n_estimators),
            lgb_max_depth=ensemble.get("lgb_max_depth", cls.lgb_max_depth),
            lgb_learning_rate=ensemble.get("lgb_learning_rate", cls.lgb_learning_rate),
            xgb_n_estimators=ensemble.get("xgb_n_estimators", cls.xgb_n_estimators),
            xgb_max_depth=ensemble.get("xgb_max_depth", cls.xgb_max_depth),
            xgb_learning_rate=ensemble.get("xgb_learning_rate", cls.xgb_learning_rate),
            cat_n_estimators=ensemble.get("cat_n_estimators", cls.cat_n_estimators),
            cat_depth=ensemble.get("cat_depth", cls.cat_depth),
            cat_learning_rate=ensemble.get("cat_learning_rate", cls.cat_learning_rate),
            min_train_rows=int(ml.get("min_train_rows", cls.min_train_rows)),
            probability_threshold=float(ml.get("probability_threshold", cls.probability_threshold)),
            n_splits=ensemble.get("n_splits", cls.n_splits),
            incremental_trees=ensemble.get("incremental_trees", cls.incremental_trees),
        )


# ---------------------------------------------------------------------------
# Drift Detection
# ---------------------------------------------------------------------------

@dataclass
class DriftStatus:
    status: str  # "healthy" | "warning" | "degraded"
    rolling_win_rate: float
    historical_avg_win_rate: float
    drop_pct: float
    recommendation: str  # "normal" | "retrain" | "disable_override"


# ---------------------------------------------------------------------------
# Ensemble Trainer
# ---------------------------------------------------------------------------

class EnsembleTrainer:
    """Stacking ensemble: LightGBM + XGBoost + Ridge → Logistic meta-learner."""

    def __init__(self, config: EnsembleConfig | None = None):
        self.config = config or EnsembleConfig()
        self.lgb_model = None
        self.xgb_model = None
        self.ridge_pipeline = None
        self.cat_model = None
        self.meta_model = None
        self.feature_columns: list[str] = []
        self.trained_at: str | None = None
        self.train_rows: int = 0
        self.metrics: dict[str, Any] = {}
        self._historical_predictions: list[dict] = []

    # -- Build base models ------------------------------------------------

    def _build_lgb(self):
        from lightgbm import LGBMClassifier
        return LGBMClassifier(
            n_estimators=self.config.lgb_n_estimators,
            max_depth=self.config.lgb_max_depth,
            learning_rate=self.config.lgb_learning_rate,
            subsample=self.config.lgb_subsample,
            colsample_bytree=self.config.lgb_colsample_bytree,
            reg_alpha=self.config.lgb_reg_alpha,
            reg_lambda=self.config.lgb_reg_lambda,
            min_child_samples=self.config.lgb_min_child_samples,
            random_state=42,
            verbose=-1,
            n_jobs=-1,
        )

    def _build_xgb(self):
        from xgboost import XGBClassifier
        return XGBClassifier(
            n_estimators=self.config.xgb_n_estimators,
            max_depth=self.config.xgb_max_depth,
            learning_rate=self.config.xgb_learning_rate,
            subsample=self.config.xgb_subsample,
            colsample_bytree=self.config.xgb_colsample_bytree,
            reg_alpha=self.config.xgb_reg_alpha,
            reg_lambda=self.config.xgb_reg_lambda,
            eval_metric="logloss",
            random_state=42,
            n_jobs=-1,
            verbosity=0,
        )

    def _build_ridge(self):
        return Pipeline([
            ("scaler", StandardScaler()),
            ("model", RidgeClassifier(alpha=self.config.ridge_alpha, random_state=42)),
        ])

    def _build_cat(self):
        from catboost import CatBoostClassifier
        return CatBoostClassifier(
            iterations=self.config.cat_n_estimators,
            depth=self.config.cat_depth,
            learning_rate=self.config.cat_learning_rate,
            l2_leaf_reg=self.config.cat_l2_leaf_reg,
            random_seed=42,
            verbose=0,
            thread_count=-1,
        )

    def _build_meta(self):
        return Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(
                C=self.config.meta_C,
                max_iter=1000,
                random_state=42,
                solver="liblinear",
            )),
        ])

    # -- Train (full) -----------------------------------------------------

    def train(
        self,
        dataset: pd.DataFrame,
        feature_cols: list[str],
        label_col: str = "net_t2_win",
    ) -> dict[str, Any]:
        """Full training with walk-forward validation."""
        self.feature_columns = list(feature_cols)
        working = dataset.dropna(subset=[label_col]).copy()
        working = working.sort_values("signal_date").reset_index(drop=True)

        if len(working) < self.config.min_train_rows:
            return {"status": "insufficient_data", "rows": len(working)}

        X = self._prepare_features(working)
        y = working[label_col].astype(int).values

        # Walk-forward: generate OOF (out-of-fold) predictions for meta-learner
        oof_lgb = np.zeros(len(y), dtype=float)
        oof_xgb = np.zeros(len(y), dtype=float)
        oof_ridge = np.zeros(len(y), dtype=float)
        oof_cat = np.zeros(len(y), dtype=float)

        tscv = TimeSeriesSplit(n_splits=self.config.n_splits)
        fold_metrics: list[dict] = []
        val_indices_set = set()

        for fold_idx, (train_idx, val_idx) in enumerate(tscv.split(X)):
            X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_tr, y_val = y[train_idx], y[val_idx]

            if len(np.unique(y_tr)) < 2:
                continue

            val_indices_set.update(val_idx)

            # Train base models on this fold
            lgb = self._build_lgb()
            xgb = self._build_xgb()
            ridge = self._build_ridge()
            cat = self._build_cat()

            lgb.fit(X_tr, y_tr)
            xgb.fit(X_tr, y_tr)
            ridge.fit(X_tr, y_tr)
            cat.fit(X_tr, y_tr, verbose=0)

            # OOF predictions
            oof_lgb[val_idx] = lgb.predict_proba(X_val)[:, 1]
            oof_xgb[val_idx] = xgb.predict_proba(X_val)[:, 1]
            oof_ridge[val_idx] = self._ridge_proba(ridge, X_val)
            oof_cat[val_idx] = cat.predict_proba(X_val)[:, 1]

            # Per-fold metrics
            fold_preds = (oof_lgb[val_idx] + oof_xgb[val_idx] + oof_ridge[val_idx] + oof_cat[val_idx]) / 4.0
            fold_selected = fold_preds >= self.config.probability_threshold
            fold_metrics.append({
                "fold": fold_idx,
                "val_size": len(val_idx),
                "selected": int(fold_selected.sum()),
                "win_rate": float(y_val[fold_selected].mean() * 100) if fold_selected.sum() > 0 else 0.0,
            })

        # Train final base models on ALL data
        self.lgb_model = self._build_lgb()
        self.xgb_model = self._build_xgb()
        self.ridge_pipeline = self._build_ridge()
        self.cat_model = self._build_cat()

        self.lgb_model.fit(X, y)
        self.xgb_model.fit(X, y)
        self.ridge_pipeline.fit(X, y)
        self.cat_model.fit(X, y, verbose=0)

        # Train meta-learner on OOF predictions
        # Only use rows that were in at least one validation fold
        meta_mask = np.zeros(len(y), dtype=bool)
        if val_indices_set:
            meta_mask[list(val_indices_set)] = True

        if meta_mask.sum() < 20:
            # Not enough OOF data — use simple average instead
            self.meta_model = None
            logger.warning("Insufficient OOF data for meta-learner; using simple average")
        else:
            meta_X = np.column_stack([oof_lgb[meta_mask], oof_xgb[meta_mask], oof_ridge[meta_mask], oof_cat[meta_mask]])
            meta_y = y[meta_mask]
            if len(np.unique(meta_y)) >= 2:
                self.meta_model = self._build_meta()
                self.meta_model.fit(meta_X, meta_y)
            else:
                self.meta_model = None

        # Evaluate on last fold (pseudo-test)
        last_fold_val_idx = list(tscv.split(X))[-1][1] if fold_metrics else []
        if len(last_fold_val_idx) > 0:
            test_X = X.iloc[last_fold_val_idx]
            test_y = y[last_fold_val_idx]
            test_proba = self._predict_proba_internal(test_X)
            test_selected = test_proba >= self.config.probability_threshold
            test_frame = working.iloc[last_fold_val_idx]

            self.metrics = {
                "threshold": self.config.probability_threshold,
                "test_size": len(test_y),
                "test_selected": int(test_selected.sum()),
                "test_win_rate": float(test_y[test_selected].mean() * 100) if test_selected.sum() > 0 else 0.0,
                "test_avg_return": float(test_frame.loc[test_selected, "net_t2_return_pct"].mean()) if test_selected.sum() > 0 and "net_t2_return_pct" in test_frame else 0.0,
                "accuracy": float(accuracy_score(test_y, (test_proba >= 0.5).astype(int))),
                "precision": float(precision_score(test_y, (test_proba >= self.config.probability_threshold).astype(int), zero_division=0)),
                "recall": float(recall_score(test_y, (test_proba >= self.config.probability_threshold).astype(int), zero_division=0)),
                "fold_metrics": fold_metrics,
                "ensemble_type": "stacking_lgb_xgb_ridge_cat",
                "meta_learner": "logistic" if self.meta_model else "simple_average",
            }
        else:
            self.metrics = {"status": "no_test_data"}

        self.trained_at = datetime.now(timezone.utc).isoformat()
        self.train_rows = len(working)

        # Compute profit factor from test predictions
        if len(last_fold_val_idx) > 0 and "net_t2_return_pct" in test_frame.columns:
            selected_returns = test_frame.loc[test_selected, "net_t2_return_pct"].tolist()
            gains = sum(r for r in selected_returns if r > 0)
            losses = abs(sum(r for r in selected_returns if r < 0))
            self.metrics["test_profit_factor"] = round(gains / losses, 4) if losses > 0 else None
            self.metrics["test_selection_metrics"] = selection_metrics(test_frame, test_selected)

        return {
            "status": "trained",
            "metrics": self.metrics,
            "train_rows": self.train_rows,
            "feature_count": len(self.feature_columns),
        }

    # -- Incremental daily update -----------------------------------------

    def daily_update(
        self,
        full_dataset: pd.DataFrame,
        label_col: str = "net_t2_win",
    ) -> dict[str, Any]:
        """Incremental retrain: add trees to LightGBM/XGBoost, retrain Ridge+Meta.

        Uses warm-start (init_model for LGB, xgb_model for XGB) to add
        a small number of additional trees on the expanded dataset.
        """
        if self.lgb_model is None or self.xgb_model is None or getattr(self, "cat_model", None) is None:
            return self.train(full_dataset, self.feature_columns, label_col)

        working = full_dataset.dropna(subset=[label_col]).sort_values("signal_date").reset_index(drop=True)
        if len(working) < self.config.min_train_rows:
            return {"status": "insufficient_data", "rows": len(working)}

        X = self._prepare_features(working)
        y = working[label_col].astype(int).values

        if len(np.unique(y)) < 2:
            return {"status": "insufficient_labels", "rows": len(working)}

        # LightGBM warm-start: add incremental trees
        from lightgbm import LGBMClassifier
        lgb_new = LGBMClassifier(
            n_estimators=self.config.incremental_trees,
            max_depth=self.config.lgb_max_depth,
            learning_rate=self.config.lgb_learning_rate * 0.5,  # Lower LR for fine-tuning
            subsample=self.config.lgb_subsample,
            colsample_bytree=self.config.lgb_colsample_bytree,
            reg_alpha=self.config.lgb_reg_alpha,
            reg_lambda=self.config.lgb_reg_lambda,
            min_child_samples=self.config.lgb_min_child_samples,
            random_state=42,
            verbose=-1,
            n_jobs=-1,
        )
        lgb_new.fit(X, y, init_model=self.lgb_model)
        self.lgb_model = lgb_new

        # XGBoost warm-start
        from xgboost import XGBClassifier
        xgb_new = XGBClassifier(
            n_estimators=self.config.incremental_trees,
            max_depth=self.config.xgb_max_depth,
            learning_rate=self.config.xgb_learning_rate * 0.5,
            subsample=self.config.xgb_subsample,
            colsample_bytree=self.config.xgb_colsample_bytree,
            reg_alpha=self.config.xgb_reg_alpha,
            reg_lambda=self.config.xgb_reg_lambda,
            eval_metric="logloss",
            random_state=42,
            n_jobs=-1,
            verbosity=0,
        )
        xgb_new.fit(X, y, xgb_model=self.xgb_model)
        self.xgb_model = xgb_new

        # Ridge: full retrain (fast for linear model)
        self.ridge_pipeline = self._build_ridge()
        self.ridge_pipeline.fit(X, y)

        # CatBoost: full retrain
        self.cat_model = self._build_cat()
        self.cat_model.fit(X, y, verbose=0)

        # Meta-learner: retrain on base predictions
        lgb_proba = self.lgb_model.predict_proba(X)[:, 1]
        xgb_proba = self.xgb_model.predict_proba(X)[:, 1]
        ridge_proba = self._ridge_proba(self.ridge_pipeline, X)
        cat_proba = self.cat_model.predict_proba(X)[:, 1] if self.cat_model else np.full(len(X), 0.5)
        meta_X = np.column_stack([lgb_proba, xgb_proba, ridge_proba, cat_proba])

        if len(np.unique(y)) >= 2:
            self.meta_model = self._build_meta()
            self.meta_model.fit(meta_X, y)

        # Quick validation on last 20% of data
        val_start = int(len(working) * 0.8)
        val_X = X.iloc[val_start:]
        val_y = y[val_start:]
        val_proba = self._predict_proba_internal(val_X)
        val_selected = val_proba >= self.config.probability_threshold

        self.trained_at = datetime.now(timezone.utc).isoformat()
        self.train_rows = len(working)

        update_metrics = {
            "status": "updated",
            "mode": "incremental",
            "total_trees_lgb": self.lgb_model.n_estimators_,
            "total_trees_xgb": self.xgb_model.get_booster().num_boosted_rounds() if hasattr(self.xgb_model, "get_booster") else "unknown",
            "total_trees_cat": self.cat_model.get_all_params().get("iterations") if self.cat_model else "unknown",
            "train_rows": self.train_rows,
            "val_size": len(val_y),
            "val_selected": int(val_selected.sum()),
            "val_win_rate": float(val_y[val_selected].mean() * 100) if val_selected.sum() > 0 else 0.0,
        }
        self.metrics.update(update_metrics)
        return update_metrics

    # -- Predict ----------------------------------------------------------

    def predict(
        self,
        features: pd.DataFrame,
        return_shap: bool = False,
    ) -> tuple[np.ndarray, list[dict] | None]:
        """Predict probabilities and optionally return SHAP values.

        Returns
        -------
        probabilities : np.ndarray
            Win probabilities for each row.
        shap_explanations : list[dict] | None
            Top 5 SHAP features per row (if return_shap=True).
        """
        X = self._prepare_features(features)
        probabilities = self._predict_proba_internal(X)

        shap_explanations = None
        if return_shap:
            shap_explanations = self._compute_shap(X)

        return probabilities, shap_explanations

    def predict_single(self, feature_row: dict[str, float]) -> tuple[float, dict]:
        """Predict for a single stock. Returns (probability, {shap_top5, agreement})."""
        df = pd.DataFrame([{col: feature_row.get(col, 0.0) for col in self.feature_columns}])
        proba, shap_vals = self.predict(df, return_shap=True)

        # Compute ensemble agreement
        X = self._prepare_features(df)
        lgb_p = float(self.lgb_model.predict_proba(X)[:, 1][0]) if self.lgb_model else 0.5
        xgb_p = float(self.xgb_model.predict_proba(X)[:, 1][0]) if self.xgb_model else 0.5
        ridge_p = float(self._ridge_proba(self.ridge_pipeline, X)[0]) if self.ridge_pipeline else 0.5
        cat_p = float(self.cat_model.predict_proba(X)[:, 1][0]) if getattr(self, "cat_model", None) is not None else 0.5
        threshold = self.config.probability_threshold
        agreement = sum(1 for p in [lgb_p, xgb_p, ridge_p, cat_p] if p >= threshold) / 4.0

        return float(proba[0]), {
            "shap_top5": shap_vals[0] if shap_vals else {},
            "ensemble_agreement": round(agreement, 3),
            "base_probabilities": {
                "lightgbm": round(lgb_p, 4),
                "xgboost": round(xgb_p, 4),
                "ridge": round(ridge_p, 4),
                "catboost": round(cat_p, 4),
            },
            "meta_probability": round(float(proba[0]), 4),
        }

    # -- Drift Detection --------------------------------------------------

    def detect_drift(self, recent_outcomes: list[dict]) -> DriftStatus:
        """Check if model performance has degraded.

        Parameters
        ----------
        recent_outcomes : list[dict]
            Each dict has: {"predicted_win": bool, "actual_win": bool}
        """
        if not recent_outcomes or len(recent_outcomes) < 5:
            return DriftStatus("healthy", 0.0, 0.0, 0.0, "insufficient_data")

        window = recent_outcomes[-self.config.drift_window:]
        predicted_wins = [o for o in window if o.get("predicted_win")]
        if not predicted_wins:
            return DriftStatus("healthy", 0.0, 0.0, 0.0, "no_predictions")

        rolling_wr = sum(1 for o in predicted_wins if o.get("actual_win")) / len(predicted_wins) * 100
        hist_wr = self.metrics.get("test_win_rate", 50.0)
        drop = max(0, hist_wr - rolling_wr)

        if drop >= self.config.drift_disable_drop_pct:
            status = "degraded"
            recommendation = "disable_override"
        elif drop >= self.config.drift_warning_drop_pct:
            status = "warning"
            recommendation = "retrain"
        else:
            status = "healthy"
            recommendation = "normal"

        return DriftStatus(
            status=status,
            rolling_win_rate=round(rolling_wr, 2),
            historical_avg_win_rate=round(hist_wr, 2),
            drop_pct=round(drop, 2),
            recommendation=recommendation,
        )

    # -- SHAP Explanations ------------------------------------------------

    def _compute_shap(self, X: pd.DataFrame) -> list[dict]:
        """Compute top-5 SHAP values per row using LightGBM model."""
        try:
            import shap
            explainer = getattr(self, "_explainer", None)
            if explainer is None:
                explainer = shap.TreeExplainer(self.lgb_model)
                self._explainer = explainer
            shap_values = explainer.shap_values(X)
            # For binary classification, shap_values might be a list [class0, class1]
            if isinstance(shap_values, list):
                shap_values = shap_values[1]  # Class 1 (win)
            elif shap_values.ndim == 3:
                shap_values = shap_values[:, :, 1]

            results = []
            for i in range(len(X)):
                row_shap = shap_values[i]
                top_indices = np.argsort(np.abs(row_shap))[-5:][::-1]
                top5 = {
                    self.feature_columns[j]: round(float(row_shap[j]), 4)
                    for j in top_indices
                    if j < len(self.feature_columns)
                }
                results.append(top5)
            return results
        except Exception as exc:
            logger.warning(f"SHAP computation failed: {exc}")
            return [{}] * len(X)

    # -- Persistence ------------------------------------------------------

    def save(self, path: Path | None = None) -> Path:
        """Save the entire ensemble to disk."""
        ENSEMBLE_DIR.mkdir(parents=True, exist_ok=True)
        save_path = path or (ENSEMBLE_DIR / f"ensemble_{self.trained_at or 'unknown'}.pkl".replace(":", ""))
        payload = {
            "lgb_model": self.lgb_model,
            "xgb_model": self.xgb_model,
            "ridge_pipeline": self.ridge_pipeline,
            "cat_model": getattr(self, "cat_model", None),
            "meta_model": self.meta_model,
            "feature_columns": self.feature_columns,
            "config": self.config,
            "trained_at": self.trained_at,
            "train_rows": self.train_rows,
            "metrics": self.metrics,
        }
        with save_path.open("wb") as f:
            pickle.dump(payload, f)

        # Update registry relative to the save path
        registry_path = save_path.parent / "ensemble_registry.json" if path else ENSEMBLE_REGISTRY_PATH
        registry = {
            "type": "ensemble_stacking",
            "trained_at": self.trained_at,
            "artifact_path": str(save_path),
            "train_rows": self.train_rows,
            "feature_columns": self.feature_columns,
            "metrics": self.metrics,
            "config": {
                "lgb_n_estimators": self.config.lgb_n_estimators,
                "xgb_n_estimators": self.config.xgb_n_estimators,
                "cat_n_estimators": self.config.cat_n_estimators,
                "probability_threshold": self.config.probability_threshold,
                "incremental_trees": self.config.incremental_trees,
            },
        }
        with registry_path.open("w", encoding="utf-8") as f:
            json.dump(registry, f, ensure_ascii=False, indent=2)

        return save_path

    @classmethod
    def load(cls, path: Path | None = None) -> "EnsembleTrainer":
        """Load ensemble from disk."""
        load_path = path
        if load_path is None:
            if not ENSEMBLE_REGISTRY_PATH.exists():
                raise FileNotFoundError("No ensemble registry found")
            with ENSEMBLE_REGISTRY_PATH.open() as f:
                registry = json.load(f)
            load_path = Path(registry["artifact_path"])

        with load_path.open("rb") as f:
            payload = pickle.load(f)

        trainer = cls(config=payload.get("config", EnsembleConfig()))
        trainer.lgb_model = payload["lgb_model"]
        trainer.xgb_model = payload["xgb_model"]
        trainer.ridge_pipeline = payload["ridge_pipeline"]
        trainer.cat_model = payload.get("cat_model")
        trainer.meta_model = payload["meta_model"]
        trainer.feature_columns = payload["feature_columns"]
        trainer.trained_at = payload["trained_at"]
        trainer.train_rows = payload["train_rows"]
        trainer.metrics = payload.get("metrics", {})
        return trainer

    # -- Optuna HPO -------------------------------------------------------

    def run_hpo(
        self,
        dataset: pd.DataFrame,
        label_col: str = "net_t2_win",
        n_trials: int = 50,
    ) -> dict[str, Any]:
        """Run Optuna hyperparameter optimization for the stacking ensemble."""
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        working = dataset.dropna(subset=[label_col]).sort_values("signal_date").reset_index(drop=True)
        X = self._prepare_features(working)
        y = working[label_col].astype(int).values

        if len(working) < self.config.min_train_rows or len(np.unique(y)) < 2:
            return {"status": "insufficient_data"}

        def objective(trial):
            lgb_params = {
                "n_estimators": trial.suggest_int("lgb_n_estimators", 100, 400),
                "max_depth": trial.suggest_int("lgb_max_depth", 3, 7),
                "learning_rate": trial.suggest_float("lgb_learning_rate", 0.02, 0.15, log=True),
                "subsample": trial.suggest_float("lgb_subsample", 0.7, 1.0),
                "colsample_bytree": trial.suggest_float("lgb_colsample", 0.7, 1.0),
            }
            xgb_params = {
                "n_estimators": trial.suggest_int("xgb_n_estimators", 100, 400),
                "max_depth": trial.suggest_int("xgb_max_depth", 3, 6),
                "learning_rate": trial.suggest_float("xgb_learning_rate", 0.02, 0.15, log=True),
                "subsample": trial.suggest_float("xgb_subsample", 0.7, 1.0),
                "colsample_bytree": trial.suggest_float("xgb_colsample", 0.7, 1.0),
            }
            cat_params = {
                "iterations": trial.suggest_int("cat_n_estimators", 100, 400),
                "depth": trial.suggest_int("cat_depth", 4, 7),
                "learning_rate": trial.suggest_float("cat_learning_rate", 0.02, 0.15, log=True),
            }
            prob_threshold = trial.suggest_float("probability_threshold", 0.45, 0.62)

            from lightgbm import LGBMClassifier
            from xgboost import XGBClassifier
            from catboost import CatBoostClassifier

            tscv = TimeSeriesSplit(n_splits=3)
            fold_scores = []

            for train_idx, val_idx in tscv.split(X):
                X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
                y_tr, y_val = y[train_idx], y[val_idx]

                if len(np.unique(y_tr)) < 2:
                    continue

                # Build base models for this trial
                lgb = LGBMClassifier(**lgb_params, random_state=42, verbose=-1, n_jobs=1)
                xgb = XGBClassifier(**xgb_params, eval_metric="logloss", random_state=42, n_jobs=1, verbosity=0)
                ridge = self._build_ridge()
                cat = CatBoostClassifier(**cat_params, random_seed=42, verbose=0, thread_count=1)

                lgb.fit(X_tr, y_tr)
                xgb.fit(X_tr, y_tr)
                ridge.fit(X_tr, y_tr)
                cat.fit(X_tr, y_tr, verbose=0)

                # Predictions
                lgb_p = lgb.predict_proba(X_val)[:, 1]
                xgb_p = xgb.predict_proba(X_val)[:, 1]
                ridge_p = self._ridge_proba(ridge, X_val)
                cat_p = cat.predict_proba(X_val)[:, 1]

                ensemble_p = (lgb_p + xgb_p + ridge_p + cat_p) / 4.0
                selected = ensemble_p >= prob_threshold

                if selected.sum() == 0:
                    fold_scores.append(0.0)
                    continue

                if "net_t2_return_pct" in working.columns:
                    returns = working.iloc[val_idx].loc[selected, "net_t2_return_pct"].tolist()
                    gains = sum(r for r in returns if r > 0)
                    losses = abs(sum(r for r in returns if r < 0))
                    # Optimize for Profit Factor with a slight penalty for low trade count
                    pf = gains / losses if losses > 0 else (gains if gains > 0 else 0.0)
                    # We multiply by a log trade count factor to encourage placing enough trades
                    trade_factor = np.log1p(selected.sum())
                    fold_scores.append(pf * trade_factor)
                else:
                    wr = y_val[selected].mean()
                    fold_scores.append(wr)

            return np.mean(fold_scores) if fold_scores else 0.0

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

        best = study.best_params
        
        # Apply the optimized parameters to config
        self.config.lgb_n_estimators = best.get("lgb_n_estimators", self.config.lgb_n_estimators)
        self.config.lgb_max_depth = best.get("lgb_max_depth", self.config.lgb_max_depth)
        self.config.lgb_learning_rate = best.get("lgb_learning_rate", self.config.lgb_learning_rate)
        self.config.lgb_subsample = best.get("lgb_subsample", self.config.lgb_subsample)
        self.config.lgb_colsample_bytree = best.get("lgb_colsample", self.config.lgb_colsample_bytree)
        
        self.config.xgb_n_estimators = best.get("xgb_n_estimators", self.config.xgb_n_estimators)
        self.config.xgb_max_depth = best.get("xgb_max_depth", self.config.xgb_max_depth)
        self.config.xgb_learning_rate = best.get("xgb_learning_rate", self.config.xgb_learning_rate)
        self.config.xgb_subsample = best.get("xgb_subsample", self.config.xgb_subsample)
        self.config.xgb_colsample_bytree = best.get("xgb_colsample", self.config.xgb_colsample_bytree)
        
        self.config.cat_n_estimators = best.get("cat_n_estimators", self.config.cat_n_estimators)
        self.config.cat_depth = best.get("cat_depth", self.config.cat_depth)
        self.config.cat_learning_rate = best.get("cat_learning_rate", self.config.cat_learning_rate)
        
        self.config.probability_threshold = best.get("probability_threshold", self.config.probability_threshold)

        return {
            "status": "completed",
            "n_trials": n_trials,
            "best_value": round(study.best_value, 4),
            "best_params": best,
        }

    # -- Internal helpers -------------------------------------------------

    def _prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract and clean feature columns."""
        cols = [c for c in self.feature_columns if c in df.columns]
        missing = [c for c in self.feature_columns if c not in df.columns]
        frame = df[cols].copy()
        for c in missing:
            frame[c] = 0.0
        frame = frame[self.feature_columns]
        return frame.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    def _predict_proba_internal(self, X: pd.DataFrame) -> np.ndarray:
        """Combine base model predictions through meta-learner."""
        lgb_p = self.lgb_model.predict_proba(X)[:, 1] if self.lgb_model else np.full(len(X), 0.5)
        xgb_p = self.xgb_model.predict_proba(X)[:, 1] if self.xgb_model else np.full(len(X), 0.5)
        ridge_p = self._ridge_proba(self.ridge_pipeline, X) if self.ridge_pipeline else np.full(len(X), 0.5)
        cat_p = self.cat_model.predict_proba(X)[:, 1] if getattr(self, "cat_model", None) is not None else np.full(len(X), 0.5)

        if self.meta_model is not None:
            meta_X = np.column_stack([lgb_p, xgb_p, ridge_p, cat_p])
            return self.meta_model.predict_proba(meta_X)[:, 1]
        else:
            # Simple average fallback
            return (lgb_p + xgb_p + ridge_p + cat_p) / 4.0

    @staticmethod
    def _ridge_proba(pipeline, X: pd.DataFrame) -> np.ndarray:
        """Convert RidgeClassifier decision_function to probabilities via sigmoid."""
        if pipeline is None:
            return np.full(len(X), 0.5)
        if hasattr(pipeline, "predict_proba"):
            return pipeline.predict_proba(X)[:, 1]
        scores = pipeline.decision_function(X)
        return 1.0 / (1.0 + np.exp(-np.clip(scores, -500, 500)))
