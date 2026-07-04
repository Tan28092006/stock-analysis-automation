from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from math import exp
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from ..config import compute_rules_hash
from ..constants import MODEL_DIR, PRICE_CACHE_DIR
from ..data.repository import read_json, write_json
from ..schemas import ModelRun, ModelSignal
from .backtest import BacktestConfig
from .calibration import _add_normalized_features, build_labeled_t2_dataset, feature_columns, selection_metrics
from .ensemble_model import EnsembleTrainer, ENSEMBLE_REGISTRY_PATH


MODEL_REGISTRY_PATH = MODEL_DIR / "model_registry.json"
DEFAULT_MODEL_FAMILIES = ["logistic", "random_forest", "svm", "mlp", "xgboost", "lightgbm", "lstm"]

_cached_ensemble_trainer = None

def clear_ensemble_cache() -> None:
    global _cached_ensemble_trainer
    _cached_ensemble_trainer = None



@dataclass
class TrainingResult:
    model_family: str
    status: str
    artifact_path: str | None = None
    metrics: dict[str, Any] = field(default_factory=dict)
    feature_columns: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def load_model_registry() -> dict[str, Any]:
    return read_json(MODEL_REGISTRY_PATH, default={"models": {}, "selected_model": None})


def model_status() -> dict[str, Any]:
    registry = load_model_registry()
    models = registry.get("models", {})
    for payload in models.values():
        path = payload.get("artifact_path")
        payload["artifact_exists"] = bool(path and Path(path).exists())

    if ENSEMBLE_REGISTRY_PATH.exists():
        try:
            ensemble_data = read_json(ENSEMBLE_REGISTRY_PATH)
            art_path = ensemble_data.get("artifact_path")
            models["ensemble"] = {
                "model_family": "ensemble",
                "status": "trained",
                "artifact_path": art_path,
                "metrics": ensemble_data.get("metrics", {}),
                "feature_columns": ensemble_data.get("feature_columns", []),
                "warnings": [],
                "trained_at": ensemble_data.get("trained_at"),
                "artifact_exists": bool(art_path and Path(art_path).exists()),
            }
            # Update selected_model based on config rule
            from ..config import load_rules
            rules = load_rules()
            configured_family = rules.get("ml", {}).get("model_family")
            if configured_family == "ensemble":
                registry["selected_model"] = "ensemble"
        except Exception as e:
            # We don't have logger imported in the global scope of this file if not defined, but logger is used elsewhere
            # Let's import logging locally just in case
            import logging
            logging.getLogger(__name__).error(f"Failed to load ensemble status: {e}")

    return registry


def train_model_suite(
    symbols: Iterable[str],
    start: date | None,
    end: date | None,
    rules: dict[str, Any],
    families: list[str] | None = None,
    price_dir: Path | None = None,
) -> dict[str, Any]:
    ml_rules = rules.get("ml", {})
    cost_config = BacktestConfig.from_rules(rules)
    dataset = build_labeled_t2_dataset(
        symbols=symbols,
        start=start,
        end=end,
        rules=rules,
        cost_config=cost_config,
        price_dir=price_dir or PRICE_CACHE_DIR,
    )
    return train_model_suite_from_dataset(
        dataset=dataset,
        rules=rules,
        families=families or list(ml_rules.get("families", DEFAULT_MODEL_FAMILIES)),
    )


def train_model_suite_from_dataset(
    dataset: pd.DataFrame,
    rules: dict[str, Any],
    families: list[str] | None = None,
) -> dict[str, Any]:
    clear_ensemble_cache()
    created_at = datetime.now(timezone.utc).isoformat()
    requested = families or list(rules.get("ml", {}).get("families", DEFAULT_MODEL_FAMILIES))
    min_rows = int(rules.get("ml", {}).get("min_train_rows", 40))
    rules_version = compute_rules_hash(rules)
    warnings: list[str] = []
    results: list[TrainingResult] = []

    if dataset.empty:
        payload = {
            "created_at": created_at,
            "rules_version": rules_version,
            "status": "insufficient_data",
            "selected_model": None,
            "dataset": {"rows": 0, "symbols": [], "date_range": {"start": None, "end": None}},
            "models": {},
            "warnings": ["no labeled rows available"],
        }
        write_json(MODEL_REGISTRY_PATH, payload)
        return payload

    working = dataset.dropna(subset=["net_t2_win", "net_t2_return_pct"]).copy()
    cols = feature_columns(working)
    working = working.sort_values(["signal_date", "symbol"]).reset_index(drop=True)
    dataset_summary = {
        "rows": int(len(working)),
        "symbols": sorted(working["symbol"].dropna().unique().tolist()) if "symbol" in working else [],
        "date_range": {
            "start": str(working["signal_date"].min()) if "signal_date" in working and not working.empty else None,
            "end": str(working["signal_date"].max()) if "signal_date" in working and not working.empty else None,
        },
        "feature_columns": cols,
    }

    if len(working) < min_rows or len(cols) < 2 or working["net_t2_win"].nunique() < 2:
        warnings.append("not enough labeled rows, feature columns, or label classes for ML training")
        payload = {
            "created_at": created_at,
            "rules_version": rules_version,
            "status": "insufficient_data",
            "selected_model": None,
            "dataset": dataset_summary,
            "models": {},
            "warnings": warnings,
        }
        write_json(MODEL_REGISTRY_PATH, payload)
        return payload

    train, validation, test = _time_split(working)
    if train.empty or validation.empty or test.empty or train["net_t2_win"].nunique() < 2:
        warnings.append("time split produced insufficient train/validation/test rows")
        payload = {
            "created_at": created_at,
            "rules_version": rules_version,
            "status": "insufficient_data",
            "selected_model": None,
            "dataset": dataset_summary,
            "models": {},
            "warnings": warnings,
        }
        write_json(MODEL_REGISTRY_PATH, payload)
        return payload

    threshold = float(rules.get("ml", {}).get("probability_threshold", 0.58))
    for family in requested:
        results.append(_train_one_family(family, train, validation, test, cols, threshold, created_at))

    selected = _select_model(results)
    models = {item.model_family: _training_result_payload(item) for item in results}
    payload = {
        "created_at": created_at,
        "rules_version": rules_version,
        "status": "trained" if selected else "no_model_trained",
        "selected_model": selected,
        "dataset": dataset_summary,
        "models": models,
        "warnings": warnings,
    }
    write_json(MODEL_REGISTRY_PATH, payload)
    return payload


def predict_model_signal(symbol: str, signal: Any, rules: dict[str, Any]) -> ModelSignal:
    ml_rules = rules.get("ml", {})
    if not bool(ml_rules.get("enabled", False)):
        return ModelSignal(status="disabled", detail="ML layer disabled by config")

    configured_family = str(ml_rules.get("model_family") or "").strip()

    # --- Ensemble path ---
    if configured_family == "ensemble":
        return _predict_ensemble(symbol, signal, rules, ml_rules)

    # --- Legacy single-model path ---
    registry = load_model_registry()
    family = configured_family or registry.get("selected_model")
    model_meta = registry.get("models", {}).get(family) if family else None
    if not family or not model_meta:
        return ModelSignal(status="unavailable", detail="No trained ML model found", warnings=["run train first"])
    if model_meta.get("status") != "trained":
        return ModelSignal(status="unavailable", model_family=family, detail="Configured model is not trained")

    artifact_path = model_meta.get("artifact_path")
    if not artifact_path or not Path(artifact_path).exists():
        return ModelSignal(
            status="unavailable",
            model_family=family,
            detail="Model artifact is missing",
            warnings=[str(artifact_path or "")],
        )

    try:
        artifact = _load_artifact(Path(artifact_path))
        cols = list(artifact["feature_columns"])
        row = _feature_row_from_signal(signal)
        frame = pd.DataFrame([{col: row.get(col, 0.0) for col in cols}]).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        probability = _predict_probability(artifact["model"], frame)
    except Exception as exc:
        return ModelSignal(
            status="error",
            model_family=family,
            detail=f"ML prediction failed: {exc}",
            warnings=[str(exc)],
        )

    threshold = float(ml_rules.get("probability_threshold", artifact.get("threshold", 0.58)))
    passed = probability >= threshold
    confidence_delta = max(-0.08, min(0.06, (probability - threshold) * 0.12))
    model_run = ModelRun(
        model_family=family,
        artifact_path=artifact_path,
        trained_at=model_meta.get("trained_at"),
        status=model_meta.get("status", "trained"),
        metrics=model_meta.get("metrics", {}),
        feature_columns=cols,
        warnings=model_meta.get("warnings", []),
    )
    return ModelSignal(
        status="available",
        model_family=family,
        probability=round(float(probability), 6),
        threshold=round(threshold, 6),
        passed=passed,
        confidence_delta=round(float(confidence_delta), 6),
        detail="ML probability is advisory evidence; deterministic rules remain the guardrail",
        model_run=model_run,
        feature_values={col: float(frame.iloc[0][col]) for col in cols[:30]},
    )


def _predict_ensemble(symbol: str, signal: Any, rules: dict, ml_rules: dict) -> ModelSignal:
    """Predict using the stacking ensemble model."""
    if not ENSEMBLE_REGISTRY_PATH.exists():
        # Fallback: no ensemble trained yet, try legacy models
        return ModelSignal(
            status="unavailable",
            model_family="ensemble",
            detail="No ensemble model trained. Run 'daily-run' or 'train' first.",
            warnings=["ensemble not trained; run daily-run --force"],
        )

    global _cached_ensemble_trainer
    try:
        if _cached_ensemble_trainer is None:
            _cached_ensemble_trainer = EnsembleTrainer.load()
        trainer = _cached_ensemble_trainer
    except Exception as exc:
        return ModelSignal(
            status="error",
            model_family="ensemble",
            detail=f"Failed to load ensemble: {exc}",
            warnings=[str(exc)],
        )

    try:
        row = _feature_row_from_signal(signal)
        return_shap = bool(ml_rules.get("return_shap", False))
        probability, details = trainer.predict_single(row, return_shap=return_shap)
    except Exception as exc:
        return ModelSignal(
            status="error",
            model_family="ensemble",
            detail=f"Ensemble prediction failed: {exc}",
            warnings=[str(exc)],
        )

    # Dynamic threshold adjustment (Symbol-specific + Volatility + Market Regime adjustment)
    base_threshold = float(ml_rules.get("probability_threshold", 0.55))
    
    # Check for static symbol override in config file
    from pathlib import Path
    symbol_thresholds_path = Path("configs/ml_symbol_thresholds.json")
    if symbol_thresholds_path.exists():
        try:
            import json
            with symbol_thresholds_path.open("r") as f:
                thresholds_map = json.load(f)
                if symbol.upper() in thresholds_map:
                    base_threshold = float(thresholds_map[symbol.upper()])
        except Exception:
            pass
            
    # Volatility adjustment (higher volatility -> higher threshold)
    atr_pct = row.get("feature_atr_pct", 0.025)
    vol_adjust = max(-0.04, min(0.04, (atr_pct - 0.025) * 1.5))
    
    # Market Regime adjustment
    regime_trend = row.get("feature_regime_trend", 0.0)
    regime_vol = row.get("feature_regime_vol_class", 1.0)
    
    regime_adjust = 0.0
    if regime_trend == -1.0:     # Bear market
        regime_adjust += 0.08    # Penalty for bear market
    elif regime_trend == 0.0:   # Sideways
        regime_adjust += 0.02
        
    if regime_vol == 2.0:       # High volatility
        regime_adjust += 0.04    # Penalty for high volatility
        
    threshold = base_threshold + vol_adjust + regime_adjust
    threshold = max(0.45, min(0.75, threshold))  # Clamp to sensible bounds
    
    passed = probability >= threshold
    confidence_delta = max(-0.10, min(0.08, (probability - threshold) * 0.15))

    override_enabled = bool(ml_rules.get("override_enabled", False))
    override_detail = ""
    if override_enabled:
        override_detail = f"ML override active (Thresh: {round(threshold, 3)}): can downgrade BUY_SETUP → WATCH"
    else:
        override_detail = f"ML advisory only (Thresh: {round(threshold, 3)}): no override"

    return ModelSignal(
        status="available",
        model_family="ensemble",
        probability=round(probability, 6),
        threshold=round(threshold, 6),
        passed=passed,
        confidence_delta=round(confidence_delta, 6),
        detail=override_detail,
        ensemble_agreement=details.get("ensemble_agreement"),
        shap_top_features=[details.get("shap_top5", {})],
        model_vintage=trainer.trained_at,
        drift_status="healthy",  # Will be updated by daily pipeline
        base_probabilities=details.get("base_probabilities", {}),
        feature_values={k: v for k, v in row.items() if isinstance(v, (int, float)) and k.startswith(("rule_", "feature_"))},
    )


def _train_one_family(
    family: str,
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
    cols: list[str],
    threshold: float,
    created_at: str,
) -> TrainingResult:
    estimator, setup_warning = _build_estimator(family)
    if estimator is None:
        return TrainingResult(family, "skipped", feature_columns=cols, warnings=[setup_warning or "adapter unavailable"])

    try:
        x_train = _feature_frame(train, cols)
        x_validation = _feature_frame(validation, cols)
        x_test = _feature_frame(test, cols)
        estimator.fit(x_train, train["net_t2_win"].astype(int))
        validation_probability = _predict_probability_array(estimator, x_validation)
        test_probability = _predict_probability_array(estimator, x_test)
        metrics = _classification_and_trade_metrics(validation, validation_probability, test, test_probability, threshold)
        artifact_path = _artifact_path(family, created_at)
        _save_artifact(
            artifact_path,
            {
                "model": estimator,
                "feature_columns": cols,
                "family": family,
                "threshold": threshold,
                "trained_at": created_at,
            },
        )
        return TrainingResult(
            model_family=family,
            status="trained",
            artifact_path=str(artifact_path),
            metrics=metrics,
            feature_columns=cols,
        )
    except Exception as exc:
        return TrainingResult(family, "failed", feature_columns=cols, warnings=[str(exc)])


def _build_estimator(family: str):
    if family == "logistic":
        return (
            Pipeline(
                [
                    ("scaler", StandardScaler()),
                    ("model", LogisticRegression(max_iter=1000, random_state=42, solver="liblinear")),
                ]
            ),
            None,
        )
    if family == "random_forest":
        return (
            RandomForestClassifier(
                n_estimators=200,
                max_depth=8,
                min_samples_leaf=3,
                random_state=42,
                class_weight="balanced_subsample",
            ),
            None,
        )
    if family == "svm":
        return (
            Pipeline(
                [
                    ("scaler", StandardScaler()),
                    ("model", SVC(C=1.0, gamma="scale", probability=True, random_state=42)),
                ]
            ),
            None,
        )
    if family == "mlp":
        return (
            Pipeline(
                [
                    ("scaler", StandardScaler()),
                    (
                        "model",
                        MLPClassifier(
                            hidden_layer_sizes=(64, 32),
                            activation="relu",
                            alpha=0.001,
                            max_iter=500,
                            random_state=42,
                            early_stopping=True,
                        ),
                    ),
                ]
            ),
            None,
        )
    if family == "xgboost":
        try:
            from xgboost import XGBClassifier  # type: ignore
        except Exception as exc:
            return None, f"xgboost unavailable: {exc}"
        return (
            XGBClassifier(
                n_estimators=250,
                max_depth=3,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                eval_metric="logloss",
                random_state=42,
            ),
            None,
        )
    if family == "lightgbm":
        try:
            from lightgbm import LGBMClassifier  # type: ignore
        except Exception as exc:
            return None, f"lightgbm unavailable: {exc}"
        return (
            LGBMClassifier(
                n_estimators=250,
                max_depth=5,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                verbose=-1,
            ),
            None,
        )
    if family == "lstm":
        return None, "lstm adapter requires sequence-window training data; skipped for daily tabular MVP"
    return None, f"unknown model family: {family}"


def _classification_and_trade_metrics(
    validation: pd.DataFrame,
    validation_probability: np.ndarray,
    test: pd.DataFrame,
    test_probability: np.ndarray,
    threshold: float,
) -> dict[str, Any]:
    validation_selected = validation_probability >= threshold
    test_selected = test_probability >= threshold
    y_test = test["net_t2_win"].astype(int).to_numpy()
    y_pred = test_selected.astype(int)
    return {
        "threshold": threshold,
        "validation": selection_metrics(validation, validation_selected),
        "test": selection_metrics(test, test_selected),
        "accuracy": round(float(accuracy_score(y_test, y_pred)), 4) if len(y_test) else 0.0,
        "precision": round(float(precision_score(y_test, y_pred, zero_division=0)), 4) if len(y_test) else 0.0,
        "recall": round(float(recall_score(y_test, y_pred, zero_division=0)), 4) if len(y_test) else 0.0,
    }


def _select_model(results: list[TrainingResult]) -> str | None:
    trained = [item for item in results if item.status == "trained"]
    if not trained:
        return None

    def key(item: TrainingResult) -> tuple[float, float, int]:
        test = item.metrics.get("test", {})
        avg_return = float(test.get("avg_net_return_pct") or 0.0)
        win_rate = float(test.get("win_rate") or 0.0)
        trades = int(test.get("selected_trades") or 0)
        return (avg_return, win_rate, trades)

    return max(trained, key=key).model_family


def _training_result_payload(result: TrainingResult) -> dict[str, Any]:
    payload = {
        "model_family": result.model_family,
        "status": result.status,
        "artifact_path": result.artifact_path,
        "metrics": result.metrics,
        "feature_columns": result.feature_columns,
        "warnings": result.warnings,
    }
    payload["trained_at"] = None
    if result.artifact_path:
        payload["trained_at"] = datetime.fromtimestamp(Path(result.artifact_path).stat().st_mtime, tz=timezone.utc).isoformat()
    return payload


def _feature_frame(frame: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    from .calibration import preprocess_features_robust
    processed, _ = preprocess_features_robust(frame, cols)
    return processed


def _time_split(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    n = len(frame)
    train_end = max(1, int(n * 0.6))
    validation_end = max(train_end + 1, int(n * 0.8))
    validation_end = min(validation_end, n - 1)
    return frame.iloc[:train_end], frame.iloc[train_end:validation_end], frame.iloc[validation_end:]


def _feature_row_from_signal(signal: Any) -> dict[str, Any]:
    row: dict[str, Any] = {}
    for ev in getattr(signal, "evidence", []):
        rule_name = ev.evidence_id.split(":", 1)[-1]
        row[f"rule_{rule_name}"] = int(bool(ev.passed))
    for key, value in getattr(signal, "features", {}).items():
        if isinstance(value, (int, float, np.integer, np.floating)) and np.isfinite(float(value)):
            row[f"feature_{key}"] = float(value)
    _add_normalized_features(row)
    return row


def _predict_probability(model: Any, frame: pd.DataFrame) -> float:
    return float(_predict_probability_array(model, frame)[0])


def _predict_probability_array(model: Any, frame: pd.DataFrame) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        values = model.predict_proba(frame)
        if values.shape[1] == 1:
            return np.asarray(values[:, 0], dtype=float)
        return np.asarray(values[:, 1], dtype=float)
    if hasattr(model, "decision_function"):
        scores = np.asarray(model.decision_function(frame), dtype=float)
        return np.asarray([1 / (1 + exp(-float(score))) for score in scores], dtype=float)
    return np.asarray(model.predict(frame), dtype=float)


def _artifact_path(family: str, created_at: str) -> Path:
    safe_time = created_at.replace(":", "").replace("+", "Z")
    return MODEL_DIR / f"t2_{family}_{safe_time}.pkl"


def _save_artifact(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump(payload, f)


def _load_artifact(path: Path) -> dict[str, Any]:
    with path.open("rb") as f:
        return pickle.load(f)
