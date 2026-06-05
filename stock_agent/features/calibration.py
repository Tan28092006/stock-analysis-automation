from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from math import ceil
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ..config import compute_rules_hash
from ..constants import PRICE_CACHE_DIR
from ..data.exchange_calendar import add_trading_days, next_trading_day
from ..data.validation import detect_corporate_action_flags, normalize_ohlcv, validate_ohlcv
from .backtest import BacktestConfig, enforce_price_limit, round_trip_cost_pct
from .signal_engine import score_symbol


MODEL_NUMERIC_FEATURES = [
    "feature_return_1d",
    "feature_log_return_1d",
    "feature_return_2d",
    "feature_return_3d",
    "feature_rsi14",
    "feature_volume_ratio_20",
    "feature_volume_z20",
    "feature_atr_pct",
    "feature_bb_width",
    "feature_score",
    "feature_macd_hist_pct",
    "feature_adx14",
    "feature_plus_di14",
    "feature_minus_di14",
    "feature_vwap20_gap_pct",
    "feature_obv_slope_5",
    "feature_adl_slope_5",
    "feature_spread_proxy_pct",
    "feature_rolling_std_5",
    "feature_rolling_std_20",
    "feature_ema9_gap_pct",
    "feature_ema21_gap_pct",
    "feature_ema50_gap_pct",
    "feature_sma5_gap_pct",
    "feature_sma20_gap_pct",
    "feature_sma50_gap_pct",
    "feature_high20_room_pct",
    "feature_low20_buffer_pct",
    "feature_bb_upper_room_pct",
    "feature_bb_lower_buffer_pct",
]


@dataclass
class CalibrationReport:
    created_at: str
    rules_version: str
    objective: str
    status: str
    calibration_status: str
    rows_total: int
    rows_used: int
    symbols_used: list[str]
    date_range: dict[str, str | None]
    baseline_metrics: dict[str, Any]
    calibrated_metrics: dict[str, Any]
    test_metrics: dict[str, Any]
    suggested_probability_threshold: float | None
    suggested_rule_weights: dict[str, float]
    indicator_correlation_flags: list[dict[str, Any]]
    warnings: list[str] = field(default_factory=list)
    negative_indicator_flags: list[dict[str, Any]] = field(default_factory=list)
    false_positive_patterns: list[dict[str, Any]] = field(default_factory=list)
    feature_columns: list[str] = field(default_factory=list)


def build_labeled_t2_dataset(
    symbols: Iterable[str],
    start: date | None,
    end: date | None,
    rules: dict[str, Any],
    cost_config: BacktestConfig,
    price_dir: Path = PRICE_CACHE_DIR,
) -> pd.DataFrame:
    from ..agents.parallel_engine import build_labeled_dataset_fast
    return build_labeled_dataset_fast(
        symbols=list(symbols),
        rules=rules,
        price_dir=price_dir,
        start=start,
        end=end,
    )


def feature_columns(dataset: pd.DataFrame) -> list[str]:
    rule_cols = [col for col in dataset.columns if col.startswith("rule_")]
    preferred = [col for col in MODEL_NUMERIC_FEATURES if col in dataset.columns]
    if preferred:
        return rule_cols + preferred
    return rule_cols + [col for col in dataset.columns if col.startswith("feature_")]


def _add_normalized_features(row: dict[str, Any]) -> None:
    close = float(row.get("feature_close") or 0.0)
    if close <= 0:
        return
    pairs = {
        "feature_macd_hist_pct": ("feature_macd_hist", "ratio"),
        "feature_ema9_gap_pct": ("feature_ema9", "gap"),
        "feature_ema21_gap_pct": ("feature_ema21", "gap"),
        "feature_ema50_gap_pct": ("feature_ema50", "gap"),
        "feature_sma5_gap_pct": ("feature_sma5", "gap"),
        "feature_sma20_gap_pct": ("feature_sma20", "gap"),
        "feature_sma50_gap_pct": ("feature_sma50", "gap"),
        "feature_vwap20_gap_pct": ("feature_vwap20", "gap"),
        "feature_high20_room_pct": ("feature_high_20", "room"),
        "feature_low20_buffer_pct": ("feature_low_20", "buffer"),
        "feature_bb_upper_room_pct": ("feature_bb_upper", "room"),
        "feature_bb_lower_buffer_pct": ("feature_bb_lower", "buffer"),
    }
    for target, (source, mode) in pairs.items():
        value = row.get(source)
        if value is None:
            continue
        value = float(value)
        if mode == "ratio":
            row[target] = value / close
        elif mode == "gap":
            row[target] = (close - value) / close
        elif mode == "room":
            row[target] = (value - close) / close
        elif mode == "buffer":
            row[target] = (close - value) / close


def audit_indicator_correlation(
    dataset: pd.DataFrame,
    columns: list[str] | None = None,
    threshold: float = 0.8,
) -> list[dict[str, Any]]:
    cols = columns or feature_columns(dataset)
    if len(cols) < 2 or dataset.empty:
        return []
    frame = dataset[cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    corr = frame.corr()
    flags: list[dict[str, Any]] = []
    for i, left in enumerate(cols):
        for right in cols[i + 1 :]:
            value = corr.loc[left, right]
            if pd.isna(value):
                continue
            if abs(float(value)) >= threshold:
                flags.append(
                    {
                        "feature_a": left,
                        "feature_b": right,
                        "correlation": round(float(value), 4),
                    }
                )
    return sorted(flags, key=lambda item: abs(item["correlation"]), reverse=True)


def meets_trade_floor(selected_count: int, total_count: int, min_trades: int) -> bool:
    if total_count <= 0:
        return False
    return selected_count >= min_trades or selected_count >= max(1, ceil(total_count * 0.05))


def selection_metrics(frame: pd.DataFrame, selected_mask: np.ndarray | pd.Series) -> dict[str, Any]:
    if frame.empty:
        return {
            "candidate_rows": 0,
            "selected_trades": 0,
            "selected_rate_pct": 0.0,
            "win_rate": 0.0,
            "avg_net_return_pct": 0.0,
            "profit_factor": None,
        }
    selected = frame.loc[np.asarray(selected_mask, dtype=bool)]
    returns = selected["net_t2_return_pct"].astype(float).tolist()
    gains = sum(ret for ret in returns if ret > 0)
    losses = abs(sum(ret for ret in returns if ret < 0))
    if not returns:
        profit_factor = None
    elif losses == 0:
        profit_factor = None if gains > 0 else 0.0
    else:
        profit_factor = gains / losses
    return {
        "candidate_rows": int(len(frame)),
        "selected_trades": int(len(selected)),
        "selected_rate_pct": round(len(selected) / len(frame) * 100.0, 3) if len(frame) else 0.0,
        "win_rate": round(float(selected["net_t2_win"].mean() * 100.0), 3) if len(selected) else 0.0,
        "avg_net_return_pct": round(float(selected["net_t2_return_pct"].mean()), 4) if len(selected) else 0.0,
        "profit_factor": round(float(profit_factor), 4) if profit_factor is not None else None,
    }


def optimize_probability_threshold(
    validation_frame: pd.DataFrame,
    probabilities: np.ndarray,
    min_trades: int,
    objective: str = "win_rate",
) -> tuple[float | None, dict[str, Any], list[str]]:
    warnings: list[str] = []
    if validation_frame.empty:
        return None, selection_metrics(validation_frame, np.array([], dtype=bool)), ["empty validation frame"]

    best_threshold: float | None = None
    best_metrics: dict[str, Any] | None = None
    best_key: tuple[float, float, int] | None = None
    for threshold in np.linspace(0.05, 0.95, 91):
        selected = probabilities >= threshold
        metrics = selection_metrics(validation_frame, selected)
        if not meets_trade_floor(metrics["selected_trades"], len(validation_frame), min_trades):
            continue
        if metrics["avg_net_return_pct"] < 0:
            continue
        if objective == "win_rate":
            key = (metrics["win_rate"], metrics["avg_net_return_pct"], metrics["selected_trades"])
        else:
            key = (metrics["avg_net_return_pct"], metrics["win_rate"], metrics["selected_trades"])
        if best_key is None or key > best_key:
            best_key = key
            best_threshold = round(float(threshold), 4)
            best_metrics = metrics

    if best_threshold is None:
        warnings.append("no validation threshold met trade-count and non-negative-return constraints")
        fallback_idx = int(np.argmax(probabilities)) if len(probabilities) else 0
        fallback_threshold = float(probabilities[fallback_idx]) if len(probabilities) else 0.5
        fallback_selected = probabilities >= fallback_threshold
        return round(fallback_threshold, 4), selection_metrics(validation_frame, fallback_selected), warnings
    return best_threshold, best_metrics or {}, warnings


def run_calibration(
    dataset: pd.DataFrame,
    objective: str = "win_rate",
    min_trades: int = 30,
) -> CalibrationReport:
    created_at = datetime.now(timezone.utc).isoformat()
    warnings: list[str] = []
    rows_total = int(len(dataset))
    rules_version = str(dataset["rules_version"].iloc[0]) if rows_total and "rules_version" in dataset else ""
    date_range = _date_range(dataset)
    symbols_used = sorted(dataset["symbol"].dropna().unique().tolist()) if rows_total and "symbol" in dataset else []
    cols = feature_columns(dataset) if rows_total else []
    corr_flags = audit_indicator_correlation(dataset, cols)
    false_patterns = _false_positive_patterns(dataset)

    empty_metrics = selection_metrics(pd.DataFrame(), np.array([], dtype=bool))
    if rows_total == 0:
        return CalibrationReport(
            created_at=created_at,
            rules_version=rules_version,
            objective=objective,
            status="insufficient_data",
            calibration_status="insufficient_data",
            rows_total=0,
            rows_used=0,
            symbols_used=[],
            date_range=date_range,
            baseline_metrics=empty_metrics,
            calibrated_metrics=empty_metrics,
            test_metrics=empty_metrics,
            suggested_probability_threshold=None,
            suggested_rule_weights={},
            indicator_correlation_flags=[],
            warnings=["no labeled rows available"],
            feature_columns=[],
        )

    working = dataset.copy()
    working = working.dropna(subset=["net_t2_win", "net_t2_return_pct"])
    rows_used = int(len(working))
    if rows_used < max(20, min_trades) or len(cols) < 2:
        warnings.append("not enough labeled rows or feature columns for calibration")
        return _report_without_model(
            created_at,
            rules_version,
            objective,
            "insufficient_data",
            rows_total,
            rows_used,
            symbols_used,
            date_range,
            corr_flags,
            false_patterns,
            cols,
            warnings,
        )

    working = working.sort_values(["signal_date", "symbol"]).reset_index(drop=True)
    y = working["net_t2_win"].astype(int)
    if y.nunique() < 2:
        warnings.append("label has only one class")
        return _report_without_model(
            created_at,
            rules_version,
            objective,
            "insufficient_data",
            rows_total,
            rows_used,
            symbols_used,
            date_range,
            corr_flags,
            false_patterns,
            cols,
            warnings,
        )

    train, validation, test = _time_split(working)
    if train.empty or validation.empty or test.empty or train["net_t2_win"].nunique() < 2:
        warnings.append("time split produced insufficient train/validation/test data")
        return _report_without_model(
            created_at,
            rules_version,
            objective,
            "insufficient_data",
            rows_total,
            rows_used,
            symbols_used,
            date_range,
            corr_flags,
            false_patterns,
            cols,
            warnings,
        )

    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=1000, random_state=42, solver="liblinear")),
        ]
    )
    x_train = _feature_frame(train, cols)
    x_validation = _feature_frame(validation, cols)
    x_test = _feature_frame(test, cols)
    model.fit(x_train, train["net_t2_win"].astype(int))
    validation_prob = model.predict_proba(x_validation)[:, 1]
    threshold, calibrated_metrics, optimizer_warnings = optimize_probability_threshold(
        validation,
        validation_prob,
        min_trades=min_trades,
        objective=objective,
    )
    warnings.extend(optimizer_warnings)

    test_prob = model.predict_proba(x_test)[:, 1]
    test_selected = test_prob >= (threshold if threshold is not None else 0.5)
    test_metrics = selection_metrics(test, test_selected)
    baseline_metrics = selection_metrics(test, test["decision"].eq("BUY_SETUP"))

    status = "passed"
    if threshold is None:
        status = "unstable"
    elif not meets_trade_floor(test_metrics["selected_trades"], len(test), min_trades):
        status = "unstable"
        warnings.append("test selection did not meet trade-count floor")
    elif test_metrics["avg_net_return_pct"] < 0:
        status = "unstable"
        warnings.append("test average net return is negative")

    coefficients = model.named_steps["model"].coef_[0]
    weights = _suggested_weights(cols, coefficients)
    negative_flags = _negative_indicator_flags(cols, coefficients)
    return CalibrationReport(
        created_at=created_at,
        rules_version=rules_version,
        objective=objective,
        status=status,
        calibration_status=status,
        rows_total=rows_total,
        rows_used=rows_used,
        symbols_used=symbols_used,
        date_range=date_range,
        baseline_metrics=baseline_metrics,
        calibrated_metrics=calibrated_metrics,
        test_metrics=test_metrics,
        suggested_probability_threshold=threshold,
        suggested_rule_weights=weights,
        indicator_correlation_flags=corr_flags,
        warnings=warnings,
        negative_indicator_flags=negative_flags,
        false_positive_patterns=false_patterns,
        feature_columns=cols,
    )


def _feature_frame(frame: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    return frame[cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)


def _time_split(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    n = len(frame)
    train_end = max(1, int(n * 0.6))
    validation_end = max(train_end + 1, int(n * 0.8))
    validation_end = min(validation_end, n - 1)
    return frame.iloc[:train_end], frame.iloc[train_end:validation_end], frame.iloc[validation_end:]


def _date_range(frame: pd.DataFrame) -> dict[str, str | None]:
    if frame.empty or "signal_date" not in frame:
        return {"start": None, "end": None}
    return {"start": str(frame["signal_date"].min()), "end": str(frame["signal_date"].max())}


def _report_without_model(
    created_at: str,
    rules_version: str,
    objective: str,
    status: str,
    rows_total: int,
    rows_used: int,
    symbols_used: list[str],
    date_range: dict[str, str | None],
    corr_flags: list[dict[str, Any]],
    false_patterns: list[dict[str, Any]],
    cols: list[str],
    warnings: list[str],
) -> CalibrationReport:
    baseline = selection_metrics(pd.DataFrame(), np.array([], dtype=bool))
    return CalibrationReport(
        created_at=created_at,
        rules_version=rules_version,
        objective=objective,
        status=status,
        calibration_status=status,
        rows_total=rows_total,
        rows_used=rows_used,
        symbols_used=symbols_used,
        date_range=date_range,
        baseline_metrics=baseline,
        calibrated_metrics=baseline,
        test_metrics=baseline,
        suggested_probability_threshold=None,
        suggested_rule_weights={},
        indicator_correlation_flags=corr_flags,
        warnings=warnings,
        negative_indicator_flags=[],
        false_positive_patterns=false_patterns,
        feature_columns=cols,
    )


def _suggested_weights(cols: list[str], coefficients: np.ndarray) -> dict[str, float]:
    positives = {col: float(coef) for col, coef in zip(cols, coefficients) if coef > 0 and col.startswith("rule_")}
    total = sum(positives.values())
    if total <= 0:
        return {}
    return {
        col.replace("rule_", ""): round(value / total * 100.0, 3)
        for col, value in sorted(positives.items(), key=lambda item: item[1], reverse=True)
    }


def _negative_indicator_flags(cols: list[str], coefficients: np.ndarray) -> list[dict[str, Any]]:
    flags = [
        {
            "feature": col,
            "coefficient": round(float(coef), 6),
            "reason": "negative logistic coefficient",
        }
        for col, coef in zip(cols, coefficients)
        if coef < 0
    ]
    return sorted(flags, key=lambda item: item["coefficient"])[:10]


def _false_positive_patterns(dataset: pd.DataFrame) -> list[dict[str, Any]]:
    if dataset.empty or "decision" not in dataset or "net_t2_win" not in dataset:
        return []
    false_positive = dataset[dataset["decision"].eq("BUY_SETUP") & dataset["net_t2_win"].eq(0)]
    if false_positive.empty:
        return []
    patterns: list[dict[str, Any]] = []
    for col in [item for item in dataset.columns if item.startswith("rule_")]:
        pass_rate = float(false_positive[col].mean()) if col in false_positive else 0.0
        patterns.append(
            {
                "rule": col.replace("rule_", ""),
                "false_positive_pass_rate": round(pass_rate * 100.0, 3),
                "false_positive_count": int(len(false_positive)),
            }
        )
    return sorted(patterns, key=lambda item: item["false_positive_pass_rate"], reverse=True)[:5]
