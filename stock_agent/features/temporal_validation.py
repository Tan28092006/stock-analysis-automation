"""Date-grouped evaluation with label-end purging; no inferred label horizons."""
from __future__ import annotations

from datetime import date
import hashlib
import json

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

from ..data.exchange_calendar import completed_session_date

TRAINING_PROTOCOL = "purged-eod-v2"


def label_times(frame: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    for col in ("signal_date", "exit_date"):
        if col not in frame:
            raise ValueError(f"Missing {col}: label availability must be explicit")
    signal = pd.to_datetime(frame["signal_date"], errors="raise").dt.normalize()
    end = pd.to_datetime(frame["exit_date"], errors="raise").dt.normalize()
    if signal.isna().any() or end.isna().any() or (end < signal).any():
        raise ValueError("Invalid signal_date/exit_date label interval")
    if "label_available_date" in frame:
        available = pd.to_datetime(frame["label_available_date"], errors="raise").dt.normalize()
        if available.isna().any() or (available < end).any():
            raise ValueError("Invalid label_available_date")
        end = available
    return signal, end


def mature_labeled(frame: pd.DataFrame, as_of: date | None = None) -> pd.DataFrame:
    signal, end = label_times(frame)
    cutoff = pd.Timestamp(min(as_of or completed_session_date(), completed_session_date()))
    keep = end <= cutoff
    if "label_resolved" in frame:
        keep &= frame["label_resolved"].eq(True)
    out = frame.loc[keep].copy()
    if "label_quality" in out and out.label_quality.ne("observed_bar_path").any():
        raise ValueError("Unverified label price path: investigate corporate actions before training")
    if "symbol" in out and out.duplicated(["symbol", "signal_date"]).any():
        raise ValueError("Duplicate symbol/signal_date labels")
    return out.sort_values("signal_date", kind="stable").reset_index(drop=True)


def model_available_at(metadata: dict, trained_at: str | None, signal_date) -> bool:
    """Availability is necessary, not proof of PIT inputs or profitable execution."""
    if not metadata or metadata.get("training_protocol") != TRAINING_PROTOCOL:
        return False
    try:
        signal = pd.Timestamp(signal_date).normalize()
        fitted = pd.Timestamp(trained_at)
        if pd.isna(signal) or pd.isna(fitted) or fitted.tzinfo is None:
            return False
        cutoff = signal.tz_localize("Asia/Ho_Chi_Minh") + pd.Timedelta(hours=16)
        label_end = pd.Timestamp(metadata["fit_label_end"])
        calibration_end = pd.Timestamp(metadata.get("calibration_label_end", metadata["fit_label_end"]))
        evaluation_end = pd.Timestamp(metadata["evaluation_label_end"])
        # Evaluation and model-family/threshold selection must also have existed.
        return bool(fitted <= cutoff and label_end < signal and calibration_end < signal and evaluation_end < signal)
    except (TypeError, ValueError, KeyError):
        return False


def purged_time_split(frame: pd.DataFrame, fractions=(.6, .8)) -> tuple[pd.DataFrame, ...]:
    signal, end = label_times(frame)
    days = np.sort(signal.unique())
    if len(days) < len(fractions) + 1:
        return tuple(frame.iloc[:0].copy() for _ in range(len(fractions) + 1))
    cuts = [min(len(days) - 1, max(1, int(len(days) * f))) for f in fractions]
    if len(set(cuts)) != len(cuts):
        return tuple(frame.iloc[:0].copy() for _ in range(len(fractions) + 1))
    bounds = [days[i] for i in cuts]
    result = []
    lower = None
    for upper in [*bounds, None]:
        keep = pd.Series(True, index=frame.index)
        if lower is not None:
            keep &= signal >= lower
        if upper is not None:
            keep &= (signal < upper) & (end < upper)
        result.append(frame.loc[keep].copy())
        lower = upper
    return tuple(result)


def purged_walk_forward(frame: pd.DataFrame, n_splits: int):
    signal, end = label_times(frame)
    days = np.sort(signal.unique())
    if len(days) <= n_splits:
        return
    for tr, val in TimeSeriesSplit(n_splits=n_splits).split(days):
        boundary = days[val[0]]
        train_mask = signal.isin(days[tr]) & (end < boundary)
        validation_mask = signal.isin(days[val])
        # A validation label must be known before the next fold begins as well.
        if val[-1] + 1 < len(days):
            validation_mask &= end < days[val[-1] + 1]
        yield np.flatnonzero(train_mask), np.flatnonzero(validation_mask)


def training_manifest(frame: pd.DataFrame, features: list[str], **extra) -> dict:
    _, end = label_times(frame)
    digest = hashlib.sha256(frame.to_json(orient="split", date_format="iso").encode()).hexdigest()
    return {"training_protocol": TRAINING_PROTOCOL, "dataset_sha256": digest,
            "feature_schema_sha256": hashlib.sha256(json.dumps(features).encode()).hexdigest(),
            "fit_signal_end": str(pd.to_datetime(frame.signal_date).max().date()),
            "fit_label_end": str(end.max().date()), "fit_rows": len(frame), **extra}
