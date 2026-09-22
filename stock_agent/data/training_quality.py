"""Fail-closed training input contract; never repair OHLC by guessing prices."""
from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from .exchange_calendar import completed_session_date


def read_training_prices(path: Path, as_of: date | None = None) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = ["date", "open", "high", "low", "close", "volume"]
    missing = set(required) - set(frame.columns)
    if missing:
        raise ValueError(f"{path.stem}: missing price columns {sorted(missing)}")
    dates = pd.to_datetime(frame.date, errors="coerce")
    if dates.isna().any():
        raise ValueError(f"{path.stem}: invalid price dates")
    cutoff = min(as_of or completed_session_date(), completed_session_date())
    frame = frame.loc[dates.dt.date <= cutoff].copy()
    frame["date"] = dates.loc[frame.index].dt.date
    values = frame[required[1:]].apply(pd.to_numeric, errors="coerce")
    bad = (~np.isfinite(values).all(axis=1) | (values[["open", "high", "low", "close"]] <= 0).any(axis=1)
           | (values.volume < 0) | (values.high < values[["open", "close", "low"]].max(axis=1))
           | (values.low > values[["open", "close", "high"]].min(axis=1))
           | frame.date.duplicated(keep=False))
    if bad.any():
        sample = ", ".join(frame.loc[bad, "date"].astype(str).head(8))
        raise ValueError(f"{path.stem}: {int(bad.sum())} invalid OHLCV/duplicate rows ({sample}); verify source before training")
    frame[required[1:]] = values
    return frame.sort_values("date").reset_index(drop=True)
