from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from ..constants import FEATURE_DIR
from ..data.repository import write_json


def _safe_float(value: Any, default: float = 0.0) -> float:
    if value is None or pd.isna(value):
        return default
    return float(value)


def build_feature_snapshot(symbol: str, features_df: pd.DataFrame, latest_row: pd.Series | None = None) -> dict[str, Any]:
    latest = latest_row if latest_row is not None else features_df.iloc[-1]
    close = _safe_float(latest.get("close"))
    senkou_a = _safe_float(latest.get("senkou_a"), close)
    senkou_b = _safe_float(latest.get("senkou_b"), close)
    cloud_top = max(senkou_a, senkou_b)
    cloud_bottom = min(senkou_a, senkou_b)
    if close >= cloud_top:
        cloud_position = "above"
    elif close >= cloud_bottom:
        cloud_position = "inside"
    else:
        cloud_position = "below"

    atr_value = _safe_float(latest.get("atr"))
    return {
        "symbol": symbol.upper(),
        "latest_date": str(latest.get("date")),
        "close": close,
        "return_1d": _safe_float(latest.get("return_1d")),
        "log_return_1d": _safe_float(latest.get("log_return_1d")),
        "return_2d": _safe_float(latest.get("return_2d")),
        "return_3d": _safe_float(latest.get("return_3d")),
        "sma5": _safe_float(latest.get("sma5")),
        "sma20": _safe_float(latest.get("sma20")),
        "sma50": _safe_float(latest.get("sma50")),
        "ema9": _safe_float(latest.get("ema9")),
        "ema21": _safe_float(latest.get("ema21")),
        "ema50": _safe_float(latest.get("ema50")),
        "rsi14": _safe_float(latest.get("rsi14"), 50.0),
        "macd": _safe_float(latest.get("macd")),
        "macd_signal": _safe_float(latest.get("macd_signal")),
        "macd_hist": _safe_float(latest.get("macd_hist")),
        "adx14": _safe_float(latest.get("adx14")),
        "plus_di14": _safe_float(latest.get("plus_di14")),
        "minus_di14": _safe_float(latest.get("minus_di14")),
        "vwap20": _safe_float(latest.get("vwap20"), close),
        "obv": _safe_float(latest.get("obv")),
        "obv_slope_5": _safe_float(latest.get("obv_slope_5")),
        "adl": _safe_float(latest.get("adl")),
        "adl_slope_5": _safe_float(latest.get("adl_slope_5")),
        "volume_ratio_20": _safe_float(latest.get("volume_ratio_20")),
        "volume_z20": _safe_float(latest.get("volume_z20")),
        "avg_volume_20": _safe_float(latest.get("avg_volume_20")),
        "value_traded": _safe_float(latest.get("value_traded")),
        "avg_value_20": _safe_float(latest.get("avg_value_20")),
        "spread_proxy_pct": _safe_float(latest.get("spread_proxy_pct")),
        "atr": atr_value,
        "atr_pct": atr_value / close if close else 0.0,
        "bb_width": _safe_float(latest.get("bb_width")),
        "bb_upper": _safe_float(latest.get("bb_upper")),
        "bb_lower": _safe_float(latest.get("bb_lower")),
        "high_20": _safe_float(latest.get("high_20")),
        "low_20": _safe_float(latest.get("low_20")),
        "rolling_mean_5": _safe_float(latest.get("rolling_mean_5")),
        "rolling_mean_20": _safe_float(latest.get("rolling_mean_20")),
        "rolling_std_5": _safe_float(latest.get("rolling_std_5")),
        "rolling_std_20": _safe_float(latest.get("rolling_std_20")),
        "close_lag_1": _safe_float(latest.get("close_lag_1")),
        "close_lag_2": _safe_float(latest.get("close_lag_2")),
        "close_lag_3": _safe_float(latest.get("close_lag_3")),
        "rsi14_lag_1": _safe_float(latest.get("rsi14_lag_1"), 50.0),
        "rsi14_lag_2": _safe_float(latest.get("rsi14_lag_2"), 50.0),
        "rsi14_lag_3": _safe_float(latest.get("rsi14_lag_3"), 50.0),
        "macd_hist_lag_1": _safe_float(latest.get("macd_hist_lag_1")),
        "macd_hist_lag_2": _safe_float(latest.get("macd_hist_lag_2")),
        "macd_hist_lag_3": _safe_float(latest.get("macd_hist_lag_3")),
        "volume_ratio_20_lag_1": _safe_float(latest.get("volume_ratio_20_lag_1")),
        "volume_ratio_20_lag_2": _safe_float(latest.get("volume_ratio_20_lag_2")),
        "volume_ratio_20_lag_3": _safe_float(latest.get("volume_ratio_20_lag_3")),
        "kijun": _safe_float(latest.get("kijun"), close),
        "tenkan": _safe_float(latest.get("tenkan"), close),
        "cloud_position": cloud_position,
    }


def save_feature_snapshot(
    scan_id: str,
    snapshots: list[dict[str, Any]],
    root: Path = FEATURE_DIR,
) -> Path:
    path = root / f"signal_snapshot_{scan_id}.json"
    write_json(path, snapshots)
    return path
