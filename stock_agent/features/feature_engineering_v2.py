"""Feature Engineering v2 — Cross-Sectional, Regime, and Temporal features.

This module adds three new feature categories that the baseline indicator-only
features cannot capture:

1. **Cross-Sectional**: How does this stock rank vs. VN30 peers *on the same day*?
2. **Regime Detection**: What is the current market state (bull/bear/sideways)?
3. **Temporal**: Day-of-week and monthly seasonal effects.

Usage
-----
>>> from stock_agent.features.feature_engineering_v2 import (
...     add_cross_sectional_features,
...     add_regime_features,
...     add_temporal_features,
...     engineer_targets,
... )
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# 1. Cross-Sectional Features (rank among VN30 peers on the same date)
# ---------------------------------------------------------------------------

def add_cross_sectional_features(
    symbol_frames: dict[str, pd.DataFrame],
) -> dict[str, pd.DataFrame]:
    """Add cross-sectional rank features to each symbol's DataFrame.

    For each trading date, ranks each symbol relative to all VN30 peers on
    the same day.  This produces *relative* features that are inherently
    normalised and capture market-wide context.

    Parameters
    ----------
    symbol_frames : dict[str, pd.DataFrame]
        Map of symbol -> OHLCV DataFrame.  Each DataFrame **must** already
        have indicator columns (rsi14, volume_ratio_20, return_1d, etc.)
        from ``add_indicators()``.

    Returns
    -------
    dict[str, pd.DataFrame]
        Same frames with new columns appended.
    """
    if not symbol_frames:
        return symbol_frames

    # Collect per-date, per-symbol metrics into a single panel
    records: list[dict[str, Any]] = []
    for symbol, df in symbol_frames.items():
        for _, row in df.iterrows():
            records.append({
                "date": row["date"],
                "symbol": symbol,
                "rsi14": _safe(row, "rsi14", 50.0),
                "volume_ratio_20": _safe(row, "volume_ratio_20", 1.0),
                "return_1d": _safe(row, "return_1d", 0.0),
                "close": _safe(row, "close", 0.0),
                "adx14": _safe(row, "adx14", 20.0),
                "macd_hist": _safe(row, "macd_hist", 0.0),
            })

    if not records:
        return symbol_frames

    panel = pd.DataFrame(records)

    # --- Market-level aggregates ---
    market_agg = panel.groupby("date").agg(
        vn30_avg_return_1d=("return_1d", "mean"),
        vn30_median_return_1d=("return_1d", "median"),
        vn30_advance_count=("return_1d", lambda x: int((x > 0).sum())),
        vn30_decline_count=("return_1d", lambda x: int((x < 0).sum())),
        vn30_total_count=("return_1d", "count"),
    ).reset_index()
    market_agg["vn30_advance_decline_ratio"] = (
        market_agg["vn30_advance_count"] /
        market_agg["vn30_total_count"].clip(lower=1)
    )

    # --- Per-symbol ranks (percentile rank 0..1) ---
    rank_cols = ["rsi14", "volume_ratio_20", "return_1d", "adx14", "macd_hist"]
    for col in rank_cols:
        panel[f"cross_{col}_rank"] = panel.groupby("date")[col].rank(pct=True)

    # Merge back into individual symbol frames
    result: dict[str, pd.DataFrame] = {}
    for symbol, df in symbol_frames.items():
        sym_panel = panel[panel["symbol"] == symbol].drop(columns=["symbol"])
        merged = df.merge(sym_panel[["date"] + [f"cross_{c}_rank" for c in rank_cols]],
                          on="date", how="left")
        merged = merged.merge(market_agg, on="date", how="left")
        result[symbol] = merged

    return result


# ---------------------------------------------------------------------------
# 2. Regime Detection Features (market-level state)
# ---------------------------------------------------------------------------

def add_regime_features(df: pd.DataFrame, market_returns: pd.Series | None = None) -> pd.DataFrame:
    """Add market regime detection features.

    Parameters
    ----------
    df : pd.DataFrame
        Single symbol OHLCV+indicators DataFrame.
    market_returns : pd.Series, optional
        Daily returns of a market proxy (e.g., average VN30 return).
        Index must be dates.  If None, uses the symbol's own return_1d.
    """
    out = df.copy()

    # Use market returns if available, otherwise fall back to symbol's own
    if market_returns is not None:
        date_to_ret = market_returns.to_dict()
        mkt_ret = out["date"].map(date_to_ret).fillna(0.0).astype(float)
    else:
        mkt_ret = out.get("return_1d", pd.Series(0.0, index=out.index)).fillna(0.0).astype(float)

    # Regime: Volatility (rolling 20-day std of returns)
    out["regime_volatility_20"] = mkt_ret.rolling(20, min_periods=5).std().fillna(0.0)

    # Classify volatility regime
    vol_q33 = out["regime_volatility_20"].quantile(0.33)
    vol_q67 = out["regime_volatility_20"].quantile(0.67)
    out["regime_vol_class"] = np.select(
        [out["regime_volatility_20"] <= vol_q33, out["regime_volatility_20"] >= vol_q67],
        [0, 2],  # 0=low, 1=medium, 2=high
        default=1,
    )

    # Regime: Trend (EMA crossover on market proxy cumulative return)
    cum_ret = (1 + mkt_ret / 100.0).cumprod()
    ema9 = cum_ret.ewm(span=9, min_periods=5).mean()
    ema21 = cum_ret.ewm(span=21, min_periods=10).mean()
    ema50 = cum_ret.ewm(span=50, min_periods=20).mean()

    out["regime_trend"] = np.select(
        [
            (ema9 > ema21) & (ema21 > ema50),  # Bull
            (ema9 < ema21) & (ema21 < ema50),  # Bear
        ],
        [1, -1],  # 1=bull, -1=bear, 0=sideways
        default=0,
    )

    # Regime: Market breadth (from cross-sectional data if available)
    if "vn30_advance_decline_ratio" in out.columns:
        out["regime_breadth"] = out["vn30_advance_decline_ratio"]
    else:
        out["regime_breadth"] = 0.5

    # Days since regime change
    regime_changes = out["regime_trend"].diff().abs() > 0
    regime_changes.iloc[0] = True  # First row is always a "change"
    groups = regime_changes.cumsum()
    out["days_since_regime_change"] = groups.groupby(groups).cumcount()

    return out


# ---------------------------------------------------------------------------
# 3. Temporal Features
# ---------------------------------------------------------------------------

# VN derivatives expiry dates: 3rd Thursday of each month
def _third_thursday(year: int, month: int) -> int:
    """Return the day-of-month for the 3rd Thursday."""
    import calendar
    cal = calendar.monthcalendar(year, month)
    thursdays = [week[calendar.THURSDAY] for week in cal if week[calendar.THURSDAY] != 0]
    return thursdays[2]  # 0-indexed, 3rd = index 2


def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add day-of-week, month, and days-to-derivative-expiry features."""
    out = df.copy()
    dates = pd.to_datetime(out["date"])

    # Day of week (0=Monday .. 4=Friday) — sin/cos encoding for cyclical
    dow = dates.dt.dayofweek.astype(float)
    out["day_of_week_sin"] = np.sin(2 * math.pi * dow / 5.0)
    out["day_of_week_cos"] = np.cos(2 * math.pi * dow / 5.0)

    # Month (1-12) — sin/cos encoding
    month = dates.dt.month.astype(float)
    out["month_sin"] = np.sin(2 * math.pi * month / 12.0)
    out["month_cos"] = np.cos(2 * math.pi * month / 12.0)

    # Days to derivative expiry (3rd Thursday of current/next month)
    days_to_expiry = []
    for d in dates:
        try:
            expiry_day = _third_thursday(d.year, d.month)
            from datetime import date as dt_date
            expiry = dt_date(d.year, d.month, expiry_day)
            current = d.date() if hasattr(d, "date") else d
            delta = (expiry - current).days
            if delta < 0:
                # Expiry already passed this month — use next month
                next_month = d.month + 1
                next_year = d.year
                if next_month > 12:
                    next_month = 1
                    next_year += 1
                expiry_day = _third_thursday(next_year, next_month)
                expiry = dt_date(next_year, next_month, expiry_day)
                delta = (expiry - current).days
            days_to_expiry.append(delta)
        except Exception:
            days_to_expiry.append(15)  # fallback ~mid-month
    out["days_to_expiry"] = days_to_expiry

    return out


# ---------------------------------------------------------------------------
# 4. Target Engineering (improved labels)
# ---------------------------------------------------------------------------

def engineer_targets(
    df: pd.DataFrame,
    net_return_col: str = "net_return_pct",
    atr_pct_col: str = "atr_pct",
    market_return_col: str = "vn30_avg_return_1d",
) -> pd.DataFrame:
    """Create improved target labels from raw net return.

    Creates:
    - label_binary: 1 if net_return > 0 else 0 (same as current)
    - label_rr_category: 0=big_loss, 1=small_loss, 2=small_win, 3=big_win
    - label_risk_adjusted: net_return / atr_pct (Sharpe-like per trade)
    - label_excess_return: net_return - market_avg_return (alpha)
    """
    out = df.copy()
    ret = out.get(net_return_col, pd.Series(0.0, index=out.index)).fillna(0.0)

    # Binary (unchanged from baseline)
    out["label_binary"] = (ret > 0).astype(int)

    # Multi-class: 4 categories
    out["label_rr_category"] = np.select(
        [
            ret < -2.0,                 # Big loss
            (ret >= -2.0) & (ret < 0),   # Small loss
            (ret >= 0) & (ret < 2.0),    # Small win
            ret >= 2.0,                  # Big win
        ],
        [0, 1, 2, 3],
        default=1,
    )

    # Risk-adjusted return (Sharpe-like per trade)
    atr = out.get(atr_pct_col, pd.Series(1.0, index=out.index)).fillna(1.0).clip(lower=0.1)
    out["label_risk_adjusted"] = ret / atr

    # Excess return (alpha over market)
    mkt_ret = out.get(market_return_col, pd.Series(0.0, index=out.index)).fillna(0.0)
    out["label_excess_return"] = ret - mkt_ret

    return out


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _safe(row: Any, col: str, default: float = 0.0) -> float:
    """Safely extract a float value from a row."""
    val = row.get(col, default) if hasattr(row, "get") else getattr(row, col, default)
    if val is None or (isinstance(val, float) and math.isnan(val)):
        return default
    return float(val)


def get_v2_feature_columns() -> list[str]:
    """Return the list of v2 feature column names for model training."""
    return [
        # Cross-sectional ranks
        "cross_rsi14_rank",
        "cross_volume_ratio_20_rank",
        "cross_return_1d_rank",
        "cross_adx14_rank",
        "cross_macd_hist_rank",
        # Market aggregates
        "vn30_avg_return_1d",
        "vn30_median_return_1d",
        "vn30_advance_decline_ratio",
        # Regime
        "regime_volatility_20",
        "regime_vol_class",
        "regime_trend",
        "regime_breadth",
        "days_since_regime_change",
        # Temporal
        "day_of_week_sin",
        "day_of_week_cos",
        "month_sin",
        "month_cos",
        "days_to_expiry",
    ]
