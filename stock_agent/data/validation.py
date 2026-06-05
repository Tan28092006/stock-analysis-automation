from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from typing import Iterable

import numpy as np
import pandas as pd

from ..schemas import DataQuality, ProviderAudit
from .exchange_calendar import trading_days_between

REQUIRED_COLUMNS = ["date", "open", "high", "low", "close", "volume"]


@dataclass
class ProviderFrame:
    provider: str
    frame: pd.DataFrame | None
    audit: ProviderAudit


def normalize_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=REQUIRED_COLUMNS)

    renamed = {col: str(col).strip().lower() for col in df.columns}
    out = df.rename(columns=renamed).copy()
    alias = {
        "time": "date",
        "tradingdate": "date",
        "datetime": "date",
        "vol": "volume",
        "value": "volume",
    }
    out = out.rename(columns={col: alias.get(col, col) for col in out.columns})

    missing = [col for col in REQUIRED_COLUMNS if col not in out.columns]
    if missing:
        raise ValueError(f"missing columns: {', '.join(missing)}")

    out = out[REQUIRED_COLUMNS].copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.date
    for col in ["open", "high", "low", "close", "volume"]:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out = out.dropna(subset=REQUIRED_COLUMNS)
    out = out.sort_values("date").drop_duplicates(subset=["date"], keep="last")
    return out.reset_index(drop=True)


def validate_ohlcv(df: pd.DataFrame, min_rows: int, today: date, max_age_days: int) -> list[str]:
    warnings: list[str] = []
    if len(df) < min_rows:
        warnings.append(f"insufficient history: {len(df)} < {min_rows}")
    if df.empty:
        warnings.append("empty dataframe")
        return warnings
    if (df[["open", "high", "low", "close"]] <= 0).any().any():
        warnings.append("non-positive price detected")
    if (df["volume"] < 0).any():
        warnings.append("negative volume detected")
    if (df["high"] < df[["open", "close", "low"]].max(axis=1)).any():
        warnings.append("high price inconsistency")
    if (df["low"] > df[["open", "close", "high"]].min(axis=1)).any():
        warnings.append("low price inconsistency")

    latest = df["date"].iloc[-1]
    age = len(trading_days_between(latest + timedelta(days=1), today))
    if age > max_age_days:
        warnings.append(f"stale price: latest={latest}, trading_age_days={age}")
    return warnings


def detect_corporate_action_flags(df: pd.DataFrame) -> list[str]:
    if df is None or df.empty or "close" not in df or "date" not in df:
        return []
    close = df["close"].astype(float)
    jumps = close.pct_change().replace([np.inf, -np.inf], np.nan).dropna()
    flags: list[str] = []
    for idx, pct_change in jumps.items():
        abs_change = abs(float(pct_change))
        if abs_change > 0.15:
            flags.append(f"possible_corporate_action:{df.loc[idx, 'date']}:{pct_change:.2%}")
        elif abs_change >= 0.07:
            flags.append(f"limit_up_or_limit_down:{df.loc[idx, 'date']}:{pct_change:.2%}")
    return flags


def classify_validation_warnings(warnings: list[str]) -> str:
    joined = " ".join(warnings).lower()
    if "stale price" in joined:
        return "stale"
    return "validation_failed"


def _provider_rejection_warning(provider: str, reason: str, warnings: list[str]) -> str:
    detail = "; ".join(warnings) if warnings else "validation failed"
    return f"{provider} rejected: {reason} ({detail})"


def pick_cross_checked_frame(
    provider_frames: Iterable[ProviderFrame],
    rules: dict,
) -> tuple[pd.DataFrame | None, DataQuality]:
    provider_frames = list(provider_frames)
    audits = [item.audit for item in provider_frames]
    cross_rules = rules["cross_check"]
    tolerance = cross_rules["latest_close_tolerance_pct"] / 100.0

    # 1. Filter out empty frames
    non_empty_frames = [item for item in provider_frames if item.frame is not None and not item.frame.empty]
    if not non_empty_frames:
        return None, DataQuality(
            status="failed",
            confidence=0.0,
            primary_provider=None,
            cross_check_status="no_valid_source",
            warnings=["no provider returned valid OHLCV"],
            provider_audits=audits,
            corporate_action_flags=[],
        )

    # 2. Find which ones are valid (pass validate_ohlcv)
    valid_frames = []
    validation_warnings = {}
    rejection_warnings: list[str] = []
    corporate_action_flags: list[str] = []
    for item in non_empty_frames:
        warns = validate_ohlcv(
            item.frame,
            min_rows=rules["min_history_rows"],
            today=date.today(),
            max_age_days=rules["max_price_age_days"],
        )
        corporate_action_flags.extend(detect_corporate_action_flags(item.frame))
        if not warns:
            valid_frames.append(item)
        else:
            reason = classify_validation_warnings(warns)
            item.audit.rejection_reason = reason
            item.audit.error = item.audit.error or "; ".join(warns)
            validation_warnings[item.provider] = warns
            rejection_warnings.append(_provider_rejection_warning(item.provider, reason, warns))

    if not valid_frames:
        # None of the providers passed validation.
        # Fall back to the first non-empty frame and return failed status.
        fallback = non_empty_frames[0]
        fallback_warnings = validation_warnings.get(fallback.provider, ["validation failed"])
        return fallback.frame, DataQuality(
            status="failed",
            confidence=0.0,
            primary_provider=fallback.provider,
            cross_check_status="primary_validation_failed",
            warnings=rejection_warnings or fallback_warnings,
            provider_audits=audits,
            corporate_action_flags=corporate_action_flags,
        )

    # 3. Find the maximum date among all valid frames and filter for the freshest frames
    max_date = max(item.frame["date"].iloc[-1] for item in valid_frames)
    freshest_frames = [item for item in valid_frames if item.frame["date"].iloc[-1] == max_date]
    
    # We prefer online providers for the primary frame if they have the freshest data
    freshest_frames = sorted(
        freshest_frames, 
        key=lambda item: (item.provider == "local_csv", item.provider)
    )
    primary = freshest_frames[0]

    if len(freshest_frames) == 1:
        allow = bool(cross_rules.get("allow_single_source", True))
        return primary.frame, DataQuality(
            status="degraded" if allow else "failed",
            confidence=float(cross_rules.get("single_source_confidence_cap", 0.65)) if allow else 0.0,
            primary_provider=primary.provider,
            cross_check_status="single_source",
            warnings=(rejection_warnings + ["only one valid price source; confidence capped"]),
            provider_audits=audits,
            corporate_action_flags=corporate_action_flags,
        )

    checks: list[str] = []
    primary_close = float(primary.frame["close"].iloc[-1])
    primary_date = primary.frame["date"].iloc[-1]
    for other in freshest_frames[1:]:
        other_close = float(other.frame["close"].iloc[-1])
        pct_diff = abs(primary_close - other_close) / max(primary_close, 1e-9)
        date_gap = abs((primary_date - other.frame["date"].iloc[-1]).days)
        if pct_diff > tolerance:
            checks.append(
                f"{primary.provider} vs {other.provider} close diff {pct_diff:.2%} > {tolerance:.2%}"
            )
            primary.audit.rejection_reason = primary.audit.rejection_reason or "conflict"
            other.audit.rejection_reason = "conflict"
        if date_gap > 3:
            checks.append(f"{primary.provider} vs {other.provider} latest date gap {date_gap} days")
            primary.audit.rejection_reason = primary.audit.rejection_reason or "conflict"
            other.audit.rejection_reason = "conflict"

    if checks:
        return primary.frame, DataQuality(
            status="failed",
            confidence=0.0,
            primary_provider=primary.provider,
            cross_check_status="conflict",
            warnings=rejection_warnings + checks,
            provider_audits=audits,
            corporate_action_flags=corporate_action_flags,
        )

    return primary.frame, DataQuality(
        status="passed",
        confidence=0.92,
        primary_provider=primary.provider,
        cross_check_status="matched",
        warnings=rejection_warnings,
        provider_audits=audits,
        corporate_action_flags=corporate_action_flags,
    )
