"""R1: source contracts, missing-state semantics and as-of join eligibility.

Every join requires ``available_at <= decision_time`` with a declared
cardinality; ambiguous versions at one cutoff and fan-out are rejected.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np
import pandas as pd

VN_TZ = "Asia/Ho_Chi_Minh"


class MissingState(str, Enum):
    OBSERVED = "observed"
    OBSERVED_ZERO = "observed_zero"  # a real zero is not missing
    MISSING = "missing"
    STALE = "stale"
    NOT_APPLICABLE = "not_applicable"
    INSUFFICIENT_HISTORY = "insufficient_history"


class ContractError(ValueError):
    """Contract violation. Callers must fail closed, never coerce."""


@dataclass(frozen=True)
class SourceContract:
    name: str
    grain: tuple[str, ...]            # business key; vintage is separate
    required: tuple[str, ...]
    numeric: tuple[str, ...] = ()
    nonnegative: tuple[str, ...] = ()
    positive: tuple[str, ...] = ()
    availability: str = "available_at"
    unit: dict[str, str] = field(default_factory=dict)  # column -> declared unit, informational


PRICE_BAR = SourceContract(
    "price_bar_vintage", ("symbol", "session_date", "provider"),
    ("symbol", "session_date", "provider", "vintage", "open", "high", "low", "close",
     "volume", "available_at"),
    numeric=("open", "high", "low", "close", "volume"), nonnegative=("volume",),
    positive=("open", "high", "low", "close"),
    unit={"open": "VND", "high": "VND", "low": "VND", "close": "VND", "volume": "shares"})

FOREIGN_FLOW = SourceContract(
    "foreign_flow_vintage", ("symbol", "session_date", "investor", "trade_scope", "provider"),
    ("symbol", "session_date", "investor", "trade_scope", "provider", "vintage",
     "buy_value", "sell_value", "available_at"),
    numeric=("buy_value", "sell_value"), nonnegative=("buy_value", "sell_value"),
    unit={"buy_value": "VND", "sell_value": "VND"})

REGISTRY = {c.name: c for c in (PRICE_BAR, FOREIGN_FLOW)}


def as_utc(series: pd.Series, name: str = "timestamp") -> pd.Series:
    """Timestamps must carry a timezone; naive values are ambiguous, not assumed."""
    try:
        stamps = [pd.Timestamp(v) for v in series]
    except (TypeError, ValueError) as exc:
        raise ContractError(f"{name}: unparseable timestamp") from exc
    if any(pd.isna(t) for t in stamps):
        raise ContractError(f"{name}: unparseable or missing timestamp")
    if any(t.tzinfo is None for t in stamps):
        raise ContractError(f"{name}: timezone-naive timestamp")
    return pd.Series(pd.DatetimeIndex([t.tz_convert("UTC") for t in stamps]), index=series.index)


def validate_source(frame: pd.DataFrame, contract: SourceContract) -> pd.DataFrame:
    missing = [c for c in contract.required if c not in frame]
    if missing:
        raise ContractError(f"{contract.name}: missing columns {missing}")
    out = frame.copy()
    out[contract.availability] = as_utc(out[contract.availability], contract.availability)
    for col in contract.numeric:
        values = pd.to_numeric(out[col], errors="coerce")
        if not np.isfinite(values.to_numpy(dtype=float)).all():
            raise ContractError(f"{contract.name}.{col}: non-finite or non-numeric value")
        out[col] = values.astype(float)
    for col in contract.nonnegative:
        if (out[col] < 0).any():
            raise ContractError(f"{contract.name}.{col}: negative value")
    for col in contract.positive:
        if (out[col] <= 0).any():
            raise ContractError(f"{contract.name}.{col}: non-positive value")
    if {"open", "high", "low", "close"} <= set(out.columns):
        bad = (out.high < out[["open", "low", "close"]].max(axis=1)) | (
            out.low > out[["open", "high", "close"]].min(axis=1))
        if bad.any():
            raise ContractError(f"{contract.name}: inconsistent OHLC bounds")
    out["session_date"] = pd.to_datetime(out["session_date"], errors="raise").dt.normalize()
    if out.duplicated([*contract.grain, "vintage"]).any():
        raise ContractError(f"{contract.name}: duplicate grain+vintage")
    return out


def select_vintage(frame: pd.DataFrame, contract: SourceContract, cutoff: pd.Timestamp,
                   *, priority: dict[str, int] | None = None) -> pd.DataFrame:
    """One version per business key knowable at ``cutoff`` (latest available_at).

    Ties at the same availability with different payloads are ambiguous -> reject,
    unless an explicit provider priority resolves them.
    """
    cutoff = pd.Timestamp(cutoff)
    if cutoff.tzinfo is None:
        raise ContractError("cutoff must be timezone-aware")
    known = frame[frame[contract.availability] <= cutoff.tz_convert("UTC")].copy()
    if known.empty:
        return known
    value_cols = [c for c in contract.numeric if c in known]
    keys = list(contract.grain)
    if priority is not None:
        known["_prio"] = known["provider"].map(priority).fillna(-1)
    else:
        known["_prio"] = 0
    known = known.sort_values([*keys, contract.availability, "_prio"], kind="stable")
    last = known.groupby(keys, sort=False)[[contract.availability, "_prio"]].transform("max")
    top = known[(known[contract.availability] == last[contract.availability])
                & (known["_prio"] == last["_prio"])]
    dup = top.duplicated(keys, keep=False)
    if dup.any():
        conflict = top[dup].groupby(keys)[value_cols].nunique(dropna=False).gt(1).any(axis=1)
        if conflict.any():
            raise ContractError(f"{contract.name}: ambiguous versions at cutoff for "
                                f"{conflict[conflict].index.tolist()[:3]}")
        top = top.drop_duplicates(keys, keep="last")
    return top.drop(columns="_prio").reset_index(drop=True)


def asof_join(left: pd.DataFrame, right: pd.DataFrame, *, on: list[str],
              decision_col: str, right_value_cols: list[str],
              right_time_col: str = "session_date", available_col: str = "available_at",
              max_staleness: pd.Timedelta | None = None, suffix: str = "") -> pd.DataFrame:
    """Latest right row per left row with ``available_at <= decision_time``.

    Picks the newest event, then its newest vintage. Output carries ``<col>_state``
    (observed / observed_zero / missing / stale) so missing is never zero. Left
    cardinality is preserved exactly; fan-out and ambiguous versions raise.
    """
    if left.duplicated([*on, decision_col]).any():
        raise ContractError("left side has duplicate key+decision_time (fan-out risk)")
    out = left.reset_index(drop=True).copy()
    decision = as_utc(out[decision_col], decision_col)
    rhs = right.copy()
    rhs["_avail"] = as_utc(rhs[available_col], available_col)
    rhs["_event"] = pd.to_datetime(rhs[right_time_col]).dt.tz_localize(None)
    groups = {k: g for k, g in rhs.groupby(on, sort=False)}
    for col in right_value_cols:
        out[col + suffix] = np.nan
        out[col + suffix + "_state"] = MissingState.MISSING.value
    out["asof_event"] = pd.NaT
    for i in range(len(out)):
        key = tuple(out.at[i, c] for c in on)
        sub = groups.get(key)
        if sub is None:
            continue
        ok = sub[sub["_avail"] <= decision.iloc[i]]
        if ok.empty:
            continue
        ok = ok[ok["_event"] == ok["_event"].max()]
        ok = ok[ok["_avail"] == ok["_avail"].max()]
        if len(ok) > 1 and ok[right_value_cols].nunique(dropna=False).gt(1).any():
            raise ContractError("ambiguous right versions at decision time")
        hit = ok.iloc[-1]
        day = decision.iloc[i].tz_convert(VN_TZ).tz_localize(None).normalize()
        stale = max_staleness is not None and (day - hit["_event"]) > max_staleness
        for col in right_value_cols:
            val = hit[col]
            state = (MissingState.STALE if stale else MissingState.MISSING if pd.isna(val)
                     else MissingState.OBSERVED_ZERO if val == 0 else MissingState.OBSERVED)
            out.at[i, col + suffix] = val
            out.at[i, col + suffix + "_state"] = state.value
        out.at[i, "asof_event"] = hit["_event"]
    if len(out) != len(left):
        raise ContractError("as-of join changed row count")
    return out


FORBIDDEN_FEATURE_TOKENS = ("target", "label", "future", "fwd", "forward", "exit_", "pnl",
                            "net_t2", "return_after", "outcome")


def assert_no_target_leak(columns: list[str], *, allowlist: set[str]) -> None:
    """Allowlist is the primary guard; token scan is a second line, not a substitute."""
    unknown = [c for c in columns if c not in allowlist]
    if unknown:
        raise ContractError(f"columns not registered as features: {unknown}")
    bad = [c for c in columns if any(tok in c.lower() for tok in FORBIDDEN_FEATURE_TOKENS)]
    if bad:
        raise ContractError(f"target-like feature names: {bad}")
