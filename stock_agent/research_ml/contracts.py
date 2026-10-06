"""R1: source contracts, missing-state semantics and as-of join eligibility.

Every join requires ``available_at <= decision_time`` with a declared
cardinality; ambiguous versions at one cutoff and fan-out are rejected.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import re

import numpy as np
import pandas as pd

VN_TZ = "Asia/Ho_Chi_Minh"


class MissingState(str, Enum):
    OBSERVED = "observed"
    OBSERVED_ZERO = "observed_zero"  # a real zero is not missing
    MISSING = "missing"
    STALE = "stale"
    NOT_APPLICABLE = "not_applicable"
    INSUFFICIENT_HISTORY = "insufficient_history"  # a.k.a. insufficient_warmup
    INVALID = "invalid"  # value failed validation; excluded, never repaired to zero


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


# --- Ported R1 semantics (earlier parallel R1 implementation) -------------------
# Additive: validate_source/asof_join/select_vintage behaviour above is unchanged.
# Row statuses, worst first: invalid > quarantined > blocked > not_yet_available > eligible.
ELIGIBLE, NOT_YET, BLOCKED, QUARANTINED, INVALID = (
    "eligible", "not_yet_available", "blocked", "quarantined", "invalid")

# Runtime stores without row provenance: never eligible for research datasets.
LEGACY_PATH_PREFIXES = ("data/raw/foreign", "data/raw/prices", "data/raw/prices_hist", "data/pipeline")
LEGACY_FLOW_PROVIDERS = ("ndtnn", "ndtnn_chart", "tudoanh_chart", "price_board_snapshots", "ndtnn_monthly",
                         "ndtnn_quarterly", "tudoanh_monthly", "tudoanh_quarterly")
FOREIGN_FLOW_SCHEMA_VERSION = 2
FOREIGN_FLOW_SCOPE = "provider_chart_aggregate"  # matched_only/unknown are never mixed in
_PATH_COLUMNS = ("source_path", "path", "raw_path")
CLOSE_HOUR = 16  # same buffer as completed_session_date() and features.CLOSE_HOUR


def is_legacy_path(path) -> bool:
    """True for anything stored under a legacy, provenance-less runtime directory."""
    text = str(path).replace("\\", "/").lower()
    return any(re.search(r"(^|/)" + re.escape(p) + r"(/|$)", text) for p in LEGACY_PATH_PREFIXES)


class _Reject(Exception):
    def __init__(self, status: str, reason: str):
        super().__init__(reason)
        self.status, self.reason = status, reason


def _absent(value) -> bool:
    return value is None or (not isinstance(value, str) and pd.isna(value)) or value == ""


def _row_ts(row: dict, name: str) -> pd.Timestamp:
    value = row.get(name)
    if _absent(value):
        raise _Reject(QUARANTINED, f"missing_timestamp:{name}")
    try:
        stamp = pd.Timestamp(value)
    except (TypeError, ValueError):
        raise _Reject(INVALID, f"unparseable_timestamp:{name}") from None
    if stamp.tzinfo is None:
        raise _Reject(INVALID, f"naive_timestamp:{name}")
    return stamp


def _check_flow_row(row: dict) -> None:
    for col in _PATH_COLUMNS:
        if not _absent(row.get(col)) and is_legacy_path(row[col]):
            raise _Reject(QUARANTINED, "legacy_storage_path")
    if _absent(row.get("schema_version")) or row["schema_version"] != FOREIGN_FLOW_SCHEMA_VERSION:
        raise _Reject(QUARANTINED, "schema_version_not_2")
    if (not _absent(row.get("research_only")) and bool(row["research_only"])) or \
            str(row.get("quality") or "").startswith("legacy"):
        raise _Reject(QUARANTINED, "legacy_research_only_row")
    if row.get("provider") in LEGACY_FLOW_PROVIDERS:
        raise _Reject(QUARANTINED, "legacy_provider")
    if row.get("trade_scope") != FOREIGN_FLOW_SCOPE:
        raise _Reject(QUARANTINED, "trade_scope_not_contracted")
    observed, available = _row_ts(row, "observed_at"), _row_ts(row, "available_at")
    sha = row.get("raw_sha256")
    if not isinstance(sha, str) or not re.fullmatch(r"[0-9a-f]{64}", sha):
        raise _Reject(QUARANTINED, "missing_provenance:raw_sha256")
    close = session_close(row.get("session_date"))
    if available < observed:
        raise _Reject(INVALID, "available_before_observed")
    if available < close:
        raise _Reject(INVALID, "available_before_session_close")
    if observed < close:
        raise _Reject(INVALID, "observed_before_session_close")
    buy, sell = _row_value(row, "buy_value"), _row_value(row, "sell_value")
    net = row.get("net_value")
    if not _absent(net) and not np.isclose(float(net), buy - sell, rtol=0, atol=1e-3):
        raise _Reject(INVALID, "inconsistent_net_value")


def session_close(session_date) -> pd.Timestamp:
    """Conservative EOD availability: 16:00 Asia/Ho_Chi_Minh on the session date."""
    try:
        day = pd.Timestamp(session_date)
    except (TypeError, ValueError):
        raise _Reject(INVALID, "invalid_date:session_date") from None
    if pd.isna(day) or day.tzinfo is not None:
        raise _Reject(INVALID, "invalid_date:session_date")
    return day.normalize().tz_localize(VN_TZ) + pd.Timedelta(hours=CLOSE_HOUR)


def _row_value(row: dict, name: str) -> float:
    value = row.get(name)
    if _absent(value) or isinstance(value, bool):
        raise _Reject(INVALID, f"missing_value_in_observed_row:{name}")  # never zero
    try:
        value = float(value)
    except (TypeError, ValueError):
        raise _Reject(INVALID, f"non_numeric:{name}") from None
    if not np.isfinite(value) or value < 0:
        raise _Reject(INVALID, f"invalid_value:{name}")
    return value


def screen_foreign_flow(frame: pd.DataFrame) -> pd.DataFrame:
    """Per-row FOREIGN_FLOW eligibility; adds ``contract_status`` and ``contract_reason``.

    Legacy/unprovenanced rows are QUARANTINED (never eligible, never coerced);
    structurally broken rows are INVALID. Missing contract columns fail closed.
    """
    missing = [c for c in FOREIGN_FLOW.required if c not in frame]
    if missing:
        raise ContractError(f"{FOREIGN_FLOW.name}: missing columns {missing}")
    out = frame.copy()
    status, reason = [], []
    for row in out.to_dict("records"):
        try:
            _check_flow_row(row)
        except _Reject as exc:
            status.append(exc.status)
            reason.append(exc.reason)
        else:
            status.append(ELIGIBLE)
            reason.append("")
    out["contract_status"], out["contract_reason"] = status, reason
    return out


def eligible_foreign_flow(frame: pd.DataFrame) -> pd.DataFrame:
    """Only contract-eligible rows, then validated by ``validate_source(FOREIGN_FLOW)``."""
    screened = screen_foreign_flow(frame)
    keep = screened[screened.contract_status == ELIGIBLE].drop(columns=["contract_status", "contract_reason"])
    if keep.empty:
        return keep.reset_index(drop=True)
    return validate_source(keep.reset_index(drop=True), FOREIGN_FLOW)


_VALUE_STATES = {MissingState.OBSERVED.value, MissingState.OBSERVED_ZERO.value}
_ABSENT_STATES = {MissingState.MISSING.value, MissingState.NOT_APPLICABLE.value,
                  MissingState.INSUFFICIENT_HISTORY.value, MissingState.INVALID.value}


def validate_feature_states(frame: pd.DataFrame, feature_ids: list[str]) -> None:
    """``<feature>_state`` must explain every absent value and never label a value as missing.

    A finite value with an absent-state (zero-fill) or a non-finite value without one
    is a contract violation. ``stale`` may carry its last observed value.
    """
    known = {m.value for m in MissingState}
    for fid in feature_ids:
        col = fid + "_state"
        if col not in frame:
            raise ContractError(f"{fid}: missing {col} column")
        states = frame[col].astype(str)
        unknown = sorted(set(states) - known)
        if unknown:
            raise ContractError(f"{fid}: unknown state {unknown}")
        values = pd.to_numeric(frame[fid], errors="coerce").to_numpy(dtype=float)
        finite = np.isfinite(values)
        if (finite & states.isin(_ABSENT_STATES).to_numpy()).any():
            raise ContractError(f"{fid}: missing state coded as a value (zero-fill)")
        if (~finite & states.isin(_VALUE_STATES).to_numpy()).any():
            raise ContractError(f"{fid}: unexplained missing value marked observed")
        if (finite & (states == MissingState.OBSERVED_ZERO.value).to_numpy() & (values != 0)).any():
            raise ContractError(f"{fid}: observed_zero state with a non-zero value")


# --- Membership (VN30 / VN100 dated timelines in configs/research) ---------------
MEMBERSHIP_INDEXES = ("VN30", "VN100")


@dataclass(frozen=True)
class MembershipResult:
    status: str                       # eligible | blocked | quarantined
    members: frozenset[str] | None    # None unless eligible
    reasons: tuple[str, ...] = ()


def _iso_day(value, name: str) -> pd.Timestamp:
    try:
        day = pd.Timestamp(str(value))
    except (TypeError, ValueError):
        raise ContractError(f"membership: invalid date {name}={value!r}") from None
    if pd.isna(day) or day.tzinfo is not None or day != day.normalize():
        raise ContractError(f"membership: invalid date {name}={value!r}")
    return day


def _has_source(value) -> bool:
    return isinstance(value, str) and bool(value.strip())


def membership_as_of(data: dict, decision_time, *, index: str, execution_date=None) -> MembershipResult:
    """Index members knowable at ``decision_time`` and effective on ``execution_date``.

    An event (and the initial list) is available at 16:00 Asia/Ho_Chi_Minh on its
    ``known_on``; ``effective_on`` is valid time and is applied separately. Decisions
    or executions outside ``coverage_start..coverage_end`` are BLOCKED: no event
    never means "unchanged" outside documented coverage. A missing source quarantines
    the whole timeline; inconsistent chronology or membership sets raise.
    """
    if index not in MEMBERSHIP_INDEXES or data.get("index", index) != index:
        raise ContractError(f"membership: index {data.get('index', index)!r} is not contracted as {index!r}")
    decision = pd.Timestamp(decision_time)
    if decision.tzinfo is None:
        raise ContractError("membership: decision_time must be timezone-aware")
    start, end = _iso_day(data["coverage_start"], "coverage_start"), _iso_day(data["coverage_end"], "coverage_end")
    initial_known = _iso_day(data["initial_known_on"], "initial_known_on")
    current, count = set(data["initial_members"]), data["member_count"]
    if (initial_known > start or start > end or count <= 0 or len(current) != count
            or len(current) != len(data["initial_members"])):
        raise ContractError("membership: invalid initial membership or coverage")
    sources_ok = _has_source(data.get("initial_source"))
    events, previous = [], start
    for i, event in enumerate(data["changes"]):
        known, effective = _iso_day(event["known_on"], "known_on"), _iso_day(event["effective_on"], "effective_on")
        added, removed = set(event["add"]), set(event["remove"])
        if not start <= known <= effective <= end or effective <= previous:
            raise ContractError(f"membership: change {i} chronology outside coverage or out of order")
        if (len(added) != len(event["add"]) or len(removed) != len(event["remove"]) or added & current
                or not removed <= current or added & removed or len((current - removed) | added) != count):
            raise ContractError(f"membership: change {i} is not a valid membership change")
        current = (current - removed) | added
        sources_ok &= _has_source(event.get("source"))
        events.append((known.tz_localize(VN_TZ) + pd.Timedelta(hours=CLOSE_HOUR), effective, added, removed))
        previous = effective
    if not sources_ok:
        return MembershipResult(QUARANTINED, None, ("missing_provenance:source",))
    local_day = decision.tz_convert(VN_TZ).tz_localize(None).normalize()
    execution = local_day if execution_date is None else _iso_day(execution_date, "execution_date")
    if execution < local_day:
        raise ContractError("membership: execution_date before decision date")
    if not start <= local_day <= execution <= end:
        return MembershipResult(BLOCKED, None, ("outside_membership_coverage",))
    if initial_known.tz_localize(VN_TZ) + pd.Timedelta(hours=CLOSE_HOUR) > decision:
        return MembershipResult(BLOCKED, None, ("initial_membership_not_yet_known",))
    members = set(data["initial_members"])
    for available_at, effective, added, removed in events:
        if available_at <= decision and effective <= execution:
            members = (members - removed) | added
    return MembershipResult(ELIGIBLE, frozenset(members))
