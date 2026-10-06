"""Port of R1 semantics from the earlier parallel R1 implementation into research_ml.

Additive only: existing research_ml contract behaviour (validate_source, asof_join,
MissingState values used by R5 bundles) is unchanged.
"""
import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from stock_agent.research_ml import contracts as C
from stock_agent.research_ml.labels import CostConfig, build_labels

TZ = "Asia/Ho_Chi_Minh"


def flow(**over):
    row = dict(symbol="MBB", session_date="2026-10-02", investor="foreign",
               trade_scope="provider_chart_aggregate", provider="vietstock_chart", vintage=1,
               buy_value=1.25e9, sell_value=5e8, net_value=7.5e8, schema_version=2,
               raw_sha256="a" * 64, observed_at="2026-10-02 17:12:00+07:00",
               available_at="2026-10-02 17:12:00+07:00")
    row.update(over)
    return row


def screen(*rows):
    return C.screen_foreign_flow(pd.DataFrame(list(rows)))


# --- 1. legacy provenance quarantine ------------------------------------------

@pytest.mark.parametrize("path", ["data/raw/foreign/ndtnn.jsonl", "data/raw/prices/FPT.csv",
                                  "data\\raw\\prices_hist\\FPT.csv", "data/pipeline/swing_scan_cache.json",
                                  "D:/Chungkhoan/data/raw/foreign/ndtnn_chart.jsonl", "DATA/Pipeline"])
def test_legacy_storage_paths_are_detected(path):
    assert C.is_legacy_path(path)


@pytest.mark.parametrize("path", ["data/foreign_flows_v2/runs/x/records.jsonl",
                                  "tests/fixtures/research/prices/FPT.csv", "data/raw/foreign_v2/x.jsonl",
                                  "mydata/pipeline/x", "data/pipelines/x"])
def test_contracted_paths_are_not_legacy(path):
    assert not C.is_legacy_path(path)


def test_clean_flow_row_is_eligible():
    out = screen(flow())
    assert out.contract_status.tolist() == [C.ELIGIBLE]
    assert out.contract_reason.tolist() == [""]


@pytest.mark.parametrize("legacy, reason", [
    (dict(available_at=None, observed_at=None, research_only=True, quality="legacy_unknown_availability",
          provider="ndtnn", trade_scope="matched_only"), None),
    (dict(available_at=None), "missing_timestamp:available_at"),
    (dict(observed_at=None), "missing_timestamp:observed_at"),
    (dict(raw_sha256=None), "missing_provenance:raw_sha256"),
    (dict(raw_sha256="xyz"), "missing_provenance:raw_sha256"),
    (dict(research_only=True), "legacy_research_only_row"),
    (dict(quality="legacy_unknown_session_and_availability"), "legacy_research_only_row"),
    (dict(trade_scope="matched_only"), "trade_scope_not_contracted"),
    (dict(provider="ndtnn_chart"), "legacy_provider"),
    (dict(schema_version=1), "schema_version_not_2"),
    (dict(source_path="data/raw/foreign/ndtnn.jsonl"), "legacy_storage_path"),
])
def test_legacy_or_unprovenanced_flow_rows_are_quarantined(legacy, reason):
    out = screen(flow(**legacy))
    assert out.contract_status.iloc[0] == C.QUARANTINED
    if reason:
        assert out.contract_reason.iloc[0] == reason
    assert C.eligible_foreign_flow(pd.DataFrame([flow(**legacy)])).empty


def test_missing_provenance_columns_quarantine_rather_than_pass():
    frame = pd.DataFrame([flow()]).drop(columns=["schema_version", "raw_sha256", "observed_at"])
    assert C.screen_foreign_flow(frame).contract_status.iloc[0] == C.QUARANTINED


def test_eligible_rows_are_returned_validated_and_quarantined_rows_dropped():
    good, legacy = flow(), flow(symbol="AAA", research_only=True)
    out = C.eligible_foreign_flow(pd.DataFrame([good, legacy]))
    assert out.symbol.tolist() == ["MBB"]
    assert str(out.available_at.dt.tz) == "UTC"  # passed through validate_source(FOREIGN_FLOW)
    assert "contract_status" not in out


# --- 3. foreign-flow row checks ----------------------------------------------------

@pytest.mark.parametrize("over, reason", [
    (dict(observed_at="2026-10-02 18:00+07:00", available_at="2026-10-02 17:30+07:00"),
     "available_before_observed"),
    (dict(observed_at="2026-10-02 11:00+07:00", available_at="2026-10-02 11:00+07:00"),
     "available_before_session_close"),
    (dict(observed_at="2026-10-02 15:59:59+07:00", available_at="2026-10-02 17:00+07:00"),
     "observed_before_session_close"),
    (dict(available_at="2026-10-02 17:12"), "naive_timestamp:available_at"),
    (dict(net_value=1.0), "inconsistent_net_value"),
    (dict(buy_value=None), "missing_value_in_observed_row:buy_value"),
    (dict(sell_value=float("nan")), "missing_value_in_observed_row:sell_value"),
    (dict(buy_value=float("inf")), "invalid_value:buy_value"),
    (dict(sell_value=-1.0, net_value=1.25e9 + 1.0), "invalid_value:sell_value"),
])
def test_flow_timing_and_value_violations_are_invalid(over, reason):
    out = screen(flow(**over))
    assert (out.contract_status.iloc[0], out.contract_reason.iloc[0]) == (C.INVALID, reason)


def test_session_close_is_vietnam_16h_even_for_utc_stamps():
    # 09:00Z == 16:00 VN: exactly at close is allowed.
    at_close = screen(flow(observed_at="2026-10-02T09:00:00Z", available_at="2026-10-02T09:00:00Z"))
    assert at_close.contract_status.iloc[0] == C.ELIGIBLE
    early = screen(flow(observed_at="2026-10-02T08:59:59Z", available_at="2026-10-02T09:30:00Z"))
    assert early.contract_status.iloc[0] == C.INVALID


def test_net_is_optional_but_checked_when_present_and_zero_flow_is_observed():
    assert screen(flow(net_value=None)).contract_status.iloc[0] == C.ELIGIBLE
    no_net = pd.DataFrame([flow()]).drop(columns="net_value")
    assert C.screen_foreign_flow(no_net).contract_status.iloc[0] == C.ELIGIBLE
    zero = flow(buy_value=0.0, sell_value=0.0, net_value=0.0)
    assert screen(zero).contract_status.iloc[0] == C.ELIGIBLE


# --- 2. missing-reason codes and value/state consistency ---------------------------

def test_missing_state_codes_reuse_existing_names_and_add_invalid():
    # insufficient_warmup is already spelled insufficient_history in research_ml (features.py).
    assert C.MissingState.INSUFFICIENT_HISTORY.value == "insufficient_history"
    assert C.MissingState.NOT_APPLICABLE.value == "not_applicable"
    assert C.MissingState("invalid") is C.MissingState.INVALID
    assert [m.value for m in C.MissingState][:6] == [  # existing values untouched (R5 bundles)
        "observed", "observed_zero", "missing", "stale", "not_applicable", "insufficient_history"]


def feats(values, states):
    return pd.DataFrame({"f": values, "f_state": states})


def test_feature_states_accept_consistent_rows():
    ok = feats([0.5, 0.0, 0.0, None, None, None, None, 1.2],
               ["observed", "observed", "observed_zero", "missing", "insufficient_history",
                "not_applicable", "invalid", "stale"])
    C.validate_feature_states(ok, ["f"])  # stale may carry its last observed value (asof_join)


@pytest.mark.parametrize("values, states, match", [
    ([0.0], ["missing"], "coded as a value"),        # zero-fill of a missing value
    ([0.0], ["insufficient_history"], "coded as a value"),
    ([None], ["observed"], "unexplained missing"),
    ([float("inf")], ["observed"], "unexplained missing"),
    ([0.3], ["observed_zero"], "observed_zero"),
    ([None], ["zero"], "unknown state"),
])
def test_feature_states_reject_zero_fill_and_unexplained_missing(values, states, match):
    with pytest.raises(C.ContractError, match=match):
        C.validate_feature_states(feats(values, states), ["f"])


def test_feature_states_require_state_column():
    with pytest.raises(C.ContractError, match="f_state"):
        C.validate_feature_states(pd.DataFrame({"f": [1.0]}), ["f"])


def test_existing_feature_builder_output_satisfies_state_contract():
    from stock_agent.research_ml import features as F
    days = pd.bdate_range("2025-01-01", periods=150)
    close = pd.Series(range(150), dtype=float) % 7 + 10
    bars = pd.DataFrame({"symbol": "AAA", "session_date": days, "close": close,
                         "volume": [0.0 if i == 30 else 1000.0 for i in range(150)]})
    out = F.build_price_features(bars)
    C.validate_feature_states(out, [c for c in F.REGISTRY if c in out])


# --- 4. membership contract ----------------------------------------------------------


VN30 = json.loads(Path("configs/research/vn30_membership_2022_2026.json").read_text(encoding="utf-8"))
VN100_PATH = Path("configs/research/vn100_membership_2026_h1.json")


def at(day, hour=16, minute=0):
    return pd.Timestamp(f"{day} {hour:02d}:{minute:02d}", tz=TZ)


def test_membership_change_is_known_only_from_16h_on_known_on():
    # LPB replaced POW with known_on == effective_on == 2025-02-03.
    before = C.membership_as_of(VN30, at("2025-02-03", 10), index="VN30")
    after = C.membership_as_of(VN30, at("2025-02-03", 16), index="VN30")
    assert before.status == after.status == C.ELIGIBLE
    assert "POW" in before.members and "LPB" not in before.members
    assert "LPB" in after.members and "POW" not in after.members
    assert len(before.members) == len(after.members) == 30


def test_membership_effective_on_is_separate_from_knowledge():
    # Known 2026-01-23 close, effective 2026-02-02.
    known_only = C.membership_as_of(VN30, at("2026-01-23"), index="VN30")
    executed = C.membership_as_of(VN30, at("2026-01-23"), index="VN30",
                                  execution_date="2026-02-02")
    unknown = C.membership_as_of(VN30, at("2026-01-23", 15, 59), index="VN30",
                                 execution_date="2026-02-02")
    assert known_only.members != executed.members
    assert unknown.members == known_only.members  # effective but not yet known -> not applied


@pytest.mark.parametrize("when, execution", [
    (at("2022-07-29"), None),           # before coverage_start
    (at("2026-09-30"), None),           # after coverage_end
    (at("2026-09-29"), "2026-10-01"),   # execution beyond coverage
])
def test_membership_outside_coverage_is_blocked(when, execution):
    result = C.membership_as_of(VN30, when, index="VN30", execution_date=execution)
    assert result.status == C.BLOCKED and result.members is None
    assert result.reasons == ("outside_membership_coverage",)


def test_membership_initial_list_unknown_before_its_close_is_blocked():
    data = dict(copy.deepcopy(VN30), initial_known_on=VN30["coverage_start"])
    assert C.membership_as_of(data, at(VN30["coverage_start"], 9), index="VN30").status == C.BLOCKED
    assert C.membership_as_of(data, at(VN30["coverage_start"]), index="VN30").status == C.ELIGIBLE


def test_membership_requires_source_and_aware_decision_time():
    unsourced = copy.deepcopy(VN30)
    unsourced["changes"][0]["source"] = " "
    result = C.membership_as_of(unsourced, at("2024-01-02"), index="VN30")
    assert result.status == C.QUARANTINED and result.members is None
    assert C.membership_as_of(dict(VN30, initial_source=""), at("2024-01-02"),
                              index="VN30").status == C.QUARANTINED
    with pytest.raises(C.ContractError, match="timezone"):
        C.membership_as_of(VN30, pd.Timestamp("2024-01-02 16:00"), index="VN30")


@pytest.mark.parametrize("mutate, match", [
    (lambda d: d["changes"][0].update(effective_on="2023-01-01"), "chronology"),
    (lambda d: d["changes"][0].update(add=["ACB"]), "membership change"),
    (lambda d: d.update(index="VN100"), "index"),
    (lambda d: d.update(member_count=31), "initial"),
])
def test_invalid_membership_timeline_fails_closed(mutate, match):
    data = copy.deepcopy(VN30)
    mutate(data)
    with pytest.raises(C.ContractError, match=match):
        C.membership_as_of(data, at("2024-01-02"), index="VN30")
    with pytest.raises(C.ContractError, match="index"):
        C.membership_as_of(VN30, at("2024-01-02"), index="HNX30")


@pytest.mark.parametrize("path, index", [("configs/research/vn30_membership_2022_2026.json", "VN30"),
                                         ("configs/research/vn30_membership_2025_2026.json", "VN30"),
                                         (str(VN100_PATH), "VN100")])
def test_membership_matches_r5_timeline_at_session_close(path, index):
    if not Path(path).exists():
        pytest.skip(f"{path} not present")
    from scripts.historical_universe import Timeline
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    tl = Timeline(data)
    for day in pd.bdate_range(tl.start, tl.end):
        d = day.strftime("%Y-%m-%d")
        result = C.membership_as_of(data, at(d), index=index)
        assert result.status == C.ELIGIBLE and result.members == tl.members(d, d), d
    beyond = (pd.Timestamp(tl.end) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    assert C.membership_as_of(data, at(beyond), index=index).status == C.BLOCKED


# --- 5. label statuses: pending / resolved / not_entered --------------------------------


def label_bars(n=60):
    rng = np.random.default_rng(0)
    close = 100 * np.exp(np.cumsum(rng.normal(0, .01, n)))
    return pd.DataFrame({"symbol": "AAA", "session_date": pd.bdate_range("2025-01-02", periods=n),
                         "open": close * .999, "high": close * 1.01, "low": close * .99, "close": close,
                         "volume": 1000.0})


def test_research_ml_label_statuses_map_to_three_classes():
    assert C.LABEL_STATUS_CLASS == {
        "matured": "resolved", "pending": "pending", "exit_unfillable": "pending",
        "no_price_history": "not_entered", "signal_bar_missing": "not_entered",
        "entry_unfillable": "not_entered"}
    assert set(C.LABEL_STATUS_CLASS.values()) == set(C.LABEL_CLASSES) == {"pending", "resolved", "not_entered"}


def test_build_labels_output_satisfies_label_contract():
    bars = label_bars()
    bars.loc[40, "volume"] = 0
    sig = pd.DataFrame({"symbol": ["AAA", "AAA", "AAA", "BBB", "AAA"],
                        "signal_date": [bars.session_date[5], bars.session_date[24], bars.session_date[55],
                                        bars.session_date[5], pd.Timestamp("2024-12-25")]})
    lab = build_labels(bars, sig, protocol="mr_fixed15", cost=CostConfig())
    classes = C.validate_label_outcomes(lab)
    assert classes.tolist() == ["resolved", "pending", "pending", "not_entered", "not_entered"]


def label_frame(**over):
    row = dict(status="matured", net_return_pct=-1.2, win=False,
               label_available_at=pd.Timestamp("2025-02-01 16:00", tz=TZ))
    row.update(over)
    return pd.DataFrame([row])


@pytest.mark.parametrize("over, match", [
    (dict(status="pending", win=None), "pending.*numeric outcome"),            # loss coerced on pending
    (dict(status="pending", net_return_pct=0.0, win=None), "pending.*numeric outcome"),  # zero-fill
    (dict(status="entry_unfillable", net_return_pct=None, win=False), "not_entered.*numeric outcome"),
    (dict(net_return_pct=None, win=None), "resolved.*without"),
    (dict(label_available_at=pd.NaT), "resolved.*without"),
    (dict(status="lost"), "unknown label status"),
])
def test_label_outcome_violations_are_invalid(over, match):
    with pytest.raises(C.ContractError, match=match):
        C.validate_label_outcomes(label_frame(**over))


def test_build_labels_data_end_pending_rows_keep_numeric_outcome_and_are_flagged():
    """Finding: labels.py re-marks rows pending via data_end but keeps net_return_pct.

    labels.py is not changed here (R5 does not pass data_end); the contract flags it.
    """
    bars = label_bars()
    sig = pd.DataFrame({"symbol": ["AAA"], "signal_date": [bars.session_date[5]]})
    lab = build_labels(bars, sig, protocol="mr_fixed15", cost=CostConfig(),
                       data_end=pd.Timestamp("2025-01-10", tz=TZ))
    assert lab.status.iloc[0] == "pending" and np.isfinite(lab.net_return_pct.iloc[0])
    with pytest.raises(C.ContractError, match="pending.*numeric outcome"):
        C.validate_label_outcomes(lab)
