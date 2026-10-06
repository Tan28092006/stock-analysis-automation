"""Port of R1 semantics from the earlier parallel R1 implementation into research_ml.

Additive only: existing research_ml contract behaviour (validate_source, asof_join,
MissingState values used by R5 bundles) is unchanged.
"""
import pandas as pd
import pytest

from stock_agent.research_ml import contracts as C

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
