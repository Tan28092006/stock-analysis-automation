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
