import pandas as pd
import pytest

from stock_agent.research_ml.contracts import (
    ContractError, FOREIGN_FLOW, MissingState, PRICE_BAR, asof_join, assert_no_target_leak,
    select_vintage, validate_source)

TZ = "Asia/Ho_Chi_Minh"


def bars(**over):
    rows = [dict(symbol="AAA", session_date="2026-01-05", provider="p", vintage=1, open=10, high=11,
                 low=9, close=10, volume=100, available_at="2026-01-05 16:00:00+07:00"),
            dict(symbol="AAA", session_date="2026-01-06", provider="p", vintage=1, open=10, high=12,
                 low=10, close=11, volume=0, available_at="2026-01-06 16:00:00+07:00")]
    df = pd.DataFrame(rows)
    for k, v in over.items():
        df[k] = v
    return df


def test_validate_rejects_naive_timestamp_and_bad_ohlc():
    with pytest.raises(ContractError, match="naive"):
        validate_source(bars(available_at="2026-01-05 16:00:00"), PRICE_BAR)
    with pytest.raises(ContractError, match="OHLC"):
        validate_source(bars(high=5), PRICE_BAR)
    with pytest.raises(ContractError, match="non-finite"):
        validate_source(bars(volume=float("inf")), PRICE_BAR)


def test_validate_rejects_duplicate_grain_vintage():
    with pytest.raises(ContractError, match="duplicate"):
        validate_source(pd.concat([bars(), bars()]), PRICE_BAR)


def test_asof_join_blocks_future_availability_intentional_leak():
    """RED/GREEN: a bar published after the decision must not be visible."""
    left = pd.DataFrame({"symbol": ["AAA"], "decision_time": [pd.Timestamp("2026-01-06 09:00", tz=TZ)]})
    right = validate_source(bars(), PRICE_BAR)
    out = asof_join(left, right, on=["symbol"], decision_col="decision_time", right_value_cols=["close"])
    assert out.close.iloc[0] == 10  # 01-06 bar is available 16:00, after the 09:00 decision
    leaky = right.copy()
    leaky["available_at"] = pd.Timestamp("2026-01-01", tz=TZ)  # backdated -> the leak the guard cannot see
    assert asof_join(left, leaky, on=["symbol"], decision_col="decision_time",
                     right_value_cols=["close"]).close.iloc[0] == 11


def test_asof_missing_zero_and_stale_are_distinct_states():
    left = pd.DataFrame({"symbol": ["AAA", "AAA", "BBB"],
                         "decision_time": [pd.Timestamp("2026-01-06 17:00", tz=TZ),
                                           pd.Timestamp("2026-02-20 17:00", tz=TZ),
                                           pd.Timestamp("2026-01-06 17:00", tz=TZ)]})
    right = validate_source(bars(), PRICE_BAR)
    out = asof_join(left, right, on=["symbol"], decision_col="decision_time",
                    right_value_cols=["volume"], max_staleness=pd.Timedelta(days=10))
    assert out.volume_state.tolist() == [MissingState.OBSERVED_ZERO.value, MissingState.STALE.value,
                                         MissingState.MISSING.value]


def test_asof_rejects_fanout_and_ambiguous_versions():
    left = pd.DataFrame({"symbol": ["AAA", "AAA"], "decision_time": [pd.Timestamp("2026-01-07", tz=TZ)] * 2})
    right = validate_source(bars(), PRICE_BAR)
    with pytest.raises(ContractError, match="fan-out"):
        asof_join(left, right, on=["symbol"], decision_col="decision_time", right_value_cols=["close"])
    dup = pd.concat([bars().iloc[[1]], bars(close=10.5, vintage=2).iloc[[1]]])
    dup["available_at"] = pd.Timestamp("2026-01-06 18:00", tz=TZ)
    one = pd.DataFrame({"symbol": ["AAA"], "decision_time": [pd.Timestamp("2026-01-07", tz=TZ)]})
    with pytest.raises(ContractError, match="ambiguous"):
        asof_join(one, validate_source(dup, PRICE_BAR), on=["symbol"], decision_col="decision_time",
                  right_value_cols=["close"])


def test_select_vintage_uses_knowledge_time_for_revisions():
    first = bars().iloc[[1]]
    revised = bars(close=10.9, vintage=2).iloc[[1]].copy()
    revised["available_at"] = pd.Timestamp("2026-01-08 10:00", tz=TZ)
    frame = validate_source(pd.concat([first, revised]), PRICE_BAR)
    before = select_vintage(frame, PRICE_BAR, pd.Timestamp("2026-01-07", tz=TZ))
    after = select_vintage(frame, PRICE_BAR, pd.Timestamp("2026-01-09", tz=TZ))
    assert before.close.iloc[0] == 11 and after.close.iloc[0] == 10.9


def test_foreign_contract_requires_scope_and_rejects_negative():
    row = dict(symbol="AAA", session_date="2026-01-05", investor="foreign", trade_scope="matched",
               provider="p", vintage=1, buy_value=5, sell_value=-1, available_at="2026-01-05 18:00+07:00")
    with pytest.raises(ContractError, match="negative"):
        validate_source(pd.DataFrame([row]), FOREIGN_FLOW)
    with pytest.raises(ContractError, match="missing columns"):
        validate_source(pd.DataFrame([row]).drop(columns="trade_scope"), FOREIGN_FLOW)


def test_target_columns_rejected_even_if_prefixed_feature():
    with pytest.raises(ContractError, match="not registered"):
        assert_no_target_leak(["feature_rsi", "feature_x"], allowlist={"feature_rsi"})
    with pytest.raises(ContractError, match="target-like"):
        assert_no_target_leak(["feature_future_ret"], allowlist={"feature_future_ret"})
