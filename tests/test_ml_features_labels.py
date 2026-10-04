import inspect

import numpy as np
import pandas as pd
import pytest

from stock_agent.features import calibration
from stock_agent.features.backtest import BacktestConfig
from stock_agent.research_ml import features as F
from stock_agent.research_ml.contracts import ContractError, MissingState
from stock_agent.research_ml.labels import CostConfig, build_labels, trainable


def make_bars(n=200, symbol="AAA", seed=0, start="2025-01-02"):
    rng = np.random.default_rng(seed)
    days = pd.bdate_range(start, periods=n)
    close = 100 * np.exp(np.cumsum(rng.normal(0, .01, n)))
    return pd.DataFrame({"symbol": symbol, "session_date": days, "open": close * .999, "high": close * 1.01,
                         "low": close * .99, "close": close, "volume": rng.integers(1000, 5000, n).astype(float)})


def test_future_mutation_does_not_change_past_features():
    bars = make_bars()
    base = F.build_price_features(bars)
    mutated = bars.copy()
    mutated.loc[150:, ["open", "high", "low", "close"]] *= 3
    mutated.loc[150:, "volume"] *= 50
    other = F.build_price_features(mutated)
    cols = [c for c in F.REGISTRY if c in base]
    pd.testing.assert_frame_equal(base.loc[:149, cols], other.loc[:149, cols])


def test_warmup_marked_insufficient_not_zero():
    out = F.build_price_features(make_bars(40))
    assert out.f0_mom_126_21_state.eq(MissingState.INSUFFICIENT_HISTORY.value).all()
    assert out.f0_mom_126_21.isna().all()
    assert out.f0_ret_1_state.iloc[0] == MissingState.INSUFFICIENT_HISTORY.value


def test_registry_rejects_target_columns():
    assert F.feature_columns(["f0_ret_1", "net_return_pct", "symbol"]) == ["f0_ret_1"]
    with pytest.raises(ContractError):
        F.assert_no_target_leak(["f0_future"], allowlist={"f0_future"})


def test_foreign_feature_keeps_missing_slots_and_requires_support():
    bars = make_bars(60)
    feats = F.build_price_features(bars)
    sessions = list(bars.session_date)
    rows = [dict(symbol="AAA", session_date=d, buy_value=2e6, sell_value=1e6) for d in sessions[:30]]
    foreign = pd.DataFrame(rows)
    foreign = foreign[~foreign.session_date.isin(sessions[10:12])]  # two missing slots, not zeros
    out = F.build_foreign_feature(feats, foreign, sessions, min_support=4)
    row = out[out.session_date == sessions[20]].iloc[0]
    assert row.f3_foreign_net_adv_5_state == "observed" and row.f3_foreign_net_adv_5 > 0
    gap = out[out.session_date == sessions[45]].iloc[0]  # no data in window
    assert gap.f3_foreign_net_adv_5_state == "missing" and np.isnan(gap.f3_foreign_net_adv_5)
    lagged_day = out[out.session_date == sessions[0]].iloc[0]
    assert lagged_day.f3_foreign_net_adv_5_state == "missing"  # lag +1 hides same-day flow


def test_labels_entry_exit_index_and_cost():
    bars = make_bars(40)
    sig = pd.DataFrame({"symbol": ["AAA"], "signal_date": [bars.session_date[5]]})
    lab = build_labels(bars, sig, protocol="mr_fixed15", cost=CostConfig())
    r = lab.iloc[0]
    assert r.entry_date == bars.session_date[6] and r.exit_date == bars.session_date[6 + 15]  # e+15
    gross = (bars.close[21] / bars.open[6] - 1) * 100
    assert r.net_return_pct == pytest.approx(gross - CostConfig().round_trip_pct)
    pricier = build_labels(bars, sig, protocol="mr_fixed15", cost=CostConfig(slippage_pct=0.5))
    assert pricier.net_return_pct.iloc[0] < r.net_return_pct  # cost config really flows


def test_pending_unfillable_and_selection_bias_rows():
    bars = make_bars(30)
    bars.loc[21, "volume"] = 0  # signal idx 5 -> entry 6 -> exit idx 21 suspended
    sig = pd.DataFrame({"symbol": ["AAA"] * 3, "signal_date": [bars.session_date[5], bars.session_date[22],
                                                              bars.session_date[29]],
                        "selected": [True, False, True]})
    lab = build_labels(bars, sig, protocol="mr_fixed15", cost=CostConfig())
    assert lab.status.tolist() == ["exit_unfillable", "pending", "pending"]
    assert lab.net_return_pct.isna().all() and lab.selected.tolist() == [True, False, True]
    assert trainable(lab, pd.Timestamp("2030-01-01", tz="Asia/Ho_Chi_Minh")).empty


def test_trainable_respects_label_availability_time():
    bars = make_bars(60)
    sig = pd.DataFrame({"symbol": ["AAA"], "signal_date": [bars.session_date[5]]})
    lab = build_labels(bars, sig, protocol="mr_fixed15", cost=CostConfig())
    avail = lab.label_available_at.iloc[0]
    assert trainable(lab, avail - pd.Timedelta(seconds=1)).empty
    assert len(trainable(lab, avail)) == 1


def test_momentum_horizons_are_separate_protocols():
    bars = make_bars(60)
    sig = pd.DataFrame({"symbol": ["AAA"], "signal_date": [bars.session_date[5]]})
    five = build_labels(bars, sig, protocol="mom_5", cost=CostConfig()).iloc[0]
    month = build_labels(bars, sig, protocol="mom_21", cost=CostConfig()).iloc[0]
    assert five.horizon == 5 and month.horizon == 21 and five.exit_date < month.exit_date


def test_cost_config_parity_with_backtest_config():
    assert CostConfig.from_backtest(BacktestConfig()).round_trip_pct == pytest.approx(0.60)


def test_legacy_t2_builder_still_drops_cost_config():
    """Characterization of the audited gap (plan §5): update this test when it is fixed."""
    src = inspect.getsource(calibration.build_labeled_t2_dataset)
    call = src[src.index("build_labeled_dataset_fast("):]
    assert "cost_config" not in call
