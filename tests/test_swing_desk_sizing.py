from datetime import date, datetime

import pytest

from stock_agent.data.exchange_calendar import VN_TIMEZONE
from stock_agent.swing_desk.config import load_config

CFG = load_config()


def test_effective_loss_is_stop_plus_gap_plus_cost():
    from stock_agent.swing_desk.sizing import effective_loss_pct
    assert effective_loss_pct(CFG) == pytest.approx(4.6)


def test_suggested_size_keeps_effective_risk_within_budget_and_lots():
    from stock_agent.swing_desk.sizing import plan
    p = plan(27350, 1_000_000_000, CFG)
    assert isinstance(p["qty"], int) and p["qty"] > 0  # lot_size 1 (odd lots)
    assert p["qty"] * 27350 * 0.046 <= 1_000_000_000 * 0.01
    assert (p["qty"] + 100) * 27350 * 0.046 > 1_000_000_000 * 0.01 or p["capped_by"]
    assert p["stop_user"] == pytest.approx(27350 * 0.97)
    assert p["stop_validated"] == pytest.approx(27350 * 0.92)
    assert p["vnd_at_risk_nominal"] == pytest.approx(p["qty"] * 27350 * 0.03)
    assert p["vnd_at_risk_effective"] == pytest.approx(p["qty"] * 27350 * 0.046)


def test_weight_and_exposure_caps_bind():
    from stock_agent.swing_desk.sizing import plan
    cfg = load_config()
    cfg["money"]["risk_per_trade_pct"] = 5.0       # risk alone would allow > 100% NAV
    p = plan(10000, 1_000_000_000, cfg)
    assert p["qty"] * 10000 <= 0.25 * 1_000_000_000
    p2 = plan(10000, 1_000_000_000, cfg, current_exposure=550_000_000)
    assert p2["qty"] * 10000 <= 50_000_000


def test_requested_amount_risk_arithmetic():
    from stock_agent.swing_desk.sizing import plan
    p = plan(27350, 1_000_000_000, CFG, amount=300_000_000)
    req = p["requested"]
    assert req["qty"] == 10968          # floor(300e6 / 27350), odd lots
    value = 10968 * 27350
    assert req["value"] == pytest.approx(value)
    assert req["vnd_at_risk_nominal"] == pytest.approx(value * 0.03)
    assert req["vnd_at_risk_effective"] == pytest.approx(value * 0.046)
    assert req["risk_effective_pct_nav"] == pytest.approx(value * 0.046 / 1e9 * 100)
    assert req["over_budget"] is True


def test_earliest_sell_skips_weekend_and_holidays():
    from stock_agent.swing_desk.calendar_events import earliest_sell
    assert earliest_sell(date(2026, 8, 28)) == date(2026, 9, 4)      # Fri; 31/8-2/9 holidays
    assert earliest_sell(date(2026, 9, 17)) == date(2026, 9, 21)     # Thu -> Mon


def test_trade_session_rolls_after_close_and_on_weekends():
    from stock_agent.swing_desk.calendar_events import trade_session
    assert trade_session(datetime(2026, 9, 21, 10, 0, tzinfo=VN_TIMEZONE)) == date(2026, 9, 21)
    assert trade_session(datetime(2026, 9, 21, 15, 0, tzinfo=VN_TIMEZONE)) == date(2026, 9, 22)
    assert trade_session(datetime(2026, 9, 19, 10, 0, tzinfo=VN_TIMEZONE)) == date(2026, 9, 21)


def test_vn30f_expiry_third_thursday_moves_back_on_holiday():
    from stock_agent.swing_desk.calendar_events import vn30f_expiry, next_vn30f_expiry
    assert vn30f_expiry(2026, 10) == date(2026, 10, 15)
    assert vn30f_expiry(2026, 2) == date(2026, 2, 13)
    assert next_vn30f_expiry(date(2026, 10, 16)) == date(2026, 11, 19)


def test_holiday_table_horizon_warning():
    from stock_agent.swing_desk.calendar_events import horizon_warning
    assert horizon_warning(date(2026, 10, 6)) is None
    assert "2026-12-31" in horizon_warning(date(2026, 12, 24))


def test_session_distance_counts_trading_sessions_not_days():
    from stock_agent.swing_desk.calendar_events import session_distance
    assert session_distance(date(2026, 9, 18), date(2026, 9, 21)) == 1   # Fri -> Mon
    assert session_distance(date(2026, 8, 28), date(2026, 9, 3)) == 1    # across holidays
