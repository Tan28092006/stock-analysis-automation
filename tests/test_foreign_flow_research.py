from datetime import date

import pytest

from scripts import foreign_flow_research as flow


def test_source_epoch_uses_vietnam_date_not_utc_date():
    assert flow.source_session('/Date(1782925200000)/') == '2026-07-02'


def test_detailed_million_to_billion_and_no_false_pit_eligibility():
    row = dict(date='2026-07-01', TradingDate='/Date(1782925200000)/',
               StockCode='AAA', BuyVal=9127.315, SellVal=1000, BuyVol=10, SellVol=5)
    actual = flow.normalize('ndtnn', row)
    assert actual['session'] == '2026-07-02'
    assert actual['net_bn'] == pytest.approx(8.127315)
    assert actual['eligible'] is False
    assert actual['date_quality'] == 'source_epoch'


def test_chart_date_is_not_silently_repaired():
    row = dict(date='2026-09-20', symbol='ACB', buy_val=1., sell_val=2., buy_vol=1, sell_vol=2)
    actual = flow.normalize('ndtnn_chart', row)
    assert actual['session'] is None
    assert actual['eligible'] is False
    assert actual['stored_date'] == '2026-09-20'


def test_missing_and_nan_values_are_quarantined_not_zero():
    with pytest.raises(ValueError):
        flow.normalize('price_board_snapshots', dict(date='2026-09-21', symbol='ACB', f_buy_val=float('nan'), f_sell_val=1))


def test_monthly_is_not_daily_and_requires_period_key():
    with pytest.raises(ValueError, match='daily'):
        flow.normalize('ndtnn_monthly', dict(period='2026-06', symbol='ACB', buy_val=1, sell_val=0))
