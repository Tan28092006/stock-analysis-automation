import copy

import pytest

from scripts.research_report import symbol_pnl, table


def test_report_symbol_contribution_includes_fees_and_open_holdings():
    trial = dict(replay=dict(fills=[
        dict(symbol='AAA', side='BUY', qty=100, price=10, fee=2),
        dict(symbol='AAA', side='SELL', qty=50, price=12, fee=3)]),
        portfolio_history=[dict(positions={'AAA': dict(value=650)})], summary=dict(pnl=245))
    assert symbol_pnl(trial) == [dict(symbol='AAA', buys=1002., sales=597., end_value=650, pnl_vnd=245.)]
    trial['summary']['pnl'] = 246
    with pytest.raises(ValueError, match='reconcile'):
        symbol_pnl(trial)


def test_report_tables_escape_untrusted_text():
    rendered = table([dict(symbol='<script>bad</script>', value=1234.5)])
    assert '<script>' not in rendered
    assert '&lt;script&gt;' in rendered
