"""Research-only VN100 H1 membership; no current-list backfill or live switch."""
import json
from datetime import date
from pathlib import Path

import pytest

from scripts.historical_universe import Timeline
from stock_agent.data.exchange_calendar import trading_days_between


PATH = Path('configs/research/vn100_membership_2026_h1.json')


def timeline():
    return Timeline(json.loads(PATH.read_text(encoding='utf-8')))


def test_january_retains_removed_names_and_excludes_future_additions():
    t = timeline()
    members = t.members('2025-12-31', '2026-01-05')
    assert len(members) == 100 and len(t.all_members) == 104
    assert {'DGC', 'PPC', 'PTB', 'TLG', 'HT1', 'NT2', 'PC1'} <= members
    assert not {'BSR', 'NVL', 'VPL', 'BAF'} & members


def test_february_change_requires_both_available_and_effective_dates():
    t = timeline()
    old = t.members('2026-01-20', '2026-02-02')
    assert t.members('2026-01-29', '2026-01-30') == old
    new = t.members('2026-01-30', '2026-02-02')
    assert new - old == {'BSR', 'NVL', 'VPL'}
    assert old - new == {'PPC', 'PTB', 'TLG'}


def test_after_close_announcement_cannot_be_known_at_same_close():
    t = timeline()
    old = t.members('2026-05-07', '2026-05-13')
    assert 'DGC' in old and 'BAF' not in old
    new = t.members('2026-05-12', '2026-05-13')
    assert old - new == {'DGC'} and new - old == {'BAF'}
    assert len(new) == 100 and 'BSR' in old & new


@pytest.mark.parametrize('as_of,execution', [
    ('2025-08-01', '2025-08-04'),
    ('2026-06-30', '2026-07-01'),
    ('2026-02-03', '2026-02-02'),
])
def test_no_membership_outside_explicit_coverage(as_of, execution):
    with pytest.raises(ValueError, match='outside|reversed'):
        timeline().members(as_of, execution)


def test_h1_vn30_is_subset_each_session_not_present_day_vn30():
    vn100 = timeline()
    vn30 = Timeline(json.loads(Path('configs/research/vn30_membership_2022_2026.json').read_text(encoding='utf-8')))
    previous = '2025-12-31'
    for session in trading_days_between(date(2026, 1, 1), date(2026, 6, 30)):
        day = str(session)
        assert vn30.members(previous, day) <= vn100.members(previous, day)
        previous = day


def test_source_lineage_and_research_only_limit_are_explicit():
    data = timeline().data
    assert data['index'] == 'VN100'
    assert data['live_eligible'] is False
    assert data['coverage_end'] == '2026-06-30'
    assert data['initial_source_sha256'] == '8573aaed26f2f087ac30567b1c943e4112a85f6c3efc6a57faa80b7b3f020eae'
    assert data['changes'][0]['source_sha256'] == 'a89a30791f450cc66a2bb81496eda3f250a75697b4102ea75401d29da0ff2814'
    assert data['changes'][1]['announced_at'] == '2026-05-07T18:03:00+07:00'
    assert data['changes'][1]['known_on'] == '2026-05-08'
    assert data['limitations']
