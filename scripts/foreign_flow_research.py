"""Read-only audit of legacy foreign data; provenance failures never become zero flow."""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.research_gate import TIMELINE, digest, paired_test
from scripts.historical_universe import Timeline
from stock_agent.data.exchange_calendar import VN_TIMEZONE, is_trading_day
from stock_agent.data.reconciliation import verify_snapshot

DAILY = ('ndtnn', 'ndtnn_chart', 'tudoanh_chart', 'price_board_snapshots')


def source_session(value):
    match = re.fullmatch(r'/Date\((-?\d+)(?:[+-]\d{4})?\)/', value)
    if not match:
        raise ValueError('Invalid source epoch')
    return datetime.fromtimestamp(int(match.group(1)) / 1000, timezone.utc).astimezone(VN_TIMEZONE).date().isoformat()


def normalize(source, row):
    if source not in DAILY:
        raise ValueError('Not a supported daily flow source')
    detailed = source == 'ndtnn'
    board = source == 'price_board_snapshots'
    keys = ('BuyVal', 'SellVal', 'BuyVol', 'SellVol') if detailed else (
        ('f_buy_val', 'f_sell_val', 'f_buy_vol', 'f_sell_vol') if board else ('buy_val', 'sell_val', 'buy_vol', 'sell_vol'))
    try:
        buy, sell, buyvol, sellvol = [float(row[k]) for k in keys]
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError('Missing numeric observation') from exc
    if not np.isfinite([buy, sell, buyvol, sellvol]).all() or min(buy, sell, buyvol, sellvol) < 0:
        raise ValueError('Invalid numeric observation')
    scale = 1000 if detailed else 1
    session = source_session(row['TradingDate']) if detailed and row.get('TradingDate') else None
    # Legacy chart discarded the epoch; price board stamped host date with no session field.
    # Even a present arbitrary observed_at field cannot retrospectively certify these sources.
    return dict(source=source, symbol=row.get('StockCode') if detailed else row['symbol'],
                stored_date=row['date'], session=session, date_quality='source_epoch' if session else 'unverified_legacy',
                buy_bn=buy / scale, sell_bn=sell / scale, net_bn=(buy - sell) / scale,
                buy_vol=buyvol, sell_vol=sellvol, pressure=(buyvol - sellvol) / (buyvol + sellvol) if buyvol + sellvol else None,
                eligible=False, reason='Legacy observed_at/raw-response provenance unavailable',
                definition='matched_only' if detailed else 'provider_aggregate_unverified')


def inventory(raw_dir):
    report, normalized = {}, {}
    names = list(DAILY) + ['ndtnn_monthly', 'ndtnn_quarterly', 'tudoanh_monthly', 'tudoanh_quarterly']
    for source in names:
        path = raw_dir / f'{source}.jsonl'
        if not path.exists():
            report[source] = dict(status='MISSING')
            continue
        rows = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]
        daily = source in DAILY
        key = lambda r: (r.get('StockCode', r.get('symbol')), r.get('date' if daily else 'period'))
        keys = Counter(key(r) for r in rows)
        dates = sorted({r.get('date' if daily else 'period') for r in rows})
        clean, invalid = [], 0
        if daily:
            for row in rows:
                try:
                    clean.append(normalize(source, row))
                except ValueError:
                    invalid += 1
        normalized[source] = clean
        report[source] = dict(rows=len(rows), symbols=len({key(r)[0] for r in rows}), periods=len(dates),
            first=dates[0] if dates else None, last=dates[-1] if dates else None,
            duplicate_rows=sum(n - 1 for n in keys.values()), invalid_numeric_rows=invalid,
            source_sha256=digest(path), observed_at_rows=sum(bool(r.get('observed_at') or r.get('fetched_at')) for r in rows),
            prospective_eligible_rows=0,
            off_calendar_stored_rows=sum(not is_trading_day(date.fromisoformat(r['date'])) for r in rows) if daily else None,
            corrected_epoch_rows=sum(r['session'] is not None and r['session'] != r['stored_date'] for r in clean),
            aggregate_warning=None if daily else 'Period aggregates cannot be used on an earlier constituent day; period completion/availability absent')
    detail = {(r['symbol'], r['stored_date']): r for r in normalized.get('ndtnn', [])}
    pairs = [(detail[(r['symbol'], r['stored_date'])], r) for r in normalized.get('ndtnn_chart', [])
             if (r['symbol'], r['stored_date']) in detail and r['buy_bn'] > 0]
    unit_evidence = dict(overlaps_nonzero_buy=len(pairs),
        matched_volume_and_unit_converted_value=sum(a['buy_vol'] == b['buy_vol'] and np.isclose(a['buy_bn'], b['buy_bn']) for a, b in pairs),
        warning='Remaining differences may be revisions or matched/put-through definitions; sources are not blindly merged.')
    return report, normalized, unit_evidence


def daily_ic(rows, frames, timeline, *, assumed_shift=False):
    dates = frames['VNINDEX']['date'].tolist()
    calendar = pd.Index(dates)
    prices = {s: f.set_index('date') for s, f in frames.items()}
    observations = []
    for row in rows:
        session = (date.fromisoformat(row['stored_date']) + timedelta(days=1)).isoformat() if assumed_shift else row['session']
        if session in calendar and row['symbol'] in prices and row['pressure'] is not None:
            observations.append(dict(date=session, symbol=row['symbol'], pressure=row['pressure']))
    if not observations:
        return dict(status='BLOCKED', reason='No traceable daily observations')
    table = pd.DataFrame(observations)
    # Any duplicate symbol-day is excluded, even if a source appended a later revision.
    table = table.loc[~table.duplicated(['date', 'symbol'], keep=False)]
    panel = table.pivot(index='date', columns='symbol', values='pressure').reindex(calendar)
    result = {}
    for span in (1, 5):
        features = panel.rolling(span, min_periods=span).mean()
        for lag in (1, 2):
            for horizon in (5, 15, 21):
                ics, partials, daily = [], [], []
                for i, session in enumerate(dates):
                    if i < 5 or i + lag + horizon >= len(dates) or not timeline.start <= session <= timeline.end:
                        continue
                    entry, exit_day = dates[i + lag], dates[i + lag + horizon]
                    members = timeline.members(session, entry)
                    x, y, price_mom = [], [], []
                    for symbol, value in features.loc[session].dropna().items():
                        p = prices[symbol]
                        if symbol not in members or not all(d in p.index for d in (entry, exit_day, session, dates[i - 5])):
                            continue
                        x.append(value)
                        y.append(float(p.loc[exit_day, 'close'] / p.loc[entry, 'open'] - 1))
                        price_mom.append(float(p.loc[session, 'close'] / p.loc[dates[i - 5], 'close'] - 1))
                    if len(x) < 10:
                        continue
                    ranked = pd.DataFrame(dict(flow=x, forward=y, price5=price_mom)).rank()
                    raw = ranked['flow'].corr(ranked['forward'])
                    z = np.column_stack([np.ones(len(x)), ranked['price5'].to_numpy()])
                    a = ranked['flow'].to_numpy(); b = ranked['forward'].to_numpy()
                    a = a - z @ np.linalg.lstsq(z, a, rcond=None)[0]
                    b = b - z @ np.linalg.lstsq(z, b, rcond=None)[0]
                    partial = float(np.corrcoef(a, b)[0, 1]) if min(np.std(a), np.std(b)) > 1e-10 else np.nan
                    if np.isfinite(raw) and np.isfinite(partial):
                        ics.append(float(raw)); partials.append(partial)
                        daily.append(dict(date=session, names=len(x), rank_ic=float(raw), partial_rank_ic=partial))
                key = f'flow{span}_lag{lag}_hold{horizon}'
                item = dict(dates=len(ics), mean_ic=float(np.mean(ics)) if ics else None,
                            partial_price5_mean_ic=float(np.mean(partials)) if partials else None, daily=daily,
                            status='EXPLORATORY_NOT_PIT_CERTIFIED', statistical_eligible=len(ics) >= 252)
                # Scale by 1/25200 because paired_test is in annualized percentage points;
                # returned interval here is on the IC scale, never annualized IC.
                if len(ics) >= 40:
                    item['ic_tests'] = {str(block): paired_test(np.array(ics) / 25200, family=48, block=block) for block in (20, 40)}
                    item['partial_ic_tests'] = {str(block): paired_test(np.array(partials) / 25200, family=48, block=block) for block in (20, 40)}
                    for tests in (item['ic_tests'], item['partial_ic_tests']):
                        for test in tests.values():
                            test['mean_ic'] = test.pop('mean_annualized_pp')
                            test['unit'] = 'rank_IC'
                result[key] = item
    return dict(date_assumption='stored_date + 1 calendar day (UNVERIFIED sensitivity only)' if assumed_shift else 'original source epoch in Asia/Saigon',
                status='EXPLORATORY_NOT_PIT_CERTIFIED', tests=result)


def run(raw_dir, manifest_path, output):
    output.mkdir(parents=True, exist_ok=False)
    files, rows, units = inventory(raw_dir)
    manifest = verify_snapshot(manifest_path)
    timeline = Timeline(json.loads(TIMELINE.read_text(encoding='utf-8')))
    frames = {s: pd.read_csv(manifest_path.parent / item['path']) for s, item in manifest['files'].items()
              if s in timeline.all_members | {'VNINDEX'}}
    result = dict(status='BLOCKED_FOR_PRODUCTION', files=files, cross_source_units=units,
        manifest_sha256=digest(manifest_path), script_sha256=digest(__file__), timeline_sha256=digest(TIMELINE),
        experiments={
            'detailed_epoch': daily_ic(rows['ndtnn'], frames, timeline),
            'chart_assumed_plus1': daily_ic(rows['ndtnn_chart'], frames, timeline, assumed_shift=True)},
        findings=['UTC date extraction shifts midnight Vietnam source epochs to previous day',
                  'Detailed values in million VND, chart/board in billion VND; old loader mixes units',
                  'No collection timestamps: retrospective corrections do not reconstruct historical availability',
                  'Chart histories are not consumed by load_flows; monthly/quarterly are not daily signals',
                  'No strategy promotion from a short, previously collected and unverified legacy sample'])
    (output / 'audit.json').write_text(json.dumps(result, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    print(json.dumps(dict(files=files, units=units, tests={source: {k: {field: v[field] for field in ('dates','mean_ic','partial_price5_mean_ic')}
        for k, v in experiment.get('tests', {}).items()} for source, experiment in result['experiments'].items()}), ensure_ascii=False), flush=True)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--raw-dir', type=Path, default=Path('data/raw/foreign'))
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    run(args.raw_dir, args.manifest, args.output)


if __name__ == '__main__':
    main()
