"""Locked VN30/VN100 H1 paired research. No production configuration or orders."""
from __future__ import annotations

import argparse
import copy
import html
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from scripts import momentum_books as books
from scripts import momentum_hypotheses as rh
from scripts import period_backtest as bt
from scripts import research_gate as gate
from scripts import research_timeslices as ts
from scripts.historical_universe import Timeline
from stock_agent.data.reconciliation import verify_snapshot

PROTOCOL = Path('configs/research/vn100_h1_v1.json')


def load_protocol(path: Path = PROTOCOL) -> dict:
    p = json.loads(Path(path).read_text(encoding='utf-8'))
    expected = dict(schema_version=1, hypothesis_id='vn100_h1_v1_20260930',
        holdout=False, live_eligible=False, start='2026-01-01', end='2026-06-30',
        membership_start='2025-12-31', family=88, min_inference_sessions=252,
        lagged_adv_sessions=20, participation_warning_pct=1.0,
        bootstrap_blocks=[20, 40], scenarios=list(books.SCENARIOS),
        snapshot_sha256='c9d046ff5d1c336284c2fc8a1bd07abc2e63c84d2493ce508a8d16bf132d7148',
        universes={'VN30': str(gate.TIMELINE).replace('\\', '/'),
                   'VN100': 'configs/research/vn100_membership_2026_h1.json'},
        event=dict(id='war_2026', kind='event', start='2026-03-01', end='2026-04-30'))
    if any(p.get(k) != v for k, v in expected.items()):
        raise ValueError('Paired universe protocol changed; explicitly version it')
    registrations = gate.load_registry()['strategy_hypotheses']
    if not any(r['id'] == p['hypothesis_id'] and r['protocol'] == PROTOCOL.as_posix() for r in registrations):
        raise ValueError('Universe hypothesis not registered')
    return p


def trial_names() -> list[str]:
    names = [f'{kind}/{variant}/normal' for kind, variants in [('mr', gate.MR), ('momentum', gate.MOMENTUM)]
             for variant in variants]
    names += ['momentum/universe_equal/normal']
    names += [f'{strategy}/{scenario}' for strategy in ts.REFERENCES for scenario in books.SCENARIOS[1:]]
    names += [f'books/{variant}/{scenario}' for variant in books.VARIANTS if variant != 'baseline'
              for scenario in books.SCENARIOS]
    return names


def bounded_timeline(data: dict, start: str, end: str) -> Timeline:
    original = Timeline(data)
    initial = sorted(original.members(start, start))
    if end > original.end:
        raise ValueError('Membership ends before experiment')
    changes = []
    for event in data['changes']:
        if start < event['effective_on'] <= end:
            event = copy.deepcopy(event)
            event['original_known_on'] = event['known_on']
            event['known_on'] = max(start, event['known_on'])
            changes.append(event)
    return Timeline(dict(coverage_start=start, coverage_end=end, member_count=original.data['member_count'],
                         initial_known_on=start, initial_members=initial, changes=changes))


def load_inputs(manifest_path: Path, protocol: dict) -> tuple:
    manifest = verify_snapshot(manifest_path)
    if gate.digest(manifest_path) != protocol['snapshot_sha256']:
        raise ValueError('Not the preregistered snapshot')
    timelines = {name: bounded_timeline(json.loads(Path(path).read_text(encoding='utf-8')),
                                       protocol['membership_start'], protocol['end'])
                 for name, path in protocol['universes'].items()}
    required = set.union(*(t.all_members for t in timelines.values()), {'VNINDEX'})
    if not required <= set(manifest['files']) or manifest['as_of'] < protocol['end']:
        raise ValueError('Incomplete paired universe input')
    frames = {s: pd.read_csv(manifest_path.parent / manifest['files'][s]['path']) for s in sorted(required)}
    frames = {s: f.loc[f.date <= protocol['end']].reset_index(drop=True) for s, f in frames.items()}
    bt.validate_frames(frames, protocol['start'], protocol['end'])
    ts.classify_regimes(frames['VNINDEX'], protocol['start'], protocol['end'])
    rules = json.loads(gate.RULES.read_text(encoding='utf-8'))
    if rules.get('ml', {}).get('enabled') or rules['backtest']['lot_size'] != 100:
        raise ValueError('Requires ML off and 100-share lots')
    return frames, timelines, rules


def execution_diagnostics(fills: list, frames: dict) -> dict:
    rows = []
    for fill in fills:
        frame = frames[fill['symbol']]
        prior = frame.loc[frame.date < fill['date'], 'volume'].tail(20)
        valid = len(prior) == 20 and np.isfinite(prior).all() and prior.mean() > 0
        adv = float(prior.mean()) if valid else None
        participation = fill['qty'] / adv * 100 if adv else None
        rows.append(dict(date=fill['date'], symbol=fill['symbol'], side=fill['side'], qty=fill['qty'],
                         lagged_adv20=adv, participation_pct=participation))
    known = [r['participation_pct'] for r in rows if r['participation_pct'] is not None]
    above = sum(value > 1.0 for value in known)
    unknown = len(rows) - len(known)
    return dict(fills=rows, above_limit=above, unknown=unknown, max_participation_pct=max(known, default=None),
                capacity_pass=not above and not unknown, limit_pct=1.0,
                interpretation='Lagged daily-volume audit only; not an auction fill or price-impact model.')


def compare_metrics(vn30: dict, vn100: dict) -> dict:
    if abs(vn30['index_return_pct'] - vn100['index_return_pct']) > 1e-9:
        raise ValueError('Paired benchmark bases differ')
    epsilon = 1e-9  # Numerical zero: 0.01 VND on the configured 1bn NAV, not alpha tuning.
    return dict(vn100_minus_vn30_pp=vn100['return_pct'] - vn30['return_pct'],
        vn100_beats_vn30=vn100['return_pct'] - vn30['return_pct'] > epsilon,
        vn100_beats_index=vn100['return_pct'] - vn100['index_return_pct'] > epsilon,
        vn100_positive_net_and_beats_both=vn100['return_pct'] - max(0., vn30['return_pct'], vn100['index_return_pct']) > epsilon,
        new_entry_delta=vn100['new_entries'] - vn30['new_entries'],
        addition_delta=vn100['additions'] - vn30['additions'],
        fees_delta_vnd=vn100['fees_vnd'] - vn30['fees_vnd'],
        drawdown_delta_pp=vn100['max_drawdown_pct'] - vn30['max_drawdown_pct'])


def run_trial(frames: dict, rules: dict, timeline: Timeline, bank: dict,
              start: str, end: str, name: str) -> dict:
    strategy, scenario = name.rsplit('/', 1)
    trial = ts.cash_reference(frames, rules, timeline, bank, start, end, strategy, scenario)
    replay, metrics = trial['replay'], trial['metrics']
    dates = frames['VNINDEX'].date.tolist()
    previous = {day: dates[i - 1] for i, day in enumerate(dates) if i}
    for fill in replay['fills']:
        if fill['side'] == 'BUY' and fill['symbol'] not in timeline.members(previous[fill['date']], fill['date']):
            raise ValueError('Purchase outside point-in-time membership')
    trial['execution'] = execution_diagnostics(replay['fills'], frames)
    initial = float(rules['backtest']['initial_capital'])
    benchmark = rh.benchmark_nav(frames, start, end, initial)
    control = gate.exposure_control(replay['nav'], rh.daily_returns(benchmark, initial), initial)
    metrics['exposure_matched_index_return_pct'] = float((np.prod(1 + control) - 1) * 100)
    metrics['two_way_turnover_multiple'] = (sum(f['qty'] * f['price'] for f in replay['fills'])
                                           / np.mean([r['nav'] for r in replay['nav']]))
    contributions = {}
    for fill in replay['fills']:
        symbol = fill['symbol']
        contributions[symbol] = contributions.get(symbol, 0.) + (
            fill['qty'] * fill['price'] * (1 if fill['side'] == 'SELL' else -1) - fill['fee'])
    last = replay['nav'][-1]['date']
    for symbol, qty in metrics['ending_positions'].items():
        contributions[symbol] += qty * float(frames[symbol].loc[frames[symbol].date == last, 'close'].iloc[0])
    if abs(sum(contributions.values()) - metrics['pnl_vnd']) > 1e-5:
        raise ValueError('Contribution reconciliation failed')
    positive = sorted((v for v in contributions.values() if v > 0), reverse=True)
    trial['contribution_pnl_vnd'] = contributions
    trial['top3_share_of_positive_pnl_pct'] = sum(positive[:3]) / sum(positive) * 100 if positive else None
    return trial


def render_report(result: dict) -> str:
    tables = []
    for block in result['blocks']:
        rows = []
        for name in block['universes']['VN30']['cash_restart']:
            values = [block['universes'][u]['cash_restart'][name] for u in ('VN30', 'VN100')]
            cells = [name]
            for value in values:
                m = value['metrics']
                cells += [f"{m['return_pct']:+.3f}% / {m['max_drawdown_pct']:.3f}%",
                          f"{m['new_entries']} / {m['additions']}"]
            cells += [f"{values[0]['metrics']['index_return_pct']:+.3f}%",
                      f"{block['comparisons']['cash_restart'][name]['vn100_minus_vn30_pp']:+.3f}",
                      str(values[1]['execution']['above_limit']) + ' / ' + str(values[1]['execution']['unknown'])]
            rows.append('<tr>' + ''.join('<td>' + html.escape(c) + '</td>' for c in cells) + '</tr>')
        title = f"{block['id']} | {block['start']} – {block['end']} | {block['regime_sessions']}"
        tables.append('<h2>' + html.escape(title) + '</h2><table><tr><th>Biến thể</th>'
            '<th>VN30 lãi / DD</th><th>Mới / thêm</th><th>VN100 lãi / DD</th><th>Mới / thêm</th>'
            '<th>VNINDEX</th><th>VN100 − VN30 (đpt)</th><th>VN100 lệnh &gt;1% ADV / thiếu ADV</th></tr>'
            + ''.join(rows) + '</table>')
    return ('<!doctype html><html lang="vi"><meta charset="utf-8"><title>VN30 vs VN100</title>'
        '<style>body{font:15px system-ui;margin:24px;color:#172333}table{border-collapse:collapse;width:100%}'
        'td,th{padding:8px;border:1px solid #ccd6e0;text-align:right}td:first-child{text-align:left}'
        'th{background:#e9eff5}h2{margin-top:36px}</style><h1>VN30 – VN100: cùng luật, cùng dữ liệu</h1>'
        '<p>Mỗi bảng bắt đầu riêng bằng tiền mặt. Lãi mô phỏng sau phí; VNINDEX giá gross. '
        'Không phải lãi tiền thật, không tự bật VN100. Mới / thêm = vị thế mới / mua thêm. '
        'DD gồm NAV ban đầu. Nhãn pha từ giá đóng cửa phiên trước, MA200 + dốc 20 phiên.</p>'
        '<p>H1 chưa đủ 252 phiên cho kiểm định; các lát chồng lấn không độc lập. ADV là kiểm tra '
        'khả năng thanh khoản, không chứng minh khớp lệnh mở cửa. Chưa kiểm chứng điều chỉnh giá/cổ tức.</p>'
        '<p><a href="results.json">Đầy đủ NAV, danh mục, lệnh, ablation, stress và carry attribution</a> · '
        '<a href="regime_labels.json">Nhãn pha VNINDEX</a></p>' + ''.join(tables) + '</html>')


def run_suite(manifest_path: Path, output: Path) -> dict:
    if output.exists():
        raise FileExistsError(f'Refusing to overwrite {output}')
    protocol = load_protocol()
    frames, timelines, rules = load_inputs(manifest_path, protocol)
    labels = ts.classify_regimes(frames['VNINDEX'], protocol['start'], protocol['end'])
    blocks = [dict(id='calendar_2026_h1', kind='calendar', start=protocol['start'], end=protocol['end']),
              protocol['event']] + ts.regime_slices(labels)
    sources = sorted(set(Path('stock_agent').rglob('*.py')) | set(Path('scripts').glob('*.py')))
    sources += [PROTOCOL, gate.REGISTRY, gate.RULES, books.PROTOCOL, ts.PROTOCOL]
    sources += [Path(path) for path in protocol['universes'].values()]
    hashes = {str(path): gate.digest(path) for path in sources}
    result = dict(schema_version=1, status='running', holdout=False, live_eligible=False, protocol=protocol,
        generated_at=datetime.now(timezone.utc).isoformat(),
        git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        manifest_sha256=gate.digest(manifest_path), source_hashes=hashes,
        membership={name: t.data for name, t in timelines.items()}, blocks=[], paired_diagnostics={})
    output.mkdir(parents=True, exist_ok=False)
    (output / 'regime_labels.json').write_text(labels.to_json(orient='records', indent=2), encoding='utf-8')
    initial = float(rules['backtest']['initial_capital'])
    benchmark = rh.benchmark_nav(frames, protocol['start'], protocol['end'], initial)
    full_names = trial_names()
    for universe, timeline in timelines.items():
        active = {s: f for s, f in frames.items() if s in timeline.all_members | {'VNINDEX'}}
        print(f'Building {universe} bank ({len(active)-1} union stocks)', flush=True)
        bank = gate.mr_signal_bank(active, rules, timeline)
        parents = {}
        for i, definition in enumerate(blocks):
            start, end = definition['start'], definition['end']
            if universe == 'VN30':
                counts = labels.loc[labels.date.between(start, end), 'regime'].value_counts()
                result['blocks'].append(dict(**definition, regime_sessions={k: int(counts.get(k, 0)) for k in ts.REGIMES},
                                             universes={}, comparisons={}))
            block = result['blocks'][i]
            clipped = {s: f.loc[f.date <= end].reset_index(drop=True) for s, f in active.items()}
            clipped = {s: f for s, f in clipped.items() if not f.empty}
            names = full_names if i == 0 else [f'{s}/{c}' for s in ts.REFERENCES for c in books.SCENARIOS]
            cash = {name: run_trial(clipped, rules, timeline, bank, start, end, name) for name in names}
            if i == 0:
                parents = cash
                for name, trial in cash.items():
                    if name in {s + '/normal' for s in ts.REFERENCES}:
                        trial['portfolio_history'] = gate.position_history(trial['replay'], clipped)
            carry = {name: ts.attribute_slice(trial['replay'], benchmark, initial, start, end,
                        slip=rules['backtest']['slippage_pct'] / 100 * (2 if name.endswith('/double_cost') else 1))
                     for name, trial in parents.items()}
            block['universes'][universe] = dict(cash_restart=cash, carry=carry)
            (output / (block['id'] + '_' + universe + '.json')).write_text(json.dumps(block['universes'][universe],
                ensure_ascii=False, allow_nan=False), encoding='utf-8')
            print(f"Completed {universe} {block['id']} ({len(cash)} replays)", flush=True)
        partition = [b['universes'][universe]['carry'] for b in result['blocks'] if b['kind'] == 'regime']
        for name, trial in parents.items():
            ending = trial['replay']['nav'][-1]['nav']
            if (abs(sum(b[name]['pnl_vnd'] for b in partition) - (ending - initial)) > 1e-5
                    or not np.isclose(np.prod([1 + b[name]['return_pct'] / 100 for b in partition]), ending / initial)):
                raise ValueError('Universe partition reconciliation failed')
    for block in result['blocks']:
        for mode in ('cash_restart', 'carry'):
            a, b = (block['universes'][u][mode] for u in ('VN30', 'VN100'))
            block['comparisons'][mode] = {name: compare_metrics(a[name]['metrics'] if mode == 'cash_restart' else a[name],
                b[name]['metrics'] if mode == 'cash_restart' else b[name]) for name in a}
    h1 = result['blocks'][0]['universes']
    for name in full_names:
        vn100 = rh.daily_returns(h1['VN100']['cash_restart'][name]['replay']['nav'], initial)
        vn30 = rh.daily_returns(h1['VN30']['cash_restart'][name]['replay']['nav'], initial)
        index = rh.daily_returns(benchmark, initial)
        result['paired_diagnostics'][name] = dict(sample_adequate=len(vn100) >= protocol['min_inference_sessions'],
            status='development_only_not_promotion', **{key: [gate.paired_test(vn100 - ref, family=protocol['family'], block=b)
                for b in protocol['bootstrap_blocks'] if len(vn100) >= b] for key, ref in [('vs_vn30', vn30), ('vs_index', index)]})
    verify_snapshot(manifest_path)
    if gate.digest(manifest_path) != result['manifest_sha256'] or any(gate.digest(Path(p)) != v for p, v in hashes.items()):
        raise ValueError('Source or snapshot changed during experiment')
    result.update(status='universe_research_complete_full_regression_required', partition_reconciled=True,
        trial_count=sum(len(u['cash_restart']) for b in result['blocks'] for u in b['universes'].values()))
    (output / 'results.json').write_text(json.dumps(result, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    (output / 'report.html').write_text(render_report(result), encoding='utf-8')
    return result


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    result = run_suite(args.manifest, args.output)
    print(json.dumps({k: result[k] for k in ('status', 'trial_count', 'partition_reconciled')}))


if __name__ == '__main__':
    main()
