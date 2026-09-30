"""Causal VNINDEX episodes and half-year diagnostics. No production rule changes."""
from __future__ import annotations

import argparse
import html
import json
import subprocess
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from scripts import momentum_books as books
from scripts import momentum_hypotheses as rh
from scripts import period_backtest as bt
from scripts import research_gate as gate
from scripts.historical_universe import Timeline
from stock_agent.data.reconciliation import verify_snapshot

PROTOCOL = Path('configs/research/timeslices_v1.json')
REFERENCES = ('mr/bracket', 'mr/fixed15', 'momentum/baseline')
REGIMES = ('uptrend', 'downtrend', 'transition')


def load_protocol(path=PROTOCOL):
    protocol = json.loads(Path(path).read_text(encoding='utf-8'))
    expected_regime = dict(index='VNINDEX', average='simple', window=200,
                           slope_sessions=20, lag_sessions=1)
    if (protocol['schema_version'] != 1 or protocol['holdout'] is not False
            or protocol['live_eligible'] is not False or protocol['regime'] != expected_regime
            or (protocol['start'], protocol['end']) != ('2022-09-01', '2026-09-29')
            or protocol['required_calendar_slice'] != dict(start='2026-01-01', end='2026-06-30')
            or tuple(protocol['cash_restart_strategies']) != REFERENCES
            or tuple(protocol['scenarios']) != books.SCENARIOS):
        raise ValueError('Timeslice protocol changed; explicitly version it')
    return protocol


def classify_regimes(index, start, end):
    """Label session t from close t-1; never backdate a subsequently detected turn."""
    frame = index.loc[index.date <= end, ['date', 'close']].reset_index(drop=True).copy()
    values = frame.close.to_numpy(dtype=float)
    if (start > end or frame.empty or frame.date.duplicated().any()
            or not frame.date.is_monotonic_increasing
            or not np.isfinite(values).all() or np.any(values <= 0)):
        raise ValueError('Invalid index chronology/prices')
    ma = frame.close.rolling(200, min_periods=200).mean()
    out = pd.DataFrame(dict(date=frame.date, known_on=frame.date.shift(1),
        prior_close=frame.close.shift(1), sma200=ma.shift(1), sma200_previous=ma.shift(21)))
    out = out.loc[out.date.between(start, end)].copy()
    if out.empty or out.isna().any().any():
        raise ValueError('Missing sessions or 220-session regime warmup')
    up = (out.prior_close > out.sma200) & (out.sma200 > out.sma200_previous)
    down = (out.prior_close < out.sma200) & (out.sma200 < out.sma200_previous)
    out['regime'] = np.select([up, down], ['uptrend', 'downtrend'], default='transition')
    return out.reset_index(drop=True)


def regime_slices(labels):
    if labels.empty or not set(labels.regime) <= set(REGIMES):
        raise ValueError('Invalid regime labels')
    episodes = []
    for _, group in labels.groupby(labels.regime.ne(labels.regime.shift()).cumsum(), sort=False):
        episodes.append(dict(id=f'regime_{len(episodes) + 1:03d}', kind='regime',
            regime=str(group.regime.iloc[0]), start=str(group.date.iloc[0]),
            end=str(group.date.iloc[-1]), sessions=len(group), short_sample=len(group) < 20))
    return episodes


def calendar_slices(start, end):
    if date.fromisoformat(start) > date.fromisoformat(end):
        raise ValueError('Reversed calendar bounds')
    slices = []
    for year in range(int(start[:4]), int(end[:4]) + 1):
        for half, (begin, finish) in enumerate([('01-01', '06-30'), ('07-01', '12-31')], 1):
            left, right = f'{year}-{begin}', f'{year}-{finish}'
            if left <= end and right >= start:
                slices.append(dict(id=f'calendar_{year}_h{half}', start=max(start, left),
                    end=min(end, right), kind='calendar', partial=start > left or end < right))
    return slices


def attribute_slice(replay, benchmark, initial, start, end, *, slip=.001):
    """Mark-to-market attribution; retain cash, holdings and fees across the boundary."""
    nav = replay['nav']
    dates = [r['date'] for r in nav]
    if dates != [r['date'] for r in benchmark] or dates != sorted(set(dates)):
        raise ValueError('Unaligned/duplicate NAV and benchmark dates')
    selected = [i for i, day in enumerate(dates) if start <= day <= end]
    if not selected:
        raise ValueError('Empty timeslice')
    first, last = selected[0], selected[-1]
    opening = float(nav[first - 1]['nav']) if first else float(initial)
    index_opening = float(benchmark[first - 1]['nav']) if first else float(initial)
    values = np.asarray([opening] + [r['nav'] for r in nav[first:last + 1]], dtype=float)
    if not np.isfinite(values).all() or np.any(values <= 0) or index_opening <= 0:
        raise ValueError('Invalid NAV for attribution')
    qty, opening_positions, fills = {}, {}, []
    new_entries = additions = 0
    for fill in replay['fills']:
        day, symbol = fill['date'], fill['symbol']
        if day > dates[last]:
            continue
        old = qty.get(symbol, 0)
        if dates[first] <= day:
            fills.append(fill)
            if fill['side'] == 'BUY':
                new_entries += int(old == 0)
                additions += int(old > 0)
        qty[symbol] = old + fill['qty'] * (1 if fill['side'] == 'BUY' else -1)
        if qty[symbol] < 0:
            raise ValueError('Negative inherited holdings')
        if day < dates[first]:
            opening_positions = {s: q for s, q in qty.items() if q}
    ret = float((values[-1] / opening - 1) * 100)
    index_return = float((benchmark[last]['nav'] / index_opening - 1) * 100)
    return dict(sessions=len(selected), opening_nav=opening, closing_nav=float(values[-1]),
        pnl_vnd=float(values[-1] - opening), return_pct=ret, index_return_pct=index_return,
        net_excess_return_pp=ret - index_return, profitable=ret > 0, beats_index=ret > index_return,
        positive_net_and_excess=ret > max(0., index_return),
        max_drawdown_pct=float((values / np.maximum.accumulate(values) - 1).min() * 100),
        average_exposure_pct=float(np.mean([r['exposure'] / r['nav'] for r in nav[first:last + 1]]) * 100),
        fees_vnd=float(sum(f['fee'] for f in fills)),
        modeled_slippage_vnd=float(sum(f['qty'] * f['price'] / (1 + slip if f['side'] == 'BUY' else 1 - slip) * slip for f in fills)),
        buys=sum(f['side'] == 'BUY' for f in fills), sells=sum(f['side'] == 'SELL' for f in fills),
        new_entries=new_entries, additions=additions, opening_positions=opening_positions,
        ending_positions={s: q for s, q in qty.items() if q}, statistical_status='descriptive_development_only')


def cash_reference(frames, rules, timeline, bank, start, end, strategy, scenario):
    kind, variant = strategy.split('/')
    if scenario not in books.SCENARIOS:
        raise ValueError('Unknown scenario')
    if kind == 'books':
        trial = books.run_one(frames, rules, timeline, start, end, variant, scenario)
    else:
        if kind not in ('mr', 'momentum'):
            raise ValueError('Unknown strategy')
        trial = gate.run_one(frames, rules, timeline, bank, start, end, kind, variant, scenario)
    initial = float(rules['backtest']['initial_capital'])
    benchmark = rh.benchmark_nav(frames, start, end, initial)
    slip = rules['backtest']['slippage_pct'] / 100 * (2 if scenario == 'double_cost' else 1)
    # Do not use annualized/variance estimates from a one-session cash experiment.
    metrics = attribute_slice(trial['replay'], benchmark, initial, start, end, slip=slip)
    return dict(metrics=metrics, replay=trial['replay'], reconciliation=trial['reconciliation'])


def parent_trials(parent, book_parent):
    block = parent['blocks']['continuous']
    trials = {f'{kind}/{variant}/normal': trial for kind in ('mr', 'momentum')
              for variant, trial in block[kind].items()}
    trials['momentum/universe_equal/normal'] = block['universe_equal']
    for scenario, stress in block['stresses'].items():
        for strategy in REFERENCES:
            trials[f'{strategy}/{scenario}'] = stress[strategy.replace('/', '_')]
    for scenario, variants in book_parent['blocks']['continuous']['trials'].items():
        for variant, trial in variants.items():
            if variant != 'baseline':
                trials[f'books/{variant}/{scenario}'] = trial
    return trials


def verify_parents(parent, book_parent, manifest_path):
    registry = gate.load_registry()
    gate.validate_evidence(parent, registry)
    if parent['registry'] != registry or book_parent['protocol'] != books.load_protocol():
        raise ValueError('Parent protocol mismatch; rerun full pinned suites')
    for result in (parent, book_parent):
        if result['status'] != 'research_complete_live_blocked' or result['manifest_sha256'] != gate.digest(manifest_path):
            raise ValueError('Incomplete/different parent snapshot')
        required = {'scripts/research_gate.py', 'scripts/period_backtest.py', 'configs/rules_mr.json',
                    'scripts/momentum_books.py', 'configs/research/research_gate_v1.json'}
        hashes = {path.replace('\\', '/'): value for path, value in result['source_hashes'].items()}
        if not required <= hashes.keys() or any(gate.digest(Path(path)) != value for path, value in hashes.items()):
            raise ValueError('Parent source drift; rerun full pinned suites')
    if set(book_parent['blocks']) != set(registry['blocks']) or book_parent['trial_count'] != 180:
        raise ValueError('Incomplete book evidence')
    for block in book_parent['blocks'].values():
        if set(block['trials']) != set(books.SCENARIOS) or any(set(v) != set(books.VARIANTS) for v in block['trials'].values()):
            raise ValueError('Missing book variant/scenario')
    return parent_trials(parent, book_parent)


def render_report(result):
    sections = []
    for kind, title in [('calendar', 'Lát 6 tháng theo lịch'), ('regime', 'Từng pha VNINDEX liên tiếp')]:
        for mode, heading in [('carry', 'Danh mục liên tục — giữ nguyên vị thế qua ranh giới'),
                              ('cash_restart', 'Bắt đầu riêng bằng tiền mặt — kiểm tra độ nhạy')]:
            rows = []
            for block in result['slices']:
                if block['kind'] != kind:
                    continue
                metrics = [block[mode][f'{strategy}/normal'] for strategy in REFERENCES]
                if mode == 'cash_restart':
                    metrics = [item['metrics'] for item in metrics]
                label = block.get('regime', 'mixed calendar')
                count = block['regime_sessions']
                label += f" (U/D/T {count['uptrend']}/{count['downtrend']}/{count['transition']})"
                cells = [block['start'] + ' → ' + block['end'], label, str(block['sessions']),
                         f"{metrics[0]['index_return_pct']:+.2f}%"]
                cells += [f"{m['return_pct']:+.2f}% / {m['max_drawdown_pct']:.2f}% / {m['net_excess_return_pp']:+.2f}đpt" for m in metrics]
                rows.append('<tr>' + ''.join('<td>' + html.escape(c) + '</td>' for c in cells) + '</tr>')
            sections.append(f'<h2>{title} — {heading}</h2><table><tr><th>Giai đoạn</th><th>Pha</th><th>Phiên</th>'
                '<th>VNINDEX</th><th>MR bracket: lãi / DD / vượt</th><th>MR 15: lãi / DD / vượt</th>'
                '<th>Momentum: lãi / DD / vượt</th></tr>' + ''.join(rows) + '</table>')
    notes = ''.join('<li>' + html.escape(note) + '</li>' for note in result['protocol']['notes'])
    return ('<!doctype html><html lang="vi"><meta charset="utf-8"><title>VNINDEX timeslices</title>'
        '<style>body{font:15px system-ui;margin:24px;color:#172333}table{border-collapse:collapse;width:100%;margin:24px 0}'
        'td,th{border:1px solid #ccd6e0;padding:8px;text-align:right}td:first-child,td:nth-child(2){text-align:left}'
        'th{background:#e9eff5}h2{margin-top:36px}li{margin:8px 0}</style><h1>Backtest theo lát thời gian / pha VNINDEX</h1>'
        '<p>Lãi mô phỏng sau phí, không phải lãi đã thực hiện trên thị trường. U/D/T = uptrend/downtrend/chuyển tiếp. '
        'VNINDEX là chỉ số giá gross; cuối lát không giả định bán hết danh mục. DD đo từ NAV mở đầu lát. '
        'Không dùng số gộp 2022–2026 để nghiệm thu, không coi lát chồng lấn là mẫu độc lập.</p>'
        '<p><a href="results.json">Toàn bộ số liệu, mọi ablation và stress</a> · '
        '<a href="regime_labels.json">Nhãn pha và dữ liệu biết trước mỗi phiên</a></p>'
        + ''.join(sections) + '<h2>Quy tắc đã khóa trước khi chạy</h2><ul>' + notes + '</ul></html>')


def run_suite(manifest_path, parent_path, books_path, output):
    protocol = load_protocol()
    manifest = verify_snapshot(manifest_path)
    parent = json.loads(parent_path.read_text(encoding='utf-8'))
    book_parent = json.loads(books_path.read_text(encoding='utf-8'))
    trials = verify_parents(parent, book_parent, manifest_path)
    timeline = Timeline(json.loads(gate.TIMELINE.read_text(encoding='utf-8')))
    rules = json.loads(gate.RULES.read_text(encoding='utf-8'))
    if rules.get('ml', {}).get('enabled'):
        raise ValueError('Retroactive ML forbidden')
    frames = {s: pd.read_csv(manifest_path.parent / manifest['files'][s]['path'])
              for s in sorted(timeline.all_members | {'VNINDEX'})}
    bt.validate_frames(frames, protocol['start'], protocol['end'])
    labels = classify_regimes(frames['VNINDEX'], protocol['start'], protocol['end'])
    slices = calendar_slices(protocol['start'], protocol['end']) + regime_slices(labels)
    output.mkdir(parents=True, exist_ok=False)
    sources = sorted(set(Path('stock_agent').rglob('*.py')) | set(Path('scripts').glob('*.py')))
    sources += [PROTOCOL, gate.REGISTRY, gate.TIMELINE, gate.RULES, books.PROTOCOL]
    hashes = {str(path): gate.digest(path) for path in sources}
    initial = float(rules['backtest']['initial_capital'])
    benchmark = rh.benchmark_nav(frames, protocol['start'], protocol['end'], initial)
    result = dict(schema_version=1, status='running', holdout=False, live_eligible=False, protocol=protocol,
        generated_at=datetime.now(timezone.utc).isoformat(),
        git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        manifest_sha256=gate.digest(manifest_path), source_hashes=hashes,
        parents={str(path.resolve()): gate.digest(path) for path in (parent_path, books_path)}, slices=[])
    (output / 'regime_labels.json').write_text(labels.to_json(orient='records', indent=2), encoding='utf-8')
    print(f'Timeslices: {len(slices)} blocks; building PIT MR bank', flush=True)
    bank = gate.mr_signal_bank(frames, rules, timeline)
    for block in slices:
        start, end = block['start'], block['end']
        counts = labels.loc[labels.date.between(start, end), 'regime'].value_counts()
        block = dict(block, sessions=int(counts.sum()), regime_sessions={k: int(counts.get(k, 0)) for k in REGIMES},
                     carry={}, cash_restart={})
        for name, trial in trials.items():
            slip = rules['backtest']['slippage_pct'] / 100 * (2 if name.endswith('/double_cost') else 1)
            block['carry'][name] = attribute_slice(trial['replay'], benchmark, initial, start, end, slip=slip)
        names = [f'{strategy}/{scenario}' for strategy in REFERENCES for scenario in books.SCENARIOS]
        if block['id'] == 'calendar_2026_h1':
            names = list(trials)
        clipped = {s: f.loc[f.date <= end].reset_index(drop=True) for s, f in frames.items()}
        clipped = {s: f for s, f in clipped.items() if not f.empty}
        for name in names:
            strategy, scenario = name.rsplit('/', 1)
            block['cash_restart'][name] = cash_reference(clipped, rules, timeline, bank, start, end, strategy, scenario)
        result['slices'].append(block)
        (output / (block['id'] + '.json')).write_text(json.dumps(block, allow_nan=False), encoding='utf-8')
        print(f"Completed {block['id']} {start}..{end} ({block['sessions']} sessions)", flush=True)
    # Every continuous trial must be exactly reconstructible from regime attribution.
    partition = [block for block in result['slices'] if block['kind'] == 'regime']
    for name, trial in trials.items():
        parts = [block['carry'][name] for block in partition]
        ending = trial['replay']['nav'][-1]['nav']
        if (abs(sum(part['pnl_vnd'] for part in parts) - (ending - initial)) > 1e-5
                or not np.isclose(np.prod([1 + part['return_pct'] / 100 for part in parts]), ending / initial)):
            raise ValueError('Timeslice accounting failed')
    if any(gate.digest(Path(path)) != value for path, value in hashes.items()):
        raise ValueError('Source changed during experiment')
    if any(gate.digest(Path(path)) != value for path, value in result['parents'].items()):
        raise ValueError('Parent evidence changed during experiment')
    result.update(status='timeslices_complete_not_live_profit',
        cash_trial_count=sum(len(block['cash_restart']) for block in result['slices']),
        carry_trial_count=len(trials), partition_reconciled=True)
    (output / 'results.json').write_text(json.dumps(result, allow_nan=False), encoding='utf-8')
    (output / 'report.html').write_text(render_report(result), encoding='utf-8')
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--gate-results', type=Path, required=True)
    parser.add_argument('--book-results', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    result = run_suite(args.manifest, args.gate_results, args.book_results, args.output)
    print(json.dumps({k: result[k] for k in ('status', 'cash_trial_count', 'carry_trial_count')}))


if __name__ == '__main__':
    main()
