"""Locked book-inspired momentum experiments and frequency diagnosis; research only."""
from __future__ import annotations

import argparse
import copy
import json
import math
import subprocess
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from scripts import period_backtest as bt, research_gate as gate, momentum_hypotheses as rh
from scripts.historical_universe import Timeline
from stock_agent.data.reconciliation import verify_snapshot
from stock_agent.features.signal_engine import prepare_signal_frame, _score_mean_reversion_from_features

PROTOCOL = Path('configs/research/momentum_books_v1.json')
VARIANTS = ('baseline','weekly','slope90','quality','breakout50','breakout50_volume')
COMPARATORS = dict(weekly='baseline',slope90='weekly',quality='weekly',breakout50='weekly',breakout50_volume='breakout50')
SCENARIOS = ('normal','double_cost','delay_one_session')


def load_protocol(path=PROTOCOL):
    p = json.loads(Path(path).read_text(encoding='utf-8'))
    if (p['holdout'] is not False or p['live_eligible'] is not False or tuple(p['variants']) != VARIANTS
            or p['comparators'] != COMPARATORS or tuple(p['scenarios']) != SCENARIOS
            or p['family'] != 27 or p['bootstrap_blocks'] != [20,40]
            or p['min_sessions'] != 252 or p['alpha'] != .05):
        raise ValueError('Protocol differs from locked implementation; version explicitly')
    gate.load_registry()
    return p


def _closes(close, minimum):
    c = np.asarray(close, dtype=float)
    if c.ndim != 1 or len(c) < minimum or not np.isfinite(c).all() or np.any(c <= 0):
        raise ValueError('Insufficient or invalid close history')
    return c


def slope_score(close):
    """Clenow-inspired ranking component only; not the book's full strategy."""
    y = np.log(_closes(close,90)[-90:])
    x = np.arange(90, dtype=float)
    x -= x.mean()
    centered = y - y.mean()
    variance = float(centered @ centered)
    if variance < 1e-20:
        return 0.
    slope = float(x @ centered / (x @ x))
    residual = centered - slope*x
    r2 = float(np.clip(1 - (residual @ residual)/variance, 0, 1))
    return float(np.expm1(slope*252)*r2)


def quality_score(close):
    """Negative daily-return information discreteness over the same 12-1 interval."""
    c = _closes(close,254)[-253:-21]
    r = c[1:]/c[:-1]-1
    return float(np.sign(c[-1]/c[0]-1)*(np.mean(r>0)-np.mean(r<0)))


def breakout_allowed(frame, *, volume=False):
    if len(frame)<51:
        return False
    f = frame.tail(51)
    a = f[['close','high','volume']].to_numpy(dtype=float)
    if not np.isfinite(a).all() or np.any(a[:,:2]<=0) or np.any(a[:,2]<0):
        return False
    breakout = float(f.close.iloc[-1]) > float(f.high.iloc[:-1].max())
    avg = float(f.volume.iloc[:-1].mean())
    return breakout and (not volume or (avg>0 and float(f.volume.iloc[-1]) >=1.5*avg))


def targets(frames, held, variant):
    if variant not in VARIANTS:
        raise ValueError('Unknown momentum book variant')
    if variant in ('baseline','weekly','breakout50','breakout50_volume'):
        return gate.momentum_targets(frames,held,'baseline')
    ranked = bt._rank({s:f for s,f in frames.items() if s!='VNINDEX'})
    eligible = {r[0] for r in ranked}
    if variant == 'quality':
        # Freeze the absolute-momentum shortlist before applying the quality ranking.
        ranked = sorted(ranked[:20], key=lambda r:(-quality_score(frames[r[0]].close),-r[1],r[0]))
    else:
        ranked = sorted(ranked, key=lambda r:(-slope_score(frames[r[0]].close),r[0]))
    selected = [r for r in ranked[:20] if r[0] in held][:10]
    for row in ranked:
        if len(selected)>=10:
            break
        if row[0] not in {r[0] for r in selected}:
            selected.append(row)
    inv = {s:1/max(vol,.05) for s,_,vol,_ in selected}
    vol = float(frames['VNINDEX'].close.pct_change().tail(20).std()*math.sqrt(252))
    if not np.isfinite(vol):
        raise ValueError('Invalid benchmark volatility')
    exposure = min(1.,.2/max(vol,1e-6))
    total = sum(inv.values()) or 1.
    return {s:v/total*exposure for s,v in inv.items()}, sorted(set(frames)-{'VNINDEX'}-eligible)


def frequency(result):
    qty, new, additions = {}, 0, 0
    buys = [f for f in result['fills'] if f['side']=='BUY']
    for fill in result['fills']:
        s = fill['symbol']
        old = qty.get(s,0)
        if fill['side']=='BUY':
            new += int(old==0)
            additions += int(old>0)
        qty[s] = old + fill['qty']*(1 if fill['side']=='BUY' else -1)
    return dict(buy_fills=len(buys),new_entries=new,additions=additions,
                buy_dates=len({f['date'] for f in buys}),unique_bought_symbols=len({f['symbol'] for f in buys}),
                rebalances=len(result.get('rebalances',[])))


def run_one(frames,rules,timeline,start,end,variant,scenario):
    if variant not in VARIANTS or scenario not in SCENARIOS:
        raise ValueError('Unknown registered experiment')
    configured = copy.deepcopy(rules)
    if configured.get('ml',{}).get('enabled'):
        raise ValueError('Retroactive ML forbidden')
    if scenario=='double_cost':
        for key in ('commission_pct','sell_tax_pct','slippage_pct'):
            configured['backtest'][key]*=2
    def members(prefix):
        prior = str(prefix['VNINDEX'].date.iloc[-1])
        return timeline.members(prior,bt.add_trading_days(date.fromisoformat(prior),1).isoformat())
    def target(prefix,held):
        active = members(prefix)
        return targets({s:f for s,f in prefix.items() if s in active|{'VNINDEX'}},held,variant)
    def entry(symbol,prefix):
        return symbol in members(prefix) and (not variant.startswith('breakout') or
                breakout_allowed(prefix[symbol],volume=variant=='breakout50_volume'))
    result = bt.replay_momentum(frames,configured,start,end,target_fn=target,entry_gate=entry,
                               execution_delay=int(scenario=='delay_one_session'),
                               rebalance='monthly' if variant=='baseline' else 'weekly')
    initial = float(configured['backtest']['initial_capital'])
    benchmark = rh.benchmark_nav(frames,start,end,initial)
    summary = rh.summarize(result,benchmark,initial,configured['backtest']['slippage_pct']/100)
    matched = gate.exposure_control(result['nav'],rh.daily_returns(benchmark,initial),initial)
    summary['exposure_matched_index_return_pct'] = float((np.prod(1+matched)-1)*100)
    return dict(summary=summary,frequency=frequency(result),replay=result,
                reconciliation=rh.reconcile(result,frames,initial))


def mr_funnel(frames,rules,timeline,start,end):
    """Counts are ordered AND-gates, not marginal causal contributions."""
    selected = copy.deepcopy(rules)
    selected['mean_reversion'].update(rules.get('vn30_mean_reversion',{}))
    counts = dict(eligible_bars=0,rsi=0,band=0,confirmation=0,rr=0,cloud=0)
    fails_alone = {k:0 for k in list(counts)[1:]}
    signal_days = {}
    for symbol,frame in sorted(frames.items()):
        if symbol=='VNINDEX':
            continue
        frame = frame.loc[frame.date<=end].reset_index(drop=True)
        prepared = prepare_signal_frame(frame.copy(),selected)
        for i in range(max(1,int(rules.get('min_history_rows',90))-1),len(frame)):
            day = str(frame.date.iloc[i])
            execution = bt.add_trading_days(date.fromisoformat(day),1).isoformat()
            if not start<=execution<=end or day<timeline.start or symbol not in timeline.members(day,execution):
                continue
            sig = _score_mean_reversion_from_features(symbol,prepared.iloc[:i+1],selected,idx=i,include_features=False)
            e = {x.name:x.passed for x in sig.evidence}
            gates = dict(rsi=e['Oversold RSI'],band=e['At lower band'],
                confirmation=(e['Reversal bar'] and e['Volume climax']) or e.get('VSA stopping volume',False),
                rr=e.get('MR risk reward',True),cloud=e.get('Cloud clearance',True))
            counts['eligible_bars']+=1
            passed = True
            for name,value in gates.items():
                passed = passed and value
                counts[name]+=int(passed)
                fails_alone[name]+=int(not value and all(v for k,v in gates.items() if k!=name))
            if passed:
                signal_days.setdefault(day,[]).append(symbol)
    return dict(sequential_pass=counts,only_gate_failing=fails_alone,
                signals=sum(len(v) for v in signal_days.values()),signal_days=signal_days,
                interpretation='Sequential counts depend on gate order; alone-fail counts are not profitability.')


def run_suite(manifest_path,output):
    protocol, registry = load_protocol(), gate.load_registry()
    timeline = Timeline(json.loads(gate.TIMELINE.read_text(encoding='utf-8')))
    rules = json.loads(gate.RULES.read_text(encoding='utf-8'))
    manifest = verify_snapshot(manifest_path)
    required = timeline.all_members|{'VNINDEX'}
    if not required<=set(manifest['files']) or max(b['end'] for b in registry['blocks'].values())>min(manifest['as_of'],timeline.end):
        raise ValueError('Snapshot does not cover locked universe/windows')
    frames = {s:pd.read_csv(manifest_path.parent/manifest['files'][s]['path']) for s in sorted(required)}
    bt.validate_frames(frames,'2022-09-01','2026-09-29')
    output.mkdir(parents=True,exist_ok=False)
    sources = sorted(set(Path('stock_agent').rglob('*.py'))|set(Path('scripts').glob('*.py')))
    sources += [PROTOCOL,gate.REGISTRY,gate.RULES,gate.TIMELINE]
    result = dict(schema_version=1,status='running',holdout=False,live_eligible=False,protocol=protocol,
        generated_at=datetime.now(timezone.utc).isoformat(),manifest=str(manifest_path.resolve()),
        manifest_sha256=gate.digest(manifest_path),git_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        source_hashes={str(p):gate.digest(p) for p in sources},blocks={})
    initial = float(rules['backtest']['initial_capital'])
    recent = registry['blocks']['recent_6m']
    result['mr_recent_funnel'] = mr_funnel(frames,rules,timeline,recent['start'],recent['end'])
    print(json.dumps(result['mr_recent_funnel'],ensure_ascii=False),flush=True)
    for name,block in registry['blocks'].items():
        print('Running books '+name,flush=True)
        clipped = {s:f.loc[f.date<=block['end']].reset_index(drop=True) for s,f in frames.items()}
        clipped = {s:f for s,f in clipped.items() if not f.empty}
        trials = {scenario:{v:run_one(clipped,rules,timeline,block['start'],block['end'],v,scenario)
                            for v in VARIANTS} for scenario in SCENARIOS}
        if name=='continuous':
            for v,ref in COMPARATORS.items():
                normal = trials['normal']
                delta = rh.daily_returns(normal[v]['replay']['nav'],initial)-rh.daily_returns(normal[ref]['replay']['nav'],initial)
                normal[v]['paired_tests'] = [gate.paired_test(delta,family=protocol['family'],block=b) for b in protocol['bootstrap_blocks']]
                normal[v]['statistical_status'] = ('candidate_development_only' if len(delta)>=protocol['min_sessions'] and all(
                    t['ci_family'][0]>0 and t['p_one_sided_bonferroni']<protocol['alpha'] for t in normal[v]['paired_tests']) else 'inconclusive_or_no_edge')
        result['blocks'][name] = dict(**block,trials=trials)
        (output/(name+'.json')).write_text(json.dumps(result['blocks'][name],allow_nan=False),encoding='utf-8')
        print(json.dumps({v:dict(return_pct=round(t['summary']['return_pct'],3),**t['frequency']) for v,t in trials['normal'].items()}),flush=True)
    if any(gate.digest(Path(p))!=digest for p,digest in result['source_hashes'].items()):
        raise ValueError('Source changed during experiment')
    result['status'] = 'research_complete_live_blocked'
    result['trial_count'] = sum(len(s) for b in result['blocks'].values() for s in b['trials'].values())
    if result['trial_count']!=180:
        raise ValueError('Incomplete evidence')
    (output/'results.json').write_text(json.dumps(result,ensure_ascii=False,allow_nan=False),encoding='utf-8')
    return result


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args = p.parse_args(argv)
    result = run_suite(args.manifest,args.output)
    print(json.dumps(dict(status=result['status'],trials=result['trial_count'])))


if __name__=='__main__':
    main()
