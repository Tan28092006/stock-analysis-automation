"""Parity guard: the fast T+2 label builder must use the canonical exit replay
(features/mr_exit.simulate_mr_exit) and keep its pre-refactor label semantics.

All data here is synthetic. `_legacy_exit` is a verbatim reference copy of the inline loop
that parallel_engine.build_labeled_dataset_fast used before the refactor; it exists only as
a characterization oracle and must not be imported by production code.
"""
import inspect
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from stock_agent.agents import parallel_engine
from stock_agent.data.exchange_calendar import add_trading_days, next_trading_day, trading_days_between
from stock_agent.features import mr_exit
from stock_agent.features.backtest import enforce_price_limit
from stock_agent.features.mr_exit import simulate_mr_exit

REASON = {"stop": "STOP_LOSS", "target": "TAKE_PROFIT_1", "time": "T2_CLOSE"}


def _legacy_exit(data, entry_idx, exit_idx, stop_loss, take_profit_1):
    exit_price = float(data.loc[exit_idx, "close"])
    exit_reason = "T2_CLOSE"
    actual_exit_idx = exit_idx
    for j in range(entry_idx, exit_idx + 1):
        if j < entry_idx + 2:
            continue
        day = data.loc[j]
        if float(day["low"]) <= stop_loss:
            exit_price, exit_reason, actual_exit_idx = stop_loss, "STOP_LOSS", j
            break
        if float(day["high"]) >= take_profit_1:
            exit_price, exit_reason, actual_exit_idx = take_profit_1, "TAKE_PROFIT_1", j
            break
    return actual_exit_idx, float(exit_price), exit_reason


def _canonical_as_called_by_builder(data, entry_idx, exit_idx, stop_loss, take_profit_1):
    j, px, reason, resolved = simulate_mr_exit(data, entry_idx, stop_loss, take_profit_1,
                                               max_hold=exit_idx - entry_idx, settle_lock=2)
    assert resolved
    return j, float(px), REASON[reason]


def _random_bars(rng, n, start_price=20000.0):
    close = start_price * np.exp(np.cumsum(rng.normal(0, 0.02, n)))
    opn = close * np.exp(rng.normal(0, 0.01, n))
    high = np.maximum(opn, close) * (1 + np.abs(rng.normal(0, 0.01, n)))
    low = np.minimum(opn, close) * (1 - np.abs(rng.normal(0, 0.01, n)))
    return pd.DataFrame({"open": opn, "high": high, "low": low, "close": close})


def test_fast_builder_calls_canonical_exit_and_has_no_inline_loop():
    src = inspect.getsource(parallel_engine.build_labeled_dataset_fast)
    assert "simulate_mr_exit(" in src
    assert "for j in range(entry_idx" not in src  # the copied loop must not return


def test_canonical_exit_equals_legacy_loop_on_random_paths():
    rng = np.random.default_rng(20261006)
    seen = set()
    for _ in range(600):
        n = int(rng.integers(5, 40))
        bars = _random_bars(rng, n)
        for _ in range(4):
            e = int(rng.integers(0, n - 1))
            x = int(rng.integers(e, n))  # includes horizons 0 and 1 (shorter than the lock)
            ref = float(bars.close[max(e - 1, 0)])
            stop = ref * (1 - rng.uniform(0.0, 0.12))
            target = ref * (1 + rng.uniform(0.0, 0.12))
            legacy = _legacy_exit(bars, e, x, stop, target)
            assert _canonical_as_called_by_builder(bars, e, x, stop, target) == legacy
            seen.add(legacy[2])
    assert seen == {"STOP_LOSS", "TAKE_PROFIT_1", "T2_CLOSE"}


def test_same_bar_stop_and_target_keeps_stop_precedence_and_settle_lock():
    bars = pd.DataFrame({"open": [100.0] * 5, "high": [101.0, 101.0, 130.0, 125.0, 101.0],
                         "low": [99.0, 99.0, 80.0, 88.0, 99.0], "close": [100.0] * 5})
    # bar 2 is inside the T+2 lock (would hit both legs) and is ignored; bar 3 hits both -> stop.
    assert _legacy_exit(bars, 1, 4, 90.0, 120.0) == (3, 90.0, "STOP_LOSS")
    assert _canonical_as_called_by_builder(bars, 1, 4, 90.0, 120.0) == (3, 90.0, "STOP_LOSS")


def test_zero_target_is_the_one_documented_divergence():
    # The only input class where the legacy loop and the canonical replay differ: with
    # take_profit_1 == 0 the legacy loop "hits" the target at price 0 on the first bar after
    # the lock, while simulate_mr_exit's falsy guard disables that leg. Unreachable from
    # signal_engine (take_profit_1 >= close * 1.005 > 0, signal_engine.py:148,463); pinned so
    # the divergence stays deliberate and visible.
    bars = pd.DataFrame({"open": [100.0] * 6, "high": [101.0] * 6, "low": [99.0] * 6,
                         "close": [100.0, 101.0, 102.0, 103.0, 104.0, 105.0]})
    assert _legacy_exit(bars, 1, 4, 90.0, 0.0) == (3, 0.0, "TAKE_PROFIT_1")
    assert _canonical_as_called_by_builder(bars, 1, 4, 90.0, 0.0) == (4, 104.0, "T2_CLOSE")


def _write_symbol(tmp_path, symbol, days, seed):
    rng = np.random.default_rng(seed)
    bars = _random_bars(rng, len(days))
    bars.insert(0, "date", [d.isoformat() for d in days])
    bars["volume"] = 1_000_000.0
    bars.to_csv(tmp_path / f"{symbol}.csv", index=False)


def test_build_labeled_dataset_fast_rows_match_legacy_semantics(tmp_path, monkeypatch):
    from stock_agent.features import signal_engine

    days = trading_days_between(pd.Timestamp("2025-03-03").date(), pd.Timestamp("2025-06-30").date())
    halted = set(days[45:48])  # synthetic suspension: bars absent while the calendar counts them
    _write_symbol(tmp_path, "AAA", days, 1)
    _write_symbol(tmp_path, "BBB", [d for d in days if d not in halted], 2)

    def fake_score(symbol, window, rules):
        close = float(window["close"].iloc[-1])
        plan = SimpleNamespace(stop_loss=round(close * 0.97, 4), take_profit_1=round(close * 1.03, 4))
        return SimpleNamespace(decision="BUY_SETUP", score=50.0, risk_plan=plan, evidence=[], features={})

    monkeypatch.setattr(signal_engine, "score_symbol", fake_score)
    monkeypatch.setattr(signal_engine, "precompute_signal_frames",
                        lambda frames, rules: {s: f.reset_index(drop=True) for s, f in frames.items()})
    rules = {"min_history_rows": 20, "max_price_age_days": 7, "time_horizon_days": 5,
             "backtest": {"commission_pct": 0.15, "sell_tax_pct": 0.1, "slippage_pct": 0.1,
                          "price_limit_pct": 6.9}}
    out = parallel_engine.build_labeled_dataset_fast(["AAA", "BBB"], rules, tmp_path,
                                                     as_of=pd.Timestamp("2025-06-30").date())
    assert not out.empty and set(out.symbol) == {"AAA", "BBB"}
    assert set(out.exit_reason) == {"STOP_LOSS", "TAKE_PROFIT_1", "T2_CLOSE"}

    expected_rows = skipped_by_calendar = 0
    for symbol in ("AAA", "BBB"):
        data = pd.read_csv(tmp_path / f"{symbol}.csv")
        data["date"] = pd.to_datetime(data["date"]).dt.date
        index = {d: i for i, d in enumerate(data["date"])}
        got = out[out.symbol == symbol].set_index("signal_date")
        for idx in range(19, len(data) - 1):
            signal_date = data.loc[idx, "date"]
            entry_date = next_trading_day(signal_date)
            entry_idx = index.get(entry_date)
            exit_idx = index.get(add_trading_days(entry_date, 5))
            if entry_idx is None or exit_idx is None:
                assert signal_date not in got.index  # calendar exit/entry bar missing -> row skipped
                skipped_by_calendar += 1
                continue
            close = float(data.loc[idx, "close"])
            stop, target = round(close * 0.97, 4), round(close * 1.03, 4)
            j, px, reason = _legacy_exit(data, entry_idx, exit_idx, stop, target)
            px = enforce_price_limit(px, float(data.loc[max(j - 1, 0), "close"]), 6.9)
            entry = enforce_price_limit(float(data.loc[entry_idx, "open"]), close, 6.9)
            row = got.loc[signal_date]
            assert (row.exit_date, row.exit_reason) == (data.loc[j, "date"], reason)
            assert row.exit_price == px
            assert row.net_t2_return_pct == pytest.approx((px - entry) / entry * 100.0 - 0.6, abs=1e-9)
            expected_rows += 1
    assert expected_rows == len(out)
    assert skipped_by_calendar > 0  # the synthetic suspension exercised the calendar-skip branch


def test_mr_exit_docstring_lists_every_caller():
    root = Path(mr_exit.__file__).resolve().parents[1]  # stock_agent/
    callers = {p.relative_to(root).as_posix() for p in root.rglob("*.py")
               if p.name != "mr_exit.py" and "simulate_mr_exit(" in p.read_text(encoding="utf-8")}
    assert "agents/parallel_engine.py" in callers
    missing = sorted(c for c in callers if c not in mr_exit.__doc__)
    assert not missing, f"mr_exit docstring does not list callers: {missing}"
