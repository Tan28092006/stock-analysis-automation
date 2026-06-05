from __future__ import annotations

from copy import deepcopy
from datetime import date
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from ..constants import PRICE_CACHE_DIR
from ..data.validation import normalize_ohlcv
from .backtest import BacktestConfig, run_portfolio_backtest, trade_metric_summary


def run_robustness(
    symbols: Iterable[str],
    rules: dict[str, Any],
    start: date | None = None,
    end: date | None = None,
    price_dir: Path = PRICE_CACHE_DIR,
    seed: int = 42,
    monte_carlo_runs: int = 200,
) -> dict[str, Any]:
    frames = _load_frames(symbols, price_dir)
    cfg = BacktestConfig.from_rules(rules)
    baseline = run_portfolio_backtest(frames, rules, cfg, start=start, end=end)
    returns = [trade.net_return_pct for trade in baseline.trades]
    return {
        "status": "ok" if frames else "insufficient_data",
        "symbols": sorted(frames),
        "baseline": _portfolio_summary(baseline),
        "parameter_sensitivity": _parameter_sensitivity(frames, rules, cfg, start, end),
        "out_of_sample_windows": _out_of_sample_windows(frames, rules, cfg, start, end),
        "monte_carlo": _monte_carlo(returns, seed=seed, runs=monte_carlo_runs),
        "stress_scenarios": _stress_scenarios(returns),
    }


def _load_frames(symbols: Iterable[str], price_dir: Path) -> dict[str, pd.DataFrame]:
    frames: dict[str, pd.DataFrame] = {}
    for symbol in [item.upper() for item in symbols]:
        path = price_dir / f"{symbol}.csv"
        if not path.exists():
            continue
        try:
            frames[symbol] = normalize_ohlcv(pd.read_csv(path))
        except Exception:
            continue
    return frames


def _portfolio_summary(result: Any) -> dict[str, Any]:
    return {
        "start": result.start,
        "end": result.end,
        "total_return_pct": result.total_return_pct,
        "total_trades": result.total_trades,
        "win_rate": result.win_rate,
        "avg_return_pct": result.avg_return_pct,
        "expectancy_pct": result.expectancy_pct,
        "max_drawdown_pct": result.max_drawdown_pct,
        "sharpe_ratio": result.sharpe_ratio,
        "sortino_ratio": result.sortino_ratio,
        "profit_factor": result.profit_factor,
    }


def _parameter_sensitivity(
    frames: dict[str, pd.DataFrame],
    rules: dict[str, Any],
    cfg: BacktestConfig,
    start: date | None,
    end: date | None,
) -> list[dict[str, Any]]:
    scenarios: list[dict[str, Any]] = []
    targets = [
        ("signal_thresholds", "volume_surge_min"),
        ("signal_thresholds", "min_rr"),
        ("risk", "stop_atr_multiple"),
        ("risk", "target_rr"),
    ]
    for section, key in targets:
        if key not in rules.get(section, {}):
            continue
        base_value = float(rules[section][key])
        for multiplier in (0.8, 0.9, 1.1, 1.2):
            scenario_rules = deepcopy(rules)
            scenario_rules[section][key] = round(base_value * multiplier, 6)
            result = run_portfolio_backtest(frames, scenario_rules, cfg, start=start, end=end)
            scenarios.append(
                {
                    "parameter": f"{section}.{key}",
                    "multiplier": multiplier,
                    "value": scenario_rules[section][key],
                    "metrics": _portfolio_summary(result),
                }
            )
    return scenarios


def _out_of_sample_windows(
    frames: dict[str, pd.DataFrame],
    rules: dict[str, Any],
    cfg: BacktestConfig,
    start: date | None,
    end: date | None,
) -> list[dict[str, Any]]:
    dates = sorted({day for frame in frames.values() for day in frame["date"]})
    if start:
        dates = [day for day in dates if day >= start]
    if end:
        dates = [day for day in dates if day <= end]
    if len(dates) < 30:
        return []
    windows = np.array_split(np.array(dates, dtype=object), 3)
    out: list[dict[str, Any]] = []
    for idx, window in enumerate(windows, start=1):
        if len(window) == 0:
            continue
        result = run_portfolio_backtest(frames, rules, cfg, start=window[0], end=window[-1])
        out.append({"window": idx, "start": str(window[0]), "end": str(window[-1]), "metrics": _portfolio_summary(result)})
    return out


def _monte_carlo(returns: list[float], seed: int, runs: int) -> dict[str, Any]:
    if not returns:
        return {"status": "insufficient_trades", "runs": 0}
    rng = np.random.default_rng(seed)
    totals: list[float] = []
    drawdowns: list[float] = []
    for _ in range(runs):
        sample = rng.choice(np.array(returns, dtype=float), size=len(returns), replace=True).tolist()
        total = float(np.prod([1 + ret / 100.0 for ret in sample]) - 1) * 100.0
        totals.append(total)
        drawdowns.append(trade_metric_summary(sample)["max_drawdown_pct"])
    return {
        "status": "ok",
        "runs": runs,
        "seed": seed,
        "total_return_pct_p05": round(float(np.percentile(totals, 5)), 4),
        "total_return_pct_p50": round(float(np.percentile(totals, 50)), 4),
        "total_return_pct_p95": round(float(np.percentile(totals, 95)), 4),
        "max_drawdown_pct_p95": round(float(np.percentile(drawdowns, 95)), 4),
    }


def _stress_scenarios(returns: list[float]) -> list[dict[str, Any]]:
    if not returns:
        return [{"scenario": "none", "status": "insufficient_trades"}]
    scenarios = {
        "bear_minus_3pct_each_trade": [ret - 3.0 for ret in returns],
        "liquidity_extra_0_5pct_cost": [ret - 0.5 for ret in returns],
        "combined_bear_liquidity": [ret - 3.5 for ret in returns],
    }
    out: list[dict[str, Any]] = []
    for name, adjusted in scenarios.items():
        metrics = trade_metric_summary(adjusted)
        out.append({"scenario": name, "metrics": metrics})
    return out
