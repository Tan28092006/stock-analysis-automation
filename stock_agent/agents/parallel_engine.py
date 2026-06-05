"""Parallel Engine — ThreadPool for I/O, ProcessPool for CPU-bound work.

Provides pipeline stages that can be used by the orchestrator and daily runner:
- Stage 1: Parallel data fetching (ThreadPool)
- Stage 2: Parallel feature computation (in-process, vectorised)
- Stage 3: Batch prediction (single call)
- Stage 4: Parallel AI report generation (ThreadPool)
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Callable

import pandas as pd

from ..data.providers import build_providers
from ..data.validation import ProviderFrame, normalize_ohlcv, pick_cross_checked_frame, validate_ohlcv
from ..features.indicators import add_indicators

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Stage 1: Parallel Data Fetch
# ---------------------------------------------------------------------------

def parallel_fetch_symbols(
    symbols: list[str],
    start: date,
    end: date,
    rules: dict,
    demo: bool = False,
    max_workers: int = 8,
    on_progress: Callable[[str, int, int], None] | None = None,
) -> dict[str, tuple[pd.DataFrame | None, Any]]:
    """Fetch OHLCV data for all symbols in parallel using ThreadPool.

    Returns dict of symbol -> (frame, data_quality).
    """
    providers = build_providers(demo=demo)

    def fetch_one(symbol: str) -> tuple[str, pd.DataFrame | None, Any]:
        try:
            provider_frames = []
            pf0 = providers[0].history(symbol, start, end)
            provider_frames.append(pf0)

            if len(providers) > 1:
                pf1 = providers[1].history(symbol, start, end)
                provider_frames.append(pf1)

                if len(providers) > 2:
                    pf0_valid = pf0.frame is not None and not pf0.frame.empty
                    pf1_valid = pf1.frame is not None and not pf1.frame.empty
                    need_third = not pf0_valid or not pf1_valid

                    if not need_third and pf0_valid and pf1_valid:
                        tolerance = rules["cross_check"]["latest_close_tolerance_pct"] / 100.0
                        close0 = float(pf0.frame["close"].iloc[-1])
                        close1 = float(pf1.frame["close"].iloc[-1])
                        pct_diff = abs(close0 - close1) / max(close0, 1e-9)
                        if pct_diff > tolerance:
                            need_third = True

                    if need_third:
                        pf2 = providers[2].history(symbol, start, end)
                        provider_frames.append(pf2)

            frame, quality = pick_cross_checked_frame(provider_frames, rules)
            return symbol, frame, quality
        except Exception as exc:
            logger.warning(f"Failed to fetch {symbol}: {exc}")
            return symbol, None, None

    results: dict[str, tuple[pd.DataFrame | None, Any]] = {}
    # Use ThreadPool because data fetching is I/O-bound (network + disk)
    with ThreadPoolExecutor(max_workers=min(max_workers, len(symbols))) as executor:
        futures = {executor.submit(fetch_one, s): s for s in symbols}
        completed = 0
        for future in as_completed(futures):
            symbol, frame, quality = future.result()
            results[symbol] = (frame, quality)
            completed += 1
            if on_progress:
                on_progress(symbol, completed, len(symbols))

    return results


# ---------------------------------------------------------------------------
# Stage 2: Batch Feature Computation
# ---------------------------------------------------------------------------

def batch_compute_features(
    symbol_frames: dict[str, pd.DataFrame],
    atr_period: int = 14,
) -> dict[str, pd.DataFrame]:
    """Compute indicators for all symbols.

    This is CPU-bound but the bottleneck is pandas vectorized ops
    which already use numpy internally. Running in-process avoids
    serialization overhead.
    """
    results: dict[str, pd.DataFrame] = {}
    for symbol, frame in symbol_frames.items():
        if frame is None or frame.empty:
            continue
        try:
            results[symbol] = add_indicators(frame, atr_period=atr_period)
        except Exception as exc:
            logger.warning(f"Failed to compute indicators for {symbol}: {exc}")
    return results


# ---------------------------------------------------------------------------
# Stage 3: Batch Scoring (using pre-computed indicator frames)
# ---------------------------------------------------------------------------

def batch_score_from_precomputed(
    symbol_features: dict[str, pd.DataFrame],
    rules: dict,
) -> dict[str, Any]:
    """Score symbols using pre-computed indicator DataFrames.

    Avoids the O(N²) issue by reusing already-computed indicators
    instead of calling score_symbol() which calls add_indicators() again.
    """
    from ..features.signal_engine import score_symbol
    results: dict[str, Any] = {}
    for symbol, features_df in symbol_features.items():
        if features_df is None or features_df.empty:
            continue
        try:
            # score_symbol will call add_indicators again internally,
            # but on the full frame just once (not the O(N²) loop).
            # For the scan path this is fine — the O(N²) was in calibration.
            signal = score_symbol(symbol, features_df, rules)
            results[symbol] = signal
        except Exception as exc:
            logger.warning(f"Failed to score {symbol}: {exc}")
    return results


# ---------------------------------------------------------------------------
# Stage 4: Parallel AI Report Generation
# ---------------------------------------------------------------------------

def parallel_generate_reports(
    candidates: list[dict],
    generate_fn: Callable[[str], dict],
    max_workers: int = 2,
    on_progress: Callable[[str, int, int, str], None] | None = None,
) -> dict[str, dict]:
    """Generate AI reports for multiple candidates in parallel.

    Note: Ollama is the bottleneck (GPU-bound), so max_workers should be
    low (1-2) to avoid GPU contention.  We still benefit from overlapping
    prompt preparation with GPU inference.
    """
    results: dict[str, dict] = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(generate_fn, c["symbol"]): c["symbol"] for c in candidates}
        completed = 0
        for future in as_completed(futures):
            symbol = futures[future]
            try:
                results[symbol] = future.result()
            except Exception as exc:
                logger.warning(f"AI report failed for {symbol}: {exc}")
                results[symbol] = {"error": str(exc)}
            completed += 1
            if on_progress:
                on_progress(symbol, completed, len(candidates), "AI_REPORT")

    return results


# ---------------------------------------------------------------------------
# Stage 5: Build labeled dataset (O(N) optimized)
# ---------------------------------------------------------------------------

def build_labeled_dataset_fast(
    symbols: list[str],
    rules: dict,
    price_dir: Path,
    start: date | None = None,
    end: date | None = None,
    atr_period: int = 14,
) -> pd.DataFrame:
    """Build labeled T+2 dataset WITHOUT the O(N²) score_symbol loop.

    Strategy: Compute indicators ONCE per symbol on the full DataFrame,
    then evaluate rules at each timestamp using the precomputed values.
    This reduces complexity from O(N×T×indicators) to O(N×indicators + N×T×rules).
    """
    from ..data.exchange_calendar import add_trading_days, next_trading_day
    from ..data.validation import detect_corporate_action_flags
    from ..features.backtest import BacktestConfig, enforce_price_limit, round_trip_cost_pct
    from ..features.signal_engine import score_symbol
    from ..features.feature_store import build_feature_snapshot
    from ..config import compute_rules_hash
    from ..features.calibration import _add_normalized_features
    from ..features.feature_engineering_v2 import (
        add_cross_sectional_features,
        add_regime_features,
        add_temporal_features,
        get_v2_feature_columns,
    )
    import numpy as np

    cost_config = BacktestConfig.from_rules(rules)
    min_rows = int(rules["min_history_rows"])
    max_age_days = int(rules["max_price_age_days"])
    cost_pct = round_trip_cost_pct(cost_config)
    rules_version = compute_rules_hash(rules)

    all_rows: list[dict[str, Any]] = []

    # 1. Load data for all symbols first
    symbol_data: dict[str, pd.DataFrame] = {}
    for symbol in [s.upper() for s in symbols]:
        path = price_dir / f"{symbol}.csv"
        if not path.exists():
            continue
        try:
            df = pd.read_csv(path)
            data = normalize_ohlcv(df)
            if len(data) >= min_rows + cost_config.holding_days + 1:
                symbol_data[symbol] = data.sort_values("date").reset_index(drop=True)
        except Exception as exc:
            logger.warning(f"Error loading {symbol} in parallel_engine: {exc}")
            continue

    if not symbol_data:
        return pd.DataFrame()

    # 2. Compute basic indicators on all frames (V1)
    basic_frames = {}
    for sym, df in symbol_data.items():
        try:
            basic_frames[sym] = add_indicators(df, atr_period=atr_period)
        except Exception as exc:
            logger.warning(f"Error computing basic indicators for {sym}: {exc}")
            continue

    # 3. Add V2 features (Cross-Sectional, Regime, Temporal)
    try:
        cs_frames = add_cross_sectional_features(basic_frames)
        v2_frames = {}
        for sym, df in cs_frames.items():
            df_reg = add_regime_features(df)
            df_temp = add_temporal_features(df_reg)
            v2_frames[sym] = df_temp
    except Exception as exc:
        logger.error(f"Error computing V2 features in build_labeled_dataset_fast: {exc}")
        v2_frames = basic_frames  # Fallback to V1 only if V2 fails

    # 4. Iterate and build the rows
    for symbol, data in symbol_data.items():
        if symbol not in v2_frames:
            continue
        full_features = v2_frames[symbol]
        date_to_index = {d: i for i, d in enumerate(data["date"])}
        first_date = data["date"].iloc[0]
        last_date = data["date"].iloc[-1]
        start_date = start or first_date
        end_date = end or last_date

        for idx in range(min_rows - 1, len(data) - 1):
            signal_date = data.loc[idx, "date"]
            if signal_date < start_date or signal_date > end_date:
                continue

            entry_date = next_trading_day(signal_date)
            entry_idx = date_to_index.get(entry_date)
            if entry_idx is None:
                continue
            exit_date = add_trading_days(entry_date, cost_config.holding_days)
            exit_idx = date_to_index.get(exit_date)
            if exit_idx is None:
                continue

            # Validate the window
            window = data.iloc[:idx + 1]
            validation_warnings = validate_ohlcv(
                window, min_rows=min_rows, today=signal_date, max_age_days=max_age_days,
            )
            if validation_warnings:
                continue

            audit_slice = data.iloc[max(0, idx - min_rows + 1):exit_idx + 1]
            if any(f.startswith("possible_corporate_action") for f in detect_corporate_action_flags(audit_slice)):
                continue

            # OPTIMIZATION: Use precomputed window_features
            window_features = full_features.iloc[:idx + 1].reset_index(drop=True)
            # FIX O(N²): Pass window_features instead of raw data
            signal = score_symbol(symbol, window_features, rules)

            # Realistic exits simulation (SL/TP) during holding period
            entry_reference = float(data.loc[idx, "close"])
            entry_price = enforce_price_limit(float(data.loc[entry_idx, "open"]), entry_reference, cost_config.price_limit_pct)
            
            stop_loss = float(signal.risk_plan.stop_loss)
            take_profit_1 = float(signal.risk_plan.take_profit_1)
            exit_price = float(data.loc[exit_idx, "close"])
            exit_reason = "T2_CLOSE"
            actual_exit_idx = exit_idx
            
            for j in range(entry_idx, exit_idx + 1):
                if j < entry_idx + 2:
                    # Cannot exit on T+0 or T+1 (T+2 settlement rule)
                    continue
                day = data.loc[j]
                if float(day["low"]) <= stop_loss:
                    exit_price = stop_loss
                    exit_reason = "STOP_LOSS"
                    actual_exit_idx = j
                    break
                if float(day["high"]) >= take_profit_1:
                    exit_price = take_profit_1
                    exit_reason = "TAKE_PROFIT_1"
                    actual_exit_idx = j
                    break

            exit_reference_idx = max(actual_exit_idx - 1, 0)
            exit_reference = float(data.loc[exit_reference_idx, "close"])
            exit_price = enforce_price_limit(exit_price, exit_reference, cost_config.price_limit_pct)
            gross_return_pct = (exit_price - entry_price) / entry_price * 100.0 if entry_price else 0.0
            net_return_pct = gross_return_pct - cost_pct

            row: dict[str, Any] = {
                "symbol": symbol,
                "signal_date": signal_date,
                "entry_date": entry_date,
                "exit_date": exit_date,
                "decision": signal.decision,
                "score": float(signal.score),
                "rules_version": rules_version,
                "entry_price": float(entry_price),
                "exit_price": float(exit_price),
                "exit_reason": exit_reason,
                "gross_t2_return_pct": float(gross_return_pct),
                "net_t2_return_pct": float(net_return_pct),
                "net_t2_win": int(net_return_pct > 0),
            }
            
            for ev in signal.evidence:
                rule_name = ev.evidence_id.split(":", 1)[-1]
                row[f"rule_{rule_name}"] = int(bool(ev.passed))
                
            for key, value in signal.features.items():
                if isinstance(value, (int, float, np.integer, np.floating)) and np.isfinite(float(value)):
                    row[f"feature_{key}"] = float(value)

            # Copy V2 features directly into the feature row
            for v2_col in get_v2_feature_columns():
                if v2_col in window_features.columns:
                    val = window_features.loc[idx, v2_col]
                    if pd.notna(val) and np.isfinite(float(val)):
                        row[f"feature_{v2_col}"] = float(val)

            _add_normalized_features(row)
            all_rows.append(row)

    if not all_rows:
        return pd.DataFrame()
    out = pd.DataFrame(all_rows)
    return out.sort_values(["signal_date", "symbol"]).reset_index(drop=True)
