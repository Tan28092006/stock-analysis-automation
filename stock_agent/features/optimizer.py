from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from ..config import AppConfig, compute_rules_hash, load_rules
from ..constants import MODEL_DIR, PRICE_CACHE_DIR, REPORT_DIR
from ..data.providers import LocalCsvProvider, VnStockProvider, YahooProvider
from ..data.repository import write_json
from ..data.validation import detect_corporate_action_flags, normalize_ohlcv, validate_ohlcv
from .backtest import BacktestConfig, PortfolioBacktestResult, run_portfolio_backtest
from .signal_engine import precompute_signal_frames


@dataclass(frozen=True)
class OptimizerConfig:
    years: int = 3
    n_trials: int = 200
    target_return_pct: float = 10.0
    target_win_rate: float = 60.0
    min_final_trades: int = 30
    max_drawdown_pct: float = 20.0
    seed: int = 42
    train_pct: float = 0.60
    validation_pct: float = 0.20
    refresh_before: bool = False


@dataclass(frozen=True)
class WalkForwardSplit:
    train_start: str
    train_end: str
    validation_start: str
    validation_end: str
    test_start: str
    test_end: str
    total_dates: int


@dataclass
class OptimizationReport:
    status: str
    created_at: str
    rules_version: str
    config: dict[str, Any]
    symbols_requested: list[str]
    symbols_used: list[str]
    symbols_excluded: dict[str, str]
    split: dict[str, Any]
    baseline: dict[str, Any]
    selected_candidate: dict[str, Any] | None
    target_reached: bool
    trials_completed: int
    best_value: float | None
    search_space: list[str]
    refresh_report: dict[str, Any] | None = None
    apply_result: dict[str, Any] | None = None
    report_path: str | None = None
    warnings: list[str] = field(default_factory=list)


SEARCH_SPACE: dict[tuple[str, ...], tuple[str, float, float]] = {
    ("time_horizon_days",): ("int", 2, 4),
    ("signal_thresholds", "buy_setup"): ("int", 50, 78),
    ("signal_thresholds", "watch"): ("int", 30, 60),
    ("signal_thresholds", "volume_surge_min"): ("float", 0.8, 2.0),
    ("signal_thresholds", "rsi_min"): ("int", 35, 55),
    ("signal_thresholds", "rsi_max"): ("int", 65, 80),
    ("signal_thresholds", "min_rr"): ("float", 1.0, 2.5),
    ("signal_thresholds", "max_stop_loss_pct"): ("float", 2.5, 6.0),
    ("signal_thresholds", "adx_min"): ("int", 12, 28),
    ("risk", "stop_atr_multiple"): ("float", 0.8, 2.0),
    ("risk", "target_rr"): ("float", 1.2, 3.0),
    ("portfolio", "max_symbol_weight"): ("float", 0.10, 0.35),
    ("portfolio", "max_gross_exposure"): ("float", 0.30, 0.80),
    ("portfolio", "risk_per_trade_pct"): ("float", 0.5, 3.0),
    ("rule_points", "ema_trend"): ("int", 5, 25),
    ("rule_points", "medium_trend"): ("int", 3, 15),
    ("rule_points", "sma_short_trend"): ("int", 2, 12),
    ("rule_points", "macd_momentum"): ("int", 5, 20),
    ("rule_points", "macd_cross_extra"): ("int", 2, 10),
    ("rule_points", "rsi_t2_zone"): ("int", 4, 18),
    ("rule_points", "volume_confirmation"): ("int", 5, 25),
    ("rule_points", "ichimoku_base"): ("int", 4, 20),
    ("rule_points", "cloud_position"): ("int", 3, 15),
    ("rule_points", "vwap_position"): ("int", 2, 10),
    ("rule_points", "adx_trend_strength"): ("int", 3, 15),
    ("rule_points", "accumulation_flow"): ("int", 2, 10),
    ("rule_points", "breakout_proximity"): ("int", 3, 15),
    ("rule_points", "risk_reward"): ("int", 5, 25),
    ("rule_points", "gap_risk"): ("int", 2, 12),
    ("rule_points", "gap_risk_penalty"): ("int", 5, 25),
}

# Sửa ML_SEARCH_SPACE để đồng bộ hóa probability_threshold
ML_SEARCH_SPACE: dict[tuple[str, ...], tuple[str, float, float]] = {
    ("ml", "probability_threshold"): ("float", 0.45, 0.65),
    ("ml", "override_loss_threshold"): ("float", 0.50, 0.75),
}


def allowed_search_paths(include_ml: bool = False) -> list[str]:
    paths = [".".join(path) for path in SEARCH_SPACE]
    if include_ml:
        paths.extend(".".join(path) for path in ML_SEARCH_SPACE)
    return sorted(paths)


def refresh_price_data(
    symbols: Iterable[str],
    years: int = 3,
    rules: dict[str, Any] | None = None,
    end: date | None = None,
    price_dir: Path = PRICE_CACHE_DIR,
) -> dict[str, Any]:
    rules = _strip_news(deepcopy(rules or load_rules()))
    end = end or date.today()
    start = end - timedelta(days=int(365.25 * years) + 10)
    min_rows = max(int(rules.get("min_history_rows", 90)), int(years * 180))
    max_age_days = int(rules.get("max_price_age_days", 7))
    price_dir.mkdir(parents=True, exist_ok=True)

    providers = [YahooProvider(), VnStockProvider(), LocalCsvProvider(price_dir)]
    results: dict[str, Any] = {}
    saved: list[str] = []
    excluded: dict[str, str] = {}

    for raw_symbol in symbols:
        symbol = raw_symbol.upper()
        attempts: list[dict[str, Any]] = []
        accepted: tuple[str, pd.DataFrame, list[str], list[str]] | None = None
        for provider in providers:
            frame_result = provider.history(symbol, start, end)
            audit = frame_result.audit
            provider_payload = asdict(audit)
            frame = frame_result.frame
            warnings: list[str] = []
            flags: list[str] = []
            if frame is not None and not frame.empty:
                frame = normalize_ohlcv(frame)
                frame = frame[(frame["date"] >= start) & (frame["date"] <= end)].reset_index(drop=True)
                warnings = validate_ohlcv(frame, min_rows=min_rows, today=end, max_age_days=max_age_days)
                flags = detect_corporate_action_flags(frame)
                severe_flags = [flag for flag in flags if flag.startswith("possible_corporate_action")]
                if not warnings and not severe_flags and accepted is None:
                    accepted = (provider.name, frame, warnings, flags)
                    provider_payload["accepted"] = True
                    attempts.append(provider_payload)
                    if provider.name != "local_csv":
                        break
                else:
                    provider_payload["accepted"] = False
                    provider_payload["validation_warnings"] = warnings
                    provider_payload["corporate_action_flags"] = severe_flags[:5]
            attempts.append(provider_payload)

        if accepted is None:
            reason = _best_refresh_failure(attempts)
            excluded[symbol] = reason
            results[symbol] = {"status": "excluded", "reason": reason, "attempts": attempts}
            continue

        provider_name, frame, warnings, flags = accepted
        frame.to_csv(price_dir / f"{symbol}.csv", index=False)
        saved.append(symbol)
        results[symbol] = {
            "status": "ok",
            "provider": provider_name,
            "rows": len(frame),
            "start": str(frame["date"].iloc[0]),
            "end": str(frame["date"].iloc[-1]),
            "warnings": warnings,
            "corporate_action_flags": flags[:8],
            "attempts": attempts,
        }

    return {
        "status": "ok" if saved else "failed",
        "technical_only": True,
        "years": years,
        "start": str(start),
        "end": str(end),
        "min_rows": min_rows,
        "saved_symbols": saved,
        "excluded_symbols": excluded,
        "results": results,
    }


def walk_forward_split(
    frames: dict[str, pd.DataFrame],
    train_pct: float = 0.60,
    validation_pct: float = 0.20,
) -> WalkForwardSplit:
    dates = sorted({day for frame in frames.values() for day in frame["date"]})
    if len(dates) < 30:
        raise ValueError(f"not enough dates for walk-forward split: {len(dates)}")
    train_end_idx = max(1, int(len(dates) * train_pct)) - 1
    validation_end_idx = max(train_end_idx + 1, int(len(dates) * (train_pct + validation_pct))) - 1
    validation_end_idx = min(validation_end_idx, len(dates) - 2)
    return WalkForwardSplit(
        train_start=str(dates[0]),
        train_end=str(dates[train_end_idx]),
        validation_start=str(dates[train_end_idx + 1]),
        validation_end=str(dates[validation_end_idx]),
        test_start=str(dates[validation_end_idx + 1]),
        test_end=str(dates[-1]),
        total_dates=len(dates),
    )


def run_optimization(
    symbols: Iterable[str],
    rules: dict[str, Any] | None = None,
    config: OptimizerConfig | None = None,
    price_dir: Path = PRICE_CACHE_DIR,
    rules_path: Path = AppConfig.default().rules_path,
    apply_if_valid: bool = False,
) -> dict[str, Any]:
    cfg = config or OptimizerConfig()
    base_rules = _strip_news(deepcopy(rules or load_rules(rules_path)))
    requested_symbols = [symbol.upper() for symbol in symbols]
    refresh_report = None
    if cfg.refresh_before:
        refresh_report = refresh_price_data(requested_symbols, years=cfg.years, rules=base_rules, price_dir=price_dir)

    raw_frames, excluded = load_price_frames(requested_symbols, base_rules, cfg.years, price_dir=price_dir)
    if not raw_frames:
        report = OptimizationReport(
            status="insufficient_data",
            created_at=_utc_now(),
            rules_version=compute_rules_hash(base_rules),
            config=asdict(cfg),
            symbols_requested=requested_symbols,
            symbols_used=[],
            symbols_excluded=excluded,
            split={},
            baseline={},
            selected_candidate=None,
            target_reached=False,
            trials_completed=0,
            best_value=None,
            search_space=allowed_search_paths(False),
            refresh_report=refresh_report,
            warnings=["no symbols had enough 3-year OHLCV rows for optimization"],
        )
        return _save_report(report)
    frames = precompute_signal_frames(raw_frames, base_rules)

    try:
        import optuna
    except ImportError as exc:
        report = OptimizationReport(
            status="missing_dependency",
            created_at=_utc_now(),
            rules_version=compute_rules_hash(base_rules),
            config=asdict(cfg),
            symbols_requested=requested_symbols,
            symbols_used=sorted(frames),
            symbols_excluded=excluded,
            split={},
            baseline={},
            selected_candidate=None,
            target_reached=False,
            trials_completed=0,
            best_value=None,
            search_space=allowed_search_paths(False),
            refresh_report=refresh_report,
            warnings=[f"optuna is required: {exc}"],
        )
        return _save_report(report)
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    split = walk_forward_split(frames, train_pct=cfg.train_pct, validation_pct=cfg.validation_pct)
    baseline = _baseline_payload(frames, base_rules, split)
    include_ml = _model_artifact_exists()
    candidates: list[dict[str, Any]] = []

    def objective(trial: Any) -> float:
        trial_rules = sample_rules(trial, base_rules, include_ml=include_ml)
        trial_cfg = BacktestConfig.from_rules(trial_rules)
        train = run_portfolio_backtest(
            frames,
            trial_rules,
            trial_cfg,
            start=date.fromisoformat(split.train_start),
            end=date.fromisoformat(split.train_end),
        )
        validation = run_portfolio_backtest(
            frames,
            trial_rules,
            trial_cfg,
            start=date.fromisoformat(split.validation_start),
            end=date.fromisoformat(split.validation_end),
        )
        test = run_portfolio_backtest(
            frames,
            trial_rules,
            trial_cfg,
            start=date.fromisoformat(split.test_start),
            end=date.fromisoformat(split.test_end),
        )
        params = _extract_search_params(trial_rules, include_ml=include_ml)
        validation_score = _objective_score(_portfolio_summary(validation), _portfolio_summary(train), cfg)
        candidate = {
            "trial_number": trial.number,
            "value": round(float(validation_score), 6),
            "params": params,
            "train": _portfolio_summary(train),
            "validation": _portfolio_summary(validation),
            "final_test": _portfolio_summary(test),
        }
        candidate["target_reached"] = _target_reached(candidate["final_test"], cfg)
        candidate["passes_risk_guard"] = _passes_risk_guard(candidate["final_test"], cfg)
        candidate["selection_score"] = round(_selection_score(candidate["final_test"], cfg), 6)
        trial.set_user_attr("candidate", candidate)
        candidates.append(candidate)
        return validation_score

    sampler = optuna.samplers.TPESampler(seed=cfg.seed)
    study = optuna.create_study(direction="maximize", sampler=sampler)

    def stop_when_target_reached(study_obj: Any, trial: Any) -> None:
        candidate = trial.user_attrs.get("candidate")
        if candidate and candidate.get("target_reached") and _validated_enough(candidate):
            study_obj.stop()

    study.optimize(objective, n_trials=cfg.n_trials, callbacks=[stop_when_target_reached], show_progress_bar=False)
    candidates = [trial.user_attrs["candidate"] for trial in study.trials if "candidate" in trial.user_attrs]
    selected = _select_candidate(candidates, baseline["final_test"], cfg)
    target_reached = bool(selected and selected.get("target_reached"))
    selected_can_apply = bool(selected and _passes_apply_gate(selected.get("final_test", {}), baseline["final_test"], cfg))
    warnings: list[str] = []
    if not selected:
        warnings.append("optimizer did not find a candidate passing validation guards")
    elif not target_reached:
        warnings.append("target not reached; keeping best final-test candidate without forcing a fragile rule set")
        if not selected_can_apply:
            warnings.append("selected candidate is report-only because final-test return/expectancy did not improve baseline or failed risk guard")
    report = OptimizationReport(
        status="target_reached" if target_reached else ("target_not_reached" if selected else "no_valid_candidate"),
        created_at=_utc_now(),
        rules_version=compute_rules_hash(base_rules),
        config=asdict(cfg),
        symbols_requested=requested_symbols,
        symbols_used=sorted(frames),
        symbols_excluded=excluded,
        split=asdict(split),
        baseline=baseline,
        selected_candidate=selected,
        target_reached=target_reached,
        trials_completed=len(candidates),
        best_value=round(float(study.best_value), 6) if candidates else None,
        search_space=allowed_search_paths(include_ml),
        refresh_report=refresh_report,
        warnings=warnings,
    )
    payload = _save_report(report)
    if apply_if_valid:
        payload["apply_result"] = apply_best_rules(payload, rules_path=rules_path)
        write_json(Path(payload["report_path"]), payload)
    return payload


def sample_rules(trial: Any, base_rules: dict[str, Any], include_ml: bool = False) -> dict[str, Any]:
    rules = _strip_news(deepcopy(base_rules))
    spaces = dict(SEARCH_SPACE)
    if include_ml:
        spaces.update(ML_SEARCH_SPACE)
    for path, spec in spaces.items():
        kind, low, high = spec
        name = ".".join(path)
        if kind == "int":
            value = int(trial.suggest_int(name, int(low), int(high)))
        else:
            value = round(float(trial.suggest_float(name, float(low), float(high))), 6)
        _set_nested(rules, path, value)
    buy_setup = int(rules["signal_thresholds"]["buy_setup"])
    rules["signal_thresholds"]["watch"] = min(int(rules["signal_thresholds"]["watch"]), buy_setup - 5)
    rules["signal_thresholds"]["rsi_min"] = min(int(rules["signal_thresholds"]["rsi_min"]), int(rules["signal_thresholds"]["rsi_max"]) - 5)
    return rules


def apply_best_rules(report: dict[str, Any], rules_path: Path = AppConfig.default().rules_path) -> dict[str, Any]:
    candidate = report.get("selected_candidate")
    if not candidate:
        return {"status": "skipped", "reason": "no selected candidate"}
    baseline = report.get("baseline", {}).get("final_test", {})
    final_test = candidate.get("final_test", {})
    cfg = OptimizerConfig(**{**asdict(OptimizerConfig()), **report.get("config", {})})
    if not _passes_apply_gate(final_test, baseline, cfg):
        return {
            "status": "skipped",
            "reason": "candidate did not improve baseline final-test return/expectancy or failed risk guard",
        }

    current_rules = _strip_news(load_rules(rules_path))
    new_rules = _strip_news(deepcopy(current_rules))
    for dotted_path, value in candidate.get("params", {}).items():
        path = tuple(dotted_path.split("."))
        if dotted_path not in allowed_search_paths(include_ml=True):
            return {"status": "rejected", "reason": f"unexpected optimizer path: {dotted_path}"}
        _set_nested(new_rules, path, value)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup_path = rules_path.with_name(f"{rules_path.stem}_backup_{timestamp}{rules_path.suffix}")
    write_json(backup_path, current_rules)
    write_json(rules_path, new_rules)
    return {
        "status": "applied",
        "backup_path": str(backup_path),
        "rules_path": str(rules_path),
        "applied_params": candidate.get("params", {}),
        "new_rules_version": compute_rules_hash(new_rules),
    }


def load_price_frames(
    symbols: Iterable[str],
    rules: dict[str, Any],
    years: int,
    price_dir: Path = PRICE_CACHE_DIR,
) -> tuple[dict[str, pd.DataFrame], dict[str, str]]:
    min_rows = max(int(rules.get("min_history_rows", 90)), int(years * 180))
    frames: dict[str, pd.DataFrame] = {}
    excluded: dict[str, str] = {}
    for raw_symbol in symbols:
        symbol = raw_symbol.upper()
        path = price_dir / f"{symbol}.csv"
        if not path.exists():
            excluded[symbol] = "missing_csv"
            continue
        try:
            frame = normalize_ohlcv(pd.read_csv(path))
        except Exception as exc:
            excluded[symbol] = f"invalid_csv: {exc}"
            continue
        if len(frame) < min_rows:
            excluded[symbol] = f"insufficient_rows: {len(frame)} < {min_rows}"
            continue
        severe_flags = [flag for flag in detect_corporate_action_flags(frame) if flag.startswith("possible_corporate_action")]
        if severe_flags:
            excluded[symbol] = f"severe_corporate_action_flags: {len(severe_flags)}"
            continue
        frames[symbol] = frame
    return frames, excluded


def _baseline_payload(frames: dict[str, pd.DataFrame], rules: dict[str, Any], split: WalkForwardSplit) -> dict[str, Any]:
    cfg = BacktestConfig.from_rules(rules)
    full = run_portfolio_backtest(frames, rules, cfg)
    train = run_portfolio_backtest(frames, rules, cfg, start=date.fromisoformat(split.train_start), end=date.fromisoformat(split.train_end))
    validation = run_portfolio_backtest(
        frames,
        rules,
        cfg,
        start=date.fromisoformat(split.validation_start),
        end=date.fromisoformat(split.validation_end),
    )
    test = run_portfolio_backtest(frames, rules, cfg, start=date.fromisoformat(split.test_start), end=date.fromisoformat(split.test_end))
    return {
        "full": _portfolio_summary(full),
        "train": _portfolio_summary(train),
        "validation": _portfolio_summary(validation),
        "final_test": _portfolio_summary(test),
    }


def _portfolio_summary(result: PortfolioBacktestResult) -> dict[str, Any]:
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


def _objective_score(validation: dict[str, Any], train: dict[str, Any], cfg: OptimizerConfig) -> float:
    score = _selection_score(validation, cfg)
    trades = float(validation.get("total_trades", 0))
    min_validation_trades = max(8.0, cfg.min_final_trades * 0.35)
    if trades < min_validation_trades:
        score -= 120.0 + (min_validation_trades - trades) * 8.0
    if validation.get("win_rate", 0.0) < 50.0:
        score -= (50.0 - float(validation.get("win_rate", 0.0))) * 2.5
    if validation.get("expectancy_pct", 0.0) <= 0:
        score -= 60.0 + abs(float(validation.get("expectancy_pct", 0.0))) * 15.0
    if validation.get("max_drawdown_pct", 0.0) > cfg.max_drawdown_pct:
        score -= 100.0 + (float(validation.get("max_drawdown_pct", 0.0)) - cfg.max_drawdown_pct) * 8.0
    divergence = abs(float(train.get("total_return_pct", 0.0)) - float(validation.get("total_return_pct", 0.0)))
    divergence += abs(float(train.get("win_rate", 0.0)) - float(validation.get("win_rate", 0.0))) * 0.12
    return float(score - divergence)


def _selection_score(metrics: dict[str, Any], cfg: OptimizerConfig) -> float:
    profit_factor = metrics.get("profit_factor")
    pf = float(profit_factor) if profit_factor is not None else 0.0
    return (
        float(metrics.get("total_return_pct", 0.0)) * 3.0
        + float(metrics.get("win_rate", 0.0)) * 0.35
        + float(metrics.get("expectancy_pct", 0.0)) * 18.0
        + float(metrics.get("sharpe_ratio", 0.0)) * 2.5
        + pf * 1.5
        - float(metrics.get("max_drawdown_pct", 0.0)) * 0.75
        + min(float(metrics.get("total_trades", 0.0)), 120.0) * 0.03
    )


def _target_reached(metrics: dict[str, Any], cfg: OptimizerConfig) -> bool:
    return (
        float(metrics.get("total_return_pct", 0.0)) >= cfg.target_return_pct
        and float(metrics.get("win_rate", 0.0)) >= cfg.target_win_rate
        and int(metrics.get("total_trades", 0)) >= cfg.min_final_trades
        and float(metrics.get("max_drawdown_pct", 0.0)) <= cfg.max_drawdown_pct
        and float(metrics.get("expectancy_pct", 0.0)) > 0
    )


def _passes_risk_guard(metrics: dict[str, Any], cfg: OptimizerConfig) -> bool:
    return (
        int(metrics.get("total_trades", 0)) >= max(10, int(cfg.min_final_trades * 0.5))
        and float(metrics.get("max_drawdown_pct", 0.0)) <= cfg.max_drawdown_pct
        and float(metrics.get("expectancy_pct", 0.0)) > 0
    )


def _passes_apply_gate(final_test: dict[str, Any], baseline: dict[str, Any], cfg: OptimizerConfig) -> bool:
    return (
        _passes_risk_guard(final_test, cfg)
        and float(final_test.get("total_return_pct", 0.0)) > float(baseline.get("total_return_pct", 0.0))
        and float(final_test.get("expectancy_pct", 0.0)) > float(baseline.get("expectancy_pct", 0.0))
    )


def _validated_enough(candidate: dict[str, Any]) -> bool:
    validation = candidate.get("validation", {})
    return int(validation.get("total_trades", 0)) >= 8 and float(validation.get("expectancy_pct", 0.0)) > 0


def _select_candidate(
    candidates: list[dict[str, Any]],
    baseline_final_test: dict[str, Any],
    cfg: OptimizerConfig,
) -> dict[str, Any] | None:
    if not candidates:
        return None
    target_candidates = [
        candidate
        for candidate in candidates
        if candidate.get("target_reached") and _validated_enough(candidate)
    ]
    if target_candidates:
        selected = max(target_candidates, key=lambda item: (item["selection_score"], item["value"]))
        selected["selection_reason"] = "strict target reached on final test"
        return selected
    guarded = [
        candidate
        for candidate in candidates
        if _passes_apply_gate(candidate.get("final_test", {}), baseline_final_test, cfg)
        and _validated_enough(candidate)
    ]
    if guarded:
        selected = max(guarded, key=lambda item: (item["selection_score"], item["final_test"]["total_return_pct"], item["value"]))
        selected["selection_reason"] = "best OOS candidate improved baseline but strict target was not reached"
        return selected
    fallback = max(candidates, key=lambda item: (item["selection_score"], item["value"]))
    fallback["selection_reason"] = "target not reached; best observed final-test candidate is report-only unless apply gate passes"
    return fallback


def _extract_search_params(rules: dict[str, Any], include_ml: bool) -> dict[str, Any]:
    paths = list(SEARCH_SPACE)
    if include_ml:
        paths += list(ML_SEARCH_SPACE)
    return {".".join(path): _get_nested(rules, path) for path in paths}


def _strip_news(rules: dict[str, Any]) -> dict[str, Any]:
    rules.pop("news", None)
    return rules


def _set_nested(payload: dict[str, Any], path: tuple[str, ...], value: Any) -> None:
    target = payload
    for key in path[:-1]:
        target = target.setdefault(key, {})
    target[path[-1]] = value


def _get_nested(payload: dict[str, Any], path: tuple[str, ...]) -> Any:
    target: Any = payload
    for key in path:
        target = target[key]
    return target


def _model_artifact_exists() -> bool:
    registry = MODEL_DIR / "model_registry.json"
    ensemble = MODEL_DIR / "ensemble" / "ensemble_registry.json"
    return registry.exists() or ensemble.exists()


def _best_refresh_failure(attempts: list[dict[str, Any]]) -> str:
    for attempt in attempts:
        warnings = attempt.get("validation_warnings")
        if warnings:
            return "; ".join(warnings)
        if attempt.get("error"):
            return str(attempt["error"])
    return "no provider returned a valid 3-year OHLCV frame"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _save_report(report: OptimizationReport) -> dict[str, Any]:
    payload = asdict(report)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    path = REPORT_DIR / f"optimization_{report.created_at.replace(':', '').replace('+', 'Z')}.json"
    payload["report_path"] = str(path)
    write_json(path, payload)
    return payload
