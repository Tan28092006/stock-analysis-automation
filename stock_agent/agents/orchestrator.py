from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from uuid import uuid4

import pandas as pd

from ..config import compute_rules_hash, load_rules, load_universe
from ..constants import LATEST_SCAN_PATH, TRAINING_EVENTS_PATH
from ..data.exchange_calendar import is_trading_day
from ..data.providers import build_providers
from ..data.repository import append_jsonl, write_json
from ..data.validation import pick_cross_checked_frame, validate_ohlcv
from ..features.backtest import BacktestConfig, run_backtest
from ..features.feature_store import save_feature_snapshot
from ..features.ml_models import predict_model_signal
from ..features.signal_engine import score_symbol
from ..schemas import HybridDecisionTrace, RuleEvidence, ScanCandidate, ScanResult, to_plain_dict
from .fundamental_filter import pass_basic_filter
from .risk_guard import equal_score_weights, hrp_weights


def _candidate_backtest_summary(symbol: str, frame: pd.DataFrame, rules: dict) -> dict:
    try:
        if frame.empty:
            return {"status": "insufficient_data"}
        start_idx = max(0, len(frame) - 160)
        result = run_backtest(
            symbol=symbol,
            df=frame,
            rules=rules,
            config=BacktestConfig.from_rules(rules),
            start=frame["date"].iloc[start_idx],
            end=frame["date"].iloc[-1],
        )
        return {
            "status": "ok",
            "start": result.start,
            "end": result.end,
            "total_trades": result.total_trades,
            "win_rate": result.win_rate,
            "avg_return_pct": result.avg_return_pct,
            "expectancy_pct": result.expectancy_pct,
            "max_drawdown_pct": result.max_drawdown_pct,
            "sharpe_ratio": result.sharpe_ratio,
            "sortino_ratio": result.sortino_ratio,
            "profit_factor": result.profit_factor,
        }
    except Exception as exc:
        return {"status": "error", "error": str(exc)}


def run_scan(
    demo: bool = False,
    symbols: list[str] | None = None,
    persist: bool = True,
) -> ScanResult:
    rules = load_rules()
    rules_version = compute_rules_hash(rules)
    universe = load_universe()
    selected_symbols = [item.upper() for item in (symbols or universe["symbols"])]
    providers = build_providers(demo=demo)
    end = date.today()
    start = end - timedelta(days=260)
    scan_id = f"scan-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-{uuid4().hex[:8]}"

    candidates: list[ScanCandidate] = []
    rejected = 0
    warnings: list[str] = []
    selected_price_frames: dict[str, pd.Series] = {}
    feature_events: list[dict] = []
    for symbol in selected_symbols:
        # Lazy provider fetching
        provider_frames = []
        pf0 = providers[0].history(symbol, start, end)
        provider_frames.append(pf0)
        
        pf0_fresh = False
        if pf0.frame is not None and not pf0.frame.empty:
            warns = validate_ohlcv(
                pf0.frame,
                min_rows=rules["min_history_rows"],
                today=end,
                max_age_days=rules["max_price_age_days"],
            )
            latest_date = pf0.frame["date"].iloc[-1]
            if latest_date < end and is_trading_day(end):
                # Cache does not have today's session yet on a weekday, so force online query
                pass
            elif not warns:
                pf0_fresh = True
                
        if len(providers) > 1:
            pf1 = providers[1].history(symbol, start, end)
            provider_frames.append(pf1)
            
            pf1_fresh = False
            if pf1.frame is not None and not pf1.frame.empty:
                warns = validate_ohlcv(
                    pf1.frame,
                    min_rows=rules["min_history_rows"],
                    today=end,
                    max_age_days=rules["max_price_age_days"],
                )
                if not warns:
                    pf1_fresh = True
            
            if len(providers) > 2:
                need_third = False
                if not pf0_fresh or not pf1_fresh:
                    need_third = True
                else:
                    tolerance = rules["cross_check"]["latest_close_tolerance_pct"] / 100.0
                    close0 = float(pf0.frame["close"].iloc[-1])
                    close1 = float(pf1.frame["close"].iloc[-1])
                    date0 = pf0.frame["date"].iloc[-1]
                    date1 = pf1.frame["date"].iloc[-1]
                    pct_diff = abs(close0 - close1) / max(close0, 1e-9)
                    date_gap = abs((date0 - date1).days)
                    if pct_diff > tolerance or date_gap > 3:
                        need_third = True
                
                if need_third:
                    pf2 = providers[2].history(symbol, start, end)
                    provider_frames.append(pf2)

        frame, quality = pick_cross_checked_frame(provider_frames, rules)
        if frame is None or quality.status == "failed":
            rejected += 1
            warnings.append(f"{symbol}: data quality failed ({quality.cross_check_status})")
            candidate = ScanCandidate(
                symbol=symbol,
                decision="REJECT",
                score=0.0,
                confidence=0.0,
                latest_close=0.0,
                latest_date=str(date.today()),
                allocation_weight=0.0,
                data_quality=quality,
                risk_plan=None,
                evidence=[],
                warnings=list(quality.warnings) + [f"data quality failed: {quality.cross_check_status}"]
            )
            candidates.append(candidate)
            continue

        pass_filter, filter_warnings = pass_basic_filter(symbol, frame, rules)
        if not pass_filter:
            rejected += 1
            warnings.append(f"{symbol}: filter failed: {'; '.join(filter_warnings)}")
            candidate = ScanCandidate(
                symbol=symbol,
                decision="REJECT",
                score=0.0,
                confidence=round(quality.confidence, 3),
                latest_close=float(frame["close"].iloc[-1]) if not frame.empty else 0.0,
                latest_date=str(frame["date"].iloc[-1]) if not frame.empty else str(date.today()),
                allocation_weight=0.0,
                data_quality=quality,
                risk_plan=None,
                evidence=[],
                warnings=list(quality.warnings) + filter_warnings + ["fundamental filter failed"]
            )
            candidates.append(candidate)
            continue

        signal = score_symbol(symbol, frame, rules)
        rule_decision = signal.decision
        model_signal = predict_model_signal(symbol, signal, rules)
        if model_signal.status == "available":
            signal.evidence.append(
                RuleEvidence(
                    evidence_id=f"{symbol}:ml_probability",
                    name="ML probability",
                    passed=bool(model_signal.passed),
                    points=0.0,
                    detail="advisory ML probability; deterministic rule guard remains authoritative",
                    value={
                        "model": model_signal.model_family,
                        "probability": model_signal.probability,
                        "threshold": model_signal.threshold,
                    },
                )
            )
        elif rules.get("ml", {}).get("enabled", False):
            signal.warnings.append(f"ML layer unavailable: {model_signal.detail}")

        symbol_warnings = list(quality.warnings) + filter_warnings + signal.warnings
        decision = signal.decision
        confidence = min(0.99, max(0.0, quality.confidence + signal.confidence_delta + model_signal.confidence_delta))
        guardrails = ["deterministic_rules", "risk_plan", "data_quality"]
        notes = [model_signal.detail] if model_signal.detail else []
        if quality.status == "degraded" and decision == "BUY_SETUP":
            decision = "WATCH"
            symbol_warnings.append("degraded data quality downgraded BUY_SETUP to WATCH")
            guardrails.append("degraded_data_downgrade")

        # --- ML Override: downgrade BUY_SETUP → WATCH if model predicts loss ---
        ml_rules = rules.get("ml", {})
        ml_override_enabled = bool(ml_rules.get("override_enabled", False))
        ml_override_applied = False
        if (
            ml_override_enabled
            and decision == "BUY_SETUP"
            and model_signal.status == "available"
            and model_signal.probability is not None
        ):
            override_loss_threshold = float(ml_rules.get("override_loss_threshold", 0.60))
            # probability < (1 - override_loss_threshold) means model thinks loss is likely
            if model_signal.probability < (1.0 - override_loss_threshold):
                decision = "WATCH"
                ml_override_applied = True
                symbol_warnings.append(
                    f"ML override: BUY_SETUP → WATCH "
                    f"(p_win={model_signal.probability:.3f} < {1.0 - override_loss_threshold:.2f})"
                )
                guardrails.append("ml_override_downgrade")
                notes.append("ML ensemble overrode rule-based BUY_SETUP to WATCH due to low win probability")

        hybrid_trace = HybridDecisionTrace(
            rule_decision=rule_decision,
            final_decision=decision,
            rule_score=signal.score,
            model_probability=model_signal.probability,
            model_threshold=model_signal.threshold,
            model_confidence_delta=model_signal.confidence_delta,
            guardrails=guardrails,
            notes=notes,
        )

        candidate = ScanCandidate(
            symbol=symbol,
            decision=decision,
            score=signal.score,
            confidence=round(confidence, 3),
            latest_close=signal.latest_close,
            latest_date=signal.latest_date,
            allocation_weight=0.0,
            data_quality=quality,
            risk_plan=signal.risk_plan,
            evidence=signal.evidence,
            warnings=symbol_warnings,
            model_signal=model_signal,
            hybrid_trace=hybrid_trace,
            backtest_summary=_candidate_backtest_summary(symbol, frame, rules),
            robustness_summary={"status": "portfolio_level_only", "endpoint": "/api/robustness"},
        )

        candidates.append(candidate)
        if decision in {"BUY_SETUP", "WATCH"}:
            selected_price_frames[symbol] = frame.set_index(pd.to_datetime(frame["date"]))["close"].astype(float)
        else:
            rejected += 1

        feature_events.append(
            {
                "scan_id": scan_id,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "rules_version": rules_version,
                "symbol": symbol,
                "decision": decision,
                "data_quality_status": quality.status,
                "cross_check_status": quality.cross_check_status,
                "features": signal.features,
                "model_signal": to_plain_dict(model_signal),
                "hybrid_trace": to_plain_dict(hybrid_trace),
                "label_status": "unlabeled_pending_t2",
            }
        )

    weights: dict[str, float] = {}
    buy_or_watch = [item for item in candidates if item.decision in {"BUY_SETUP", "WATCH"}]
    if buy_or_watch:
        try:
            prices = pd.concat(selected_price_frames, axis=1).dropna(how="all").ffill().tail(120)
            weights = hrp_weights(prices, max_weight=float(rules["risk"]["max_position_weight"]))
        except Exception as exc:
            warnings.append(f"HRP allocation failed; using score weights: {exc}")
            weights = equal_score_weights(
                {item.symbol: item.score for item in buy_or_watch},
                max_weight=float(rules["risk"]["max_position_weight"]),
            )
    for item in candidates:
        item.allocation_weight = weights.get(item.symbol, 0.0)

    result = ScanResult(
        scan_id=scan_id,
        created_at=datetime.now(timezone.utc).isoformat(),
        mode="demo" if demo else "real",
        universe_name=universe["name"],
        symbols_scanned=len(selected_symbols),
        candidates=sorted(candidates, key=lambda item: (item.decision != "BUY_SETUP", item.decision != "WATCH", -item.score)),
        rejected_count=rejected,
        rules_version=rules_version,
        warnings=warnings,
    )

    if persist:
        payload = to_plain_dict(result)
        write_json(LATEST_SCAN_PATH, payload)
        save_feature_snapshot(scan_id, feature_events)
        for event in feature_events:
            append_jsonl(TRAINING_EVENTS_PATH, event)
    return result
