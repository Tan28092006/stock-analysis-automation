from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from uuid import uuid4

import pandas as pd

from ..config import compute_rules_hash, load_rules, load_universe
from ..constants import LATEST_SCAN_PATH, TRAINING_EVENTS_PATH
from ..data.repository import append_jsonl, write_json
from ..features.backtest import BacktestConfig, run_backtest
from ..features.feature_store import save_feature_snapshot
from ..features.ml_models import predict_model_signal
from ..features.signal_engine import score_symbol
from ..schemas import DataQuality, HybridDecisionTrace, RuleEvidence, ScanCandidate, ScanResult, to_plain_dict
from .fundamental_filter import pass_basic_filter
from .parallel_engine import parallel_fetch_symbols
from .risk_guard import equal_score_weights, hrp_weights


def _candidate_backtest_summary(symbol: str, frame: pd.DataFrame, rules: dict) -> dict:
    try:
        if frame.empty:
            return {"status": "insufficient_data"}
        start_idx = max(0, len(frame) - 160)
        # Display-only recent-performance snapshot: run the rule engine WITHOUT the
        # ML override. Running the ensemble per BUY_SETUP bar inside this mini-backtest
        # meant ~1200 ML predictions per scan (the dominant cost); the summary reflects
        # rule performance, so ML is disabled here for speed.
        summary_rules = dict(rules)
        summary_rules["ml"] = {**rules.get("ml", {}), "enabled": False, "override_enabled": False}
        result = run_backtest(
            symbol=symbol,
            df=frame,
            rules=summary_rules,
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


def _load_index_frame(start, end, rules, demo):
    """Load VNINDEX for the market-regime filter: expanded historical cache first,
    then a live provider fetch. Returns a DataFrame or None (caller fails open)."""
    from pathlib import Path

    p = Path("data/raw/prices_hist/VNINDEX.csv")
    if p.exists():
        try:
            df = pd.read_csv(p)
            df["date"] = pd.to_datetime(df["date"]).dt.date
            return df
        except Exception:
            pass
    try:
        res = parallel_fetch_symbols(["VNINDEX"], start, end, rules, demo=demo, max_workers=1)
        frame, _ = res.get("VNINDEX", (None, None))
        return frame
    except Exception:
        return None


def _apply_market_regime(fetched: dict, start, end, rules, demo, warnings: list) -> None:
    """Attach a ``market_regime`` column to fetched frames in place when the filter is
    enabled. Date-keyed by ISO string to avoid date/Timestamp type mismatches. Fail-open."""
    mf = rules.get("market_filter", {})
    if not mf.get("enabled", False):
        return
    try:
        from ..features.market_regime import market_regime_map

        index_frame = _load_index_frame(start, end, rules, demo)
        if index_frame is None or index_frame.empty:
            warnings.append("market_filter enabled but VNINDEX unavailable; filter skipped (fail-open)")
            return
        universe = {s: f for s, (f, q) in fetched.items() if f is not None}
        regime = market_regime_map(
            index_frame, universe,
            ema_period=int(mf.get("ema_period", 50)),
            breadth_ma=int(mf.get("breadth_ma", 20)),
            breadth_min=float(mf.get("breadth_min", 0.4)),
        )
        regime_str = {str(k)[:10]: float(v) for k, v in regime.items()}
        for s, (f, q) in list(fetched.items()):
            if f is None:
                continue
            f = f.copy()
            f["market_regime"] = f["date"].astype(str).str.slice(0, 10).map(regime_str).fillna(1.0)
            fetched[s] = (f, q)
    except Exception as exc:
        warnings.append(f"market_filter skipped: {exc}")


def run_scan(
    demo: bool = False,
    symbols: list[str] | None = None,
    persist: bool = True,
) -> ScanResult:
    rules = load_rules()
    rules_version = compute_rules_hash(rules)
    universe = load_universe()
    selected_symbols = [item.upper() for item in (symbols or universe["symbols"])]
    end = date.today()
    start = end - timedelta(days=260)
    scan_id = f"scan-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-{uuid4().hex[:8]}"

    candidates: list[ScanCandidate] = []
    rejected = 0
    warnings: list[str] = []
    selected_price_frames: dict[str, pd.Series] = {}
    feature_events: list[dict] = []

    # Fetch all symbols concurrently (I/O-bound). Replaces the previous
    # one-symbol-at-a-time provider loop; ~5-10x faster on a full VN30 scan.
    # Cross-check + freshest-source selection is handled inside parallel_fetch_symbols
    # via pick_cross_checked_frame.
    fetched = parallel_fetch_symbols(selected_symbols, start, end, rules, demo=demo, max_workers=8)
    _apply_market_regime(fetched, start, end, rules, demo, warnings)

    for symbol in selected_symbols:
        frame, quality = fetched.get(symbol, (None, None))
        if quality is None:
            quality = DataQuality(
                status="failed",
                confidence=0.0,
                primary_provider=None,
                cross_check_status="fetch_error",
                warnings=["provider fetch raised an exception"],
            )
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
