from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any


@dataclass
class ProviderAudit:
    provider: str
    status: str
    rows: int = 0
    latest_date: str | None = None
    latest_close: float | None = None
    error: str | None = None
    source_type: str = "unknown"
    rejection_reason: str | None = None


@dataclass
class DataQuality:
    status: str
    confidence: float
    primary_provider: str | None
    cross_check_status: str
    warnings: list[str] = field(default_factory=list)
    provider_audits: list[ProviderAudit] = field(default_factory=list)
    corporate_action_flags: list[str] = field(default_factory=list)


@dataclass
class RuleEvidence:
    evidence_id: str
    name: str
    passed: bool
    points: float
    detail: str
    value: Any | None = None


@dataclass
class RiskPlan:
    entry_reference: float
    stop_loss: float
    take_profit_1: float
    take_profit_2: float
    stop_loss_pct: float
    reward_risk: float
    holding_period_days: int = 2


@dataclass
class ModelRun:
    model_family: str
    artifact_path: str | None
    trained_at: str | None
    status: str
    metrics: dict[str, Any] = field(default_factory=dict)
    feature_columns: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


@dataclass
class ModelSignal:
    status: str
    model_family: str | None = None
    probability: float | None = None
    threshold: float | None = None
    passed: bool | None = None
    confidence_delta: float = 0.0
    detail: str = ""
    model_run: ModelRun | None = None
    feature_values: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    # Ensemble-specific fields
    ensemble_agreement: float | None = None
    shap_top_features: list[dict[str, Any]] = field(default_factory=list)
    model_vintage: str | None = None
    drift_status: str | None = None
    base_probabilities: dict[str, float] = field(default_factory=dict)


@dataclass
class HybridDecisionTrace:
    rule_decision: str
    final_decision: str
    rule_score: float
    model_probability: float | None = None
    model_threshold: float | None = None
    model_confidence_delta: float = 0.0
    guardrails: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


@dataclass
class ScanCandidate:
    symbol: str
    decision: str
    score: float
    confidence: float
    latest_close: float
    latest_date: str
    allocation_weight: float
    data_quality: DataQuality
    risk_plan: RiskPlan | None
    evidence: list[RuleEvidence]
    warnings: list[str] = field(default_factory=list)
    model_signal: ModelSignal | None = None
    hybrid_trace: HybridDecisionTrace | None = None
    backtest_summary: dict[str, Any] = field(default_factory=dict)
    robustness_summary: dict[str, Any] = field(default_factory=dict)


@dataclass
class ScanResult:
    scan_id: str
    created_at: str
    mode: str
    universe_name: str
    symbols_scanned: int
    candidates: list[ScanCandidate]
    rejected_count: int
    rules_version: str = ""
    warnings: list[str] = field(default_factory=list)


def to_plain_dict(obj: Any) -> Any:
    import math
    if hasattr(obj, "__dataclass_fields__"):
        return to_plain_dict(asdict(obj))
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    if isinstance(obj, datetime):
        return obj.isoformat()
    if isinstance(obj, list):
        return [to_plain_dict(item) for item in obj]
    if isinstance(obj, dict):
        return {key: to_plain_dict(value) for key, value in obj.items()}
    return obj


@dataclass
class ExperimentManifest:
    command: str
    timestamp: str
    rules_hash: str
    data_start: str
    data_end: str
    symbols: list[str]
    symbols_excluded: list[str] = field(default_factory=list)
    report_links: dict[str, str] = field(default_factory=dict)
    data_hash: str = ""
    data_hashes: dict[str, str] = field(default_factory=dict)

