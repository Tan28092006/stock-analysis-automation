from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any

from ..constants import DATA_DIR
from ..data.repository import read_json, write_json
from ..reports.contradiction_detector import detect_contradictions
from ..reports.evidence import build_evidence_bundle, validate_ai_summary
from .ollama_client import OllamaChatClient

REPORT_CACHE_DIR = DATA_DIR / "reports" / "ai_cache"


# ── Data classes ────────────────────────────────────────────────

@dataclass
class ChecklistItem:
    criterion: str
    status: str        # PASS / FAIL / WARNING
    detail: str


@dataclass
class StockAnalysis:
    symbol: str
    decision: str
    summary: str
    contradictions: list[str] = field(default_factory=list)
    entry_checklist: list[ChecklistItem] = field(default_factory=list)
    key_evidence: list[str] = field(default_factory=list)
    risk_notes: str = ""
    confidence_note: str = ""


@dataclass
class ScanReport:
    scan_id: str
    model: str
    created_at: str
    market_overview: str
    analyses: list[StockAnalysis] = field(default_factory=list)
    top_picks: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


# ── Prompts ─────────────────────────────────────────────────────

_SYSTEM_PER_STOCK = """Bạn là chuyên gia phân tích kỹ thuật chứng khoán Việt Nam, chuyên chiến lược T+2 (mua-bán trong 2 phiên).
Phân tích mã {symbol} dựa HOÀN TOÀN trên evidence bundle bên dưới.

QUY TẮC BẮT BUỘC:
1. Giữ nguyên decision "{decision}" — KHÔNG ĐƯỢC thay đổi.
2. Mọi nhận định phải cite evidence_id cụ thể.
3. KHÔNG bịa số liệu — chỉ dùng giá, volume, indicator values từ bundle.
4. Không dùng tin tức, tin đồn, vĩ mô, cảm tính thị trường, hoặc dữ liệu ngoài OHLCV.
5. Viết tiếng Việt tự nhiên, ngắn gọn.

NHIỆM VỤ:
- Viết "summary": tóm tắt 2-3 câu về tình hình kỹ thuật của mã.
- Viết "risk_notes": rủi ro chính cần lưu ý.
- Viết "confidence_note": giải thích ngắn mức tin cậy.
- Nếu có model_signal, backtest_summary hoặc robustness_summary, dùng chúng như bằng chứng phụ; không được để ML lấn át rule guard.
- Chọn "key_evidence": danh sách evidence_id quan trọng nhất (tối đa 5).
{checklist_instruction}
{contradiction_instruction}

Trả về JSON duy nhất theo schema:
{{
  "symbol": "{symbol}",
  "decision": "{decision}",
  "summary": "string",
  "key_evidence": ["evidence_id", ...],
  "risk_notes": "string",
  "confidence_note": "string"{checklist_schema}{contradiction_response_schema}
}}"""

_CHECKLIST_INSTRUCTION_BUY = """
- Tạo "entry_checklist" cho BUY_SETUP — đánh giá từng tiêu chí:
  * Trend: EMA alignment (EMA9 > EMA21 > EMA50)?
  * Momentum: RSI trong vùng an toàn + MACD histogram dương?
  * Volume: surge >= 1.25x trung bình 20 phiên?
  * Risk: R:R >= 1.45, stop loss <= 4.5%?
  * Data: cross-check passed?
  Mỗi tiêu chí ghi status PASS/FAIL/WARNING + giải thích ngắn."""

_CHECKLIST_INSTRUCTION_WATCH = """
- Tạo "entry_checklist" cho WATCH — ghi rõ thiếu điều kiện gì để thành BUY_SETUP."""

_CHECKLIST_SCHEMA_BUY = """,
  "entry_checklist": [
    {{"criterion": "Trend alignment", "status": "PASS|FAIL|WARNING", "detail": "string"}},
    {{"criterion": "Momentum (RSI+MACD)", "status": "PASS|FAIL|WARNING", "detail": "string"}},
    {{"criterion": "Volume confirmation", "status": "PASS|FAIL|WARNING", "detail": "string"}},
    {{"criterion": "Risk/Reward", "status": "PASS|FAIL|WARNING", "detail": "string"}},
    {{"criterion": "Data quality", "status": "PASS|FAIL|WARNING", "detail": "string"}}
  ]"""

_CHECKLIST_SCHEMA_WATCH = """,
  "entry_checklist": [
    {{"criterion": "string", "status": "FAIL|WARNING", "detail": "thiếu gì để BUY"}}
  ]"""

_CONTRADICTION_INSTRUCTION = """
- Hệ thống phát hiện các mâu thuẫn sau, hãy bình luận từng điểm:
{issues_text}
  Viết "contradiction_comments": bình luận ngắn cho từng mâu thuẫn."""

_CONTRADICTION_RESPONSE_SCHEMA = """,
  "contradiction_comments": ["string", ...]"""

_SYSTEM_REJECT_BATCH = """Bạn là chuyên gia phân tích kỹ thuật chứng khoán Việt Nam.
Tóm tắt NGẮN GỌN lý do từ chối cho mỗi mã REJECT bên dưới (1-2 câu mỗi mã).

QUY TẮC: Giữ nguyên decision REJECT. Chỉ dùng evidence kỹ thuật từ bundle. Không dùng tin tức/vĩ mô/tin đồn. Viết tiếng Việt.

Trả về JSON:
{{"analyses": [{{"symbol": "string", "decision": "REJECT", "summary": "string", "key_evidence": ["string"]}}]}}"""

_SYSTEM_OVERVIEW = """Bạn là chuyên gia phân tích thị trường chứng khoán Việt Nam.
Dựa trên kết quả scan VN30 bên dưới, viết tổng quan kỹ thuật 3-5 câu tiếng Việt.
Chỉ dùng số lượng BUY/WATCH/REJECT và tín hiệu kỹ thuật trong scan. Không dùng tin tức, vĩ mô, tin đồn, hoặc dữ liệu ngoài OHLCV.

Trả về JSON:
{{"market_overview": "string", "top_picks": ["symbol", ...]}}"""


# ── Core engine ─────────────────────────────────────────────────

def _build_per_stock_evidence(candidate: dict[str, Any]) -> str:
    """Build compact evidence JSON for a single stock."""
    compact = {
        "symbol": candidate["symbol"],
        "decision": candidate["decision"],
        "score": candidate["score"],
        "confidence": candidate.get("confidence", 0),
        "latest_close": candidate["latest_close"],
        "latest_date": candidate.get("latest_date", ""),
        "allocation_weight": candidate.get("allocation_weight", 0),
        "data_quality": candidate.get("data_quality", {}),
        "risk_plan": candidate.get("risk_plan"),
        "model_signal": candidate.get("model_signal"),
        "hybrid_trace": candidate.get("hybrid_trace"),
        "backtest_summary": candidate.get("backtest_summary", {}),
        "robustness_summary": candidate.get("robustness_summary", {}),
        "warnings": candidate.get("warnings", []),
        "evidence": candidate.get("evidence", []),
    }
    return json.dumps(compact, ensure_ascii=False, indent=2)


def _analyze_single_stock(
    candidate: dict[str, Any],
    client: OllamaChatClient,
) -> StockAnalysis:
    """Run per-stock AI analysis."""
    symbol = candidate["symbol"]
    decision = candidate["decision"]

    # Detect contradictions (rule-based, instant)
    contradictions = detect_contradictions(candidate)

    # Build prompt
    if decision == "BUY_SETUP":
        checklist_instruction = _CHECKLIST_INSTRUCTION_BUY
        checklist_schema = _CHECKLIST_SCHEMA_BUY
    elif decision == "WATCH":
        checklist_instruction = _CHECKLIST_INSTRUCTION_WATCH
        checklist_schema = _CHECKLIST_SCHEMA_WATCH
    else:
        checklist_instruction = ""
        checklist_schema = ""

    if contradictions:
        issues_text = "\n".join(f"  - {issue}" for issue in contradictions)
        contradiction_instruction = _CONTRADICTION_INSTRUCTION.format(issues_text=issues_text)
        contradiction_schema = _CONTRADICTION_RESPONSE_SCHEMA
    else:
        contradiction_instruction = ""
        contradiction_schema = ""

    system = _SYSTEM_PER_STOCK.format(
        symbol=symbol,
        decision=decision,
        checklist_instruction=checklist_instruction,
        checklist_schema=checklist_schema,
        contradiction_instruction=contradiction_instruction,
        contradiction_response_schema=contradiction_schema,
    )

    user_payload = _build_per_stock_evidence(candidate)

    try:
        result = client.chat_json(system, user_payload, temperature=0.1)
    except Exception as exc:
        # Graceful fallback — return basic analysis without AI
        return StockAnalysis(
            symbol=symbol,
            decision=decision,
            summary=f"[AI không phản hồi: {exc}]",
            contradictions=contradictions,
            risk_notes="Không có phân tích AI — xem evidence bundle trực tiếp.",
            confidence_note="",
            key_evidence=[],
        )

    allowed_evidence = {ev.get("evidence_id") for ev in candidate.get("evidence", []) if isinstance(ev, dict)}
    key_evidence = [item for item in result.get("key_evidence", []) if item in allowed_evidence]
    ai_warnings = []
    unknown_evidence = [item for item in result.get("key_evidence", []) if item not in allowed_evidence]
    if unknown_evidence:
        ai_warnings.append(f"AI cited unknown evidence IDs and they were dropped: {unknown_evidence[:5]}")
    if result.get("decision") and result.get("decision") != decision:
        ai_warnings.append(f"AI attempted to change decision to {result.get('decision')}; deterministic decision kept")

    # Parse checklist items
    checklist = []
    for item in result.get("entry_checklist", []):
        if isinstance(item, dict):
            checklist.append(ChecklistItem(
                criterion=item.get("criterion", ""),
                status=item.get("status", ""),
                detail=item.get("detail", ""),
            ))

    return StockAnalysis(
        symbol=symbol,
        decision=decision,
        summary=result.get("summary", ""),
        contradictions=contradictions + result.get("contradiction_comments", []) + ai_warnings,
        entry_checklist=checklist,
        key_evidence=key_evidence,
        risk_notes=result.get("risk_notes", ""),
        confidence_note=result.get("confidence_note", ""),
    )


def _analyze_reject_batch(
    candidates: list[dict[str, Any]],
    client: OllamaChatClient,
) -> list[StockAnalysis]:
    """Batch analysis for REJECT stocks — short summary each."""
    if not candidates:
        return []

    compact_bundle = []
    for c in candidates:
        passed = [ev["evidence_id"] for ev in c.get("evidence", []) if ev.get("passed")]
        failed = [ev["evidence_id"] for ev in c.get("evidence", []) if not ev.get("passed")]
        compact_bundle.append({
            "symbol": c["symbol"],
            "decision": "REJECT",
            "score": c["score"],
            "latest_close": c["latest_close"],
            "passed_evidence": passed[:3],
            "failed_evidence": failed[:5],
            "warnings": c.get("warnings", [])[:3],
        })

    user_payload = json.dumps(compact_bundle, ensure_ascii=False)

    try:
        result = client.chat_json(_SYSTEM_REJECT_BATCH, user_payload, temperature=0.1)
    except Exception as exc:
        # Fallback — return basic without AI
        return [
            StockAnalysis(
                symbol=c["symbol"],
                decision="REJECT",
                summary=f"Score {c['score']:.0f} — không đạt ngưỡng BUY/WATCH.",
                contradictions=detect_contradictions(c),
            )
            for c in candidates
        ]

    analyses = []
    result_by_symbol = {
        a["symbol"]: a for a in result.get("analyses", result.get("items", []))
        if isinstance(a, dict) and "symbol" in a
    }

    for c in candidates:
        ai = result_by_symbol.get(c["symbol"], {})
        allowed_evidence = {ev.get("evidence_id") for ev in c.get("evidence", []) if isinstance(ev, dict)}
        key_evidence = [item for item in ai.get("key_evidence", []) if item in allowed_evidence]
        analyses.append(StockAnalysis(
            symbol=c["symbol"],
            decision="REJECT",
            summary=ai.get("summary", f"Score {c['score']:.0f} — không đạt ngưỡng."),
            contradictions=detect_contradictions(c),
            key_evidence=key_evidence,
        ))

    return analyses


def _generate_overview(
    scan: dict[str, Any],
    analyses: list[StockAnalysis],
    client: OllamaChatClient,
) -> tuple[str, list[str]]:
    """Generate market overview from scan results."""
    summary_data = {
        "scan_id": scan.get("scan_id"),
        "universe": scan.get("universe_name"),
        "symbols_scanned": scan.get("symbols_scanned"),
        "buy_count": sum(1 for a in analyses if a.decision == "BUY_SETUP"),
        "watch_count": sum(1 for a in analyses if a.decision == "WATCH"),
        "reject_count": sum(1 for a in analyses if a.decision == "REJECT"),
        "top_buy": [
            {"symbol": a.symbol, "summary": a.summary[:80]}
            for a in analyses if a.decision == "BUY_SETUP"
        ][:5],
    }
    user_payload = json.dumps(summary_data, ensure_ascii=False)

    try:
        result = client.chat_json(_SYSTEM_OVERVIEW, user_payload, temperature=0.3)
        return (
            result.get("market_overview", ""),
            result.get("top_picks", []),
        )
    except Exception:
        buy_symbols = [a.symbol for a in analyses if a.decision == "BUY_SETUP"]
        return (
            f"Scan VN30: {len(buy_symbols)} mã BUY_SETUP, "
            f"{sum(1 for a in analyses if a.decision == 'WATCH')} WATCH.",
            buy_symbols[:5],
        )


# ── Public API ──────────────────────────────────────────────────

def generate_full_report(
    scan: dict[str, Any],
    client: OllamaChatClient | None = None,
    on_progress: Any = None,
) -> ScanReport:
    """Generate per-stock AI analysis for all candidates in a scan.

    Args:
        scan: The full scan payload dict.
        client: Ollama client. Created from env if None.
        on_progress: Optional callback(symbol, index, total) for progress.
    """
    client = client or OllamaChatClient()
    candidates = scan.get("candidates", [])

    buy_candidates = [c for c in candidates if c.get("decision") == "BUY_SETUP"]
    watch_candidates = [c for c in candidates if c.get("decision") == "WATCH"]
    reject_candidates = [c for c in candidates if c.get("decision") == "REJECT"]

    analyses: list[StockAnalysis] = []
    total_individual = len(buy_candidates) + len(watch_candidates)
    idx = 0

    # 1. Analyze BUY_SETUP individually (deep)
    for c in buy_candidates:
        idx += 1
        if on_progress:
            on_progress(c["symbol"], idx, total_individual, "BUY_SETUP")
        analyses.append(_analyze_single_stock(c, client))

    # 2. Analyze WATCH individually
    for c in watch_candidates:
        idx += 1
        if on_progress:
            on_progress(c["symbol"], idx, total_individual, "WATCH")
        analyses.append(_analyze_single_stock(c, client))

    # 3. Analyze REJECT in batches of 8
    batch_size = 8
    for batch_start in range(0, len(reject_candidates), batch_size):
        batch = reject_candidates[batch_start : batch_start + batch_size]
        if on_progress:
            syms = ", ".join(c["symbol"] for c in batch)
            on_progress(syms, -1, -1, "REJECT_BATCH")
        analyses.extend(_analyze_reject_batch(batch, client))

    # 4. Market overview
    if on_progress:
        on_progress("overview", -1, -1, "OVERVIEW")
    overview, top_picks = _generate_overview(scan, analyses, client)

    # Reorder: BUY first, then WATCH, then REJECT
    order = {"BUY_SETUP": 0, "WATCH": 1, "REJECT": 2}
    analyses.sort(key=lambda a: (order.get(a.decision, 9), a.symbol))

    return ScanReport(
        scan_id=scan.get("scan_id", ""),
        model=client.config.model,
        created_at=datetime.now(timezone.utc).isoformat(),
        market_overview=overview,
        analyses=analyses,
        top_picks=top_picks,
        warnings=scan.get("warnings", []),
    )


def generate_single_report(
    scan: dict[str, Any],
    symbol: str,
    client: OllamaChatClient | None = None,
) -> StockAnalysis | None:
    """Generate AI analysis for a single stock from a scan."""
    client = client or OllamaChatClient()
    for c in scan.get("candidates", []):
        if c["symbol"].upper() == symbol.upper():
            return _analyze_single_stock(c, client)
    return None


# ── Caching ─────────────────────────────────────────────────────

def _cache_path(scan_id: str, model: str):
    safe_model = model.replace("/", "_").replace(":", "_")
    return REPORT_CACHE_DIR / f"{scan_id}_{safe_model}.json"


def get_or_generate_report(
    scan: dict[str, Any],
    client: OllamaChatClient | None = None,
    force: bool = False,
    on_progress: Any = None,
) -> dict[str, Any]:
    """Get cached report or generate a new one."""
    client = client or OllamaChatClient()
    scan_id = scan.get("scan_id", "unknown")
    cache = _cache_path(scan_id, client.config.model)

    if cache.exists() and not force:
        return read_json(cache)

    report = generate_full_report(scan, client, on_progress)
    payload = _to_dict(report)

    REPORT_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    write_json(cache, payload)
    return payload


def _to_dict(obj: Any) -> Any:
    """Recursively convert dataclasses to dicts."""
    if hasattr(obj, "__dataclass_fields__"):
        return {k: _to_dict(v) for k, v in asdict(obj).items()}
    if isinstance(obj, list):
        return [_to_dict(item) for item in obj]
    if isinstance(obj, dict):
        return {k: _to_dict(v) for k, v in obj.items()}
    return obj
