from __future__ import annotations

import json
from typing import Any

from ..constants import LATEST_SCAN_PATH
from ..data.repository import read_json
from ..reports.evidence import build_evidence_bundle, validate_ai_summary
from .analyst import generate_full_report, generate_single_report, get_or_generate_report
from .ollama_client import OllamaChatClient


def synthesize_latest_report(
    symbol: str | None = None,
    force: bool = False,
    on_progress: Any = None,
) -> dict[str, Any]:
    """Generate AI analysis report for the latest scan.

    Args:
        symbol: If provided, only analyze this one stock.
        force: Bypass cache and regenerate.
        on_progress: Optional callback(symbol, idx, total, phase) for CLI progress.

    Returns:
        Full report dict (ScanReport) or single StockAnalysis dict.
    """
    scan = read_json(LATEST_SCAN_PATH, default={})
    if not scan:
        raise RuntimeError("No latest scan found. Run a scan first.")

    client = OllamaChatClient()

    if not client.is_available():
        raise RuntimeError(
            "Ollama is not running. Start it with: ollama serve\n"
            "Or install from: https://ollama.com/download"
        )

    if symbol:
        # Single stock analysis
        from ..schemas import to_plain_dict
        analysis = generate_single_report(scan, symbol, client)
        if analysis is None:
            raise RuntimeError(f"Symbol {symbol} not found in latest scan.")
        result = to_plain_dict(analysis)
        return result

    # Full report with caching
    return get_or_generate_report(
        scan, client, force=force, on_progress=on_progress,
    )
