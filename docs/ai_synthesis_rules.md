# Evidence-First AI Synthesis Rules

These rules are for any LLM that writes the human-readable report.

## Hard rules

1. Use only the provided `evidence_bundle` JSON.
2. Never invent prices, dates, financial metrics, or reasons.
3. Every claim about a symbol must reference an `evidence_id`.
4. If data quality is `failed`, `conflict`, or `stale`, the output must say the symbol is not tradeable.
5. If source cross-check is missing, confidence must be capped by `confidence_cap`.
6. The LLM cannot override rule decisions. It may explain `BUY_SETUP`, `WATCH`, or `REJECT`.
7. External news, rumors, macro commentary, and sentiment are out of scope.
8. Output must keep the same symbol order as the evidence bundle.
9. No price target can be shown unless it exists in `risk_plan`.
10. No macro, rumor, or external-news commentary is allowed.

## Output schema

```json
{
  "scan_id": "string",
  "summary": "string",
  "candidates": [
    {
      "symbol": "string",
      "decision": "BUY_SETUP|WATCH|REJECT",
      "confidence": 0.0,
      "reason": "string",
      "evidence_ids": ["string"]
    }
  ],
  "data_warnings": ["string"]
}
```
