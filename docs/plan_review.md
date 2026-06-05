# Plan Review For VN30 T+2 MVP

## Verdict

The original architecture is directionally sound for an auditable swing-trading assistant: controlled data, deterministic rules, risk guards, and an AI report layer. For T+2, the plan needs stricter freshness, liquidity, and verification gates than a 3-15 day swing system.

## Main gaps fixed in this MVP

- VN100 is reduced to VN30 to lower API load and false positives.
- The model horizon is explicitly T+2, so scoring favors fresh momentum, volume confirmation, gap risk, ATR stop distance, and short-term reward/risk.
- LLM usage is moved out of the decision path. It can only summarize a structured evidence bundle.
- Every candidate carries source provenance, validation status, and rule evidence.
- Data cross-check is explicit. A single source is allowed only as degraded confidence in MVP; live trading should set `allow_single_source` to `false`.
- Portfolio P&L is implemented as a data product that can feed later supervised labels and position-aware features.

## Real-world barriers

- Free VN data APIs change without notice. Provider adapters must be monitored.
- Yahoo symbols for Vietnam can be inconsistent. Local CSV cache should be kept as a fallback.
- External news has been removed from the decision path; the scanner is price/volume technical analysis only.
- T+2 labels require an exchange-calendar-aware outcome builder, not calendar-day offsets.
- Corporate actions and adjusted prices must be verified before training.
- Backtests must include fees, tax, slippage, lot size, price bands, and non-trading days.

## Accuracy improvements for later phases

- Add official HOSE index refresh ingestion instead of a static VN30 config.
- Add T+2 forward-return labels and train a calibrated classifier/regressor only after enough observations.
- Track market regime from price/volume only: VNINDEX/VN30 trend, breadth, sector rotation, and volatility.
- Add intraday close auction data if the strategy enters near ATC.
- Add walk-forward validation and model drift monitoring.
- Keep model output as one signal inside the rule framework, not an unrestricted decision maker.
