# R1–R4 implementation record (04/10/2026)

Base: `codex/data-integrity-audit` @ `4b859cd`. New package `stock_agent/research_ml/`;
no trading runtime, data, schedule, model or CI change. All evidence is synthetic
fixtures: **no backtest, no profitability or leakage-free claim**.

| Item | Module | What the tests prove |
|---|---|---|
| R1 | `contracts.py` | tz-aware availability, grain/vintage uniqueness, OHLC/unit checks; as-of join with `available_at <= decision_time`, fan-out and ambiguous-version rejection; observed/zero/missing/stale states; target-column rejection |
| R2 | `catalog.py` | content-addressed immutable raw, one-transaction publish, idempotent duplicates, correction = new vintage with knowledge-time reads, partial = not ready, crash before commit invisible, corrupt/missing raw fails closed, rebuild from raw, backup/restore, writer lock |
| R3 | `features.py`, `labels.py` | registry hash; future-mutation invariance; warmup = `insufficient_history`; foreign window over session skeleton with min support; labels entry `e`, exit `e+horizon`, cost flows, pending/unfillable explicit, trainable only after `label_available_at` |
| R4 | `harness.py`, `evidence.py` | purged grouped date folds on mandatory timestamps, fold-fitted serialized transform (mask before imputation, schema fail-closed, batch/single parity), forward OOF with uncovered rows, meta on available labels only, nested stack outer-label invariance, model timestamp contract v2, sealed bundle validator that cannot return a pass |

Verification: `python -m pytest tests/test_ml_*.py tests/test_pytest_collection_safety.py tests/test_exchange_calendar.py -q` → 51 passed (pandas 2.x, numpy<2, scikit-learn; LightGBM/XGBoost/CatBoost not installed here, tests use sklearn bases).

Known limits / not done
- R2 covers price vintages only; foreign-flow vintages, pagination proof, RPO/RTO and migration of the existing manifests are open.
- `asof_join` is a Python loop (correct, not fast); benchmark before real-size use.
- Legacy `build_labeled_t2_dataset` still ignores `cost_config`; pinned by a characterization test, not fixed (trading pipeline untouched).
- Existing `temporal_validation.model_available_at` is unchanged; contract v2 is a separate validator, not yet wired into `model_release.py`.
- F1/F4 BLOCKED (no PIT breadth/sector sources). R5 needs the user's numerical activity floor and frozen sample floors; the validator returns BLOCKED without them.
- Full legacy suite not re-run here (heavy deps missing); only new tests and two guards.
