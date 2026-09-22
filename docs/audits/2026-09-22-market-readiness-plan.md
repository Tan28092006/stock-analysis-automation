# Real-market paper readiness: iteration compact

Goal: one reproducible EOD paper run on actual public market prices, as quickly as data gates allow. No broker orders, money movement, or automatic model promotion.

Decision owner: user. New runs can record forward paper recommendations only within the ex-ante window (signal-day 16:00 Vietnam to next trading-day 09:00). Before/after that window, provide a clearly labelled preview without backdating a ledger.

Acceptance: explicit current universe; completed-session data for every requested symbol and VNINDEX; validated OHLCV and source-byte manifest; deterministic scans on one snapshot; isolated immutable per-session run with idempotent retry; rule-only fallback when ML is unverified; timestamps/config/code/model hashes; machine-readable blocked status and nonzero exit for actionable failure. Existing prices/models/ledger are preserved. Timing and invalid/partial data are tested before code changes.

Parallel slices: (1) provider snapshot/reconciliation, (2) staged MR candidate with predeclared eval/release gates, (3) safe scheduled launcher, (4) root paper readiness/orchestration and end-to-end verification.

Scope limits: paper recommendations are not actual fills, momentum returns are a forward proxy not realized portfolio P&L, swing remains outside the first paper ledger contract. Missing historical PIT membership and adjustment lineage cannot be invented. Offline profitability does not certify live profitability. Known historical bad rows remain blocked/quarantined unless independently reconciled.

Fallback: record why a run abstained, keep rules preview available on locally valid data, retain existing production artifacts unchanged. Candidate models stay outside the production artifact path. The safe launcher must not automatically commit/rebase/push Git state.
