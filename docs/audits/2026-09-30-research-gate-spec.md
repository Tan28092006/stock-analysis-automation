# Research regression gate — specification locked before new replays

User objective: evaluate bottom-fishing and especially momentum, show the latest
six-month portfolio/trades, test signals across appropriate regimes, and make
future theory changes pass a reproducible research gate. Foreign-flow data is
explicitly included. No change to production selection thresholds is implied.

## Deliverables and fixed assumptions

- Latest six months: 2026-04-01 through verified completed EOD 2026-09-29.
- Separate initial cash of VND 1bn per strategy (current configured baseline).
- MR comparison: existing bracket exit vs holding to entry + 15 trading sessions.
  Fixed-hold retains the same entry gates and initial sizing; its nominal stop is
  ONLY a sizing reference, not an active loss cap. Unmatured trades remain open.
- Full daily NAV/cash/positions, entry/exit history, monthly and symbol P&L.
- PIT universe wherever documented; source hashes and code/config hashes retained.
- Every variant shares its block's source vintage, universe, costs and execution.
- Current rules and all already viewed historical periods are development data,
  never an untouched holdout. Earliest new forward boundary: 2026-10-01.

## Research design

Freeze a machine-readable registry of calendar blocks, overlapping event slices,
controls, one-rule ablations, cost/delay stresses and statistical thresholds.
The registry will include the 2022 selloff/rebound, 2023 and 2024 full regimes,
2025 tariff shock and recovery, March 2026 shock and subsequent months, plus
continuous multi-year runs. Event slices are descriptive and must not be pooled
as independent replications when they overlap calendar blocks.

MR: remove RSI, band, confirmation, RR and cloud gates one at a time; disable VSA;
compare strict gates and fixed holding horizons. Momentum: compare index vs own
portfolio volatility, market-adjusted ranking, reversal entry, equal weighting,
no buffer, positive momentum gate, market trend gate, formation horizon and
52-week-high proximity. Keep variants separate; no winner-selected combination.

Report net return and excess over VNINDEX, drawdown, exposure, turnover, trade
count, contribution concentration, monthly returns and independent accounting.
Use paired, date-block bootstrap of daily excess returns with at least 20-session
blocks, simultaneous/adjusted family inference, minimum sample and sensitivity
checks. Include cash and exposure-matched index controls so de-risking cannot be
mistaken for stock-selection alpha. A profitable backtest is not a statistical
pass. Sparse MR trades must be marked inconclusive.

Foreign-flow audit must preserve raw files, distinguish matched/put-through,
million/billion VND, session dates vs observed timestamps, revisions/duplicates,
daily vs unfinished monthly/quarterly aggregates. Missing observations are not
zero flows. Legacy rows without provenance cannot pass prospective eligibility.
Exploratory lagged tests, if data quality permits, are labeled as such and cannot
promote a strategy; unknown date shifts must not be guessed into place.

## Engineering acceptance

Meaningful RED/GREEN tests for fixed hold, old exchange holidays, temporal
causality, accounting, registry completeness, bootstrap multiplicity, and flow
data quarantine. CI runs deterministic synthetic tests and protocol validation;
the full data gate requires pinned verified snapshots and fails visibly if
required inputs/evidence are absent. Missing historical evidence is BLOCKED,
never a silently skipped green check. Generated reports are research artifacts;
production scans, models, positions and ledgers are not rewritten by this suite.
