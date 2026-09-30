# MR / momentum / foreign-flow reassessment — 2026-09-30

## Verdict

Research infrastructure is implemented; **live-money edge is NOT established**.
No production thresholds, model, positions or trade ledger were promoted/rewritten.
ML was disabled throughout these rule-only trials, so this report does not certify
the previously trained ML artifact as leakage-free.

The first full run completed 290 portfolio replays: 10 windows × (12 MR variants +
10 momentum variants + 6 baseline cost/delay stresses + 1 equal-weight PIT control).
All cash, receivables, share lots and NAV were independently reconciled. Maximum
observed error was below 0.000001 VND. A second deterministic run and coverage
instrumentation are retained separately; hashes distinguish evidence vintages.

## Latest six months

Window 2026-04-01–2026-09-29; separate VND1bn initial cash, no opening holdings,
100-share lots, buy commission 0.15%, sell commission/tax 0.25%, slippage 0.10%
each side, conservative T+3 shares/cash, no borrowing or interest on cash.
Return includes unrealized ending holdings at close, without hypothetical exit
costs on unsold positions. Index comparator is gross VNINDEX **price** return from
first-session open; it is not an investable total-return product.

| Strategy | Net portfolio P&L VND | Return | Max drawdown | Mean exposure |
|---|---:|---:|---:|---:|
| MR current brackets | 30,653,719 | 3.0654% | -2.8945% | 3.8169% |
| MR fixed 15 sessions | 16,902,100 | 1.6902% | -2.8925% | 7.4678% |
| Momentum current monthly replay | -1,052,761 | -0.1053% | -12.3531% | 85.9893% |
| VNINDEX gross price | 37,230,442 equivalent | 3.7230% | -13.4553% | 100% |

Fixed 15 means exit at the close on **entry + 15 trading sessions**. The signal
is from an earlier close, entry at next open. Stop/target remain initial entry
and sizing references, but no longer trigger an exit. Unmatured lots stay open;
the simulation never fabricates completed returns after the snapshot cutoff.

| Symbol | Entry | Exit | Shares | Net P&L VND |
|---|---|---|---:|---:|
| MSN | 2026-07-09 | 2026-07-30 | 3,600 | -4,714,500 |
| VRE | 2026-07-24 | 2026-08-14 | 5,200 | 10,460,851 |
| BID | 2026-07-24 | 2026-08-14 | 5,100 | 9,672,109 |
| HPG | 2026-07-24 | 2026-08-14 | 3,500 | 1,483,640 |

No MR15 positions remain at September 29. April–June and September MR15 monthly
return is zero; July +2.3038%, August -0.5998% (mark-to-market monthly NAV, not
closed-trade-only returns). A 75% win rate on four trades is not reliable edge
evidence. Scaling to small actual capital is not linear because of board lots.

## Regimes and ablations

| Period (cash start per row) | MR brackets | MR fixed15 | Momentum |
|---|---:|---:|---:|
| Sep–Dec 2022 | -1.738% | -5.062% | -24.336% |
| 2023 | 2.844% | 0.554% | 4.523% |
| 2024 | 0.320% | -0.514% | 20.764% |
| 2025 | 0.260% | 2.268% | 34.575% |
| Jan–Sep 29 2026 | 5.011% | 3.591% | -1.593% |
| Mar–Apr 2026 event diagnostic | 1.856% | 1.849% | -2.019% |
| Continuous Sep 2022–Sep 2026 | 7.904% | 3.087% | 43.436% |

Continuous momentum beats VNINDEX's +38.857% by 4.579 percentage points over the
whole period, with -27.438% drawdown versus index -28.772%. This is neither a
4.579% annual alpha estimate nor a statistically established improvement.

- Removing MR confirmation: -21.050% in late 2022, -30.606% continuously. This
  is economically concerning falling-knife exposure. It is not a causal proof
  that the remaining confirmation rule will always work.
- MR fixed15 has only 27 closed lots in the continuous replay, below the locked
  30-trade minimum. Brackets have 30. No rare-trade win rate is treated as robust.
- Momentum no-buffer looks better in recent six months (+4.155%) but worse
  continuously (+29.505% vs +43.436% baseline). Turning over more is not generally
  better. Monthly buffering changes path dependence and realized costs.
- The 52-week-high variant is +31.940% in 2024 but -12.418% in 2026-to-date;
  6-1 formation is +50.606% in 2025 but -19.068% in 2026-to-date. This is why
  selecting the best recent variant is not an acceptable deployment rule.
- Equal-weight momentum (+52.235%), own-portfolio volatility (+48.370%) and
  market-adjusted rank (+44.050%) are development observations only. None passes
  the registered statistical gate. They are not combined into an optimized winner.

Primary inference uses paired daily returns in the continuous run, circular
20- and 40-session blocks, 4,000 seeded draws and Bonferroni correction across
22 primary comparisons. Baselines compare against index; variants against the
same strategy's baseline. All family intervals include zero. Event/calendar
slices are descriptive; overlapping windows are not counted as independent
replications. Cash and prior-close-exposure-matched gross index are separate
controls. The equal-weight PIT VN30 comparator is actually replayed with costs.
No global DSR or untouched holdout claim is possible: the historical trial
count is unknown and these dates have already been inspected.

## Foreign and proprietary-flow data collected so far

| File | Rows | Symbols | Stored daily span | Findings |
|---|---:|---:|---|---|
| ndtnn | 2,500 | 51 | Apr 2–Jul 1 2026 | All 2,500 source epochs map to a Vietnamese session one calendar day later; only 2–3 PIT VN30 names/day |
| ndtnn_chart | 7,400 | 100 | Jun 4–Sep 20 2026 | Original epoch discarded; 1,600 stored off-calendar rows |
| tudoanh_chart | 7,500 | 100 | Jun 3–Sep 20 2026 | Same provenance/calendar concern; separate investor category |
| price_board_snapshots | 2,630 | 100 | Jul 2–Sep 21 2026 | 130 duplicate rows; 300 off-calendar; host-date stamp, no session proof |

None has observed/fetched timestamps. The last stored chart/board rows precede
the price snapshot; the current collection is not proven fresh. Monthly files
cover Aug 2025–Jul 2026 and quarterly files Q4/2023–Q3/2026, including unfinished
periods at their July collection vintage. These aggregates cannot be assigned
to the start of a month/quarter in a causal test.

The current loader merges detailed BuyVal/SellVal (million VND) with board values
(billion VND) without conversion. Of 145 nonzero overlapping detailed/chart buys,
135 have identical volumes and matching values after dividing detailed values by
1,000. The remaining differences are unresolved, potentially revisions or
matched/put-through definitions. `load_flows()` ignores the two chart files;
old scratch IC code uses stored dates and naive IID t-statistics on overlapping
forward returns. Those old estimates cannot certify a foreign-flow edge.

The new **research-only** normalization preserves raw data, fixes dates only
where original epochs exist, labels matched-only versus unknown aggregation,
and sets prospective eligibility false. The legacy production collector/loader
is not silently rewritten as part of this research request.

Exploratory chart sensitivity assumes stored date +1 calendar day, explicitly
unverified. Signal is signed foreign **volume imbalance**, `(buy-sell)/(buy+sell)`,
and its complete 5-session average; entry next open or one extra session later;
hold 5/15/21 sessions. It is not a direct test of nominal net billion-VND flow.
Cross-sectional rank IC requires 10 PIT VN30 names per date. Conditional IC
residualizes ranks against contemporaneous 5-session price momentum; this does
not establish causality or remove all sector/size/ETF effects.

- Source-epoch detailed data: zero eligible IC dates at the 10-name threshold;
  observed PIT breadth is only ACB/BID and, after admission, BSR. This looks like
  truncated cross-sectional harvesting; verify pagination before relying on it.
- Chart flow1 / next-open / hold15: 64 dates, mean IC **-0.0881**, price5-conditioned
  IC **-0.0714**. One extra session: mean IC **-0.0898** on 63 dates.
- Chart flow5 / next-open / hold15: 60 dates, mean IC **-0.0982**, conditional
  **-0.0687**. Family-adjusted interval: **[-0.2242, 0.0591]** for block20,
  **[-0.1975, 0.0010]** for block40. Minimum 252 dates is not met.

There is no supported "foreign buying implies buy" rule in this sample. The
negative means do not authorize flipping the signal after seeing the data.
Neither prospective eligibility nor statistical evidence passes. Proprietary
flow is inventoried only; do not conflate it with foreign flow or pretend an
untested combination has been validated.

## Concrete remaining risks / next hypotheses

1. **Execution parity:** momentum UI checks top20 exit daily; replay rebalances
   monthly. MR UI/canonical exits use a different settlement lock; gap-stop
   handling differs. Do not claim the current dashboard is the tested simulator.
2. **Risk concentration:** index volatility is not portfolio volatility;
   inverse-vol weights do not equalize covariance-based risk contributions.
   Sector/group caps require historical sector data, not today's labels applied
   retrospectively. Monthly risk scaling may react too slowly inside a panic.
3. **Execution feasibility:** no order-book queue, auction fill, ADV participation
   cap, suspension management for held names, or price-impact model. The 6.9%
   opening gate uses adjusted previous close, not official reference prices.
   Stop-loss orders do not guarantee exits on locked limit-down sessions.
4. **Data lineage:** public PIT membership is reconstructed manually; prices are
   current adjusted vintage, not archival corporate-action-certified data.
   The Jan2025 membership availability bound is conservative, so first-session
   inclusion may lag the actual announced list. No full spring2022 comparison
   is possible: VNINDEX 2021-08-23 is malformed; separate valid snapshot starts
   Aug24 with sufficient warmup only for September2022 onward.
5. **Behavioral hypotheses:** anchoring near highs, overtrading, disposition
   effect, panic liquidity and momentum rebound crashes motivate tests but are
   not proven psychological causes of these results. Avoid changing thresholds
   to rescue a disappointing period; register new trials and retain failures.
6. **Not-yet-testable signals:** earnings/news publication times, ETF rebalance
   flows, rates/FX, broad-market breadth, sector rotation, spread/depth, insider
   or ownership changes. Obtain availability timestamps and historical coverage
   before adding features. Missing observations must not be zeros.

Paper → hypothesis → limitation and primary event/calendar sources are in
[research-sources](2026-09-30-research-sources.md). The next genuinely untouched
forward boundary is October 1, 2026; freezing a rule now does not undo earlier
selection on already-seen history.

## Reproduction and CI status

From repository root (PowerShell or shell, one command each):

```text
python -m scripts.research_gate --validate
python -m pytest -q --tb=short
python -m scripts.research_gate --manifest data/paper/research_gate/20260930/snapshot_20210824/manifest.json --output data/paper/research_gate/20260930/new_run
python -m scripts.foreign_flow_research --manifest data/paper/research_gate/20260930/snapshot_20210824/manifest.json --output data/paper/research_gate/20260930/new_foreign_audit
python -m scripts.research_report --results data/paper/research_gate/20260930/run_v2/results.json --foreign data/paper/research_gate/20260930/foreign_v3/audit.json --output data/paper/research_gate/20260930/new_report
```

Each output directory must be new. Never overwrite a previous trial. Keep the
blocked original snapshot, source raw bytes and all failed trials. Results
record the source hashes plus Git HEAD; the content hashes also capture any
uncommitted source present when a run began. Re-run after relevant code changes.

`.github/workflows/research-gate.yml` adds synthetic contract tests and a required-
input full replay job. **Remote CI has not been configured/executed here.** Its
full job intentionally fails BLOCKED until repository variable
`RESEARCH_SNAPSHOT_RUN_ID` identifies a GitHub Actions run with the immutable
`research-snapshot-v1` artifact. That artifact must have `manifest.json`, `raw/`
and `prices/` at its root. It is an input transfer, not a fresh provider fetch.
Manifest/raw/canonical hashes are verified again by the runner. Configure branch
protection separately if these checks must block merging; adding YAML alone does
not enable branch protection or approve deployment.

The full suite succeeding means **engineering/research evidence complete**, not
strategy approval. `live_eligible` remains false. No live orders, remote pushes,
model retraining or automatic promotion were performed in this work.
