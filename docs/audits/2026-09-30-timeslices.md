# Causal VNINDEX timeslices — evaluation extension v1

The user explicitly rejected a pooled 2022–2026 return as the main acceptance
criterion and requested separate up/down phases plus 2026 H1. This changes the
evaluation/reporting contract, not selection, orders, costs or production state.

Protocol: `configs/research/timeslices_v1.json`; registered in the parent registry
before examining new slice results. Source: the same immutable raw-backed price
snapshot used by the existing 290-trial gate and 180-trial book suite.

## Fixed definition

For session t, use only the completed close of t-1:

- Uptrend: close[t-1] > SMA200[t-1] and SMA200[t-1] > SMA200[t-21].
- Downtrend: both inequalities reversed.
- Transition: every other case, including equality. No forced binary label.

Every contiguous label run is a separate episode. Short runs remain visible;
there is no hindsight pivot detection, smoothing or merging chosen after seeing
strategy performance. The end of an episode is known only when the next label
arrives; an episode's cash restart is a sensitivity experiment, not a claim that
its future exit boundary could be traded in advance.

SMA200 measures long-term trend. A sudden crash can occur before it turns down;
therefore the original event windows (including March–April 2026) remain in the
parent gate. They are not replaced by these labels or excluded as inconvenient.

## Two distinct accounting views

**Carry attribution:** partition each unchanged continuous parent replay. Include
all holdings, cash/receivables, original rebalance dates, fees and settlement.
Opening NAV is the prior session's closing NAV (initial capital for the first
slice). VNINDEX uses the matching prior close; the first full-evaluation day
retains the original first-open basis. Both VND profits summed across disjoint
episodes and compounded returns must reconstruct the parent result exactly.

**Cash restart:** replay MR bracket, MR fixed15 and monthly momentum independently
from cash at each slice, each under normal, double-cost and extra-session-delay
scenarios. Price history before the slice remains available as warmup. The first
rebalance clock restarts too; this path dependence is why the two views must not
be mixed. Benchmark starts at the first session's open. No forced liquidation at
the slice end and no fabricated exit fees on still-held stocks.

All 12 MR variants, 10 momentum variants, universe control, six reference
stresses and 15 non-duplicate book trials are attributed in every slice: **44
paths** per slice. For **2026 H1**, all 44 also receive independent cash replays.
Other slices have the nine reference cash replays, not a false claim of full
independent restarts for every ablation.

The current pinned index defines 46 episodes (637 uptrend, 211 downtrend and 166
transition sessions) and nine calendar half-years. The first/last calendar
halves are partial; 2026-01-01–2026-06-30 is complete. Total expected new cash
replays: **530**, plus 2,420 attributed path/slice summaries. Overlapping
calendar and regime slices are not independent statistical replications.

## Output and gates

The runner exports a standalone HTML report, complete per-slice JSON, daily
regime labels with the prior source date/close/SMA, source hashes and upstream
result hashes. Metrics include net profit, excess over gross VNINDEX price,
within-slice drawdown, exposure, commissions/tax, slippage, buys/sells,
flat-to-long entries, add-on buys and opening/ending quantities.

`profitable`, `beats_index` and `positive_net_and_excess` are separate descriptive
flags. For example, -10% versus -20% beats the index but is NOT positive profit.
Short-slice significance or a global winner-selected p-value is not claimed.
Historical data already inspected remains development data.

The runner rejects incomplete parents, changed code/config/data hashes, missing
registered trials and accounting discrepancies. The new CI step follows both
complete parent suites, and a regression test requires it to remain present.
CI YAML installation is not proof that remote CI ran: remote execution still
requires an accessible pinned snapshot artifact and verification of the run.

```text
python -m scripts.research_gate --manifest <manifest.json> --output <new-parent-directory>
python -m scripts.momentum_books --manifest <manifest.json> --output <new-books-directory>
python -m scripts.research_timeslices --manifest <manifest.json> --gate-results <new-parent-directory>/results.json --book-results <new-books-directory>/results.json --output <new-timeslices-directory>
```

No automatic strategy promotion, real-money profit claim, or production universe
change follows from generating this report. VN100 expansion is separately
specified in `2026-09-30-vn100-expansion.md` and still requires its own price and
historical membership evidence.

## Completed local evidence — September 30

- Full test suite: **511 passed**, 23 existing dependency/fixture warnings.
  Timeslice module: 31 dedicated tests, 95% coverage from unit plus synthetic
  end-to-end tests (220 statements, 12 uncovered; no full-replay coverage claim).
- Both parent suites completed: all 290 original trial blocks exactly match
  `run_v2`; all 180 book trial blocks exactly match `books_v1`.
- Extension completed with **55 slices, 530 independent cash replays and 2,420
  attributed summaries** (44 unchanged parent paths per slice). All 46 causal
  regime episodes reconcile to each parent, in both summed VND and compounded
  returns; no source-hash drift. Strict JSON parsing found no NaN/Infinity.
- One-session runs emit existing variance warnings inside the parent summary
  helper. Those unused annualized/beta summaries are not published in this
  extension; one-session returns/accounting remain included, not dropped.
- Local report: `data/paper/research_gate/20260930/timeslices_v1/report.html`.
  Full daily NAV, fills, closed lots and variants are in `results.json` and each
  slice JSON. `regime_labels.json` retains every prior-close label input.
- Remote CI remains unexecuted/unconfigured for the pinned input artifact.
  This local completion does not mean CI on GitHub is green or live trading ready.

Evidence SHA256:

| File under `data/paper/research_gate/20260930/` | SHA256 |
|---|---|
| `timeslices_parent_gate/results.json` | `f29f42754c0c17ee0d51e92058df232e18d4024068dca68176a40fcb45381082` |
| `timeslices_parent_books/results.json` | `afc617615402ed13d5017f909c0a7dcc7b2ef7c74e0b1a3daa0fd48d87505129` |
| `timeslices_v1/results.json` | `486b1a7bfb248f90417393745b2f8bc6041cb15043c176f31fb8fe3f6cde19b1` |

### Fixed half-year cash-restart results

Each row starts separately from VND1bn cash. Returns are after modeled costs;
the VNINDEX price comparator is gross. Do not compound independent restarts to
claim a continuously executable portfolio. The HTML separately shows the actual
continuous-portfolio attribution and every up/down/transition episode.

| Half-year | MR bracket | MR fixed15 | Momentum monthly | VNINDEX |
|---|---:|---:|---:|---:|
| 2022 H2, September onward only | -1.738% | -5.062% | -24.336% | -21.337% |
| 2023 H1 | -0.636% | +0.147% | -0.664% | +10.757% |
| 2023 H2 | +3.481% | +0.408% | -0.285% | +0.389% |
| 2024 H1 | +0.214% | -0.617% | +13.591% | +9.586% |
| 2024 H2 | -0.327% | -0.327% | +7.173% | +1.641% |
| 2025 H1 | +0.260% | +2.268% | +0.622% | +8.441% |
| 2025 H2 | 0.000% | 0.000% | +45.531% | +29.485% |
| 2026 H1 | +1.856% | +1.849% | +1.534% | +4.078% |
| 2026 H2, through September 29 only | +3.065% | +1.690% | +0.706% | -4.421% |

H1 2026 has 119 sessions (115 long-term uptrend, four transition). SMA200's
long-term classification does not erase the short violent March drawdown;
the original March-April crisis block remains mandatory.

### H1 2026 details

All 44 registered paths have independent cash replays in this slice.

| Reference | Net P&L VND | Maximum drawdown | New entries / add-on buys |
|---|---:|---:|---:|
| MR bracket | 18,558,009 | -0.915% | 2 / 0 |
| MR fixed15 | 18,491,186 | -1.711% | 2 / 0 |
| Momentum monthly | 15,340,245 | -14.351% | 10 / 27 |

Both MR15 entries were March 11 and exits April 1 (entry + 15 trading sessions):
MSN 1,700 shares, net +12,168,070 VND; VPB 7,100 shares, net +6,323,116 VND.
No MR15 holdings remain at June 30. The momentum path still holds ten names;
its return includes unrealized closing marks, not hypothetical liquidation.

Increasing frequency alone fails this slice: weekly momentum returns -2.290%
with 13 new entries and 99 add-on buys, versus monthly +1.534%, 10 and 27.
Book-inspired slope90, quality, breakout50 and breakout50+volume return
-3.994%, -3.258%, -3.571% and -3.760% respectively. The own-portfolio-volatility
ablation is +4.310%, just 0.232 percentage points above the index here; it is
not a promoted winner or statistical proof. Every other ablation and stress
remains visible in the complete slice JSON.

Inherited-portfolio momentum attribution is a different result: H1 +2.555%
versus matching previous-close-basis index +4.232%, opening NAV 1,417,339,028
VND with ten carried positions. It must not replace the +1.534% independent
cash-start result. Neither reference view beats the index in H1.
