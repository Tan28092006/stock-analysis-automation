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
