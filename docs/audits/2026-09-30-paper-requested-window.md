# Paper scoring: requested rolling window contract

## Locked before implementation / new replay results

Hypothesis: `paper_requested_window_v1_20260930`. Baseline code: `03af68b`.
Goal: score archived ex-ante paper observations against subsequent source-backed
snapshots without confusing an intentional rolling request boundary with a
provider revision. This does not introduce a trading signal or certify an edge.

The refresh command requests 650 calendar days before the completed session.
Current scoring instead reindexes the new data onto every old history row. As
the requested start advances, absent leading rows become NaN and incorrectly
look like revisions. Original archived snapshot bytes are still present.

Observed September 29 source request starts December 18, 2024. September 25 and
28 paper snapshots start December 16 and 17, respectively. Nine of the ten
recommended stocks per old run have exactly equal shared OHLCV. GAS has a real
last-session volume revision: Sep 25 1,042,700 -> 1,046,966 shares; Sep 28
1,158,100 -> 1,162,223. These observations were diagnosed before the fix. The
GAS differences must remain quarantined; the fix is not expected to make the
entire real ledger green.

## Required behavior

1. Reverify original raw/CSV hashes, manifest binding, record identity, ex-ante
   recording time and universe exactly as before. Never regenerate old signals,
   overwrite paper records, rewrite snapshots or import revised history into the
   original feature computation.
2. Compare every original OHLCV row on or after the **declared requested start**
   in the verified outcome manifest, through the original signal date, exactly.
   Do not use the first available returned date to excuse missing requested bars.
3. Only rows strictly before that requested start may be absent from the new
   snapshot. Report their archived count explicitly as not re-observed; do not
   claim the entire original history was re-confirmed or corporate actions certified.
4. The outcome request must include the signal session. Internal/request-boundary
   gaps, missing signal/future outcome sessions and all changes to shared open,
   high, low, close **or volume** must continue to block the entire affected run.
   Preserve existing run-atomic staging; do not hide a bad recommendation behind
   healthy recommendations from the same selection event.
5. Full-window and legitimate prefix-rolled snapshots with identical future bars
   must produce identical outcomes, excluding the explicit history-check metadata.
   Pending observations remain null-return pending, never zero/profit/loss.
6. No fee, settlement, exit, ranking, sizing, universe or signal-frequency changes.
   Momentum paper returns remain informational, not a funded portfolio backtest.
   Existing UI/replay parity gaps remain visible, not silently declared resolved.

## Verification plan

Use RED/GREEN unit and CLI integration tests with source-backed synthetic
snapshots. Cover real prefix movement, an outcome-only post-signal crash, each
OHLCV revision, missing first requested row, internal history/signal/outcome
gaps, an outcome window starting after the signal, corrupt originals and atomic
quarantine. Check raw snapshots and paper records remain byte-identical.
Require at least 80% scoring coverage and include these tests in remote CI.

Run all local tests and the full pinned research workflow: 290 parent, 180 books,
530 causal timeslices and 196 VN30/VN100 replays. Compare complete economic
payloads to preceding verified results, retaining code/config/input hashes.
Apply scoring to the real archived records only after synthetic tests pass;
preserve the previous score report as evidence and keep genuine revisions blocked.

An engineering pass removes a false failure mode. It is not enough to meet the
user's separate activity, after-cost profitability and live-readiness requirements.

## RED/GREEN and real-record verification

Two RED checkpoints precede implementation: `16f4264` captures five failing
rolling-window/CI regressions (19 related checks already passed); `ba09ce5`
captures two additional failing CLI and whole-run quarantine cases. The GREEN
implementation is `8234c28`. All **43** paper scoring, runner, CLI and runtime
hardening tests pass together. Statement coverage for `paper_scoring.py` is
**88%** (101 statements, 12 uncovered). Tests retain exact checks for all five
OHLCV fields, including real volume revisions; no comparison tolerance changed.
The complete local suite then passed: **545 tests**, 23 warnings, 129.85 seconds.
Warnings are reported rather than suppressed; no failed or skipped tests were
converted into a pass. Fresh remote pinned-snapshot replay evidence is required
for this source revision and is not inferred from the earlier VN100 CI badge.

After those tests passed, a read-only score of the September 29 snapshot checked
194 original-source/record/report file hashes before and after: no changes.
The normal scoring command then updated only the derived score report, after
preserving its predecessor. A wider check of **380** archived snapshot, record
and scan-report files again found **zero** changed hashes.

The result is intentionally still **blocked**: the September 25 and 28 runs
now identify GAS as having a real revision inside the requested overlap. The
September 29 run has ten pending momentum observations, zero resolved outcomes,
and null returns. Nine unaffected stocks are not cherry-picked out of either
quarantined run. CLI exit code 2 is the expected data-quality failure, not a
failure to execute the scorer. These are paper observations, not account P&L.

Preserved reports live under
`data/paper/research_gate/20260930/paper_requested_window_v1/`:

| Report | SHA256 |
|---|---|
| `scores_before.json` | `da0aba50b902b4e8e01d82f4765afd2b3ce2bd0f819f4eaaf1f66ecdd8a7de9b` |
| `scores_after.json` | `f07d63a625ccfb32df973fce21cdfad8ba11472308c953fcdbc67007a3ea8e6d` |

The real volume differences remain a source-reconciliation requirement. Their
cause is not established by this fix. Neither historical snapshots nor recorded
decisions were rewritten, and no daily bar was fabricated.

## Fresh remote engineering gate

[Run 36689514737](https://github.com/Tan28092006/stock-analysis-automation/actions/runs/36689514737)
completed successfully in both jobs for pushed source
`788401f12dc4b22e5587e8a635a1d2b062728270`: all contract tests and **1,196**
pinned replays (290 parent + 180 books + 530 timeslices + 196 VN30/VN100).
The replay job took 17m43s. This is not a validation-only job or an inference
from the preceding CI badge. Subsequent local edits at `8e40c9c` change only
AGENTS.md and the VN100 audit, not any runtime, research, test, config or CI code.

GitHub reports artifact `research-output-36689514737`, ID `11086355816`, size
18,552,542 bytes and archive digest
`686946613d463c09b28c3d346529853213657a9fc0f61f2f1529a44d0dde6c4f`.
The artifact was downloaded to
`data/paper/research_gate/20260930/github_36689514737/`. All four complete
economic block/slice payloads, book funnel, paired diagnostics and trial counts
were compared to the preceding audited run `36685094314`. There are no
structural, categorical or substantive numeric differences. Maximum absolute
roundoff is **5.551115123125783e-16** for the parent/book diagnostics; timeslices
and VN100 payloads are exactly equal. All **85 distinct source blobs** recorded
across these results match pushed Git contents at `788401f`; manifest hashes
also match. Git-blob bytes, rather than Windows checkout line endings, are the
cross-platform source reference.
An additional exact-equality check of each embedded `replay` object passes for
all 290 + 180 + 530 + 196 paths: fills, cash/settlement records and daily NAV
are unchanged, not merely within a profit-summary tolerance.

| Remote result | SHA256 |
|---|---|
| Parent gate | `fcf51ae936d79d33ee95f16029bf874a6a66b6a9a7e3c1db427da30627c28483` |
| Books | `d23c25617689ac7d01d09d951f32bc6606e102b378b5d2527d72baac57fd57d2` |
| Timeslices | `a0a061836c9ca7673064a6e01712cb0deeb8ae331a724d5f42eb7666d8c610c9` |
| VN100 | `0d54bb8a6c99ce9681f2aa117702ebac4d2734e2eaef238adada4c2e19a60b7c` |

CI used PR test-merge commit `23225ef511f72a2de40a67f186ad8331042d249f`;
the pull request remains open and draft, not merged into main. Every result
retains `live_eligible: false`. The activity and economic failures in the VN100
audit are unchanged. No new signal hypothesis, model training or live order
was introduced by the paper-scoring repair.

The workflow reports upcoming runner/action-runtime migration warnings. Its
dependency versions are bounded ranges, not an exact environment lock, and
no branch-protection or live-promotion claim is implied by this green run.

## September 30 archived-source reconciliation

The existing September 30 score report is still `blocked`: the September 25,
28 and 29 runs are quarantined, with ten September 30 observations pending
and zero resolved. No fresh signal scan was requested or run for this audit.
The investigation compared all **31 files** in each archived snapshot, not
only the ten recommended stocks. Each manifest first passed `verify_snapshot`,
including raw-source hashes and raw-to-CSV reconstruction.

The September 30 outcome snapshot is
`data/paper/snapshots/20260930T101058138662Z/manifest.json`, SHA256
`9f6d4ca079269f03b7ac828c2ca50ede32637cc2c4d1ccb701d2b986beda3ab2`.
Comparison starts at its declared requested start, December 19, 2024, and ends
at each original signal date. There are no missing rows within that overlap.

| Original session | Snapshot directory | Compared symbol-date rows | Changed rows |
|---|---|---:|---:|
| 2026-09-25 | `20260926T022757171239Z` | 13,235 | 4 |
| 2026-09-28 | `20260928T124557138267Z` | 13,266 | 4 |
| 2026-09-29 | `20260929T100504472404Z` | 13,297 | 4 |
| 2026-09-30, self-check | `20260930T101058138662Z` | 13,328 | 0 |

All twelve changed rows are last-session **volume-only** revisions. Open,
high, low and close are unchanged throughout the compared overlap. Volumes
below are shares, original archived value followed by the September 30 value:

| Session | FPT | GAS | TCX | VPB |
|---|---|---|---|---|
| Sep 25 | 3,543,900 → 3,568,808 | 1,042,700 → 1,046,966 | 1,794,200 → 1,797,273 | 46,844,000 → 46,877,134 |
| Sep 28 | 5,148,700 → 5,185,502 | 1,158,100 → 1,162,223 | 2,624,700 → 2,631,441 | 20,503,800 → 20,523,441 |
| Sep 29 | 3,118,000 → 3,141,350 | 1,601,600 → 1,605,028 | 2,629,000 → 2,635,172 | 16,588,300 → 16,599,712 |

Original manifest SHA256 values, in chronological order:

- `bf4a310704db4a2124f9b3e47ab59c3698347170f9f77d1853582018933d5ffd`
- `ed9e6a66b536820ee81fd71534fe1825765036739146fb60e0ec7b29e0fb4ad3`
- `b94b3f86e7c016993f23381b0a9b57b5318029c034bac2d9e3d065fd19776d76`

The raw VCI responses contain these differences; they are not CSV rounding
or the rolling-prefix bug. For the four affected symbols in all four
snapshots, raw `v` equals `accumulatedVolume` on every returned row, so merely
switching between those fields does not resolve this discrepancy. The observed
pattern suggests a latest-bar aggregation/finality issue, but does not establish
its cause or prove odd-lot inclusion. Snapshot verification proves integrity of
the archived response, not that a provider's latest daily bar is final.

Volume feeds MR confirmation, momentum volume filters and liquidity sizing.
These revisions must therefore remain visible even though OHLC is unchanged.
No comparison tolerance was relaxed, no unaffected recommendations were scored
selectively, and no original snapshot, decision or derived score was changed
by this read-only reconciliation. Provider finality semantics and their effect
on live/replay parity remain unresolved; the September 30 bar's future revision
status is unknown. This audit does not introduce a new strategy or trading rule.

A read-only feature sensitivity check on these same twelve archived rows used
the existing event feature function, aligned to the same December 19 requested
start and truncated at each original signal date. Maximum volume revision is
0.7488775%; maximum 20-session ADV revision is 0.02990184%. None of the twelve
rows changes the existing `volume >= 1.5 * prior_20_session_mean` Boolean gate.
This small sample does **not** prove invariant MR decisions, rounded quantities,
future data or entire portfolios. No scanner or new trade replay was run, and
the exact revision quarantine remains unchanged.

## Decision versus outcome lineage: code review follow-up

Read-only inspection of `paper_scoring._outcome` and `simulate_mr_exit` confirms
that scoring consumes the sealed recommendation; it does not regenerate the
historical signal from the outcome vintage. Momentum's descriptive result uses
the recorded signal close and the close 21 sessions later. MR uses the recorded
stop/target/holding horizon, the next open and future OHLC bars; its volume tests
are on entry/exit bars, not the pre-signal historical volume series.

The exact shared-history check is a separate, earlier gate, so a positive-to-
positive revision of historical volume can block an entire run even when no
field read by the outcome calculation changes. This is not evidence that the
original strategy, quantities or features would be unchanged under revised
inputs. Decision-input finality and outcome-measurement comparability are two
different questions. Neither one establishes actual fills.

A possible future contract should preserve original decision/raw integrity,
retain and expose later revisions, and test any narrowly permitted outcome
comparison separately from signal readiness. It must not silently replace the
old signal with a hindsight rerun, drop affected trades selectively, accept
price/missing-session changes, or relabel informational returns as funded P&L.
No scoring condition or derived score has changed during this review. Any
implementation first requires registry entry, explicit refusal cases and
RED/GREEN/CI evidence; the existing strict quarantine remains in force.

## Preregistered outcome-only volume revision contract

Registered September 30, 2026, before implementation or rescoring archived
outcomes: `paper_volume_revision_outcomes_v1_20260930`. This explicitly extends
the earlier requested-window policy, not its original verification evidence.
User journey: inspect delayed observations of an immutable paper decision even
when a later source changes only positive historical volumes. No signal is
regenerated, no trade is added, and no old decision or source bytes are replaced.

Acceptance contract:

- Continue verifying both source vintages, the ex-ante record, exact requested
  dates and all non-volume columns. Missing dates, price/other-column revisions,
  source corruption, duplicates and invalid risk plans remain run-atomic errors.
- Only finite, strictly positive original AND outcome historical volume may
  differ. Positive/zero transitions remain errors. No fitted size tolerance:
  increases and decreases of any positive magnitude are reported, not hidden.
- Record each revised date with original/outcome volume, symbol and original
  run path. Deduplicate counts within a run across MR and momentum tracks.
  Each scored observation exposes `decision_inputs_revised`; this is NOT a
  declaration that the revised inputs would have generated the same decision.
- Reports with accepted revisions use `ok_with_revisions`, not plain `ok`.
  Any invalid run takes precedence as `blocked`; discard that run's staged
  outcomes AND warnings, retaining the explicit error. Other valid runs remain
  separately visible. CLI prints revision counts for score-only and run modes.
- Existing MR simulation, future-bar volume checks, costs, settlement assumptions
  and informational momentum return do not change. Actual execution and funded
  portfolio P&L remain unverified. No price or missing-session relaxation.
- Synthetic RED/GREEN tests cover both tracks, pending/resolved outcomes,
  increase/decrease/large revisions, refusal boundaries, atomicity, duplicate
  counts and byte-for-byte original evidence preservation. CLI end-to-end tests
  use synthetic sources, with no network, fresh real scan or broker access.
- Run all contract tests and all 2,618 pinned research paths. Their economics
  must remain unchanged within the existing cross-platform 1e-12 tolerance.
  If examining archived paper outcomes after validation, preserve the previous
  blocked score and write a separately versioned report, not a hindsight record.

This is a measurement/data contract, not a new profit hypothesis. The twelve
already observed source revisions above are development evidence motivating
the contract; they are not an untouched validation sample. Provider finality,
feature/sizing sensitivity, paper/replay parity and live eligibility are not
resolved by allowing a flagged descriptive outcome measurement.

### Implementation and local verification

- RED checkpoint `8be690d`: 24 intended failures, 32 passes on paper scoring/CLI.
  Fixture-only issues (synthetic future clock and a historical zero-volume bar)
  were corrected before this checkpoint; they are not counted as valid RED.
- GREEN checkpoint `aab449b`: the identical target passed all 56 tests in 51.77s.
  `447975d` then removed a fixture dtype warning; four refusal tests passed.
- Full local suite at `447975d`: **683 passed**, 23 pre-existing warnings,
  201.54s. No failed or skipped tests. Compile checks and `git diff --check` pass.
- Operational coverage run: 67 tests passed in 84.90s; scorer 90%, runner 91%,
  combined 91% (309 statements, 29 missing). Every new history-comparison and
  revision-reporting statement is covered. This is statement coverage, not a
  claim of complete branch coverage or certification of every pre-existing path.
- Existing CI already includes both changed test files. The latest patch still
  requires its complete 2,618-path CI replay and independent economic comparison;
  neither local unit tests nor protocol validation substitute for that evidence.

Coverage JSON: `data/paper/paper_volume_revision_verification/coverage.json`,
SHA256 `95ab122effcaadc1d6be5334b8bdab102050ca00ca77660af0c20f47528c40d3`.

### Offline archived observations under the new contract

After local verification, a separately versioned report was generated at
`data/paper/paper_volume_revision_verification/archived_outcomes_v1/scores.json`.
It uses the already preserved September 30 outcome snapshot identified above:
no provider request, new signal scan or broker interaction. The previous
`data/paper/scores/latest.json` remains unchanged and still records the old
strict-policy result; it has not been silently replaced.

The new report is `ok_with_revisions`: four original runs, **40 pending momentum
observations**, zero resolved, zero not-entered and zero errors. These are ten
informational picks per session across September 25/28/29/30, not 40 independent
stocks, new entries, filled orders or funded positions. No MR observation is
present in these four records. All 21-session outcomes are still immature;
there is no forward profit estimate yet.

Three unique recommended-symbol revisions are flagged, all GAS: September 25,
28 and 29, with original/outcome volumes listed in the reconciliation above.
The scorer compares the requested history of **recommended symbols only**,
not every field used by the entire scanner/universe. This scope explains three
warnings versus twelve revisions in the earlier all-31-files audit. A false
`decision_inputs_revised` on one observation means no permitted volume revision
in that compared symbol/window, not proof that all scanner inputs are invariant.

All **260** enumerated input/source/record/latest-report/cache files matched
their pre-scoring SHA256 values afterward. Output evidence:

- Scores SHA256:
  `6c7827b576c4b20e37a729818337230aef0302579e0069ad684a24638320abb6`.
- `verification.json` SHA256:
  `bedf118ce8328e5ff1f74fe731b0c0b1760521187ad1bbb088702e34d46bcad2`;
  contains all input hashes, source hashes, revision scope and Git `447975d`.
- Offline generator: `data/paper/paper_volume_revision_verification/audit_archived.py`,
  SHA256 `64c9fd451ec94a18d493f1ea25c87f6569525e7b3fd76bc85a637d9a8637305a`.
  Its destination is exclusive; an existing report directory is never replaced.

The retained original decision vintage remains authoritative. No hindsight
signal recomputation, newly manufactured orders or strategy/model promotion
occurred. Provider finality and execution readiness remain separate blockers.
