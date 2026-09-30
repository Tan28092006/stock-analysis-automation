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
