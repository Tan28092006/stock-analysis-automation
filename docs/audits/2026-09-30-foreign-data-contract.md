# Foreign flow v2 — contract frozen before implementation

Scope: normalize legacy data without overwriting it; collect public Vietstock
daily per-symbol foreign and proprietary series automatically via the existing
17:05 Windows EOD launcher. No model, strategy thresholds or live orders change.

Grain: session, symbol, investor category, source, provider aggregation definition.
Canonical values VND and shares; explicit billion-VND convenience columns only
at the compatibility boundary. Do not merge matched trades with unknown chart
aggregates. Raw source TradingDate is interpreted in UTC then Asia/Saigon, never
host date. Source DataTime is metadata, not historical feature availability.

Availability: response receipt time in UTC, never backdated to the session.
Filter incomplete/future sessions at the Vietnam 16:00 cutoff. Historical rows
fetched now are backfill available now, not point-in-time knowledge of the past.
Legacy rows lack observed_at: normalized research-only, never included by the
default production loader. Recover dates only from retained source epochs.
Missing values, undefined periods and unverified dates are quarantine, not zero.

Storage: immutable run folders with raw market-response bytes, SHA256, receipt
times, normalized rows and a manifest committed last. No cookies/tokens saved.
Atomic status pointer and process lock. Keep revisions; as_of selection filters
availability BEFORE choosing the latest source revision. Same response is
idempotent at the loader's session/symbol/category/source key. Tampered or
incomplete runs fail closed. No previously good dataset is overwritten.

Collection: retain the prior 100-symbol collection basket and union current
configured trading universe, so newly admitted VN30 names are covered. Explicit
per-symbol endpoints avoid the old first-page-only market-table truncation.
Bounded timeouts/retries/rate, recent-window overlap to recover missed sessions,
coverage/freshness/gap report, partial failure returns nonzero. No silent stale
fallback. Empty payload is not all-zero activity. Runtime failure in flow job
must not prevent independent paper job, but the combined launcher cannot print
success if either fails. Legacy callers route through the same collector.

Verification: RED/GREEN normalization, timezone, intraday, raw/hash, revisions,
as-of, duplicates, missing symbols, retries, partial failures, concurrent writes,
legacy quarantine and launcher exit-code tests; actual bounded public-source
run; default loader verification; complete registered 290 price-strategy replay
must remain unchanged because these flows are not yet strategy features.

Limitations: public chart aggregation may differ from detailed matched/put-through
series. Collection readiness is not alpha validation or exchange certification.
Legacy chart rows cannot be magically certified; refetched overlap is a new
vintage. All automated runs require this machine/network/task account available.

## Collection readiness extension — registered before implementation

Hypothesis `foreign_collection_readiness_v1_20260930`. Two observed operational
failures motivate this extension, not a new trading hypothesis: the default
runtime basket omits TAL/VCK/VPX from the verified July HOSE table, and the
September 30 17:05 batch received valid but previous-session proprietary data
for 26 names. A later source probe returned those missing rows without changing
1,014 shared rows. Evidence is retained in the foreign-data rollout audit.

Keep all prior 100 collection names and explicitly add MCH, TAL, TCX, VCK, VPX;
continue unioning the current configured VN30. This is a 105-name collection
basket, not a historical membership change. July HOSE attachment SHA256 is
`441799b3afbc4f29f8b53ce6ef3b500b9121a4d032ca4cfeb1d4aa6e14407879`:
https://staticfile.hsx.vn/Uploads/UploadDocuments/2479382/15072026%20CBTT%20-%20Danh%20muc%20thanh%20phan%20HOSE-Index%20thang%207.2026.pdf
No announcement/effective date is inferred for backtesting. Do not drop retired
names, certify adjusted prices, promote foreign-flow signals or rewrite H1 inputs.

Add opt-in `foreign_refresh --retry-stale` and use it in the existing EOD BAT.
The first batch still collects the full requested basket. If source responses
are valid but lack the current completed session, retry only missing names,
with at most three extra batches after waits of 60, 300 and 900 seconds.
Both investor categories remain separately verified; existing endpoint, network
retry, rate, raw storage and source-time rules remain unchanged. Each batch is
immutable, and later data becomes available only at its own batch completion.
Recompute readiness against the entire originally requested basket, never the
last small retry alone. Report batch manifest paths and attempt count.

Do not retry away schema/quarantine errors, internal historical gaps, source
hash failures, unfinished batches or concurrent writers. Abort if the completed
session changes or the clock reverses during a retry wait. Remaining missing
data after the bounded attempts is nonzero/partial, never fabricated zero or
silent success. Normal manual collection and read-only status do not wait.
The schedule stays 17:05; delayed retries can defer the subsequent paper job by
up to 21 minutes of waits plus bounded request durations. The BAT must still run
paper independently after a flow failure and must preserve its exit code.

Acceptance: RED/GREEN collection coverage, bounded selective retries, persistent
staleness, invalid-source fail-closed, full-basket readiness, cutoff/clock changes,
immutable vintages/as-of and CLI/Windows launcher tests; at least 80% coverage.
Re-run the full repository tests and all 2,618 pinned replays, preserving every
economic result. No fresh live scan is part of this change. Source-backed smoke
is restricted to collection and cannot prove a future scheduled run has occurred.

### Candidate implementation evidence (full replay still pending)

Isolated branch `codex/foreign-collection-readiness` retains checkpoints:
`d636a4f` executed 16 intended RED cases (nine prior checks passed), followed by
`6612f08` with all 25 GREEN for coverage/retry/CLI/Windows launcher behavior.
The combined foreign-flow suite passes 51 tests. Coverage is 93% of 405
statements (collector 94%, refresh CLI 89%); no coverage pass is claimed for
the entire repository. No installed Ruff/Pyright is available; compilation
and diff whitespace checks pass, not a type/lint certification.

A full clean-checkout run initially found one existing ML unit-test dependency
on a developer's ignored model registry (658 passed, one failed). `99ce6e2`
makes that test supply an isolated synthetic registry/cache and verifies the
missing-registry refusal before testing thresholds. It also prevents local
symbol overrides from affecting the test and restores the cache on failure.
No ML runtime, threshold, production artifact or activation changed. The test
is now included in the remote contract job. All **659 tests** then passed,
with 23 existing warnings, in 175.12 seconds; 54 focused checks also pass.

Real-source CLI smoke used `--retry-stale --symbols TAL VCK VPX` with a separate
output root and completed ready on its first batch: 120 rows, no missing latest
session or gaps, no quarantine. It tests real initial success, not a fabricated
claim that a future delayed-source retry has already occurred. Manifest:
`data/paper/foreign_readiness_new_code_smoke/runs/20260930T112509688349Z_a756e283/manifest.json`,
SHA256 `43f34c32e761d34c00dc09f26cedc902c1c8f8e99c2d475687f2e17e1a598927`.
The deterministic late-publication test uses the actual immutable store and
checks that the old as-of query still lacks the late value while the new one
contains it. Retry timing itself is injected in tests, with no real test sleeps.

The new full 2,618-replay chain runs from the isolated checkout against the
unchanged verified market snapshots. It must finish and be compared before
claiming full regression acceptance. The old receipt revision's successful CI
is separately audited; it is not reused as proof for this candidate. No local
signal scan, broker order or production pipeline activation is part of this
candidate's smoke tests.

### Evidence retention and partial comparison

The candidate parent result (290 paths) has been compared with audited CI
`36703736150`: complete economic blocks match to a maximum absolute difference
of 1.1102230246251565e-15, with unchanged snapshot hash, all 85 recorded local
source hashes verified, and `live_eligible=false`. Result SHA256:
`2003f757583fdb1d8718a5824c6bc24933cbd79408ebc90612966da6bc45c4bd`.
Its recorded Git commit is `99ce6e2`; subsequent `1685d14` only adds documentation
and does not change the runtime/config content. The remaining candidate suites
and the fresh remote artifact are not yet accepted by this partial comparison.

Coverage evidence was copied without modification out of the temporary worktree
to `D:/Chungkhoan/data/paper/foreign_readiness_verification/.coverage`.
Both files have SHA256
`fc9512c9fe908af1ca95dbf8cb1467113b49ddc6ca55a75510ba5bec27c55f94`.
Reading the retained file from the candidate checkout reproduces collector 94%,
refresh CLI 89%, combined 93% (405 statements, 27 uncovered).

A read-only candidate `--status` against the runtime store reports ready for
September 30 over its default 105-name basket: 9,090 vintage rows, no missing
latest foreign/proprietary values and no internal gaps. No collection, signal
scan or broker action was made by this check. Windows Task Scheduler still
points to `D:/Chungkhoan/run_eod_update.bat`, with next run October 1 at 17:05;
the last scheduled run remains exit 2. The later manual recovery does not
rewrite that failed run or prove the next scheduled retry will succeed.

### Completed candidate CI and downloaded-artifact audit

[CI 36709652180](https://github.com/Tan28092006/stock-analysis-automation/actions/runs/36709652180)
passed both contract and pinned-data-replay jobs for pushed head
`1685d148b996cf06775ee85958f2c7f30de9118a`. All **2,618** paths completed.
The downloaded results record PR test-merge commit
`4a3a1dd8b854e6ed90ae69a54b262ebb44602847`; no merge into main occurred.
All 92 distinct recorded source blobs match the pushed candidate, and both
input snapshot hashes remain unchanged.

Complete economic payloads of all five suites match audited CI `36703736150`
using an absolute numeric tolerance of 1e-12.
Maximum absolute delta is 5.551115123125783e-16, not a change in return or orders.
The independent event audit passes for 1,422 paths, 23,131 fills and 69 registered
comparisons, including PIT membership, signal/entry timing, prior-bar volume
caps, board lots, add-on limits, monthly activity and regime attribution.
Every result still has `live_eligible=false`; no strategy was promoted.

| Remote result | SHA256 | Maximum absolute economic delta |
|---|---|---:|
| Parent | `54e35b16060d59b7688bbfb91031c910987fcf8f22ce8b25d8be44d13caaec1d` | 5.551115123125783e-16 |
| Books | `592d92e7007dd3a924026d872a2bc6639139539d966bfc12cfb76d4daf9af745` | 5.551115123125783e-16 |
| Timeslices | `a54cd7c0d7c5c2944e688f2d5ba34e0022cd43616adf59a1a050a7113ecf7825` | 0 |
| VN100 | `edacad1d351355a1727f1cbf442ed1fd269e3b31ac0f0879161481e923007502` | 0 |
| Events | `867875e3abdc151a076d30c292038daab5c818f9d19ff501e475afbc28b4263e` | 2.220446049250313e-16 |

Artifacts are retained in `data/paper/research_gate/20260930/github_36709652180/`.
GitHub reports artifact ID 11095220442, 37,448,250 bytes, archive digest
`37568da9dd58d77ac3429573f333f78f1e3df3669a3ba02c4b3cab5aa2abc4a0`.
That archive digest is provider-reported, not a locally retained ZIP checksum.
The result hashes above were independently calculated from downloaded files.

### Local candidate checkpoint: 1,000 paths verified, chain still running

The local books suite (180 paths) and timeslices suite (530 cash restarts plus
44 carry attributions) have also completed. Full economic-object comparisons
against CI `36703736150`, snapshot checks and local source-byte checks pass:

- Books SHA256 `68841c57f4ea058d8fe74194e648e649d612113148c9231056fa5d5367c34486`;
  85 sources, maximum delta 1.1102230246251565e-15.
- Timeslices SHA256 `947a0b4e34923f76b21547a41414839ba7f0225e1ba2cb788981f283e2e79cdc`;
  86 sources, maximum delta 5.551115123125783e-17.

Both record `1685d14`; the earlier parent records `99ce6e2`, whose relevant
source/config bytes are unchanged by the intervening docs-only commit.
The Windows VN100 and events suites are still pending; the successful remote
audit does not silently mark this unfinished local process as complete.

Subsequent local VN100 checkpoint: all 196 paths finished, bringing verified
local replay paths to **1,196**. Full blocks, paired diagnostics, partition flag
and trial count match the newly audited candidate CI `36709652180` with maximum
absolute difference 1.1102230246251565e-16. The snapshot matches and all 88
recorded local source hashes were verified against the frozen checkout.
Result SHA256 is `cdce2b95886e4be5910ec53816146ee0f68d12c4096eaffaa5c306afc4701f02`;
recorded source commit is `1685d14`, with `live_eligible=false`.
Only the 1,422-path event suite remains pending in the local chain.

### Completed Windows regression and final source audit

The local chain has now exited zero after all **2,618** paths. This supersedes
the earlier pending checkpoints. The event result SHA256 is
`e554cdde1fd8ce86d7fbc7c45e5ad24c68588d294ece59f8c01b11edc4d081de`.
Its complete protocol, panels, paired universe tests and live blockers match
candidate CI `36709652180` within 8.881784197001252e-16 absolute roundoff.
The independent auditor passes 1,422 event paths, 23,131 fills and all 69
registered comparisons, with reconciled regime partitions for VN30, H1/VN30
and H1/VN100. This is simulated-fill accounting, not brokerage evidence.

Before releasing the source freeze, every recorded local source hash across
all five suites was rechecked: 92 distinct paths, no byte mismatch. Full
economic objects were re-compared against the new candidate CI; the maximum
absolute difference across all five suites is 1.1102230246251565e-15, below
the fixed comparison tolerance of 1e-12. Both pinned snapshot hashes match.
All results retain `live_eligible=false` and the prior economic rejection.

Acceptance here is for the collection-readiness implementation and unchanged
historical economics. It does not certify provider finality, resolve the old
paper volume-revision quarantine, change strategy-to-order parity, authenticate
broker receipts or demonstrate real-money profit. No live scan occurred.
