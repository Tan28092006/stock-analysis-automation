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
