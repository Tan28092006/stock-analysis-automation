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
