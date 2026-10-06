# Price history store (10-year daily, VCI)

`stock_agent/data/history_store.py` keeps a long daily OHLCV history that is
**separate** from the daily paper snapshot (`data/paper/snapshots/<id>/`, 650
calendar days, strict: one bad row blocks the snapshot). The paper runner, the
swing desk freshness gate (G0) and `verify_snapshot` are unchanged and never
read this store.

## Usage

Run from the repo root. `fetch` is the only command that uses the network.

```text
# universe + duration estimate, no network
python -m stock_agent.data.history_store fetch --years 10 --universe all --symbols PNJ TPB --out data/history --dry-run

# fetch a new store (one VCI request per symbol, >= 3.1 s apart)
python -m stock_agent.data.history_store fetch --years 10 --universe all --symbols PNJ TPB --out data/history

# re-hash every file and re-derive CSV + issues from the raw bytes
python -m stock_agent.data.history_store verify data/history/<UTC>_<id>

# newest complete store, ok/partial/failed counts, sessions behind the completed session
python -m stock_agent.data.history_store status --root data/history
```

`--universe` takes `all` or any of `vn30` (current `configs/universe_vn30.json`),
`vn30-history` (every symbol ever listed in the two VN30 membership configs),
`vn100` (every symbol in `vn100_membership_2026_h1.json`) and `vnindex`. The
union is deduplicated; the manifest records which set brought each symbol in.
`--as-of` defaults to the completed session (16:00 Vietnam cutoff).

On 2026-10-06, `all` + PNJ/TPB is 107 symbols: at least 107 x 3.1 s = 5.5 min,
in practice about 6-10 min depending on response time.

Exit codes: `fetch` 1 if any symbol failed (partial symbols are expected, e.g.
VNINDEX); `verify` 1 on any mismatch; `status` 1 if stale or no store exists.

## Layout and guarantees

```text
data/history/<UTC>_<id>/      new directory per fetch, never reused (git-ignored)
  raw/<SYM>.json              provider bytes exactly as received
  prices/<SYM>.csv            valid rows only: date,open,high,low,close,volume
  issues/<SYM>.json           quarantined rows with reasons and raw values
  manifest.json               written last, atomically (tmp + replace)
```

- A directory without `manifest.json` is an interrupted run. `status` lists it
  under `incomplete_dirs` and never uses it.
- The per-symbol manifest entry holds sha256 values for raw, CSV and issues, plus
  the request, `fetched_at`, attempts, first/last date, row count, quarantined
  count and reasons, `ends_before_as_of` and calendar coverage.
- Status per symbol: `ok` means no quarantined rows. `partial` means at least
  one row was quarantined; the other rows are kept unchanged. `failed` means an
  HTTP error, a structural or wrong-symbol response, or no valid row.
- Row rules match `reconciliation.parse_vci_history`: non-finite values,
  prices <= 0, volume < 0, high/low inconsistent with OHLC, duplicate dates and
  non-session dates. A timestamp that cannot be parsed is also quarantined.
  Rows outside `[start, as_of]` are counted in `excluded_out_of_window`. Rows
  are **never** filled, repaired or re-adjusted. The kept rows are re-checked
  against the strict snapshot rules.
- Pacing is at least 3.1 s between the starts of all calls, including retries.
  Only timeouts, connection errors and 5xx are retried, up to 3 attempts.
  429 and other 4xx are never retried. A 429 stops the run, and the remaining
  symbols are recorded as `not attempted`. Five consecutive failed fetches also
  stop the run.

## Limits (read before using the data)

- **Survivorship before 2022.** Point-in-time membership exists only from
  2022-08 (VN30) and 2025-08 (VN100). The universe is a union of current and
  recorded members. Delisted or demoted names that never appear in those configs
  are absent, so pre-2022 cross-sections are survivorship-biased. No historical
  member lists were invented to fill this gap.
- **Provider-adjusted prices at fetch vintage.** VCI applies corporate-action
  adjustments as of the fetch date. A later fetch can change old prices, and the
  data is not point-in-time certified (`adjustment_basis` in the manifest).
- **Calendar coverage.** `HOSE_HOLIDAYS` covers 2021-01-01 to 2026-12-31. Only
  that window reports `missing_sessions`. Earlier weekday gaps are
  `UNVERIFIED`; they may be holidays. Pre-2021 rows on weekday holidays cannot
  be detected as non-session.
- **First row is not a listing date.** Coverage starts at the first valid row.
  It does not use a verified listing date.
- `status` does not re-hash files; run `verify` before relying on a store.

## Research-gate note

This store is an engineering artifact. It is not wired into `research_ml`, R5
or `scripts.research_gate`. Using it in research results changes the data
contract, so the hypothesis and protocol version must first be registered in
`configs/research/research_gate_v1.json` (see `AGENTS.md`). A successful fetch
or verify does not approve trading or deployment.
