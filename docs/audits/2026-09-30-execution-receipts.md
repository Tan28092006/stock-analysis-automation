# Execution receipts v1 — contract registered before implementation

## Outcome and scope

The user wants a local backend usable for manual trading, not a UI. Existing
event research conditions purchases on the final opening price while assuming
that very opening fill. Its economic results are not executable-order evidence.
Goal of this change: accept an immutable, pre-session limit-order plan and audit
external timestamped order/fill records against it. No broker connection, order
submission, live scan, collector schedule change, strategy activation or retrain.

The proposed opening-only limit-order replay is **not** implemented: daily OHLC
also cannot establish cancellation timing or residual intraday fills. Adding a
new opening-only simulator would not close the identified real-execution gap.
This is an execution-evidence data contract, not a new alpha hypothesis.

Primary sources checked 2026-09-30:
[SSI conditional orders](https://www.ssi.com.vn/khach-hang-ca-nhan/giao-dich-chung-khoan-ib-web)
use continuous-session market-price triggers, excluding periodic auctions;
[SSI KRX FAQ](https://www.ssi.com.vn/khach-hang-ca-nhan/krx-thi-truong-co-so-faq)
documents order amendments and changed auction priority. Neither source proves
that our simulated orders filled. A submitted limit buy does not encode a lower
execution-price rejection. Cancel requests do not themselves prove cancellation.

## Frozen schema and invariants

- One plain LO plan, BUY or SELL, fixed whole-board-lot quantity and VND limit;
  no conditional lower price, auction-only expiry, resizing after open or order
  transmission. Unknown fields fail closed. Plan digest binds all plan contents.
- Require explicit strategy/signal IDs, source digest, timezone-aware signal
  availability, creation and validity timestamps; signal must be from an earlier
  Vietnamese session. Creation precedes validity. Session is an exchange trading
  date; validity cannot span dates. No inferred timestamps from daily CSV dates.
- Require unadjusted VND prices and declared official reference/floor/ceiling,
  tick size, observation time and source digest. Reference observation and cash/
  sellable-share resource snapshot cannot be later than plan creation. Tick,
  board-lot, price-band and resource checks precede admission. Declared provenance
  is not independently authenticated market data. Historical adjusted snapshots
  cannot silently be relabeled as certified raw prices.
- The resource snapshot must be from the plan-creation date, not a stale earlier
  cash balance. The reference observation must be from the target session date.
  This contract checks the declared tick grid; it does not certify historical
  exchange-specific tick schedules or allow adjusted-price tick rounding.
- Cash available must be net of other reservations; sellable quantity must be
  net of reserved shares. This is a **single-plan boundary**, not a batch allocator
  or whole-account solvency proof. Require an explicit fee reserve rate for buys;
  do not invent the user's capital. No sale proceeds or unsettled shares assumed
  available. Settlement availability is externally supplied, not inferred from
  T+3 research bookkeeping.
- Receipt origin must be broker execution report, not daily-bar simulation. Each
  receipt has a stable event ID, broker order ID, sequence, occurred/observed time,
  source digest and exact plan digest. Audit cutoff is explicit; no future
  observed receipt admitted. Exact duplicates are idempotent; conflicting IDs or
  sequence numbers fail visibly. One broker order per plan.
- ACCEPTED precedes FILL. A rejection before acceptance is terminal. CANCEL_REQUEST
  keeps residual quantity live; fills may arrive before CANCELLED. No fills after
  terminal acknowledgement, before order validity or above fixed quantity. Expiry
  needs a receipt at/after validity end; absence of records is unconfirmed, not
  inferred zero fills or a successful cancellation.
- CANCELLED, REJECTED and EXPIRED receipts must contain an integer
  `cumulative_filled_quantity` exactly equal to the supplied individual fills.
  Otherwise a missing fill could incorrectly become a clean terminal audit.
  Actual buy cash debits including recorded fees cannot exceed the declared cash.
- BUY fills cannot exceed the limit; SELL fills cannot be below it. Fills must
  satisfy declared band/tick/board-lot checks. Record actual fees and signed cash
  movement, not modeled slippage. A validated receipt file is not cryptographic
  broker authentication, funded-account P&L, or live strategy eligibility.
  Plan timestamps are also supplied evidence, not an independent pre-market
  timestamp seal. `fully_reconciled` means only that the supplied single-order
  lifecycle reaches a consistent terminal state, not verified account history.
- CLI reads supplied JSON files and exclusively creates one audit output. No
  overwrites, network writes, fresh market fetch or private data committed. User
  supplies records only if/when they want actual execution reconciled.

## Verification and acceptance

TDD for timezone/availability, adjusted prices, invalid resources, illegal price
conditions, tampered plans, duplicate receipts, partial fills, cancellation races,
overfills, costs, cutoff boundaries and a complete local CLI round trip. Minimum
80% coverage. CI retains all prior causality/accounting tests and 2,618 pinned
replays; their economic payloads must remain unchanged. No new return comparisons
are registered: this patch must not change any existing strategy.

Completion of this boundary does not establish profitability or activate v1.
Still required: strategy-to-order parity with a tested executable strategy, raw
reference/quote lineage, portfolio-wide reservations, authentic broker receipt
provenance and prospective net outcomes. The broader user goal stays unachieved.

## Local use and implementation evidence

```text
python -m scripts.execution_audit --plan <fixed-plan.json> --receipts <broker-receipts.json> --as-of <timezone-aware-cutoff> --output <new-audit.json>
```

The plan schema is illustrated by the **synthetic** `plan` fixture in
`tests/test_execution_receipts.py`; its `event` helper shows normalized receipts.
No real account data is present in these fixtures. Receipt files are JSON arrays,
not daily OHLC files. The CLI retains exact input-byte and implementation hashes,
rejects duplicate JSON keys/nonfinite numbers, caps each input at 8 MiB and uses
exclusive output creation. Invalid evidence or an existing output exits 2;
valid supplied evidence exits 0, even when its truthful state is `unconfirmed`.
An exit 0 is never a trading or profit approval.

- `7421708`: 44 executed RED cases for missing implementation; preregistration
  and schema expectations committed before implementation.
- `e5b6d2e`: same 44 GREEN; immutable plan/receipt content validation.
- `04bd210`: 12 executed RED cases (terminal cumulative quantity, stale cash,
  malformed kind types, fee overdraft and missing CLI/CI), 46 existing cases green.
- `e0920ca`: all 58 GREEN, including subprocess CLI round trip; terminal fixtures
  extended to supply the newly required cumulative totals, not to bypass checks.
- `6906119`: 68 GREEN with additional schema/lifecycle/error-path tests. Coverage
  on 235 statements: core 100%, CLI 98%, combined 99%; the uncovered line is the
  module entry-point invocation, separately exercised by the subprocess test.

Full repository verification: **644 passed**, 23 existing dependency/test-fixture
warnings, 180.34 seconds. Both modules and both test modules compile; diff whitespace
check passes. No installed Pyright/Ruff modules were found, so type/lint passes are
not claimed. Filename-only secret-pattern scan of the four new Python files found
no known key patterns; it is not a comprehensive security certification.

The unchanged 2,618-replay regression remains required on this source revision.
No new return metric or actual broker fill was evaluated. Existing strategies,
live pipeline, ML, refresh schedules, user logs/cache and UI are unchanged.

### Partial local regression checkpoint (not full acceptance)

In `data/paper/research_gate/20260930/execution_receipts_8dcced0_regression/`,
the completed parent 290, books 180 and cash-timeslice 530 results have been
compared against downloaded CI `36694782314`. Comparison covers complete
economic objects, not only headline returns: parent blocks; book blocks/funnel/
trial count; timeslice slices/cash and carry counts/partition reconciliation.
The timeslice report also retains 44 carry-attribution paths. Their manifest
hashes agree, `live_eligible` remains false and all recorded source hashes
still match the local bytes (85 sources in parent/books, 86 in timeslices).

| Result | SHA256 | Maximum absolute numeric difference |
|---|---|---:|
| Parent | `9709019260357f78cc5acc23ac205130e7b160a746578b43be9bcb397a242081` | 1.1102230246251565e-15 |
| Books | `c9853a646d39c2eb32448dbd34f872a79bbd126cf500f229c5c6269f959db30c` | 1.1102230246251565e-15 |
| Timeslices | `4fcf0fc3eeea7947283192b5b2f26bba294bf58ae2616383aeec9421d5826f66` | 5.551115123125783e-17 |

The structural comparator's rejection self-tests also pass. These are floating
roundoff differences, not a changed strategy. The VN100 and event outputs and
fresh remote artifact still need their own checks; 1,000 checked replay paths
must not be presented as the complete 2,618-pass acceptance result.

### Completed remote regression and artifact audit

[CI run 36703736150](https://github.com/Tan28092006/stock-analysis-automation/actions/runs/36703736150)
completed successfully for pushed head `8dcced00c3ac76c566fffa2d06a35ea0c43ce048`:
contract checks and all **2,618** pinned replays. The downloaded outputs report
PR test-merge source commit `879b7bf1260326d7db14972cfb7897486194a6ed`; this is not
a merge into main. All 92 distinct recorded source blobs across the five suites
match Git content at `8dcced0`, with snapshot hashes unchanged.

The complete economic payloads of all five suites are **exactly equal** to the
previous audited CI `36694782314` (maximum numeric delta zero). Fields include
parent/book blocks, funnel, timeslice/carry results, universe paired diagnostics,
event protocol/panels/comparisons/live blockers and reconciliation/trial counts.
The independent event auditor checked 1,422 paths, 23,131 fills, 69 registered
comparisons, PIT membership, prior-bar volume caps, timing, board lots, activity
classification and regime attribution. All checks pass; `live_eligible=false`.

| Remote result | SHA256 |
|---|---|
| Parent | `64d2d09571b3d3e8fd645dd99563a62956cf97bfe6abde3069a9cb46a0eb17e4` |
| Books | `968cf783e875289912891f8511ac18b6c746099d314205ffac126a6b96d8f2d7` |
| Timeslices | `404cc9b66f3beeb34afe1d0be7d21c2456c7dc854bc915514df4950178395f70` |
| VN100 | `070804d0de89ae27e1d550437a7f872f9b520563881b615b44baab93d4d7342b` |
| Events | `54ef2511aafaf3245bc89dc179c3f9eba664ebfdbe0ff4ab0041eb596b4c6040` |

Artifact directory: `data/paper/research_gate/20260930/github_36703736150/`.
GitHub reports artifact ID 11091789036, 37,448,188 bytes, archive digest
`4aa0300be11e35175a47560e0d2f3def4fdf2d9c818a558468c3934158fdeb5f`;
the archive digest is provider-reported, not a locally retained ZIP checksum.
No broker records or real-money return were evaluated. The economic rejection
and execution/data/prospective limitations above remain unchanged.
