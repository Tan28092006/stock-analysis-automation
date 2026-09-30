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
- BUY fills cannot exceed the limit; SELL fills cannot be below it. Fills must
  satisfy declared band/tick/board-lot checks. Record actual fees and signed cash
  movement, not modeled slippage. A validated receipt file is not cryptographic
  broker authentication, funded-account P&L, or live strategy eligibility.
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
