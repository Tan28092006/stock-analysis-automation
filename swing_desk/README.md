# Swing desk (VN30, advisory)

A pre-trade discipline filter, trade journal review and prospective evaluation log for VN30 swing
trades. **Advisory only:** it never places, routes or authorizes broker orders and never changes
data-collection schedules. PASS means "no structural or behavioural red flag and a known setup",
not "will win". The edge of the LLM desk flow is an unproven hypothesis being measured
(`configs/swing_desk/prospective_v1.json`).

Rules, sources and limits: `docs/audits/2026-10-06-swing-desk-signal-research.md` (design) and
`configs/swing_desk/desk_v1.json` (every threshold). Agent procedure: `.claude/skills/swing-desk/SKILL.md`
(Codex copy: `.codex/skills/swing-desk/SKILL.md`).

## Commands (run from the repository root)

```text
python -m stock_agent.swing_desk status                      # read-only freshness report (+ history_store line)
python -m stock_agent.swing_desk refresh [--with-flows] [--history]  # reuse/fetch verified snapshot, build state/
python -m stock_agent.swing_desk check HPG --price 27350 --amount 50000000 [--reason S1] [--emotion calm] [--ack]
python -m stock_agent.swing_desk scan                        # gate table for all VN30 (needs fresh state)
python -m stock_agent.swing_desk review [--show-nav]         # portfolio review of portfolio.txt
python -m stock_agent.swing_desk events HPG --session 2026-10-07 [--event 2026-10-09:ex_dividend] --source <URL>
python -m stock_agent.swing_desk freeze --source baseline     # baseline + control, before 09:00 VN
python -m stock_agent.swing_desk freeze --source desk|single --file rec.json --model-id <id> --n-rollouts 4
python -m stock_agent.swing_desk score                       # resolve matured rows; writes recommendations/scores/
python -m stock_agent.swing_desk verify-log                  # hash-chain check of both logs
```

`check` exit codes: 0 PASS, 1 WARN, 2 BLOCK/error. Wrappers in `triggers/` only forward arguments.
Use `--data-root <path>` (or `SWING_DESK_DATA_ROOT`) when `data/` lives in another checkout.

## What `check` prints

Verdict, failed rule ids with reasons (G0-G11 gate, B1-B7 behaviour, COOL), earliest sell time
(afternoon of T+2), the user 3% stop and the validated -8% stop, suggested quantity (lot 100) sized so
the *effective* loss (3% + 1 pp gap allowance + 0.6% round-trip cost = 4.6%) is 1% of NAV, and VND at
risk at both 3% and the effective loss, for the suggested size and for the `--amount` you typed.

Behavioural rules (on by default): revenge buying after a realized loss in the previous session (B3)
and averaging down (B4) are BLOCK; any behavioural BLOCK holds until the next session's ATO (COOL);
two consecutive losses pause one extra session (B3P). A WARN asks for a written `THESIS` line in
`portfolio.txt` before `--ack` records the acknowledgment. The desk records; it cannot stop a trade.

G9 (risk budget) needs `portfolio.txt`; G10 (event calendar) needs `events` to be recorded for the
ticker and session. Without them those rules are UNVERIFIED, which counts as WARN, never as PASS.

## Data

- Prices: newest **verified** VCI snapshot in `data/paper/snapshots/` or `swing_desk/snapshots/`
  (G0: re-verified hashes, `as_of` = completed session, 30 VN30 + VNINDEX). Before 16:00 VN the
  completed session is the previous one ("EOD data as of ...").
- Foreign/proprietary flows: `data/foreign_flows_v2` through `foreign_flows.health/load_flows`
  (read-only); annotations only, `INSUFFICIENT_HISTORY` below 20 sessions, never zero-filled.
- `data/raw/prices_hist` is not used (status reports its staleness); legacy `data/raw/foreign` is
  quarantined and never read.
- `refresh` fetches only through `paper_runner --refresh --check --output-dir swing_desk/snapshots`
  (no paper record) and collects flows only with `--with-flows`, outside 16:30-18:30 VN, through the
  existing collector (fails closed on its lock). No schedule or BAT change.

## Long history (10-year store)

The verified daily snapshot stays the freshness authority (G0 / BF-G0) and the source of every
short-window field. Long-window fields come from the newest verified 10-year store under
`data/history/` (`stock_agent.data.history_store`, `docs/HISTORY-STORE.md`) when it matches:

| Mode | Snapshot (650 days, ~440 sessions) | History store (10 years), when consistent |
|---|---|---|
| swing | G0, RSI2/14, EMA, SMA5-50, ATR, ADX, BB, returns, liquidity, gate, sizing | SMA200, high_250, locked MAE p50/p80/p90, breach rate, day-2 return, MFE/MAE 5 |
| bottom-fish | BF-G0, floors/locks, 60-session drawdown, beta, ATR, ret_10d, liquidity | high_250, drawdown_250, episode start/sessions/depth, panic_low, new episode low, stab_state |

Selection, per ticker: store `as_of` = snapshot `as_of`, `verify` passes, ticker status ok/partial
(quarantined rows stay out and are listed in `history_data_issues`), non-aborted runs first, then the
newest. The store must equal the snapshot over the snapshot's sessions (exact VND prices and volume);
otherwise the snapshot is used and the ticker is flagged `history_mismatch` (never stitched). A failed,
missing or mismatched ticker falls back to the snapshot with a WARN note.

Which source a verdict used: `check` prints "Lịch sử dài ...: history_store <id> ..." or
"snapshot_650d ..." plus any WARN; the JSON `history` block, state `tickers/<SYM>.json`, the state
`manifest.json` (`history`), `checks.jsonl` and frozen rows carry `history_source`,
`history_store_id`, `history_first_date`, `history_rows` and `history_store_manifest_sha256`.

```text
python -m stock_agent.swing_desk status                      # includes the history_store line
python -m stock_agent.swing_desk refresh                     # as before + history status and per-ticker coverage
python -m stock_agent.swing_desk refresh --history           # + fetch VN30+VNINDEX+held (portfolio.txt) if stale, then verify
python -m stock_agent.swing_desk --history off check TCB     # snapshot only (pre-history behaviour)
```

`refresh --history` runs `python -m stock_agent.data.history_store fetch --years 10 --universe vn30
vnindex --symbols <held> --as-of <session> --out data/history` (paced >= 3.1 s per VCI call, stops
on HTTP 429) and then `verify <new store>`. It skips the fetch when verified stores for the completed
session already cover every symbol, and when a store directory without a manifest younger than an
hour suggests a fetch is running. No schedule change. Protocol note:
`configs/swing_desk/protocol_amendments.md`.

## Privacy and files

Tracked: this README, `portfolio.example.txt`, `triggers/`, `prompts/`. Git-ignored: `portfolio.txt`,
other `*.txt`, `state/`, `snapshots/`, `checks/`, `recommendations/`, `sentiment/`, `runs/`, `events/`,
`logs/`. `review` hides absolute NAV/VND unless `--show-nav`; no state file except
`portfolio_review.md` is derived from journal lines. Do not paste the journal or NAV into chat unless
the user asks.

## Prospective log

`recommendations/recommendations.jsonl` and `checks/checks.jsonl` are append-only and hash-chained
(`prev_sha256`, `row_sha256`, plus a `.head` sidecar). Rows are frozen before 09:00 VN of the target
session and never edited or deleted. Arms: `desk` (VeriHarness), `single` (one analyst, measures what
VeriHarness adds), `baseline` (rule-only), `control` (random structural PASS). Activity floor: 4 desk
ideas per month, reported, never forced. No conclusion before 90 resolved rows per arm and 40 sessions.

## Bottom-fish mode (bắt đáy)

Advisory filter with **no prospective evidence of edge**; firm-specific fundamental falls (type b) have
literature evidence of continued drift. Rules: `docs/audits/2026-10-06-bottom-fishing-research.md`;
thresholds: `configs/swing_desk/bottomfish_v1.json` (own `bf_params_hash`, `desk_v1.json` untouched);
protocol: `configs/swing_desk/bottomfish_prospective_v1.json`; checklist: `prompts/BOTTOMFISH_CHECKLIST.md`.

```text
python -m stock_agent.swing_desk check HPG --mode bottomfish --profile P1 --price 25000 --amount 500000 [--why cheap] [--tranche 2]
python -m stock_agent.swing_desk scan --mode bottomfish            # VN30 price-only table (no news/events/margin)
python -m stock_agent.swing_desk bf-news HPG --session 2026-10-07 --class none_found --searched <HOSE/IR URL> --searched <URL> \
    --margin "hose_ineligible=no|broker_cut=none|checked=2026-10-07|<URL>" [--item "date|FACT_PRIMARY|-2|type|<URL>|quote"]
python -m stock_agent.swing_desk events PNJ --session 2026-10-07 --event 2026-10-21:egm_capital --pending-dilution --source <URL>
python -m stock_agent.swing_desk events PNJ --session 2026-10-22 --resolved 2026-10-21:egm_capital --source <URL>
python -m stock_agent.swing_desk refresh --held                    # VCI snapshot for held non-VN30 symbols (asks first)
python -m stock_agent.swing_desk freeze --source bf_baseline       # bf_baseline + bf_knife + bf_random, before 09:00 VN
```

- Profiles: **P0 (default) = no bottom-fishing**; P1 short bounce after a panic (types a, c with half
  size); P2 up to 3 tranches written in a `plan=scalein` THESIS before the first lot (type a); P3 special
  situation (type b) only after the event is resolved, always labelled "no evidence of edge; measurement only".
- Size: 1% NAV at the effective loss = stop + one extra limit-down (7%) + 0.6% costs (P1 19.6%, P2 32.6%,
  P3 assumed 50% + 0.6%); type c x0.5 (P1), high index volatility x0.5. Uses the journal NAV when present.
- Fail-closed: missing snapshot, news class (`bf-news`), event calendar (`events`), margin status or
  journal is a BLOCK in this mode, never UNVERIFIED. BF-B2 anchoring and BF-B3 "cheap" are WARN.
- Held symbols outside VN30 (e.g. from `refresh --held` into `snapshots/extra/`) are check/review only:
  BF-G1 blocks any new bottom-fish entry, so the desk only reports HOLD/SELL/AVOID facts, never BUY.
- A bottom-fish PASS needs a THESIS line with `bf_type=`, `profile=`, `invalid=`, `time_stop=`,
  `max_attempts=` and `checklist=done` (BF-G12); `--ack` never replaces it.
