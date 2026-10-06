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
python -m stock_agent.swing_desk status                      # read-only freshness report
python -m stock_agent.swing_desk refresh [--with-flows]      # reuse/fetch verified snapshot, build state/
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
