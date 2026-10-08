---
name: swing-desk
description: "VN30 swing desk for this repository (advisory only). Use for: (1) a pre-trade check when the user names a VN30 ticker they want to buy now (PASS/WARN/BLOCK, failed rule ids, earliest sell T+2 afternoon, 3% and -8% stops, lot-100 size, VND at risk); (2) the daily desk run that produces at most 3 swing ideas through a deterministic data step, 4 analyst rollouts with cited news sentiment and VeriHarness verification, then freezes them in the prospective log; (3) a review of the user's journal swing_desk/portfolio.txt by gate, emotion tag and activity; (4) a bottom-fish check after sharp falls. Never places or authorizes broker orders and never changes data schedules."
---

# Swing desk (VN30, advisory)

Authority: root `AGENTS.md`. Rules, rationale and sources:
`docs/audits/2026-10-06-swing-desk-signal-research.md`; thresholds: `configs/swing_desk/desk_v1.json`;
evaluation protocol: `configs/swing_desk/prospective_v1.json`; folder guide: `swing_desk/README.md`.

This skill never places orders, never edits `run_eod_update.bat` or Task Scheduler, never re-enables
`eod_update`, never reads `data/raw/foreign` (quarantined legacy rows) and never uses
`data/raw/prices_hist` for live decisions. The desk's main job is pre-trade discipline; picking
tickers is secondary. Whether LLM analysis improves win rate is an **unproven hypothesis** that the
prospective log measures; say so whenever ideas are reported.

All commands run from the repository root: `python -m stock_agent.swing_desk <command>`. Add
`--data-root <checkout>` when `data/` lives in another checkout (for example a git worktree).

## Mode 1: pre-trade check (default when the user names a ticker to buy now)

1. `status --no-flows`. If `snapshot.ok` is false (stale or unverified), tell the user and ask before
   running `refresh`; never refresh silently. Before 16:00 VN the data is the previous session's EOD.
2. `check <TICKER> --price <price the user sees> --amount <VND> [--reason <reason>] [--emotion <tag>]`.
   Omit `--price` or `--amount` if the user gave none; ask for the reason and emotion tag if missing.
3. Report the output verbatim or nearly so: verdict, each failed/UNVERIFIED rule id with its reason,
   earliest sell time (afternoon of T+2; the 3% stop cannot be executed earlier), the 3% stop and the
   validated -8% stop, suggested quantity, and VND at risk at **both** the 3% stop and the effective
   loss (3% + gap allowance + costs = 4.6%), for the suggested size and for the user's amount.
4. WARN: ask for one written `THESIS` line in `swing_desk/portfolio.txt` (setup, stop, target, what
   invalidates it), then rerun with `--ack`. BLOCK from a behavioural rule (B1, B3, B3P, B4, COOL):
   say a re-check is allowed only from the next session's ATO (`recheck_from`). Other BLOCKs: say why.
5. Do not soften a BLOCK, do not add LLM opinions to the verdict, do not turn WARN into PASS. G9
   needs the journal and G10 needs `events <TICKER> --session <S> --source <URL>` after you check the
   event calendar (cite it); without them they are UNVERIFIED (= WARN).

## Mode 2: portfolio review

`review` (add `--json` for structured output). Summarize open positions (actions such as stop
breach, broken thesis, time stop, S1 exit), portfolio risk, closed-trade statistics, results by gate
and by emotion tag (descriptive below 30 trades per group), behaviour flags B8/B9, and monthly activity
(new entries, add-on buys, distinct entry days, zero-entry months). Never paste raw journal lines or
NAV into chat unless the user asks (`--show-nav`).

## Mode 4: bottom-fish check (when the user wants to buy after a sharp fall, "bắt đáy", or names a ticker at/near the floor)

Rules, sources and profiles: `docs/audits/2026-10-06-bottom-fishing-research.md`; thresholds:
`configs/swing_desk/bottomfish_v1.json`; protocol: `configs/swing_desk/bottomfish_prospective_v1.json`.
Bottom-fishing has **no prospective evidence of edge**; firm-specific fundamental shocks (type b) have literature
evidence of continued drift. Say this every time, with the caveat the desk prints:
"Chỉ tư vấn; bắt đáy chưa có bằng chứng edge tiến cứu; loại (b) có bằng chứng trôi giảm."

User decisions (2026-10-06): the default recommendation is **P0 = no bottom-fishing**; P1 (short bounce after a
panic), P2 (up to 3 tranches, plan written before the first lot) and P3 (special situation, only after the event
is resolved) are enabled when the user chooses them; P3 always says "no evidence of edge; measurement only".
Risk per bottom-fish trade is 1% NAV at the effective loss (stop + one extra limit-down + costs). Held tickers
outside VN30 can only get HOLD/SELL/AVOID facts, never BUY. BF-B2 (anchoring) and BF-B3 ("cheap now") are WARN.
Missing data, news class or event calendar is BLOCK.

1. `status --no-flows`. If the ticker is outside VN30 and held, ask before `refresh --held` (it calls VCI for the
   held non-VN30 symbols plus VNINDEX into `swing_desk/snapshots/extra/`; no `data/` write, no schedule change).
2. Classify the fall with sources you opened (never from memory). Search the company IR / HOSE disclosure page
   and at least one reputable outlet for the last 20 sessions. Record what you found:
   `bf-news <T> --session <S> --class fact_negative|none_found|opinion_only --item "date|evidence_type|impact|type|<URL>|<quote>" ...`
   and the margin/warning status (`--margin "hose_ineligible=no|broker_cut=<date|none>|checked=<date>|<URL>"`).
   `none_found` needs at least two `--searched <URL>` including one HOSE/HNX/SSC/VSD page or a company page whose domain (or a subdomain) is listed for that ticker in `configs/swing_desk/company_ir_domains.json`; add a missing domain there after checking it on the HOSE profile. Check the event
   calendar (EGM agenda, private placement, rights, ex-dates, earnings) and record it with
   `events <T> --session <S> ... --source <URL>`; mark a pending dilution vote with `--pending-dilution` and,
   once the result and terms are published, `events <T> --session <S> --resolved <date>:<type> --source <URL>`.
   Unopened sources are UNVERIFIED and cannot support `none_found`.
3. `check <T> --mode bottomfish --profile <P> --price <price the user sees> --amount <VND> [--why <code>]
   [--reason bottomfish] [--emotion <tag>] [--tranche <k>]`. Without `--profile` the configured default P0 applies
   (always BLOCK: no bottom-fishing).
4. Report verbatim: verdict, fall type and its evidence, every failed BF-G/BF-B id, earliest sell (T+2 afternoon),
   invalidation level, catastrophe stop, time-stop date, suggested quantity, VND at risk at the effective loss
   (stop + one floor + costs) and the "3 more floors while locked" scenario. Then show the 10-question checklist
   from `swing_desk/prompts/BOTTOMFISH_CHECKLIST.md`. A bottom-fish entry is not PASS until a THESIS line with
   `bf_type=`, `profile=`, `invalid=`, `time_stop=`, `max_attempts=`, `checklist=done` exists (BF-G12); `--ack`
   never replaces it.
5. Never turn BLOCK into advice to buy, never suggest averaging down outside a pre-written `plan=scalein`, never
   recommend buying on a floor day, and never reclassify a fall from (b) to (a)/(c) without a cited source. For a
   held position, give the review facts (earliest sell, invalidation, time stop, upcoming events) and ask the user
   for their plan; do not tell them to sell or hold.
6. Daily desk run: bottom-fish ideas are **not** part of the 3 swing ideas. Phase 0 runs
   `freeze --source bf_baseline` with the swing baseline (bf_baseline, bf_knife, bf_random, before 09:00 VN and
   before any analyst row); analysts may add at most 1 bottom-fish idea, type (a) only, frozen with
   `freeze --source bf_desk --file <rec.json> --model-id <model>`. No activity quota applies to bottom-fishing.

Bottom-fish checklist (ask in this order, answers go in the THESIS line):
- [ ] Fall type (a)/(b)/(c) with opened sources recorded by `bf-news`
- [ ] Next event and date recorded by `events` (pending dilution marked; resolved only with a source)
- [ ] Margin/warning status checked within 20 sessions
- [ ] Profile chosen by the user (P0 default; P3 = no evidence of edge; measurement only)
- [ ] Invalidation close, catastrophe stop and time-stop date written
- [ ] VND loss if the stock locks at the floor 3 more sessions, and the user accepts it
- [ ] Attempt number in this episode; for P2 the tranche plan written before tranche 1
- [ ] THESIS line with `checklist=done`, then rerun `check`

## Mode 3: daily desk run (only when the user asks for swing ideas or a full desk review)

Decide after EOD (completed session = the session just closed, after 16:00 VN); everything must be
frozen before 09:00 VN of the next session. Let `S` = state session (`as_of`), `T` = trade session.

**Phase 0 — data steward (deterministic; main session):**
1. `status`; if the snapshot is not for the completed session, `refresh` (add `--with-flows` only if
   the user allows it; the desk skips flows during 16:30-18:30 VN by itself). Stop if G0 fails.
2. `scan`; `review` if `swing_desk/portfolio.txt` exists.
3. `freeze --source baseline` (freezes the rule-only baseline and random control for `T` **before any
   analyst runs**).
4. Create `swing_desk/runs/<S>/`: `MISSION.md` (N=4, user goal, output contract), `spec/` (copies of
   `swing_desk/prompts/ANALYST.md`, `OUTPUT_CONTRACT.md`, `SENTIMENT_RUBRIC.md`), `workspace/` (a
   read-only copy of `swing_desk/state/<S>/`), `rollouts/a1..a4/`.

**Phase 1 — 4 analyst rollouts:** same model, same prompt (`ANALYST.md` + contract + rubric), separate
contexts, no access to each other. Each reads `workspace/` and searches the web independently for
macro, policy, sector and company news and event calendars, then writes
`rollouts/aN/deliverables/{recommendations.json, sentiment.md, desk_report.md}`. Rules for analysts:
- BUY only if the structural gate is not BLOCK; a WARN needs a reason per failed rule id.
- Every news claim is an `S-xx` line with URL, publisher, publication time (VN), access time, a verbatim
  quote of at most 2 sentences and the rubric fields; unopened sources are UNVERIFIED with weight 0.
- Sentiment can only veto; a FACT with impact <= -1 and weight >= 0.7 blocks a BUY.
- `p_win_stated` is mandatory and will be Brier-scored.

**Single-analyst arm (measures what VeriHarness adds):** immediately after Phase 1 and before any
verification, freeze rollout `a1` (fixed in advance, never the "best" one):
`freeze --source single --file swing_desk/runs/<S>/rollouts/a1/deliverables/recommendations.json --model-id <model> --n-rollouts 1`.

**Phase 2 — VeriHarness investigations** (separate contexts; they never see each other). Use the
`veriharness` skill's prompts (user-level skill: `CHARTER.md`, `INVESTIGATE_ELIM.md`,
`INVESTIGATE_FALS.md`, `ADJUDICATE.md`, `REPAIR.md`, ledger formats); paste CHARTER plus the phase
prompt verbatim into each brief and restrict work to `swing_desk/runs/<S>/`.
- Disagreement resolver -> `ledger_elim.json`: settle disagreements on tickers, numbers and news with
  the snapshot CSVs and the cited URLs, never by vote.
- Consensus challenger -> `ledger_fals.json`. Mandatory: rerun `status`; recompute 3 random tickers'
  indicators from the snapshot CSV named in `workspace/manifest.json` and compare with the ticker JSON;
  open every URL behind a load-bearing sentiment claim and find an independent or primary source;
  confirm no BUY violates a BLOCK, G9 sizing or the T+2.5 sell constraint; validate the output contract.

**Phase 3 — adjudication** (fresh context; only the two ledgers + spec + workspace) -> `finish.json`
with `base`, `work` (each change with evidence) and `open` forks.

**Phase 4 — repair** -> `out/deliverables/` + `repair.json`. Untraceable values stay empty.

**Phase 5 — freeze and report (before 09:00 VN of `T`):**
`freeze --source desk --file swing_desk/runs/<S>/out/deliverables/recommendations.json --model-id <model> --n-rollouts 4 --adjudication-base <aN> --open-forks <k>`.
Then report to the user in Vietnamese: regime, gate-table summary, at most 3 ideas with entry, stop,
target, size, VND at risk and earliest sell, sentiment summary with citations, open forks, and the
standing caveat: "Chỉ tư vấn; edge chưa được chứng minh; đang đánh giá tiến cứu (n = ...)." Get `n`
from `score`. Report the activity floor (>= 4 desk ideas per month) as met or not; never add ideas to
meet it. Run `score` on later sessions to resolve matured rows.

Cheap path (if the user wants it): N=2 and only the consensus challenger; record `--n-rollouts 2`.

## Running the 4-analyst flow

Host-specific delegation; the phases, files and freeze order above are identical on every host.

- **Claude Code:** use the Agent tool. Launch the four Phase 1 analysts in one message as background
  subagents (`run_in_background`), each with the same brief (ANALYST.md + OUTPUT_CONTRACT.md +
  SENTIMENT_RUBRIC.md + its own `rollouts/aN/` path) and web search allowed. Wait for all four
  notifications, freeze `a1`, then launch the two Phase 2 investigators as two separate background
  subagents in one message. Phase 3 and Phase 4 are each a new subagent with no Phase 2 context.
- **Codex:** use native subagents if this Codex build has them, mapped one-to-one onto the same
  roles. Otherwise run each rollout and each verifier phase as a separate sequential pass that reads
  only its own inputs from disk, and **disclose in the report that contexts were not isolated** (later
  passes may be influenced by earlier ones), so the run's independence is weaker than designed.
- Any host: never show one analyst another's output; never let a verifier see the other verifier's
  ledger; never count analyst agreement as evidence or probability.

## Long history (10-year store) and which data a verdict used

- Freshness authority is unchanged: G0 / BF-G0 use the verified daily snapshot (650 calendar days,
  about 440 sessions). Every short-window field (RSI, EMA, ATR, ADX, SMA5-50, returns, flows, floors,
  beta) comes from the snapshot in both modes.
- Long-window fields (swing: SMA200, high_250, locked MAE / MFE and breach stats; bottom-fish:
  high_250 / drawdown_250 and the fall-episode stats incl. stab_state) come from the newest verified
  `data/history/<UTC>_<id>/` store whose `as_of` equals the snapshot's and where the ticker is ok/partial
  (per ticker; non-aborted runs preferred), only if the store equals the snapshot over the snapshot's
  window. Otherwise the snapshot is used with a WARN note; a store/snapshot difference is
  `history_mismatch` and is never stitched.
- Tell the user the source: the check output line "Lịch sử dài ..." and the JSON `history` block
  (`history_source` = `history_store` or `snapshot_650d`, `history_store_id`, `history_first_date`,
  `history_rows`, store manifest sha256). The same keys are in state ticker JSON, the state manifest
  (`history`), checks.jsonl and frozen recommendation rows. Report any WARN / `history_mismatch` verbatim.
- `status` shows the `history_store` line; `refresh` reports it with per-ticker coverage. Ask before
  `refresh --history`: it calls VCI (paced >= 3.1 s per symbol, about 2 minutes) for VN30 + VNINDEX + held
  tickers into a new `data/history/` store, then verifies it; it never runs when a fetch is in progress.
- `--history off` (before the command, e.g. `--history off check TCB`) restores the snapshot-only
  behaviour; use it only when the user asks or to compare.

## Honest limits (say them when relevant)

- Advisory filter, not validated alpha. Only S1 (RSI(2) < 5 and close > SMA50) has prior backtest
  support, measured with a -8% stop, development data, survivorship-affected; the 3% stop is inside
  daily noise (about 1.1 x ATR) and about 1 in 5 entries breach -3% before they can be sold.
- PASS means no red flag and a known setup, not a winning trade. The desk cannot stop a real order.
- Prices are provider-adjusted and may differ from the board on ex-rights days; the holiday table ends
  2026-12-31; foreign-flow history is short (INSUFFICIENT_HISTORY below 20 sessions).
- No claim that the desk or VeriHarness beats rule-only until the registered floor (>= 90 resolved rows
  per arm, >= 40 sessions, both regimes) and the bootstrap test are met. A green result still does not
  approve live trading.

## Never

- Recommend outside VN30, against a BLOCK, or with a stop that assumes selling before T+2 afternoon.
- Invent news, dates or numbers; fill gaps with UNVERIFIED.
- Edit or delete frozen rows, freeze after 09:00 VN, or cherry-pick which rollout is the single arm.
- Place or suggest placing broker orders, change schedules, or refresh data without telling the user.

## Codex notes

- This file is the Codex copy of `.claude/skills/swing-desk/SKILL.md`; keep the two identical apart
  from this section (`tests/test_swing_desk_skill.py` checks the frontmatter and body).
- The `veriharness` prompts are a user-level skill; in Codex look for it under `~/.codex/skills/veriharness/`
  (or the Claude path `~/.claude/skills/veriharness/`). If neither exists, run the phases from the
  descriptions above and say that the VeriHarness prompts were not available.
- Without native subagents, the flow is sequential passes (see "Running the 4-analyst flow"): start
  each pass fresh from files on disk, keep pass outputs in their own folders, and state in the report
  that analyst and verifier contexts were not isolated.
- Use PowerShell-compatible commands on this Windows workspace; the triggers in `swing_desk/triggers/`
  only forward arguments to `python -m stock_agent.swing_desk`.
- `refresh --held` makes VCI calls; on Codex ask the user before running it, as on Claude Code.
