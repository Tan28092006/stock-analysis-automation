---
name: swing-desk
description: "VN30 swing desk for this repository (advisory only). Use for: (1) a pre-trade check when the user names a VN30 ticker they want to buy now (PASS/WARN/BLOCK, failed rule ids, earliest sell T+2 afternoon, 3% and -8% stops, lot-100 size, VND at risk); (2) the daily desk run that produces at most 3 swing ideas through a deterministic data step, 4 analyst rollouts with cited news sentiment and VeriHarness verification, then freezes them in the prospective log; (3) a review of the user's journal swing_desk/portfolio.txt by gate, emotion tag and activity. Never places or authorizes broker orders and never changes data schedules."
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
