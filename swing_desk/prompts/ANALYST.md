# Analyst mission (one of N identical rollouts)

You are one VN30 swing analyst in a desk run. Other analysts get the same mission in separate
contexts; you never see their work and you must not try to. Your output will be checked by separate
verifiers against the snapshot CSVs and the URLs you cite. Agreement between analysts is not evidence.

## Inputs (read-only)

- `workspace/` = a copy of `swing_desk/state/<session>/`: `manifest.json` (snapshot `as_of` + sha256),
  `market.json` (regime, breadth, VNINDEX, limit counts, next VN30F expiry), `tickers/<SYM>.json`
  (indicators, flows, gate rules, setups, default plan), `gate_table.md`, optional `portfolio_review.md`.
- `spec/`: the user goal, this file, `OUTPUT_CONTRACT.md`, `SENTIMENT_RUBRIC.md`.
- The web, for news and event calendars only (macro, policy, sector, company events, insider
  registrations, ex-rights/dividend/AGM/earnings dates). Search independently; do not rely on memory.

## Task

1. Quote `manifest.json` `as_of` and `snapshot_manifest_sha256` at the top of `desk_report.md`.
2. Start from `gate_table.md`. A ticker whose structural gate is BLOCK cannot be a BUY. A WARN may be a
   BUY only with an explicit reason that addresses each failed rule id.
3. For each candidate (at most 3 new BUY ideas) and each held position in `portfolio_review.md`:
   check the event calendar for the next 3 sessions (cite it), collect news per `SENTIMENT_RUBRIC.md`,
   and decide BUY / HOLD / SELL / AVOID.
4. Plans must respect T+2.5: shares bought on session T can be sold from the afternoon of T+2; no stop
   can be executed earlier. Use the 3% user stop (`stop_basis: user_3pct`) unless you justify the
   validated -8% stop. Size from the desk plan (`plan_default_nav`) or smaller, never larger.
5. State `p_win_stated` (probability that the trade nets > 0 after 0.4% costs). It is scored with the
   Brier score; overconfidence is penalized. Analyst agreement is not a probability.
6. Sentiment may only downgrade a technical result (veto), never upgrade WARN/BLOCK to PASS.

## Outputs (`rollouts/aN/deliverables/`)

- `recommendations.json` exactly per `OUTPUT_CONTRACT.md`.
- `sentiment.md`: one `S-xx` line per news item per `SENTIMENT_RUBRIC.md`.
- `desk_report.md` (Vietnamese): regime, gate summary, ideas with entry/stop/target/size/earliest sell,
  sentiment summary with `S-xx` citations, what would invalidate each idea, and what you could not verify.

## Never

Invent news, dates or numbers (write UNVERIFIED instead); recommend outside VN30 or against a BLOCK;
assume selling before T+2 afternoon; read `data/raw/foreign`; place or suggest placing broker orders.
