# Locked follow-up: historical VN30 membership sensitivity

Specified before examining this follow-up's returns, after the fixed-current-basket
36-run experiment. This is a correction/sensitivity study on already researched
periods, NOT a new untouched holdout or evidence of live profitability.

## Design

Same four variants, three scenarios and three blocks as the original
[hypothesis protocol](2026-09-26-momentum-hypotheses-protocol.md). No threshold,
parameter, cost, ranking, risk, settlement, or exit changes. Two eligibility
policies, hence **72 replays**, each independently starting with 1 billion VND:

- `fixed_current`: the current 30 members, kept fixed throughout each block.
- `historical`: the manually reconstructed, source-linked timeline in
  `configs/research/vn30_membership_2025_2026.json`.

Both use the SAME new snapshot of the union of all historical/current members
plus VNINDEX, from January 2024 through September 25, 2026. Historical rows are
clipped to each decision. The snapshot's generic fixed-universe metadata is not
altered; this runner separately binds the membership timeline and its hash.
The fixed-policy rerun controls for the changed snapshot/history length; changes
relative to the earlier report are not attributed to membership without this
same-snapshot control.

## Eligibility and knowledge timing

At each existing monthly decision close, use the membership known by that close
and effective on the next trading session. Daily known-on dates mean available
at CLOSE, not that day's open. Do not anticipate unannounced changes. Require
coverage through execution, all union price files, and 30 distinct members after
every transition. Fail closed outside documented coverage or on invalid changes.

Keep monthly rebalancing, including the existing initial-period rebalance. No
extra rebalance on index-change days; this is a momentum selection strategy,
not VN30 ETF tracking. A removed holding is sold at the next monthly rebalance
(subject to ordinary execution constraints), not at an invented off-cycle exit.
However, each retry purchase must still be eligible for its actual execution
session using information available by the previous close. This gate also applies
under the one-session delay stress. Newly added members wait for the next monthly
selection and must meet the unchanged price-history requirement.

In particular, the August 1, 2025 monthly decision does not anticipate the
August 4 composition: BVH is eligible on August 1, DGC is not. BVH cannot be
incrementally bought on August 4 retries. BSR's May 13, 2026 addition is not used
at the May 4 monthly execution; it can first enter monthly targets in June.
No future returns are used to select these policies.

## Evidence and interpretation

Retain all original metrics, paired variant/control bootstrap diagnostics and
independent cash/share/NAV reconciliation. Record monthly membership decisions
and every blocked entry. Verify every historical-policy BUY against membership
available at its actual preceding close. Report same-snapshot membership return
deltas, not a new strategy-selection contest. Original decision rules remain
diagnostic; these additional 72 runs increase the research trial history and
do not justify reusing the earlier three-comparison interval as a global search
correction. No strategy promotion based on these sensitivity runs.

Reconstructing membership only addresses one source of look-ahead/survivorship
bias. Prices were retrieved now and have unverified provider adjustments and no
dividend cash ledger. Daily opening-price fills remain proxies. There is still
no independent unseen forward portfolio record or actual execution evidence.
`live_promotion` must remain `blocked` regardless of these returns.
