# R5 weekly momentum lane — result: INCONCLUSIVE, negative control failed (04/10/2026)

Protocol `configs/research/ml_protocol_v2_weekly.json` (commit `a7099a4`, before results; v1 monthly
result stays BLOCKED). Same user floors: ≥120 train dates, ≥40 test dates. Independence rule: decisions
every 5th VNINDEX session, entry next open, exit close at entry+5, so lot windows do not overlap.
Data: pinned snapshot (provider-adjusted as retrieved, not archival PIT), PIT VN30, 2022-09-01..2026-09-29,
development data. Bundle: `docs/audits/2026-10-04-r5-weekly-bundle/` (sealed, validator `engineering_complete`,
decision `INCONCLUSIVE`, `live_eligible=false`).

Floors met: 202 matured decision dates (6060 lot labels, 30 pending); 2 forward folds, train 120 and 160 dates,
40 test dates each (80 out-of-sample dates). Ridge on fold-fitted features, top 10 vs PIT equal-weight universe.
Per-lot diagnostics only: full weekly turnover cost on every lot, no NAV, capacity or delay.

| Strategy (80 OOS dates) | Top10 net %/wk | Universe % | Excess %/wk | Rank IC | 95% CI (block 4) | p (≤0) | Excess at cost ×2 |
|---|---|---|---|---|---|---|---|
| M0 formation 126-21 | −0.20 | −0.30 | +0.10 | −0.019 | [−0.47, +0.65] | 0.34 | −0.50 |
| M1 ridge, all features | −0.03 | −0.30 | +0.27 | −0.010 | [−0.13, +0.68] | 0.09 | −0.33 |
| M1 drop reversal | −0.05 | −0.30 | +0.25 | −0.020 | [−0.17, +0.68] | 0.11 | −0.35 |
| M1 drop momentum | −0.16 | −0.30 | +0.15 | −0.014 | [−0.31, +0.60] | 0.25 | −0.45 |
| M1 drop risk | −0.02 | −0.30 | +0.29 | −0.003 | [−0.09, +0.67] | 0.07 | −0.31 |
| M1 drop volume | −0.05 | −0.30 | +0.25 | −0.017 | [−0.16, +0.68] | 0.11 | −0.35 |
| M1 drop liquidity | −0.04 | −0.30 | +0.27 | −0.012 | [−0.12, +0.66] | 0.09 | −0.33 |
| **Control: train labels shuffled within date** | **+0.15** | −0.30 | **+0.45** | −0.003 | [+0.07, +0.86] | **0.01** | −0.15 |

(Block 8 gives the same picture; full numbers in `metrics.json`. Bonferroni threshold over the two primary
comparisons is 0.025.)

What this shows
1. No ranking skill: rank IC is ≈0 or slightly negative for M0 and every M1 variant. The model does not rank next-week returns.
2. The negative control "wins" more than the real model (+0.45% excess, p≈0.01) with no information in the target.
   So top-10-vs-equal-weight excess in this window is produced by which kind of stock the score favours (style/size/volatility
   exposure, and a universe mean that is itself negative), not by prediction. A test that passes on shuffled labels cannot support an edge claim.
3. Neither primary comparison clears the corrected threshold (M0 p≈0.34, M1 p≈0.07–0.09). The positive means are not evidence.
4. Ablations are descriptive: drop-momentum lowers excess by 0.12 points, the other drops move it by <0.03; all inside noise.
5. At double cost every excess is negative or near zero; weekly lots pay ~0.6% round trip, versus a universe move of ~0.3%/week.

Verdict: INCONCLUSIVE at best, and the lane's own control argues against reading it as positive. Nothing here supports R6
(ensembles). One control seed was run (registered); more seeds would only quantify how large style-exposure excess can be.

Not run: NAV replay, delay stress, VN100 and cash-restart slices (R7); F3 foreign (no verified vintages); F1/F4 (no PIT sources).
Next decisions for you: (a) stop ML on this universe and spend effort on the data (≥10y PIT, foreign v2 vintages) or on execution/cost
(cost is the binding constraint for weekly turnover), or (b) a registered v3 with exposure-matched controls (size/vol-matched random top-10) before any further model.
