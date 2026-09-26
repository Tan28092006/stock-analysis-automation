# Locked experiment protocol: momentum hypotheses, 2026-09-26

Status: specified before running the new variants. This is a conditional research
replay, NOT a preregistered external study or an untouched out-of-sample test.
User objective: outperform VNINDEX net of modeled costs; greater volatility is
acceptable. No leverage, live deployment, retraining, or production config edits.

## Controls and trial registry

Four strategies, three execution scenarios, three non-overlapping evaluation
blocks: **36 replays**, with three primary hypothesis comparisons on H1 2026.
No parameter search and no combined variant in this experiment. Do not remove
losing stocks, change thresholds, or rerun changed parameters under the same ID.

- Control: existing 12-1 momentum, top 10, held-name top-20 buffer, inverse 60-day
  stock volatility, 20% annualized VNINDEX volatility target (20 daily returns),
  exposure capped at 100%, monthly prior-close orders.
- A / `market_adjusted`: change ranking only. For each baseline-eligible stock,
  align VNINDEX closes to its dates, calculate the last 252 daily log returns,
  beta = demeaned covariance(stock,index) / variance(index). Score is the sum of
  stock minus beta*index log returns over the first 231 of those 252 observations,
  divided by their sample standard deviation. Do NOT subtract an estimated alpha.
  Beta uses only information available at the decision close. Stock weights,
  buffer, exposure rule and execution remain the control's. Zero residual
  variance scores zero; zero market variance implies beta zero.
- B / `reversal_entry`: leave monthly targets unchanged; only allow incremental
  purchases when the stock's preceding 5-session log return minus VNINDEX's is
  <= 0. Check each execution morning using data through the previous close only.
  Keep the control's four-session execution/retry window; no later catch-up.
  Existing holdings and sells are not screened. This tests a natural-rebalance
  timing overlay, not the rare RSI/Bollinger mean-reversion sleeve.
- C / `own_portfolio_vol`: same ranking, buffer and inverse-vol stock weights.
  Estimate the volatility of the unscaled target portfolio from the last 126
  common daily simple returns (including correlations), annualized by sqrt(252).
  Monthly exposure = min(1, 0.20 / estimated volatility). Fail closed if common
  history is insufficient. This is capped volatility targeting, not leverage or
  the original papers' inverse-variance/long-short implementations.

Evaluation blocks (separate cash starts, 1 billion VND each; not concatenated):

1. H2 2025, July 1–December 31: historical robustness diagnostic using the verified
   H1 snapshot fetched September 26, 2026.
2. **Primary: H1 2026, January 1–June 30**, same snapshot and execution as the
   already examined baseline. Development/replay, never relabel as untouched OOS.
3. H2 2026 partial, July 1–September 25: recent robustness diagnostic using the
   verified September 25 snapshot. This period has already informed other work
   and is not an untouched holdout either.

Scenarios: normal; all modeled commission, sell tax and slippage amounts doubled
(cost sensitivity, not a claim that legal tax rates change); and one extra
session execution delay with the ORIGINAL monthly decision/quantities frozen.
Retry window shifts with first execution. Each variant is compared with the
control run under the SAME scenario. No change to conservative T+3 cash/share
availability, board lots, price-limit proxy, nonnegative cash or ending MTM.

## Measurements and decision rule

Report total net return, gross price-only VNINDEX return, net excess percentage
points, maximum drawdown, relative-wealth drawdown versus the index, daily beta,
tracking error, information ratio, average invested exposure, two-way traded
notional / average NAV, fees, modeled slippage, fills, and month-by-month returns.
Reconcile every daily NAV independently from fills and closing holdings; check
cash, lots, settlement, order causality, source hashes and baseline parity.

For each variant/control pair, bootstrap paired daily return DIFFERENCES using
circular moving blocks of 10 observations, 4,000 draws, seed 20260926. Report
annualized mean daily difference and its marginal 95% interval, plus a 98.3333%
interval (Bonferroni across the THREE primary variants). These are approximate
dependent-sample diagnostics, not a proof of alpha or an adjustment for unknown
earlier strategy searches. They are NOT confidence intervals on compounded
six-month excess return. Monthly/supplementary/stress results are exploratory.

An adaptation is a promising conditional candidate only if: primary net return
exceeds BOTH control and VNINDEX; primary improvement over control stays positive
in both stresses; it improves on control in at least two of the three normal
blocks; and the primary adjusted bootstrap lower bound is above zero. Otherwise
record which conditions fail. In all cases live promotion remains BLOCKED by
point-in-time membership/adjustment lineage and absence of an untouched forward
test. Failure here rejects a particular implementation in these data, not a
universal research finding. Limited sample size may make results inconclusive.

## Paper-to-test mapping and non-replications

| Research | Testable implication here | What this experiment does NOT verify |
|---|---|---|
| [Blitz, Huij & Martens, Residual Momentum](https://repub.eur.nl/pub/22252/ResidualMomentum-2011.pdf) | A: removing common-market movement from momentum ranking improves net active performance | Original monthly FF3, 36-month regression and long-short construction. We lack PIT size/value factors and sufficient matching history. A is market-adjusted, not an exact residual-momentum replication. |
| [Vo & Truong, Vietnam momentum](https://www.sciencedirect.com/science/article/pii/S2214635017300965) | Existing momentum control and active-return benchmark test | Their 6-month formation/9-month holding construction is not implemented here; this paper alone does not validate our 12-1 strategy. |
| [Nguyen et al., short-term technical trading in Vietnam](https://yoksis.bilkent.edu.tr/pdf/files/14964.pdf) | Counter-hypothesis to B: short-term continuation can make a pullback screen miss winners | We do not claim a separate replication of all their 1–5-day rules or historical market arrangements. |
| [Da, Liu & Schaumburg, short-term reversal](https://academicweb.nd.edu/~zda/Reversal.pdf) | B is only a price-relative timing proxy | Fundamental-news/analyst-revision decomposition is NOT TESTABLE with current data. A price decline is not proof of a liquidity shock. |
| [Dai et al., Reversals and Returns to Liquidity Provision](https://rpc.cfainstitute.org/research/financial-analysts-journal/2024/reversals-and-the-returns-to-liquidity-provision), [author implementation discussion](https://www.dimensional.com/se-en/insights/q-and-a-on-short-run-reversals-with-mamdouh-medhat-and-robert-novy-marx) | B as a screen on natural rebalancing; measure forgone exposure and net costs | No PIT shares outstanding/true turnover or order book, so volatility/turnover mechanism and executable liquidity provision remain unverified. |
| [Moreira & Muir, Volatility-Managed Portfolios](https://onlinelibrary.wiley.com/doi/10.1111/jofi.12513), Barroso & Santa-Clara, Momentum Has Its Moments | C: own-portfolio risk estimate rather than market proxy improves active return | Not leveraged long-short WML, inverse-variance allocation, or a claim of eliminating crashes. |
| [Barroso & Detzel, limits to arbitrage](https://www.sciencedirect.com/science/article/pii/S0304405X21000775), [de Groot et al., costs and reversals](https://repub.eur.nl/pub/25718/AnotherLook_2011.pdf) | Cost stress, turnover and timing stress may overturn apparent benefits | Doubling linear modeled costs is not a calibrated nonlinear impact/queue model. |
| [Bailey & Lopez de Prado, Deflated Sharpe Ratio](https://www.davidhbailey.com/dhbpapers/deflated-sharpe.pdf) | Fixed trial registry and multiple-comparison caution | Do not publish a global DSR without the unknown historical trial count. Bootstrap intervals are not DSR. |

## Data limitations that remain binding

Current fixed universe (effective August 2026), not historical constituent
membership. Raw/canonical hash integrity is not proof of point-in-time knowledge.
Provider-adjusted prices retrieved now have unverified corporate-action lineage;
no separate dividend cash ledger. Snapshot vintages/history lengths differ
between blocks but never between variants within a block. No re-use of the
September-trained ML candidate in historical decisions. Daily opens are a fill
proxy, not order-book evidence. No total-return investable index comparator.
Previous H1 result and all generated experiment artifacts must remain immutable.
