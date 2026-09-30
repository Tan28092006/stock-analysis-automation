# Preregistered daily-event momentum v1

Registered before examining any new historical outcome on 2026-09-30. Research
only; no production strategy, ML, live scan, scheduler, broker order or UI change.
The user wants more new positions **and** add-ons, effective after fees. A minimum
monthly new-position count has not been agreed: activity cannot receive a pass.

## Hypothesis and source

The previous monthly/weekly target-weight engine largely adds to existing names.
Test a distinct event-driven engine, without claiming increased frequency is alpha.
Source: Curtis Faith, [Original Turtle Trading Rules](https://c.mql5.com/3/131/Curtis_Faith_-_Original_Turtle_Rules.pdf),
2003, chapters on volatility, entries, stops and exits. The source describes
20/55-session channels, smoothed 20-session true range and favorable-price unit
additions. It traded futures and used intraday signals. This is a long-only,
close-confirmed Vietnamese equity adaptation, not a replication of either Turtle
system. In particular daily20 retains the 20-session exit to isolate entry lookback.

## Locked mechanics

- Close strictly above preceding 55 highs (20 only in daily20); exclude signal bar
  from channel. N is first 20 true ranges' mean, then `(19*N_previous+TR)/20`.
  First bar TR is high minus low. No 254-bar momentum warmup; features must be finite.
- A buy intent uses completed prior close, prior-close NAV and PIT membership.
  Normal next-session open; delay stress one extra trading session. Quantity,
  signal N, stop and priority remain frozen. Entry attempt expires after one try.
  First observed session of an ISO week is the only weekly55 buy opportunity;
  all variants evaluate exits daily. No same-stock duplicate pending buy.
- Initial stop is signal close minus 2N. Reject execution below breakout level,
  at/below stop, or above signal close +1N. Slipped buy price is used for these
  tests. Quantity may only shrink at execution. These are conditional opening
  price assumptions, not proof of auction fills or protection against gaps.
- Initial unit nominal risk 0.5% NAV, at most ten names, 20% per-name and 100%
  gross, portfolio nominal stop risk at most 6%. All variants share caps. Caps
  constrain new buys, not forced rebalancing when prices drift. Conservative
  execution risk charges existing quantity times max(open, prior close)-stop.
- pyramiding55 alone adds when signal close >= last actual fill +0.5 frozen N;
  execution must also exceed that trigger. Maximum four filled orders per campaign,
  one buy per name/day, each add no bigger than first actual filled unit. Common
  stop becomes max(old stop, actual new fill minus 2 frozen N). Exit latches cancel
  pending buys and prevent averaging down. New entries precede adds, then descending
  breakout-distance/N (add-distance/N), then symbol; no ex-post return ranking.
- Sell when completed close falls below preceding 20 lows, at/below campaign
  stop, or membership removal is known and effective for execution. Exit intent
  latches until flat, even if price recovers. Sell only next open (extra session in
  stress), respecting T+3, limit-price proxy and liquidity. This is NOT an intraday
  guaranteed stop. Campaign unit count never resets on a partial sale.
- Both buy and sell capped at 1% preceding 20-session mean volume, rounded down to
  100 shares. Delayed buy cap is min(signal-known cap, latest prior-close cap).
  No execution-day volume, high, low or close for fills. Zero/unavailable ADV
  blocks orders. volume55 additionally requires signal volume >=1.5 times the
  preceding 20 volumes (excluding signal bar). market55 requires prior-close
  VNINDEX > SMA200 for buys, not forced liquidation.
- Broker is shared with existing research: 1bn VND initial cash, 100-share lots,
  buy commission 0.15%, sell commission+tax 0.25%, slippage 0.10% each side,
  conservative T+3 shares/proceeds, no leverage or advance. Double-cost stress
  doubles commissions, tax and slippage. Sale proceeds cannot fund immediate buys.
- All calendars, fees, state transitions and ledgers independently reconciled.
  Monthly new positions, add-ons, entry days, first-session construction, months
  without new entries, longest quiet streak, candidate/rejection funnel retained.

## Evaluation and limitations

Exact variants, controls, scenarios and family accounting are in
`configs/research/momentum_events_v1.json`. Timeslices are the existing causal
VNINDEX classification (prior close vs SMA200 and prior 20-session SMA slope),
never hindsight market labels. Every registered comparison stays in the report.
Pooled performance cannot replace H1 or up/down/transition diagnostics. All
history is already observed development data. Short episodes are descriptive,
not independent statistical discoveries. Statistical success needs both 20/40
block tests and >=252 sessions, but never implies live eligibility.

Remaining limitations: provider-adjusted vintage/corporate actions, manually
reconstructed membership, no historical auction queues/official reference-price
archive, and no prospective verified fills. A profitable historical result is
not real-money profit. Current pipeline stays unchanged pending evidence.

### Specific opening-auction execution gap (reviewed before new outcomes)

The conditional opening-price gate and quantity shrink are NOT an executable ATO
order specification: they inspect the completed opening price and still simulate
a fill at that opening price plus fixed slippage. A pre-submitted limit buy can
cap its maximum price, but does not implement a simultaneous minimum-price reject
or post-open quantity adjustment. Waiting until the open is known instead requires
a subsequent executable quote, which daily OHLC does not provide. This can create
optimistic selection/execution, despite the absence of future high/low/close use.

SSI's [conditional-order guidance](https://www.ssi.com.vn/khach-hang-ca-nhan/giao-dich-chung-khoan-ib-web)
says market-price triggers use continuous-session prices, not periodic-auction
prices. Its [KRX FAQ](https://www.ssi.com.vn/khach-hang-ca-nhan/krx-thi-truong-co-so-faq)
also documents changed ATO/ATC priority; do not assume generic older broker pages
describe current queue mechanics. Read on 2026-09-30; no account/API access used.

Therefore this v1 is a signal/economic research screen only. An executable-order
model (precommitted quantities/order types or timestamped post-open quotes) and
prospective fill reconciliation remain mandatory before activation. Delaying by
one whole session does not by itself cure this conditional-price assumption.
Do not change the locked v1 rules in response to outcomes; register any execution
ablation separately and retain v1 as a rejected/unverified assumption if needed.

Read-only backend inspection confirms the current price lane is daily:
`VnStockProvider.history` requests `interval="1D"`, and immutable VCI snapshot
requests use `timeFrame="ONE_DAY"`. `normalize_ohlcv` converts timestamps to dates
and retains one row per day; `read_eod_csv` also removes the time component.
These functions cannot preserve an execution-time intraday sequence. Do not feed
minute bars through them or change their interval to create an apparent fill audit.
A separate timestamp-preserving execution-data contract would be required (bar
open/close time, exchange timezone, observed-at/source vintage, price basis and
official reference/limit prices). No fetch, collector change or live scan was
performed in this inspection.

## Implementation evidence

- RED `6473418`: 19 executed missing-engine cases, before implementation.
- GREEN `7ade6c7`: all 19 pass. Synthetic rally increment was corrected to remain
  within the registered delayed-open price cap; small-volume pyramid fixture
  isolates four units without confusing the existing 20% name cap.
- RED `8710cf7`: future-listing empty-prefix regression plus seven missing suite
  contracts. An additional cancellation fixture initially missed the pending-order
  day; that test-setup failure was not counted as a strategy bug.
- GREEN `49ab891`: 30 contracts pass, including partial exits, caps and latches.
- RED `07b69b6`, GREEN `947083c`: full synthetic run caught the real paired-parent
  schema mismatch; 31 contracts pass, 54 synthetic paths reconciled and immutable
  inputs checked. Existing output directories cannot be overwritten.
- RED `7d5f8d8`, GREEN `45c00b2`: remove unregistered H1 within-panel inference.
  Exactly 51 continuous plus 18 paired-universe comparisons remain. H1 retains
  descriptive outcomes and histories but cannot meet the 252-session requirement.
- Locked inventory: 65 long-history windows (10 parents + 9 half-years + 46 causal
  episodes) x 18 = 1,170 new paths. Each H1 universe has seven windows (H1 + five
  causal episodes + March-April) x 18 = 126. Total **1,422 new paths**, in addition
  to **1,196 unchanged parent paths**. Carry attribution is not counted as an
  independent replay.

Local verification on source `45c00b2`: **576 passed**, 23 existing dependency/test
warnings, 175.54 seconds. Focused coverage run: **31 passed**, 26.40 seconds;
`momentum_events.py` 91%, `momentum_event_suite.py` 82%, combined **87%** (362
statements). `coverage report --fail-under=80` exits 0. Source pushed to
`codex/data-integrity-audit` for remote regression; no merge or deployment.

The source was frozen before the historical runs. The opening-auction limitation
above was recorded before inspecting their economic outcomes. Completed CI
evidence below does not imply activity, economic or live-trading acceptance.

Execution checkpoint:

- Local output root: `data/paper/research_gate/20260930/events_45c00b2_regression`.
  The parent, books, timeslices, paired-universe and event commands completed
  serially with exit 0; all 2,618 paths completed, existing snapshots read-only.
- [CI 36694782314](https://github.com/Tan28092006/stock-analysis-automation/actions/runs/36694782314)
  **succeeded** on pushed source `45c00b28aabc5818aba10eb52b8d0d499d443bf1`.
  Contracts and all **2,618 historical paths** passed. The historical job ran
  09:14:57-09:50:37 UTC on 2026-09-30. Its GitHub test-merge SHA is
  `10e16694f3fe2a1c6e7e524d926a4b160f63ba4b`; this is not a branch merge/deployment.
- All 1,196 completed local parent paths compared recursively with previously
  audited CI `36689514737`. Maximum absolute economic-field differences:
  baseline 290 and books 180: `1.1102230246251565e-15`; timeslices 530 plus carry
  attribution: `5.551115123125783e-17`; paired-universe 196:
  `1.1102230246251565e-16`. Counts, partitions and manifest identities match.
  No baseline economic change observed beyond floating-point roundoff.
  A second comparison of CI `36694782314` directly against CI `36689514737`
  is exactly equal in every compared economic field for all four parent suites
  (maximum absolute difference zero); 87 distinct recorded Git blobs verified.
- CI artifact `research-output-36694782314`, ID `11088493448`, downloaded under
  `data/paper/research_gate/20260930/github_36694782314`. GitHub reports archive
  SHA-256 `e4bd968c49a4bd079c5ba42447eef0346eb26e9d7536db91e03dc2382b58ac42`;
  the extracted download does not independently verify the original ZIP digest.
  Locally verified event-result SHA-256:
  `cd0791c5055b69eddbe738c4954ef3442acdb9818ecf21f2e7fc49a62556bdef`.
- All 89 recorded CI source blobs match exact Git blob bytes at `45c00b2`.
  Raw Windows working-tree bytes differ for 24 files (including line endings),
  so Git blob comparison is required for the CI artifact, not relaxed hashes.
  Both immutable snapshots pass their verification and recorded manifest hashes.
- `data/paper/research_gate/20260930/audit_event_evidence.py` is a read-only
  secondary audit of the completed CI artifact: all 1,422 paths and 23,131 fills,
  PIT buys, signal/fill timing, lagged participation caps, unit counts, monthly
  activity, 69 comparisons and regime-P&L partition pass. These overlapping
  replay fill counts are not counts of independent investments or real trades.
- The same secondary audit passes on the completed local event artifact:
  89 raw working-tree source hashes, 23,131 fills, 69 registered comparisons and
  all three panels. Local result SHA-256:
  `b873929f278d482ff980024e02ace612dceeaed5360c6369f60438f1868f8f1f`.
  Full recursive comparison against CI (all panels, histories, ledgers, activity,
  carry, controls and inference) has maximum absolute numeric difference
  `8.881784197001252e-16`; protocol, paired tests and blocker lists are identical.
  Metadata such as timestamps, GitHub test-merge SHA and raw OS source-byte hashes
  are intentionally not economic comparison fields. The 12 H1 outcome-table rows
  below were also checked programmatically against the artifact's rounded values.

Verification-loop review (continuation, no source edits):

| Check | Evidence/status |
|---|---|
| Python build/syntax | `py_compile` succeeds for both new modules and both test modules |
| Types | Not run: no project type-check configuration or installed `pyright` command found |
| Lint | No installed `ruff` or project Ruff config; not claimed as a lint pass. `git diff --check bf21ba2..HEAD` passes |
| Tests | 576 local tests pass; focused new-module coverage 87% |
| Security scope | Filename-only scan of changed Python/config files found no known private-key or common API-key patterns; not a comprehensive security certification |
| Diff | Eight changed tracked files; no production strategy, scheduler or UI changes |
| Research readiness | All 2,618 paths complete locally and in CI; cross-platform event outputs match within roundoff. Live readiness remains blocked independently of engineering checks |

## Observed results (development data, not strategy promotion)

All returns below are modeled end-of-period NAV after the registered costs, not
bank-account profits. Open positions are marked to the final close rather than
forcibly liquidated. Initial capital is 1bn VND. VNINDEX is a gross price-return
benchmark, not an investable net-cost account. H1 has 119 sessions; its causal
labels contain 115 uptrend sessions, four transition sessions and no downtrend
sessions. It therefore cannot validate behavior in a prolonged bear market.

### H1/2026, paired VN100 universe, cash restart

VNINDEX: **+4.0780%**. New positions and add-ons are counted separately.

| Variant | Normal NAV return | Excess vs index (pp) | Max drawdown | New / add-ons | Double-cost return | One-session-delay return |
|---|---:|---:|---:|---:|---:|---:|
| daily55 | +3.9858% | -0.0923 | -13.9055% | 33 / 0 | +2.8089% | +1.1357% |
| weekly55 | -6.6880% | -10.7661 | -11.1679% | 23 / 0 | -7.3502% | -4.6175% |
| daily20 | +7.0210% | +2.9429 | -13.5163% | 28 / 0 | +5.9923% | +2.6458% |
| pyramiding55 | -4.8058% | -8.8838 | -24.8170% | 41 / 38 | -7.0657% | -1.9253% |
| market55 | +3.8864% | -0.1917 | -14.0328% | 33 / 0 | +2.7026% | +1.1983% |
| volume55 | +0.2573% | -3.8208 | -14.2475% | 32 / 0 | -0.7442% | -1.3693% |

| Variant | Jan new | Feb new | Mar new | Apr new | May new | Jun new | First-session new | Entry days Jan-Jun | Longest no-new-entry streak (sessions) |
|---|---:|---:|---:|---:|---:|---:|---:|---|---:|
| daily55 | 9 | 0 | 2 | 9 | 6 | 7 | 3 | 3, 0, 2, 8, 4, 6 | 47 |
| weekly55 | 10 | 1 | 1 | 1 | 8 | 2 | 3 | 3, 1, 1, 1, 3, 1 | 29 |
| daily20 | 10 | 0 | 5 | 8 | 2 | 3 | 4 | 3, 0, 4, 6, 2, 3 | 47 |
| pyramiding55 | 10 | 0 | 4 | 9 | 12 | 6 | 3 | 4, 0, 3, 8, 9, 5 | 37 |
| market55 | 9 | 0 | 2 | 9 | 6 | 7 | 3 | 3, 0, 1, 8, 4, 6 | 50 |
| volume55 | 10 | 0 | 2 | 9 | 5 | 6 | 2 | 4, 0, 2, 7, 4, 6 | 45 |

Pyramiding monthly add-ons: **4, 0, 7, 10, 12, 5**. Thus 79 buys does not mean
79 new opportunities. daily20 includes four initial entries and 24 later entries;
February still has none. There is no agreed numerical activity floor, so the
activity verdict remains null rather than passed.

daily20's normal simulated gain is **70,209,718 VND**: approximately **9,139,818
VND** closed-lot realized P&L and **61,069,900 VND** remaining unrealized P&L. Fees
are 5,707,826 VND and modeled slippage 3,005,408 VND. Average exposure is 53.9682%.
MSB contributes 41,034,821 VND, larger than the entire 29.4m VND excess over the
index. This is concentration evidence, not a valid leave-MSB-out counterfactual:
removing a stock would change subsequent allocations and needs a new experiment.

Its 436 new-entry candidates reconcile to **28 filled, 222 portfolio-risk rejects,
145 position-limit rejects and 41 opening-gap rejects**. Scarcity is not simply
the absence of signals. Existing risk measures appreciation to the static campaign
stop, although a separate 20-session-low exit also exists. Whether effective-exit
risk accounting improves this is an untested follow-up hypothesis, not permission
to loosen risk/slot caps after looking at returns.

Pyramiding loses 48,057,515 VND despite more fills. Its fees plus modeled slippage
are about 23.16m VND, versus 9.98m for daily55; the approximately 87.92m VND net
performance gap cannot be explained by extra modeled costs alone. None of these
three normal H1 paths has a fill exceeding the entire execution session's volume,
but that does not prove auction liquidity. Maximum actual-day participation is
1.1583% for daily20, 1.0065% for daily55 and 2.7949% for pyramiding55: a 1% **lagged
ADV** cap does not promise a 1% share of unknown execution-day volume.

### Matched H1/2026 VN30 control

| Variant | Normal NAV return | Max drawdown | New / add-ons | Double-cost return | One-session-delay return |
|---|---:|---:|---:|---:|---:|
| daily55 | +4.3765% | -9.2019% | 18 / 0 | +3.6428% | -4.4371% |
| weekly55 | -3.9117% | -5.7987% | 9 / 0 | -4.1518% | -2.5608% |
| daily20 | -0.6352% | -12.4211% | 29 / 0 | -2.8550% | -5.5037% |
| pyramiding55 | +1.2288% | -20.3591% | 20 / 22 | -0.9864% | -6.6305% |
| market55 | +4.3765% | -9.2019% | 18 / 0 | +3.6428% | -4.4371% |
| volume55 | +3.6073% | -9.9322% | 14 / 0 | +2.9360% | -5.2293% |

VN100 expansion is not a uniform improvement: daily20 improves in this slice,
whereas several other variants deteriorate. There is no H1 variant beating the
index under every registered scenario. All 18 H1 paired inferential comparisons
are below the 252-session minimum; H1 descriptive profits cannot override this.

### VN30 fixed-calendar/crisis slices, normal costs, cash restart

Returns in percent. H2/2022 starts in September; H2/2026 ends September 29.
Overlapping crisis windows are diagnostics, not additional independent samples.

| Slice | VNINDEX | daily55 | weekly55 | daily20 | pyramiding55 | market55 | volume55 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 2022 H2 partial / panic | -21.3371 | -3.9843 | -2.5696 | +0.9023 | -5.6218 | 0.0000 | -1.5205 |
| 2023 H1 | +10.7565 | +1.2243 | -2.3913 | +7.6472 | -2.6572 | +1.6165 | +0.4582 |
| 2023 H2 | +0.3891 | -3.1849 | -2.2300 | -3.9181 | -9.0368 | -3.9493 | -0.4261 |
| 2024 H1 | +9.5856 | +21.3442 | +4.9705 | +24.9628 | +24.5601 | +21.3442 | +18.2377 |
| 2024 H2 | +1.6408 | -3.3319 | -6.5944 | -9.5042 | -5.6161 | -4.3513 | -4.6404 |
| 2025 H1 | +8.4408 | +3.1875 | +2.6816 | +1.9791 | -0.7850 | +9.3166 | +3.2199 |
| 2025 H2 | +29.4854 | +26.6457 | +12.9574 | +18.0761 | +24.0652 | +26.6457 | +24.3248 |
| 2026 H1 | +4.0780 | +4.3765 | -3.9117 | -0.6352 | +1.2288 | +4.3765 | +3.6073 |
| 2026 H2 partial | -4.4211 | -3.6082 | -0.4340 | -5.1894 | -5.6464 | -2.7223 | -1.9560 |
| Apr-Sep 2026 | +3.7230 | -0.5689 | -1.8642 | -4.8505 | -8.2400 | +0.6681 | -2.5851 |
| Apr-Jun 2025 tariff window | +4.8011 | +4.7906 | +1.4447 | +7.0208 | +3.3111 | +2.7242 | +4.7367 |
| Mar-Apr 2026 crisis window | -1.1679 | -1.3339 | 0.0000 | -5.7348 | -2.3354 | -1.3339 | -1.8911 |

The complete artifact also retains all 46 causal VN30 episodes, both H1 panels'
five causal episodes, every stress and inherited-portfolio carry attribution.

### Causal VNINDEX regimes, not hindsight return labels

For readability, this table shows **every VN30 episode with at least 20 sessions**;
the other 36 shorter episodes remain in the full result and are not dropped from
testing. This is a display rule only, not a new sample-selection/acceptance rule.
Returns are cash-restart percentages, normal costs. A lagged uptrend label can
precede a subsequent loss: do not rename such episodes after seeing their returns.

| Period | Causal regime | Sessions | VNINDEX | daily55 | weekly55 | daily20 | pyramiding55 | market55 | volume55 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 2022-09-05 to 2023-06-01 | downtrend | 185 | -15.7679 | -2.5929 | -5.1076 | +2.2229 | -9.2031 | 0.0000 | -5.3208 |
| 2023-06-02 to 2023-07-28 | transition | 41 | +11.3891 | +8.2297 | +6.8419 | +11.2112 | +7.1268 | +8.2297 | +8.4381 |
| 2023-07-31 to 2023-10-18 | uptrend | 56 | -9.5322 | -8.3115 | -6.2993 | -8.4040 | -9.9466 | -8.3115 | -5.8248 |
| 2023-10-19 to 2023-11-15 | transition | 20 | +1.7744 | -0.0964 | -0.0964 | +0.0285 | -0.5049 | 0.0000 | +0.3111 |
| 2023-12-29 to 2024-04-19 | uptrend | 74 | +3.8083 | +16.2886 | +5.3606 | +12.9031 | +16.9852 | +16.2886 | +13.2313 |
| 2024-04-23 to 2024-08-05 | uptrend | 72 | -0.2091 | +0.7725 | -3.2740 | +2.0724 | +1.8568 | +0.7725 | +0.3846 |
| 2024-08-07 to 2024-11-04 | uptrend | 62 | +2.2803 | -6.2894 | -3.9088 | -5.1307 | -5.0721 | -6.2894 | -5.9268 |
| 2025-02-13 to 2025-04-03 | uptrend | 36 | -2.9521 | +0.6177 | -2.5710 | +2.3530 | +1.0682 | +0.6177 | -0.6879 |
| 2025-05-23 to 2026-03-20 | uptrend | 207 | +25.4108 | +13.1355 | +8.6158 | +3.3202 | +16.1239 | +13.1355 | +18.1323 |
| 2026-03-30 to 2026-07-20 | uptrend | 78 | +5.9189 | -0.7235 | -1.2620 | -5.1847 | -4.4192 | -0.7235 | -1.0495 |

All five H1 VN100 episodes are shown below. In contrast to the cash-restart tables,
these are **daily20 inherited-portfolio contributions** from the continuous H1
run. Their P&L sums to 70,209,718 VND, and their returns compound to +7.0210%.
They must not be summed as percentage returns or treated as independent trials.

| Period | Causal regime | Sessions | daily20 carry return | Carry index return | P&L contribution (VND) |
|---|---|---:|---:|---:|---:|
| 2026-01-05 to 2026-03-20 | uptrend | 50 | +6.0321% | -7.7957% | +60,321,459 |
| 2026-03-23 to 2026-03-25 | transition | 3 | +0.3767% | +0.6299% | +3,993,755 |
| 2026-03-26 | uptrend | 1 | -0.7482% | -0.8178% | -7,963,315 |
| 2026-03-27 | transition | 1 | +0.6502% | +1.7128% | +6,868,628 |
| 2026-03-30 to 2026-06-30 | uptrend | 64 | +0.6574% | +11.1914% | +6,989,191 |

The apparent H1 winner therefore lagged the index sharply in the final 64-session
uptrend portion. No downtrend period exists in this H1 classification, even though
part of the sample's realized index return is negative.

Continuous VN30 daily55 returns +41.0549% versus index +38.8570%, but **all 51
continuous inferential comparisons are `inconclusive_or_no_edge`** under the
preregistered block tests and family accounting. The 69 new comparisons use the
declared family of 184, not a correction for all unknown research ever attempted.

Decision: retain these variants as development evidence, **promote none**. More
entries are available, but profitable add-ons, robust excess return, adequate
monthly activity and executable fills have not jointly passed. Resolve the order
execution contract before a new risk-accounting ablation; preserve all v1 failures.

## Goal acceptance audit (not a completion claim)

| User requirement | Authoritative evidence still needed/current state |
|---|---|
| Local backend, no UI | Implemented research CLI and synthetic end-to-end path; no new UI or live scan |
| Adequate new entries and profitable add-ons after costs | More new entries demonstrated; pyramiding loses in H1 VN100, zero-entry February persists, numerical activity floor unagreed |
| Beat VNINDEX by appropriate timeslice | Some descriptive winners but no robust statistical pass; H1 daily20 edge disappears under delay stress, recent VN30 window underperforms |
| Clean data/no leakage | Synthetic chronology tests pass; legacy ML remains disabled, current adjusted-price vintage and historical corporate-action/reference-price lineage not certified |
| CI workflow | Contracts and all 2,618 historical paths pass on pushed source; downloaded artifact audited |
| Real execution | Conditional opening-price simulation is not an executable order contract; timestamped execution evidence and live/replay parity missing |
| Demonstrated real-money profit | No broker fills or actual funded account ledger evaluated here; historical simulated NAV cannot satisfy this requirement |

Do not mark the overall goal complete because code/tests pass or because one
historical variant wins. No amount of rerunning the same past prices creates
prospective fills or a guarantee of future profit.
