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

## Implementation evidence

Pending TDD and full regression. No new historical results inspected yet.
