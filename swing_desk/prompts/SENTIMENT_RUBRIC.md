# Sentiment rubric and citation rules

Source: research doc §9 (`docs/audits/2026-10-06-swing-desk-signal-research.md`). Evidence that LLM
news sentiment predicts VN large-cap returns is weak; sentiment is a veto, never an upgrade.

## News categories and preferred sources

| Category | Examples | Preferred sources |
|---|---|---|
| MACRO | policy rate, FX, credit, FDI, Fed | SBV, Tạp chí Thị trường Tài chính Tiền tệ, VnExpress/VnEconomy/Reuters |
| POLICY | laws, decrees, circulars, tax, trade remedies | original text (vbpl, ministries), Ministry of Finance press |
| SECTOR | bank credit/NIM, steel and oil prices, consumption | reputable press + broker reports (opinion) |
| COMPANY_MAJOR | earnings, issuance/dividend, M&A, management, litigation, VN30 review | HOSE/HNX/VSDC/SSC disclosures, company IR site; CafeF, Vietstock, Tin Nhanh Chứng Khoán, VnEconomy, Báo Đầu Tư (tiers and tips: `NEWS_SOURCES.md`) |
| COMPANY_MINOR | small contracts, insider trades, operations | disclosures, Vietstock/CafeF |
| MARKET_FLOW | foreign/proprietary, margin, ETF, FTSE/MSCI | local flows v2 + press |
| FORUM | F247, Zalo/Facebook groups | player opinion only, never evidence; use to detect hype (B6) |

## Score each item

- `evidence_type` and weight: FACT_PRIMARY 1.0 (original document, disclosure, statistics);
  FACT_REPORTED 0.7 (reputable press reporting an event); OPINION 0.3 (brokers, experts, commentary);
  RUMOR 0 (forums, social media, unconfirmed; record only to flag hype).
- `tone`: optimistic / objective / pessimistic, separate from impact (pessimistic tone tends to revert).
- `impact` in {-2, -1, 0, +1, +2}: effect on this ticker over 2-10 sessions, scored after restating the
  event in neutral words. Optional: score once with and once without the company name and record the gap.
- `novelty`: new or repeated; repeated items weigh 0.
- Decay `w = 0.5^(age_sessions / h)`: h = 1 for COMPANY_MINOR and market commentary, 5 for COMPANY_MAJOR,
  10-20 for POLICY/MACRO (design parameters, not estimates).
- Ticker score `clip(sum(w x weight x impact), -3, +3)`: <= -1.5 strong negative; -1.5..-0.5 negative;
  within ±0.5 neutral; symmetric for positive.

## Using the score

- A FACT with impact <= -1 and weight >= 0.7 -> veto: the idea becomes AVOID or at most WARN.
- Only pessimistic tone or negative OPINION without a FACT -> note only (may be the S1 dip condition).
- Large positive score with ret_1d >= 4% -> hype flag (B6).
- A positive score never raises WARN/BLOCK to PASS.

## Citation line format (`sentiment.md`)

```text
S-01 | URL | publisher | published (VN time) | accessed (VN time) | opened: yes|no
     | "verbatim quote, at most 2 sentences"
     | category | evidence_type | tone | impact | novelty | age_sessions | w
```

Forbidden: inventing news; using model memory instead of a source; citing without a publication time.
Unopened sources are UNVERIFIED with weight 0. Find at least one independent or primary source for any
load-bearing claim (analysts often read the same headline).
