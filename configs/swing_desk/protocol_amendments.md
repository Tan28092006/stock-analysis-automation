# Swing desk protocol amendments (sidecar)

The registered protocol files `prospective_v1.json` and `bottomfish_prospective_v1.json` are not
edited, so their pinned hashes (golden test) stay unchanged. Amendments are recorded here, dated.

## 2026-10-07 - long-window history source (T16)

- Applies to both protocols (`swing_desk_prospective_v1`, `bottomfish_prospective_v1`).
- From this date, rows written by the desk carry `history_mode`, `history_source`
  (`history_store` or `snapshot_650d`), `history_store_id`, `history_store_manifest_sha256`,
  `history_first_date`, `history_rows` and `history_mismatch`: frozen recommendation rows (all arms),
  `checks.jsonl` rows, desk state ticker JSON and the state manifest (`history` block).
- Long-window fields (swing: SMA200, high_250, locked MAE/MFE and breach statistics; bottom-fish:
  high_250, drawdown_250, fall-episode statistics and stab_state) are computed on the verified
  10-year history store when its `as_of` equals the snapshot's and its series equals the snapshot over
  the snapshot window; otherwise on the snapshot (WARN; `history_mismatch` when they differ).
- The daily verified snapshot remains the freshness authority (G0 / BF-G0). Short-window fields,
  thresholds, `desk_v1.json` / `bottomfish_v1.json` and their params hashes are unchanged.
- Where the snapshot already covers a window (VN30 names with ~440 sessions: SMA200, high_250 and the
  250-entry excursion statistics) the values are bit-identical to `--history off`; they differ only
  for shorter snapshot series and for fall episodes that started before the snapshot window.
- `--history off` restores the previous behaviour; the swing golden test runs in that mode.
- Rows before this date have no `history_source`: treat them as `snapshot_650d`.
- This is an engineering data-source change, not evidence of edge, and approves no trading.

## 2026-10-08 — primary-source recognition for `bf-news`

- `none_found` still needs >= 2 opened sources including one primary source. "Primary" now means an
  exchange/regulator host (hsx.vn, hose.vn, hnx.vn, ssc.gov.vn, vsd.vn, vsdc.com.vn) or a company domain
  listed for that ticker in `configs/swing_desk/company_ir_domains.json`, matched exactly or as a parent of a
  subdomain. The old "ticker appears in the host" test is kept only as a host-label prefix fallback.
- Fixes two defects: companies whose domain does not contain the ticker (vinhomes.vn, vincom.com.vn,
  seabank.com.vn, techcombank.com, ...) could never satisfy the rule, and look-alike hosts (evilhsx.vn,
  hsx.vn.attacker.com) were accepted by the old suffix/substring test.
- No threshold, profile or params hash changed; bf-news records written before this date are unaffected.

## 2026-10-08 — market-scope news is not a firm-specific shock

- `fact_negative_items` (fall classification and BF-G6) now skips items whose `type` is market, sector or
  group scope or the ticker's own price action (`MARKET_SCOPE_TYPES` in `bottomfish.py`: market,
  market_group, market_macro, market_flows, group, group_contagion, group_price, sector, foreign_flow,
  technical, macro, policy_macro), unless the type is also in `fact_negative_types`.
- Reason: the research taxonomy treats market/group contagion as evidence for type (a); before this change
  a cited "Vingroup dragged the index" item classified VRE's fall as type (b) and blocked P1.
- Firm-specific types (provision, dilution, legal, fraud, insider, earnings, margin_cut, ...) behave as before.
  No threshold or params hash changed.

## 2026-10-08 - desk universe VN30 -> current VN100 (T19)

- User request 2026-10-08 ("mở rộng ra cho VN100 ... VN30 ít quá"). Applies to both protocols
  (`swing_desk_prospective_v1`, `bottomfish_prospective_v1`).
- The active universe is set in `configs/swing_desk/universe.json` (default `vn100`; `--universe vn30` keeps the
  previous desk). VN100 = `configs/universe_vn100.json`: HOSE CBTT 15/07/2026, HOSE-Index review July 2026,
  effective 2026-08-03, VN100 table pages 4-5, source sha256
  `441799b3afbc4f29f8b53ce6ef3b500b9121a4d032ca4cfeb1d4aa6e14407879`.
- From this date, `checks.jsonl` rows, frozen recommendation rows (all arms, swing and bottom-fish) and the state
  manifest carry `universe` (`vn30` / `vn100`) and `universe_source_sha` (LF-normalized sha256 of the universe
  symbols file); checks rows also carry `ticker_data_source` and `ticker_snapshot_manifest_sha256`. Rows before
  this date have no `universe`: treat them as `vn30`.
- Where the registered protocols say "VN30" for the candidate pool (baseline, control, bf_baseline, bf_knife,
  bf_random, analyst BUY scope) and `bottomfish_v1.json` `scope.new_entries = "vn30"`, read "the active universe
  recorded in the row's `universe`". Evaluate and report results per `universe`; do not pool VN30-era and
  VN100-era rows without saying so.
- Unchanged definitions: market regime and breadth (G2, BF classification), limit counts and S2 (VN30 MR scan)
  stay on the VN30 paper basket; S2 is NA for names outside VN30. Foreign-flow health stays VN30.
- Daily data for the 70 names outside the paper snapshot comes from desk-owned verified snapshots
  (`swing_desk/snapshots/universe/`, same VCI builder, 650 days). G0 is per ticker; a name without fresh verified
  data is G0 UNVERIFIED (WARN) and cannot enter a code arm or an analyst BUY; BF-G0 stays BLOCK.
- G9 adds `configs/swing_desk/sectors_vn100.json` with VN100: at most 2 positions per HOSE level-1 sector, and the
  bank group adds NAB and OCB to `desk_v1` `sectors.bank`. Bottom-fish `sector_mean_ret_10d` uses the same
  widened groups (more peer banks with daily data in VN100 mode).
- `desk_v1.json`, `bottomfish_v1.json`, `prospective_v1.json`, `bottomfish_prospective_v1.json` and their hashes are
  unchanged; G7 liquidity is unchanged (many VN100 names fail it). The swing golden runs with
  `--universe vn30 --history off` and is unchanged.
- The S1 backtest evidence is VN30-based and survivorship-affected; VN100 names outside VN30 have no prior
  evidence of their own. This is a scope change, not evidence of edge, and approves no trading.

