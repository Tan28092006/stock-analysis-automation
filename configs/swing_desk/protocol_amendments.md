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
