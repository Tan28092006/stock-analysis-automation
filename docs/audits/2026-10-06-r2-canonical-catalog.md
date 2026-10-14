# R2 canonical aggregation/catalog: đối chiếu thiết kế với audits hiện có

Ngày: 06/10/2026. Reviewer: Firstmate run `2026-10-06-ml-roadmap` / T4.
Phạm vi: research design read-only cho ledger row R2 trong
`docs/plans/2026-10-04-data-ml-research-plan.md` §10; đối chiếu flow raw→canonical,
idempotency, receipts, pagination, revisions, restore, rủi ro và đề xuất SQLite
catalog với các audit đã có. Không sửa runtime/config/data/schedule, không chạy
collector, scan, backtest hoặc test; không commit.

- Source revision: `4b859cd` (nhánh `codex/data-integrity-audit`), working tree chỉ có
  hai file user-owned chưa track (`data/pipeline/local_server.log`,
  `data/pipeline/swing_scan_cache.json`); không dùng chúng làm bằng chứng.
- R1 tham chiếu: nhánh `firstmate/r1-source-contracts` tại worktree `D:/Chungkhoan-r1`,
  HEAD `204e06b`, module `stock_agent.data.contracts`
  (`CONTRACT_VERSION = 'source-dataset-contracts-v1'`, `REGISTRY_SHA256` pin
  `0584d394…cba1d`). R1 **chưa merge** vào `4b859cd`; line numbers `contracts.py:N`
  dưới đây là của `204e06b`. Mọi line khác là của `4b859cd`.
- Code line numbers đọc trực tiếp; không phải kết quả chạy test. Thiết kế dưới đây là
  proposal, không phải capability đã có.

## Audits đã đọc

| File | Nội dung liên quan R2 |
|---|---|
| `docs/audits/2026-09-30-foreign-data-contract.md` | Contract v2 đóng băng trước implementation: grain, availability, storage immutable, retry-stale extension, regression 2.618 replay |
| `docs/audits/2026-09-30-foreign-data-rollout.md` | Tải thật, migration legacy, late-publication probe 30/09, 9.090 vintage rows |
| `docs/audits/2026-10-04-system-design-review.md` | Bảng câu hỏi Alex Xu (04/05/06/12/13), P1/P2 issues |
| `docs/audits/2026-10-04-ml-plan-review-data.md` | DE01–DE17 claim matrix, sáu clarification cho plan |
| `docs/audits/2026-09-22-market-readiness-plan.md` | Acceptance paper readiness: snapshot manifest, idempotent retry, quarantine |
| `docs/audits/2026-09-22-reconciliation/source-probes.json` (file duy nhất) | 76 dòng OHLCV legacy invalid, probe VCI/KBS, decision `QUARANTINE` |

## 1. Flow raw→canonical hiện tại (tóm tắt evidence)

- **Giá VCI**: `paper_runner --refresh` (`stock_agent/pipeline/paper_runner.py:217-223`)
  → `build_market_snapshot` (`stock_agent/data/reconciliation.py:92-139`): thư mục mới
  `mkdir(exist_ok=False)` (:103), `raw/<SYM>.json` + `prices/<SYM>.csv`, manifest có
  raw/CSV SHA256, request, `fetched_at`, unit (:129-133); `verify_snapshot` đọc lại raw
  và so CSV (:142-163). Cache legacy `data/raw/prices/*.csv` bị ghi đè, nuốt lỗi
  (`stock_agent/data/providers.py:19-25`).
- **Khối ngoại/tự doanh**: `foreign_refresh --retry-stale`
  (`stock_agent/pipeline/foreign_refresh.py:18-76`) → `collect_flows`
  (`stock_agent/data/foreign_flows.py:178-265`) → `runs/<UTC>_<id>/{raw/,records.jsonl,
  quarantine.json,manifest.json}`; `verified_records` re-normalize mọi run (:282-304);
  `load_flows` lọc `available_at <= as_of` rồi dedup giữ vintage cuối (:341-348).
- **SQLite/DuckDB/Parquet**: không có trong `stock_agent/`, `scripts/`, `tests/`.

## 2. Bảng đối chiếu claim của audits với findings R2

Verdict: **MATCHES** = code/finding xác nhận; **CONFLICTS** = code hoặc R1 mâu thuẫn
claim; **EXTENDS** = finding bổ sung điều kiện/khoảng thiếu chưa được ghi;
**NOT-COVERED** = audit không đề cập, R2 phải tự xử lý.

| # | Audit claim (file §) | Verdict | Evidence code / ghi chú |
|---|---|---|---|
| C1 | Storage là immutable run folders, raw bytes + SHA256 + receipt time, manifest ghi sau cùng (foreign-data-contract, đoạn "Storage") | MATCHES | `foreign_flows.py:193-194, 218-220, 262`; `reconciliation.py:103, 138` |
| C2 | Status pointer atomic + process lock (foreign-data-contract "Storage"); "Unique temp + fsync + replace cho paper/flow JSON" (system-design-review 04-DD1, L79) | EXTENDS | `atomic_json` đúng cho `quarantine.json`/`manifest.json`/`latest_status.json` (`foreign_flows.py:45-56, 255, 262, 264`), nhưng `records.jsonl` dùng `write_bytes` (:254), raw dùng `write_bytes` (:218). Được giảm nhẹ vì manifest atomic có `records_sha256` ghi sau. `research_only.jsonl` legacy dùng `write_text` (:409). Review chỉ nêu snapshot manifest `write_text` (`reconciliation.py:138`) |
| C3 | Same response idempotent ở loader key session/symbol/category/source (foreign-data-contract "Storage"; FOREIGN-FLOWS L97-99) | MATCHES, EXTENDS | `foreign_flows.py:347-348`. Không có idempotency tại ingest: cùng payload được lưu lại mỗi run. Tie-break là sort `available_at, observed_at, raw_sha256` và **không reject** khi `available_at` bằng nhau nhưng nội dung khác; R1 `select_as_of` reject trường hợp đó (`contracts.py:415-441`). Hiện vô hại vì `available_at` = completion của từng run, nhưng R2 phải dùng ngữ nghĩa R1 và có parity test |
| C4 | Revisions giữ lại; as-of lọc availability trước khi chọn revision (foreign-data-contract; rollout "Bổ sung ba mã": as-of 10:11Z vẫn thiếu 26 giá trị muộn) | MATCHES | `foreign_flows.py:342-348`; test `test_store_preserves_revisions_and_filters_asof_before_dedup`, `test_real_store_keeps_old_missing_value_and_new_availability` |
| C5 | 9.090 dòng vintage "không phải 9.090 quan sát độc lập" (rollout) | MATCHES | Cần tách `vintage` và business key trong catalog; xem §4 |
| C6 | Late probe: 1.014 dòng chung giá trị trùng, 26 dòng mới (rollout "Theo dõi phiên") | EXTENDS | Không có cơ chế tính revision diff; kết luận được làm thủ công. R2 `revision_event` tự động hóa so sánh (không đổi collector) |
| C7 | Per-symbol endpoint tránh truncation first-page của market table (foreign-data-contract "Collection") | MATCHES, EXTENDS | `ENDPOINTS` per symbol (`foreign_flows.py:25-28, 162-172`). Không có pagination; nguồn trả cửa sổ ~20 phiên (FOREIGN-FLOWS L28; rollout "Giới hạn"). Không có proof coverage ngoài `gaps`/`coverage` (:240-250). VCI một request `countBack` (`reconciliation.py:68-70`), chỉ kiểm `min_rows` + ngày cuối (:124). Plan §3.2 L100 "pagination có proof coverage" là **chưa có** |
| C8 | Receipt `started/succeeded/partial/failed`, watermark, expected sessions (plan §3.2 L98; ml-plan-review DE08) | CONFLICTS (với code hiện tại) | Manifest chỉ có `sources` thành công (`foreign_flows.py:219`) và `errors` không tham chiếu raw (:231). Chuỗi retry chỉ nằm ở stdout `attempt_manifest_paths` (`foreign_refresh.py:58, 106`). Snapshot giá: failure list (`reconciliation.py:135`), không attempt ID. R2 chỉ có thể dựng receipt **suy ra** từ manifest, không phải receipt đầy đủ |
| C9 | Tách `ingestion_attempt(attempt_id)` khỏi `raw_object(sha256)`, hash dedup không gộp receipts (ml-plan-review DE03, correction 1) | MATCHES (design) | Thiết kế §4 dùng bảng nối nhiều-nhiều; R1 `vintage_key=('raw_sha256',)` (`contracts.py:96, 119`) **không đủ** làm unique key quan sát; xem C15 |
| C10 | Publication protocol: một visibility boundary duy nhất, SQLite không rollback file writes (plan §3.2 L103-113; ml-plan-review DE10) | EXTENDS | Bước R2 đầu tiên **không** đổi boundary: manifest filesystem vẫn là authority, catalog là projection dẫn xuất, rebuild được. Như vậy không có hai current pointer cạnh tranh. Chuyển catalog thành publication boundary là bước sau, cần sửa collector và crash tests |
| C11 | Crash trước commit để lại orphan không visible (plan §3.2) | MATCHES | `health` báo `incomplete_runs` khi thiếu manifest (`foreign_flows.py:327`); `verified_records` chỉ glob `*/manifest.json` (:284). Snapshot giá: thư mục thiếu/hỏng manifest → `verify_snapshot` raise (`reconciliation.py:145`). R2 phải ghi các trường hợp này vào `ingest_issue`, không crash toàn rebuild |
| C12 | Replay có, nhưng chưa retention/backup/restore drill/RPO-RTO (system-design-review 04-TĐ2 L77, P2 L116-122; ml-plan-review DE12) | MATCHES | Không có code backup/restore. Runtime data trong `.gitignore`. R2 thêm backup→restore→rebuild parity test cho catalog; không giải quyết backup raw |
| C13 | `verified_records` đọc toàn lịch sử mỗi lần; cần projection có checkpoint + parity (system-design-review 05-TĐ2 L86, P2 L124-127) | MATCHES | `foreign_flows.py:282-304` gọi từ `health` (:313) và `load_flows` (:342). Catalog incremental theo `manifest_sha256`. Không tuyên bố cải thiện tốc độ khi chưa đo |
| C14 | Legacy không provenance không vào loader mặc định; 17.400 research-only, 7.421 cách ly (rollout "Migration"); legacy không thể certify (foreign-data-contract "Limitations") | MATCHES | `migrate_legacy` đặt `available_at=None`, `research_only` (`foreign_flows.py:400-405`); R1 `LEGACY_PATH_PREFIXES`, `is_legacy_path`, `_check_flow` quarantine (`contracts.py:50-52, 230-235, 310-322`). Catalog chỉ inventory vào `ingest_issue`, không projection |
| C15 | Vintage giữ riêng mỗi lần nhận (plan §3.1 bảng `foreign_flow_vintage`; FOREIGN-FLOWS L97-99) | EXTENDS R1 | R1 `vintage_key=('raw_sha256',)`. Cùng payload refetch ở hai run có cùng `raw_sha256` nhưng `observed_at/available_at` khác. Unique key catalog phải thêm `generation_id`, nếu không sẽ mất quan sát đầu (vi phạm DE03). `select_as_of` vẫn đúng vì chọn theo `available_at`. Đề xuất ghi chú này cho R1 v2, không tự sửa pin |
| C16 | Giá: adjustment provider chưa xác minh, universe current fixed (reconciliation.py manifest; ml-plan-review spot check; readiness plan "Scope limits") | MATCHES | `reconciliation.py:109-111`; R1 `claims_blocked=('pit_adjustment_history',)` và rule `available_at = max(16:00 VN, fetched_at, snapshot_completed_at)`, BLOCKED nếu decision nằm giữa close và retrieval (`contracts.py:103-114, 364-379`) |
| C17 | Dòng OHLCV legacy invalid giữ quarantine, không splice provider khác (reconciliation `source-probes.json` `decision`; readiness plan "Known historical bad rows") | MATCHES | 76 invalid rows; `reconcile_invalid_rows` chỉ trả staged copy (`reconciliation.py:166-193`); `data/raw/prices*` là legacy path ở R1 (`contracts.py:50`). Catalog không đọc chúng vào `vci_ohlcv_daily_vintage` |
| C18 | 29/09 scoring lỗi raw/canonical OHLC mismatch ở snapshot 25/09, 28/09 (rollout "Tự động cập nhật") | NOT-COVERED (bởi design cũ) | Rebuild phải ghi lỗi `verify_snapshot` từng snapshot vào `ingest_issue` và tiếp tục; snapshot đầu `data/paper/snapshots/20260922T140208` có naming khác, cần kiểm schema trước |
| C19 | Paper record idempotent/conflict per session (readiness plan; system-design-review 04-TĐ3 L78) | MATCHES, ngoài phạm vi R2 | `paper_runner.py:94-131`. R2 không catalog paper records ở bước đầu |
| C20 | Timestamp có timezone, session theo lịch VN (plan §3.1; foreign-data-contract "Grain") | MATCHES | `aware` (`foreign_flows.py:32-38`), `source_time` → `VN_TIMEZONE` (:59-63, 108); VCI epoch → VN date (`reconciliation.py:49`). R1 `_ts` reject naive (`contracts.py:253-263`). Snapshot giá không có per-row `available_at`; `eod.scan_input_snapshot` phụ thuộc đồng hồ (`stock_agent/data/eod.py:36`) |
| C21 | Store legacy mutable, `write_json` temp tên cố định (system-design-review 12-DD1 L84) | MATCHES | `stock_agent/data/repository.py:24-29` (không fsync, `.tmp` cố định). R2 không ghi qua helper này |

## 3. Rủi ro cho R2

1. **Legacy promotion**: `data/raw/foreign/*.jsonl`, `data/raw/prices`, `data/raw/prices_hist`
   (101 CSV), `data/pipeline/*` không có manifest/receipt time. Chỉ được ghi
   `ingest_issue(kind='legacy_inventory')` với hash file; không vào bảng vintage. Kiểm
   bằng `contracts.is_legacy_path` và `check_source_row` (QUARANTINED).
2. **Overwrite in-place**: `providers._save_cache` (`providers.py:23`),
   `repository.write_json` (`repository.py:24-29`), `latest_status.json` chỉ mô tả batch
   cuối (FOREIGN-FLOWS L49-51). Catalog không đọc những file này làm nguồn sự thật.
3. **Non-atomic writes**: `records.jsonl`, raw files (`foreign_flows.py:218, 254`),
   snapshot raw/CSV/manifest (`reconciliation.py:123, 128, 138`). Rebuild xử lý manifest
   không parse được, hash mismatch, file thiếu = orphan/issue; không "sửa" hash.
4. **Timezone/as-of**: giá chỉ có `fetched_at` theo file và `completed_at` snapshot;
   `available_at` phải tính bằng rule R1, không tự gán. Snapshot đầu có naming không có
   `Z`, phải xác minh `fetched_at` timezone-aware trước khi ingest, nếu không → issue.
5. **Catalog drift**: projection có thể lệch raw khi code normalize đổi. Ghi
   `builder_sha256`, `contract_version`, `registry_sha256` vào `catalog_meta`; khác thì
   rebuild từ đầu, không update tăng dần.
6. **Concurrency**: rebuild chạy đồng thời collector có thể thấy run chưa có manifest.
   Không lấy `.collect-lock`; chỉ ingest run có manifest, ghi run chưa commit là issue
   `incomplete_run` (cùng ngữ nghĩa `health`).
7. **Scope creep**: catalog không được trở thành publication boundary hay thay
   `load_flows` trong bước đầu (C10); thay reader là contract change cần preregister.

## 4. Thiết kế R2 tối thiểu, căn theo R1

Engine: stdlib `sqlite3`, `PRAGMA journal_mode=WAL`, `foreign_keys=ON`, một transaction
cho mỗi generation. Raw bytes **không** copy vào DB: filesystem raw vẫn là authority,
DB giữ pointer + hash. Tên bảng vintage lấy theo `contract_id` của R1.

| Bảng | Unique key | Cột chính / quy tắc |
|---|---|---|
| `catalog_meta` | `key` | `contract_version='source-dataset-contracts-v1'`, `registry_sha256`, `builder_sha256`, `catalog_schema_version`, `built_at` |
| `generation` | `generation_id` (tên thư mục run/snapshot); UNIQUE `manifest_sha256`; UNIQUE `manifest_relpath` | `contract_id` (`vietstock_flow_chart` / `vci_ohlcv_daily`), manifest `status`, `started_at`, `completed_at` (UTC), `expected_session`/`as_of`, `collector_sha256`, `verify_status` |
| `ingestion_attempt` | (`generation_id`, `contract_id`, `symbol`, `investor`) — `investor=''` cho giá để UNIQUE không bị NULL vô hiệu | `outcome` (`succeeded`/`failed`/`circuit_open`, suy từ `sources`/`errors`/`failures`), `raw_sha256` NULL khi failed, `observed_at`/`fetched_at`, `error_type`, `attempts`; cột `derived_from_manifest=1` để không giả là receipt gốc (C8) |
| `raw_object` | `raw_sha256` | `byte_len`, `first_relpath`; nhiều attempt tham chiếu cùng hash, không gộp `observed_at` (DE03) |
| `vietstock_flow_chart_vintage` | R1 business key + vintage + generation: (`symbol`, `date`, `investor`, `source`, `trade_scope`, `raw_sha256`, `generation_id`) | Đúng các FieldSpec R1 (`contracts.py:120-126`): VND/shares, `source_timestamp`, `observed_at`, `available_at` |
| `vci_ohlcv_daily_vintage` | (`symbol`, `date`, `source`, `raw_sha256`, `generation_id`) | FieldSpec R1 (`contracts.py:97-101`), `unit`, `snapshot_completed_at`=manifest `completed_at`, `available_at` tính bằng rule R1, `adjustment_basis` |
| `revision_event` | (`contract_id`, `business_key_json`, `prev_generation_id`, `new_generation_id`) | `changed_fields`; chỉ ghi khi nội dung khác; trùng giá trị không phải revision |
| `ingest_issue` | (`path`, `kind`, `reason`) | `kind`: `orphan_run`, `incomplete_run`, `hash_mismatch`, `canonical_mismatch`, `verify_failed`, `legacy_inventory`, `naive_timestamp`; không bao giờ được promote |

As-of read **không** viết lại logic bằng SQL: hàm `catalog.select_as_of(db, contract_id,
decision_time)` SELECT các vintage có `available_at <= decision_time` rồi gọi
`contracts.select_as_of` (`contracts.py:415-441`) để giữ ambiguity rejection và
eligibility của R1.

**Lệnh** (module đề xuất `stock_agent/data/catalog.py`, chỉ đọc store):

```text
python -m stock_agent.data.catalog rebuild --out <new.db> [--flows-root data/foreign_flows_v2] [--snapshots-root data/paper/snapshots]
python -m stock_agent.data.catalog verify --db <db>      # rebuild vào temp và so content hash
python -m stock_agent.data.catalog update --db <db>      # chỉ ingest manifest_sha256 chưa có; sau bước đầu
```

`rebuild` từ chối `--out` đã tồn tại, ghi vào temp cùng thư mục rồi `os.replace`; không
lấy collector lock; không network; không đổi `run_eod_update.bat` hay lịch 17:05. Vị trí
DB runtime ngoài Git, chọn khi implement.

**Tests (fixtures `tmp_path`, không dùng data thật)**:

1. Rebuild hai lần → content hash bảng giống nhau; `update` lặp lại không thêm dòng.
2. Parity: `catalog.select_as_of(..., t)` == tập dòng `load_flows(as_of=t)` dạng long trên
   fixtures kiểu `tests/test_foreign_flows_v2.py`, gồm case late publication.
3. Cùng payload ở hai run → một `raw_object`, hai `ingestion_attempt`, hai vintage
   (C15); giá trị đổi → một `revision_event`; trùng giá trị → không có event.
4. Tamper raw/`records.jsonl`/manifest, path escape → `ingest_issue`, không có vintage.
5. Run thiếu manifest hoặc manifest JSON dở dang → `incomplete_run`/`orphan_run`.
6. Legacy paths (`data/raw/foreign`, `data/raw/prices*`, `data/pipeline`) và
   `legacy/*/research_only.jsonl` → không xuất hiện trong as-of; R1 trả QUARANTINED.
7. Equal `available_at` khác nội dung → rejected `ambiguous_version_at_cutoff` (R1).
8. Giá: decision giữa 16:00 và `fetched_at` → BLOCKED; trước close → not_yet_available.
9. Kill giữa transaction (inject exception sau N rows) → DB cũ/không có DB, không
   partial generation.
10. `sqlite3.Connection.backup` → restore → `verify` parity bằng hash.
11. Drift: đổi `builder_sha256`/`registry_sha256` → `update` từ chối, yêu cầu rebuild.

**Phụ thuộc R1**: R2 import `CONTRACT_VERSION`, `SOURCES[...].business_key`,
`vintage_key`, `FieldSpec`, `check_source_row`, `select_as_of`, `is_legacy_path`,
`session_coverage`; không tự định nghĩa grain/unit/availability. R2 không bắt đầu
implementation trước khi R1 được review và merge vào nhánh làm việc. Điểm cần R1 quyết
định: vintage identity quan sát = (`raw_sha256`, `generation_id`) (C15).

## 5. Bước bounded đầu tiên

Flow-only, read-only projection:

- Bảng `catalog_meta`, `generation`, `ingestion_attempt`, `raw_object`,
  `vietstock_flow_chart_vintage`, `ingest_issue` dựng từ
  `data/foreign_flows_v2/runs/*/manifest.json`, tái dùng logic `verified_records`
  (`foreign_flows.py:282-304`) mà không sửa module đó.
- Tests 1, 2, 3 (phần flow), 4, 5, 6, 7 ở §4; RED trước GREEN.
- Không có bảng giá, không thay reader `load_flows`/`health`, không đổi collector,
  BAT hoặc lịch.
- Nếu output được dùng cho research/feature, đăng ký data-contract change theo
  `AGENTS.md` trước, và chạy research gate đầy đủ như quy định; bước projection thuần
  không thay đổi kết quả kinh tế.

## Handoff

Path thay đổi duy nhất: file này. Không chạy test, không commit, không fetch/refresh.
Unresolved: receipt đầy đủ cho failed attempts cần sửa collector (ngoài bước đầu);
pagination/coverage proof của nguồn chưa có; RPO/RTO chưa chốt; R1 chưa merge.
Next bounded action: coordinator xác nhận C15 với owner R1, rồi giao bước §5.

## Cập nhật sau merge research_ml (06/10, 4403fc7)

Bối cảnh: `stock_agent/research_ml/catalog.py` (237 dòng) là SQLite catalog **chỉ cho price
vintages**, chỉ được kiểm bằng synthetic fixtures (`tests/test_ml_catalog.py`), chưa nối vào
collector hay manifest thật. Ledger ghi R2 PARTIAL (`docs/plans/2026-10-04-data-ml-research-plan.md:398`;
`docs/audits/2026-10-04-r1-r4-implementation.md:16`). Collector/loader legacy
(`foreign_flows.py`, `reconciliation.py`, `repository.py`, `providers.py`, `eod.py`) không đổi
giữa `4b859cd` và `4403fc7`. `stock_agent/data/contracts.py` sẽ không được merge; R1 được port
vào `research_ml/contracts.py` (branch `firstmate/r1-port-into-research-ml`, `0f299f4` đã có
legacy quarantine, chưa merge).

| ID | Trạng thái | Evidence @ `4403fc7` / ghi chú |
|---|---|---|
| C1, C4, C5, C17, C19 | N/A (đã MATCHES, legacy không đổi) | Không cần hành động mới |
| C2 ghi `records.jsonl`/raw/snapshot manifest không atomic | STILL OPEN | `foreign_flows.py:218,254`, `reconciliation.py:138` không đổi. Mẫu tmp + replace có ở `catalog.py:79-82` (chưa fsync) |
| C3 idempotency khi ingest; tie-break | RESOLVED (catalog giá) / STILL OPEN (flow loader) | Raw content-addressed `catalog.py:76-83,126`; trùng nội dung → `duplicates` `:141-143`; test `test_ml_catalog.py:29`. `foreign_flows.py:348` không đổi |
| C6 `revision_event` tự động | STILL OPEN | Không có bảng; chỉ có vintage counter (`catalog.py:137-147`) |
| C7 pagination/coverage proof | STILL OPEN | `expected_sessions` do caller khai báo (`catalog.py:108,117-119`); không có proof từ nguồn |
| C8 receipt `started/succeeded/partial/failed` | RESOLVED (API catalog) / STILL OPEN (collector) | `catalog.py:24-29,95-105,128-131`. Collector thật vẫn không ghi attempt cho lỗi |
| C9 tách attempt khỏi `raw_object` | RESOLVED | `catalog.py:23-29,126` |
| C10 publication boundary | CHANGED | Audit đề xuất catalog là projection dẫn xuất và manifest filesystem là authority. research_ml làm ngược lại: một transaction catalog là boundary (`catalog.py:1-8,123-155`), reader chỉ thấy generation `ready=1` (`:167-179`). Runtime vẫn dùng manifest, nên khi migrate phải hòa giải hai authority |
| C11 crash trước commit → orphan không visible | RESOLVED (catalog) | `catalog.py:115-116,148-153`; test `test_ml_catalog.py:52,61` |
| C12 backup/restore; RPO/RTO | RESOLVED (backup → restore → verify) / STILL OPEN (RPO/RTO, store thật) | `catalog.py:219-237`; test `:88` |
| C13 `verified_records` đọc lại toàn lịch sử | STILL OPEN | Legacy không đổi. `read_prices` cũng gọi `verify()` và rehash mọi raw object mỗi lần đọc (`catalog.py:160-169`), nên chi phí vẫn tăng theo archive |
| C14 legacy inventory (`ingest_issue`) | STILL OPEN | Không có `ingest_issue`. Catalog cũng không có đường promote legacy nào. Quarantine nằm ở `0f299f4` (chưa merge) |
| C15 vintage identity, `generation_id` | SUPERSEDED | PK là (`symbol`, `session_date`, `provider`, `vintage` số nguyên) (`catalog.py:33-37`), `generation_id` chỉ là FK. Refetch cùng nội dung chỉ tăng `duplicates`; quan sát đầu (available_at sớm nhất) được giữ, receipt sau vẫn có `ingestion_attempt` riêng. Vì vậy lỗi mất quan sát đầu mà C15 lo ngại không xảy ra. Bảng flow chưa có, nên phải chốt lại identity khi làm flow |
| C16 adjustment chưa xác minh; rule price `available_at` | STILL OPEN | Không có `adjustment_basis`; `available_at` do caller cấp (`catalog.py:144-146`); rule R1 chờ port |
| C18 lỗi `verify_snapshot` → issue khi rebuild | STILL OPEN | Chưa migrate manifest hiện có (`r1-r4-implementation.md:16`) |
| C20 timezone | RESOLVED (research_ml) / STILL OPEN (`eod.py:36`) | `contracts.py:60-70`; `catalog.py:170-172` |
| C21, rủi ro 2–3: ghi mutable/không atomic ở legacy | STILL OPEN | `repository.py:24-29`, `providers.py:23` không đổi |
| Rủi ro 5: catalog drift (`catalog_meta`, builder/registry sha) | STILL OPEN | Không có `catalog_meta`; schema dùng `CREATE TABLE IF NOT EXISTS` và không có version (`catalog.py:22-38,67-68`) |
| Rủi ro 6: concurrency | RESOLVED (catalog) | `BEGIN IMMEDIATE` + `timeout=30` (`catalog.py:71,125`); test `:99` |
| §4 schema (`catalog_meta`, `revision_event`, `ingest_issue`, `vietstock_flow_chart_vintage`, WAL, CLI `rebuild/verify/update`) | SUPERSEDED (thiết kế nhỏ hơn) / STILL OPEN (flow, issue, meta) | research_ml có 4 bảng (`catalog.py:22-38`). `rebuild_projection` chạy in-place, DELETE + insert lại trong một transaction (`:187-217`), thay cho `rebuild --out`. Không có WAL, không có CLI |
| §4 tests 1–11 | RESOLVED: 1, 3 (giá), 9, 10 / STILL OPEN: 2, 4, 5, 6, 8, 11 / test 7 nằm ở contracts | `test_ml_catalog.py:29,52,73,88`. Raw hỏng thì fail closed (`:61`) nhưng không ghi issue; ambiguity được test ở `test_ml_contracts.py:60` |
| §5 bước đầu: flow-only projection | SUPERSEDED (đã làm price-only trước) / STILL OPEN (flow) | Nên làm lại trên `research_ml/catalog.py` với `FOREIGN_FLOW` (`contracts.py:50-55`) và screen của `0f299f4` |
| Handoff "R1 chưa merge" | CHANGED | `research_ml/contracts.py` đã merge; `stock_agent/data/contracts.py` bị bỏ |

Còn actionable:
1. Thêm `foreign_flow_vintage` vào `research_ml/catalog.py`, dùng `screen_foreign_flow` (`0f299f4`) và chốt identity theo C15.
2. Thêm `ingest_issue` và legacy inventory, migrate read-only các manifest `foreign_flows_v2`/`paper/snapshots`, kèm parity test với `load_flows`.
3. Còn mở: pagination proof, `revision_event`, `catalog_meta`/drift guard, RPO/RTO. `read_prices` cần verify incremental thay vì rehash toàn bộ mỗi lần đọc.
