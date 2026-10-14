# R1: đối chiếu source/dataset contracts với các audit hiện có

Ngày: 06/10/2026. Reviewer: Firstmate run `2026-10-06-ml-roadmap` / T1.
Phạm vi: so sánh kết quả R1 (registry source/dataset contracts + as-of eligibility)
với các claim/yêu cầu data-contract trong audit đã có. Không đánh giá
profitability, không cấp trading approval, không chạy backtest/research gate,
không fetch network, không sửa runtime/config/data/plan trong lần review này.

- Source revision: base `4b859cd`, branch `firstmate/r1-source-contracts`
  (worktree `D:/Chungkhoan-r1`), commits `0b81772` (RED tests), `a02d83c`
  (implementation), `204e06b` (ghi chú IN PROGRESS dưới hàng R1 trong ledger).
  Branch chưa push/merge; checkout chính `D:/Chungkhoan` không chứa code này.
- Artifact được review: `stock_agent/data/contracts.py` (`CONTRACT_VERSION =
  'source-dataset-contracts-v1'`, `REGISTRY_SHA256 = 0584d394…cba1d`) và
  `tests/test_data_contracts.py` (40 tests). Line numbers dưới đây là bản tại `a02d83c`.
- **Verdict: R1 IN PROGRESS, chưa DONE.** Contract và eligibility check đã có
  test RED/GREEN, nhưng chưa có builder/catalog/scan/gate nào dùng module, và còn
  hai quyết định cần người dùng chốt (preregistration và price-vintage policy).

## Audit đã đọc

| Audit | Phần dùng để đối chiếu |
|---|---|
| `2026-09-30-foreign-data-contract.md` | Toàn bộ, trọng tâm Grain/Availability/Storage/Limitations |
| `2026-09-26-pit-universe-protocol.md` | Toàn bộ, trọng tâm "Eligibility and knowledge timing" |
| `2026-09-26-pit-validation-and-readiness.md` | Toàn bộ, trọng tâm §1 timeline, §3 legacy source, §5 corporate actions |
| `2026-10-04-ml-plan-review-data.md` | Toàn bộ claim matrix DE01–DE17 và current-code spot checks |
| `2026-10-04-system-design-review.md` | Toàn bộ, các dòng 06-TĐ1, 06-DD1, 12-TĐ2, 04-DD1, 05-TĐ2 và mục P2 |
| `2026-09-30-research-gate-spec.md` | Chỉ phần data contract: "Foreign-flow audit…" và "Engineering acceptance" |

## Tóm tắt thiết kế R1 (để đọc bảng)

- Source contracts: `vci_ohlcv_daily` v1 (`contracts.py:93–114`),
  `vietstock_flow_chart` v2 (`:116–138`), `vn30_membership` v1 (`:140–155`).
  Dataset contracts: `feature_row` v1 (`:159–172`), `label_row` v1 (`:174–188`).
  Mỗi contract khai báo grain, business key, vintage key, fields/units, availability
  rule, missing/zero/stale semantics, coverage, source-priority tie-break, quarantine rules.
- Status theo mức nghiêm trọng: `invalid` > `quarantined` > `blocked` >
  `not_yet_available` > `eligible`; label còn có `pending`/`not_entered`.
- `check_source_row` (`:399–412`): eligible chỉ khi `available_at <= decision_time`
  (`:410`); decision time phải có timezone và nằm trong phạm vi lịch (`:292–295`).
- `select_as_of` (`:415–443`): lọc eligibility **trước**, sau đó chọn version có
  `available_at` muộn nhất cho mỗi business key; hai nội dung khác nhau có cùng
  `available_at` thì reject `ambiguous_version_at_cutoff` (`:437`).
- Registry hash được pin (`:33`, test `tests/test_data_contracts.py:41`): sửa contract
  mà không bump version/pin thì test fail.

## Bảng đối chiếu

Ký hiệu: **MATCHES** = R1 thực hiện đúng claim; **EXTENDS** = R1 thêm rule/ràng buộc
chặt hơn nhưng không trái claim; **CONFLICTS** = R1 khác hành vi/giả định của audit
hoặc code hiện có; **NOT-COVERED** = R1 chưa xử lý. Test path là `tests/test_data_contracts.py`.

| # | Audit / mục | Claim hoặc yêu cầu | Verdict | Evidence trong R1 |
|---|---|---|---|---|
| 1 | foreign-data-contract, Grain | Grain session × symbol × investor × source × provider aggregation; không trộn matched với chart aggregate | MATCHES | business_key `contracts.py:116–138`; chỉ chấp nhận `provider_chart_aggregate`, `matched_only` bị quarantine `:318–319`; test `:117` |
| 2 | foreign-data-contract, Grain | Canonical VND và shares; bn-VND chỉ ở compatibility boundary | MATCHES (record) / NOT-COVERED (panel) | Units VND/shares, kiểm tra net = buy − sell `:337–341`. Panel bn-VND `load_flows` không có contract riêng; ml-plan-review "spot checks" cũng yêu cầu phân biệt tên/đơn vị panel |
| 3 | foreign-data-contract, Availability | Availability = thời điểm nhận response, không lùi về session; lịch sử fetch hôm nay chỉ available từ hôm nay | MATCHES | Dùng `available_at` của run, phải ≥ `observed_at` ≥ session close `:330–334`; test leak `:75`, GREEN `:82`, backdate `:87` |
| 4 | foreign-data-contract, Availability; research-gate-spec "Foreign-flow audit" | Legacy không có provenance không bao giờ eligible | MATCHES | Quarantine theo schema_version, `research_only`, `legacy_*`, tên source legacy, thiếu `observed_at`/`available_at`/`raw_sha256` `:310–326`; test tham số hóa `:117`, row thật từ `migrate_legacy` `:123` |
| 5 | foreign-data-contract, Availability | TradingDate đọc theo UTC rồi đổi Asia/Saigon, không dùng ngày của host | NOT-COVERED | Contract chỉ yêu cầu `source_timestamp` có mặt (`:325–326`), **không** đối chiếu lại `date` với epoch. Việc chuyển đổi vẫn do `normalize_chart`; test `:95` chỉ xác nhận row đã normalize thì pass |
| 6 | foreign-data-contract, Storage | as-of lọc availability **trước** khi chọn revision mới nhất; trùng response thì idempotent | MATCHES | `select_as_of` `:415–443`; test revision tương lai, duplicate collapse `:222` |
| 7 | foreign-data-contract, Storage (loader hiện có) | `load_flows` dedup với `keep='last'` sau khi sort `available_at, observed_at, raw_sha256` (`foreign_flows.py:348`) | **CONFLICTS** | Gặp hai vintage khác nội dung có cùng `available_at`, loader hiện tại chọn ngầm theo thứ tự hash; R1 reject vì mơ hồ (`contracts.py:437`). Trong vận hành bình thường hiếm xảy ra (mỗi run một response cho mỗi symbol/kind), nhưng hai đường sẽ cho kết quả khác nhau khi R2 dùng contract |
| 8 | foreign-data-contract, Collection; research-gate-spec | Empty payload không phải all-zero activity; missing không phải zero flow | MATCHES / EXTENDS | NaN/None/inf/âm trong một observed row thì invalid `:274–285`, test `:152`; observed zero hợp lệ và value/volume nhất quán `:342–343`, test `:157` |
| 9 | research-gate-spec "Foreign-flow audit" | Phân biệt daily với monthly/quarterly chưa kết thúc; không đoán date shift | MATCHES | Tên source monthly/quarterly legacy bị quarantine `:51–52`, `:316–317`; legacy không có date thì quarantine (test `:117`) |
| 10 | research-gate-spec "Engineering acceptance" | Thiếu evidence lịch sử = BLOCKED, không âm thầm green | MATCHES / EXTENDS | Status `blocked` khi ngoài phạm vi lịch `:292–295` và khi price vintage được lấy sau decision `:377–378`; test `:180`, `:202` |
| 11 | pit-universe-protocol, "Eligibility and knowledge timing" | known_on có nghĩa available lúc CLOSE; hiệu lực ở phiên kế tiếp; không đoán trước thay đổi chưa công bố | MATCHES | `available_at = known_on 16:00 VN` `:382–396`; test trước/sau close `:209`. `effective_on` là valid time, áp bởi `scripts/historical_universe.Timeline.members`, không phải bởi contract |
| 12 | pit-universe-protocol; pit-validation §1 | Fail-closed ngoài coverage timeline (coverage_start…coverage_end) | NOT-COVERED (lệch giữa mô tả và code) | Contract mô tả "after coverage_end are BLOCKED" (`:151`) nhưng `_check_membership` chỉ kiểm tra phạm vi lịch HOSE, không kiểm tra coverage của timeline. Hiện chỉ `Timeline.members` thực thi. Cũng chỉ chấp nhận `index == 'VN30'` (`:383`); `configs/research/vn100_membership_2026_h1.json` chưa có contract |
| 13 | pit-validation §3 | `data/raw/prices_hist` là nguồn legacy của dashboard, không được chứng nhận | MATCHES | `LEGACY_PATH_PREFIXES` gồm `data/raw/foreign`, `data/raw/prices`, `data/raw/prices_hist`, `data/pipeline` `:50`; mọi row có path legacy đều quarantine `:298–301`; test `:138`, `:144` |
| 14 | pit-validation §5.3; pit-universe "Evidence"; system review 06-TĐ1 | Hash đúng không chứng minh dữ liệu đã được biết trong quá khứ; adjustment của provider chưa xác minh | MATCHES (marker) / NOT-COVERED (lineage) | Price row eligible vẫn mang `claims_blocked=('pit_adjustment_history',)` `:114`; test `:169`. Chưa có contract `reference_price`/`corporate_action` |
| 15 | ml-plan-review DE06 | Bound bảo thủ phải dựa trên receipt/commit thật, không bịa timestamp EOD | MATCHES | Price `available_at = max(session close, fetched_at, snapshot_completed_at)` `:367–379`; fetch trước close thì invalid `:365–366`; test `:180`, `:191` |
| 16 | pit-universe "Evidence"; research_gate replay hiện có | Snapshot lấy hôm nay được dùng cho replay lịch sử, adjustment chưa xác minh được ghi là **limitation** | **CONFLICTS** | Theo rule R1, mọi decision lịch sử trước `fetched_at` của fixture (30/09/2026) đều `blocked` (`:377–378`). Gate/PIT protocol hiện chấp nhận vintage-adjusted kèm caveat. Cần quyết định (xem mục dưới) |
| 17 | ml-plan-review DE05; plan §3.1 | Valid time tách knowledge time; half-open intervals; tối đa một version mỗi key tại knowledge cutoff; tie-break đã freeze; không reject tràn lan mọi overlap | MATCHES (knowledge-axis) / NOT-COVERED (valid-time intervals) | Một version mỗi key, tie-break theo `available_at`, reject ambiguity `:415–443`. Prices/flows chưa có `effective_from/to`; R1 dùng `available_at <= decision_time` (closed) đúng như plan §3.1, chưa có test retroactive correction trên trục valid time |
| 18 | ml-plan-review DE07; plan §4 bước 1 | Missing, zero, stale, invalid, not-applicable không được gộp làm một | EXTENDS | `MISSING_REASONS` `:46`; `feature_row` reject `missing_coded_as_value`, missing không có lý do, lý do không có trong enum, giá trị non-finite `:476–492`; test `:273` |
| 19 | ml-plan-review spot check `feature_store.py:12` | `_safe_float` mặc định trả 0.0 cho None/NaN | **CONFLICTS (tiềm ẩn)** | Khi R3 dùng `check_feature_row`, mọi zero-fill qua `_safe_float` sẽ bị reject là `missing_coded_as_value` (`:487`). Hiện chưa nối nên không vỡ test; R3 phải sửa đường build hoặc ghi lý do missing |
| 20 | ml-plan-review DE11; plan §3.2 | Rolling N phiên bám lịch giao dịch; báo present/expected; không nén missing slots | EXTENDS (coverage) / NOT-COVERED (rolling) | `session_coverage` đếm theo `symbol_trading_days_between` và liệt kê slot thiếu `:552–561`; test `:329`. Chưa có rolling window/min-support (R3) |
| 21 | ml-plan-review DE15; plan §5 | Pending không phải zero/loss; cần cả label_end và label_available_at | MATCHES / EXTENDS | `check_label_row` yêu cầu decision ≤ exit close ≤ `label_available_at` ≤ as_of; pending/not_entered phải có outcome null `:505–538`; `partition_labels` `:541–549`; test `:299`, `:305`, `:320`. `temporal_validation.label_times` vẫn coi `label_available_date` là optional, chưa nối (R4) |
| 22 | ml-plan-review DE03; plan §3.1 | Tách `ingestion_attempt(attempt_id)` khỏi `raw_object(content_sha256)` | NOT-COVERED | R1 chỉ có row contracts; chưa có contract cho receipt/attempt (R2) |
| 23 | ml-plan-review DE08–DE10; system review 04-DD1 | Receipt, watermark, expected population, publication boundary, snapshot manifest ghi không atomic | NOT-COVERED | Ngoài phạm vi R1 (R2 catalog). `verify_snapshot`/`collect_flows` không thay đổi |
| 24 | ml-plan-review DE13; system review 12-TĐ2 | Hash closure của builder/calendar/universe/feature/cost và environment; paper code_hash chưa đủ | EXTENDS (một phần) / NOT-COVERED | Registry hash pinned là một thành phần closure (`:204–216`). Không bao phủ calendar table, environment hay paper identity |
| 25 | system review 06-TĐ1/06-DD1 | Fields/identity/event time; late/duplicate/revision tách riêng | MATCHES | Business/vintage key mỗi contract; duplicate collapse, revision tương lai bị loại `:415–443` |
| 26 | system review 06-DD1 (4 volume revision rows) | Revision của volume dương được báo riêng, không im lặng | NOT-COVERED | `select_as_of` chọn vintage mới nhất nhưng không phát cảnh báo revision; `paper_scoring` vẫn là nơi báo |
| 27 | system review 05-TĐ2 | `verified_records` đọc lại toàn bộ archive, cost tăng theo lịch sử | NOT-COVERED | R1 kiểm tra row được đưa vào, không đọc store; chưa có projection/index (R2) |
| 28 | plan §3.1 bảng `price_bar_vintage` | Ghi loại raw/adjusted, unit, session completeness cho từng row | NOT-COVERED (một phần) | Unit chỉ mô tả ở cấp contract ("VND\|index_points"), không có field unit/adjustment theo row; adjustment chỉ thể hiện qua `claims_blocked` |
| 29 | exchange_calendar (dùng trong mọi audit) | Bảng ngày nghỉ HOSE cần làm mới cho các năm tới | EXTENDS | `CALENDAR_COVERAGE` lấy từ min/max năm trong `HOSE_HOLIDAYS` `:37–38`; decision sau 2026-12-31 thì `blocked`, test `:202` |

## Xung đột cần quyết định

### C1. Preregistration cho thay đổi data contract

AGENTS.md và plan §10 (R1 phụ thuộc "prereg contract change") yêu cầu đăng ký
giả thuyết trong `configs/research/research_gate_v1.json` trước khi xem kết quả mới
khi đổi data contract. R1 **không** sửa file này vì module chưa được gate/scan/
backtest nào dùng và không có kết quả nào thay đổi (full suite 723 passed, không
chạy replay). Cần người dùng chốt một trong hai:

1. R1 (library-only) không cần preregistration; đăng ký bắt buộc tại bước R2/R3
   khi contract bắt đầu lọc input của dataset/gate; hoặc
2. Thêm ngay entry/version protocol cho contract v1 trước khi merge, kèm full
   pinned suite + books + timeslices theo AGENTS.md.

### C2. Price-vintage policy (hàng 15–16)

R1 dùng cách bảo thủ mà DE06 yêu cầu: bar public lúc close, nhưng giá đã adjust
của vintage này chỉ tồn tại từ lúc retrieve. Vì vậy decision sau close mà trước
`fetched_at`/`snapshot_completed_at` là `blocked`, không bịa timestamp. Hệ quả:
mọi replay lịch sử từ fixture lấy ngày 30/09/2026 sẽ bị chặn nếu builder R3 áp
rule này nguyên trạng, trong khi research gate và PIT protocol hiện chấp nhận
vintage-adjusted như một limitation đã ghi. Lựa chọn cần chốt:

- (a) Giữ BLOCKED nghiêm ngặt: R3/R5 chỉ được dùng dữ liệu prospective hoặc vintage lưu
  trữ có bằng chứng; hoặc
- (b) Thêm research mode khai báo rõ "accepted vintage-adjusted, PIT not certified":
  row được dùng cho development replay nhưng luôn mang `pit_adjustment_history` và
  status riêng, không bao giờ được tính là PIT-eligible hay bằng chứng prospective.
  Mode này phải version contract (v2) và cập nhật `REGISTRY_SHA256`.

### C3. Khác biệt nhỏ cần giải quyết khi nối vào runtime

- Hàng 7: tie-break của `load_flows` (ngầm theo hash) khác với reject ambiguity của
  R1. R2 cần chọn một rule; khuyến nghị dùng rule của contract và thêm test cho loader.
- Hàng 19: zero-fill trong `feature_store._safe_float` sẽ fail `feature_row`; R3 phải
  thay bằng missing reason hoặc khai báo neutral-fill như một transform có version.
- Hàng 12: mô tả của `vn30_membership` nói BLOCKED ngoài coverage nhưng code chưa
  enforce; cần thêm check coverage (bump contract version) hoặc sửa mô tả.

## Khoảng trống còn lại

- R1: quyết định C1/C2; enforce coverage của membership timeline; contract cho
  VN100 membership, VNINDEX/breadth, reference price/corporate action, sector;
  đối chiếu `date` với `source_timestamp` của flow; field unit/adjustment theo row cho price.
- R2: contract `ingestion_attempt`/`raw_object`, receipt/watermark, catalog
  dùng business/vintage key làm unique key và `select_as_of` khi đọc, cảnh báo revision,
  publication boundary, restore tests. Không promote legacy.
- R3/R4: builders xuất `feature_row`/`label_row` và pass `check_feature_row`/
  `partition_labels`; rolling window theo lịch giao dịch với min-support;
  `label_available_date` bắt buộc trong `temporal_validation`.

## Kiểm chứng đã thực hiện (trong task R1, không chạy lại cho báo cáo này)

Chạy từ `D:/Chungkhoan-r1` bằng `C:\Users\acer\anaconda3\python.exe` (python trên
PATH là msys64, không có pytest):

```text
python -m pytest tests/test_data_contracts.py -q
# RED tại 0b81772: ImportError khi collect (chưa có stock_agent.data.contracts)
# GREEN lần đầu: 39 passed, 1 failed (placeholder REGISTRY_SHA256, có chủ đích); sau khi pin: 40 passed

python -m pytest tests/test_data_contracts.py tests/test_reconciliation.py tests/test_foreign_flows_v2.py tests/test_training_integrity.py tests/test_pytest_collection_safety.py tests/test_exchange_calendar.py tests/test_validation.py -q
# 134 passed, 14 warnings in 55.86s

python -m pytest -q -x -p no:cacheprovider
# 723 passed, 23 warnings in 312.05s (warnings là pandas date-parsing UserWarning đã có)
```

Không chạy research gate, momentum books, timeslices, replay, training, scan,
network fetch hay thay lịch. Không có kết quả chiến lược mới nào được xem. Test
xanh không phải trading approval và không chứng minh dữ liệu lịch sử là PIT.

## Handoff

File thay đổi duy nhất trong bước này: audit này. Code R1 vẫn chỉ nằm trên branch
`firstmate/r1-source-contracts` (chưa push/merge). Bước tiếp theo có giới hạn:
người dùng chốt C1 và C2, sau đó R2 import `stock_agent.data.contracts` theo rule đã chốt.

## Cập nhật sau merge research_ml (06/10, 4403fc7)

Bối cảnh: `stock_agent/data/contracts.py` (branch `firstmate/r1-source-contracts`, `204e06b`)
**sẽ không được merge**. R1 đã vào nhánh là `stock_agent/research_ml/contracts.py`
(194 dòng: `SourceContract` `PRICE_BAR`/`FOREIGN_FLOW`, `MissingState`, `validate_source`,
`select_vintage`, `asof_join`, `assert_no_target_leak`). Ledger ghi R1 IMPLEMENTED
(engineering, synthetic) tại `docs/plans/2026-10-04-data-ml-research-plan.md:397`. Phần còn
thiếu đang được port vào `research_ml/contracts.py` trên branch
`firstmate/r1-port-into-research-ml` (worktree `D:/Chungkhoan-r1port`). Commit `0f299f4`
(chưa merge) đã port quarantine cho legacy path và legacy foreign-flow row; các mục khác
đang làm. Mọi `contracts.py:N` dưới đây là `research_ml/contracts.py` @ `4403fc7`. Code legacy
(`foreign_flows.py`, `feature_store.py`, `temporal_validation.py`, `exchange_calendar.py`)
không đổi giữa `4b859cd` và `4403fc7`.

| ID | Trạng thái | Evidence @ `4403fc7` / ghi chú |
|---|---|---|
| Verdict "R1 IN PROGRESS" | CHANGED | R1 = IMPLEMENTED ở `plan:397`, nhưng là bản tối giản: 2 source contract, chưa có dataset contract và status ladder |
| #1 grain foreign; quarantine `matched_only` | RESOLVED (grain) / chờ merge port (scope) | Grain có `trade_scope` (`contracts.py:50-55`). `0f299f4` quarantine mọi scope khác `provider_chart_aggregate` (`trade_scope_not_contracted`) |
| #2 unit VND/shares; net = buy − sell; panel bn-VND | STILL OPEN (chờ port) | Unit chỉ mang tính mô tả (`contracts.py:39,55`), không kiểm net; `load_flows` không đổi |
| #3 `available_at` ≥ `observed_at` ≥ close | STILL OPEN (chờ port) | @`4403fc7` chỉ bắt tz-aware (`contracts.py:60-70,78`). `0f299f4` thêm yêu cầu có timestamp tz-aware cho `observed_at`/`available_at`, nhưng chưa kiểm thứ tự |
| #4, #9, #13 quarantine legacy (schema, `research_only`, path prefix, provider monthly/quarterly) | STILL OPEN @`4403fc7`; đã port trong `0f299f4` (chưa merge) | `0f299f4`: `LEGACY_PATH_PREFIXES`, `LEGACY_FLOW_PROVIDERS` (gồm `*_monthly`/`*_quarterly`), `schema_version_not_2`, `legacy_research_only_row`, `missing_provenance:raw_sha256`; test `tests/test_ml_contracts_port.py`. @`4403fc7`, F3 bị BLOCKED ở cấp protocol (`configs/research/ml_protocol_v1.json:19`) |
| #5 TradingDate UTC → VN, đối chiếu `source_timestamp` | STILL OPEN | Vẫn chỉ nằm trong `normalize_chart` |
| #6 lọc availability trước khi chọn revision; duplicate idempotent | RESOLVED | `select_vintage` `contracts.py:101-131`, `asof_join` `:134-180`; dedup `catalog.py:141-143`; `tests/test_ml_contracts.py:36,73` |
| #7 tie-break `load_flows` vs reject ambiguity | STILL OPEN | `foreign_flows.py:348` vẫn `keep='last'`; research_ml reject (`contracts.py:124-129,166-167`). Hai đường vẫn có thể cho kết quả khác nhau |
| #8 empty ≠ zero; observed zero hợp lệ | RESOLVED (record/as-of) | `OBSERVED_ZERO` `contracts.py:17-23,172-174`; reject non-finite/âm/không dương `:79-89`; `test_ml_contracts.py:48`. Chưa kiểm value/volume |
| #10 thiếu evidence = BLOCKED (phạm vi lịch) | STILL OPEN (chờ port) | Contract chưa có calendar coverage; BLOCKED mới chỉ có ở bundle (`evidence.py:389-404`) |
| #11, #12 membership `known_on` 16:00, coverage timeline, VN100 | STILL OPEN (chờ port) | Chưa có membership contract; R5 dùng `Timeline` (`ml_protocol_v1.json:16`) |
| #14 marker `pit_adjustment_history` | STILL OPEN (chờ port) | Hiện chỉ ghi bằng lời (`2026-10-04-r5-results.md:5-6`; bundle `source_manifest.json` "limitations") |
| #15 price `available_at` = max(close, fetched_at, snapshot_completed_at) | STILL OPEN (chờ port) | `publish_prices` lưu `available_at` do caller cấp (`catalog.py:144-146`) |
| #16 / C2 price-vintage policy | CHANGED | R5 thực tế đi theo phương án (b) nhưng không có version: snapshot "provider-adjusted as retrieved, not archival PIT", dùng làm development data (`r5-results.md:5-7`; `ml_protocol_v1.json:5,15`). Port cần biến lựa chọn này thành mode/status có version |
| #17 một version/key; valid-time intervals | RESOLVED (knowledge axis) / STILL OPEN (valid time) | `contracts.py:101-131`, provider priority `:116-119`; chưa có `effective_from/to` |
| #18 tách missing/zero/stale/invalid/n.a. | RESOLVED một phần | Enum `contracts.py:17-23`, cột `_state` `features.py:404-413`. Feature chưa có `INVALID` (inf → `missing`, `features.py:409-413`); `MISSING_REASONS`/`check_feature_row` chờ port |
| #19 `_safe_float` zero-fill | STILL OPEN (legacy) | `feature_store.py:12-15` không đổi; research_ml không dùng feature_store |
| #20 rolling theo lịch + min-support; coverage | RESOLVED (F3) / STILL OPEN (F0, `session_coverage`) | F3 tính trên session skeleton (`features.py:418-451`); F0 vẫn rolling theo dòng bar (`features.py:383-397`) |
| #21 pending ≠ 0/loss; `label_available_at` | RESOLVED (research_ml) / STILL OPEN (legacy) | `labels.py:284-317`, `trainable` `:320-324`, folds bắt buộc timestamp `harness.py:48-50`. `temporal_validation.py:25` vẫn optional; `check_label_row`/`partition_labels` chờ port |
| #22 `ingestion_attempt` ≠ `raw_object` | RESOLVED (giá) | `catalog.py:22-29` |
| #23 receipt/watermark/expected/publication boundary | RESOLVED (giá, synthetic) / STILL OPEN (collector thật) | `catalog.py:107-157`; `verify_snapshot`/`collect_flows` không đổi |
| #24 hash closure | RESOLVED một phần | `registry_hash` `features.py:372-374`; seal `evidence.py:331-337`. Còn thiếu hash calendar/universe/cost và device/threads |
| #25 identity/event time | RESOLVED | `contracts.py:30-57,95-97` |
| #26 cảnh báo revision | STILL OPEN | Catalog tạo vintage mới (`catalog.py:141-147`) nhưng không phát revision event |
| #27 đọc lại toàn archive | STILL OPEN | Legacy không đổi; `read_prices` cũng rehash mọi raw mỗi lần đọc (`catalog.py:160-169`) |
| #28 unit/adjustment theo row | STILL OPEN | `catalog.py:33-37` không có các cột này |
| #29 lịch giao dịch chỉ tới 2026 | STILL OPEN (chờ port) | `exchange_calendar.py` không đổi |
| C1 preregistration | CHANGED | research_ml được merge dạng library-only. ML protocol đăng ký riêng trước khi có kết quả (`ml_protocol_v1.json:4-5`, commit `db6aa97`; v2 `a7099a4`); `research_gate_v1.json` không sửa. Tức là thực tế theo phương án 1 |
| C3 (hàng 7, 19, 12) | STILL OPEN | Xem các hàng tương ứng |
| Handoff "R2 import `stock_agent.data.contracts`" | SUPERSEDED | `catalog.py:20` import `research_ml.contracts` |

Hạng mục còn chờ port (ngoài phần đã có trong `0f299f4`):
- source registry `vci_ohlcv_daily`, `vn30_membership` (+ VN100);
- dataset contracts `feature_row`/`label_row` (`check_feature_row`, `check_label_row`, `partition_labels`) và `MISSING_REASONS`;
- status ladder `invalid` > `quarantined` > `blocked` > `not_yet_available` > `eligible` cho price/membership;
- thứ tự availability, rule price `available_at`, calendar coverage → `blocked`, membership coverage;
- marker `claims_blocked`, kiểm net = buy − sell và value/volume, `session_coverage`, `CONTRACT_VERSION` + pin `REGISTRY_SHA256`.

Còn actionable:
1. Merge `0f299f4` và hoàn tất phần port còn lại. Quyết định C2 chính thức dưới dạng mode có version.
2. Chọn một rule tie-break duy nhất cho `load_flows` (#7), kèm parity test.
3. Bắt buộc `label_available_at` trong legacy `temporal_validation`, hoặc ghi rõ legacy path không được dùng cho R5–R7.
