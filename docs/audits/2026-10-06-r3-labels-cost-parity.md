# R3: đối chiếu label builder / cost parity với các audit hiện có

Ngày kiểm tra: 06/10/2026. Reviewer: Firstmate run `2026-10-06-ml-roadmap` / T3.
Phạm vi: scout read-only cho roadmap item **R3 Feature and strategy label
builders** (`docs/plans/2026-10-04-data-ml-research-plan.md` §4, §5, §7, ledger R3).
Không chạy test, backtest, training, data refresh hay network fetch; không xem
performance number mới. Lịch sử đã xem là development data.

## Input cố định và verdict

- Source revision: `4b859cd` (branch `codex/data-integrity-audit`). Dirty tree lúc
  đọc chỉ có `data/pipeline/local_server.log`, `data/pipeline/swing_scan_cache.json`
  (untracked, không liên quan code). Line numbers dưới đây thuộc revision này.
- Audit đã đọc toàn văn: `2026-09-30-research-gate-spec.md`,
  `2026-09-30-strategy-reassessment.md`, `2026-10-04-ml-plan-review-features.md`,
  `2026-09-30-momentum-books.md`, `2026-09-30-momentum-events.md`,
  `2026-09-30-execution-receipts.md`, `2026-09-22-training-remediation.md`;
  cùng `configs/research/research_gate_v1.json`.
- Verdict: **R3 CHƯA THỂ BẮT ĐẦU IMPLEMENTATION LABEL** trước R1 schema. Cost model
  của gate spec (T+3, multiplicative slippage, opening fill gate) **không phải**
  cost model mà fast label builder dùng. Training labels hiện tại và simulator
  dùng để đánh giá là hai hệ kế toán khác nhau. Đây là training/evaluation skew,
  chưa được audit nào trước đây liệt kê đầy đủ.

## Câu hỏi trọng tâm: fast label builder có dùng cost model của gate không?

**Không.** Gate spec/strategy reassessment §Latest six months khóa: lô 100, buy
commission 0,15%, sell commission+tax 0,25%, slippage 0,10% mỗi chiều, T+3
shares/cash, next open, opening-price fill gate. Đường thực thi là
`scripts/period_backtest.Broker` + `replay_mr` (`scripts/research_gate.py:179-180`).
Fast builder `build_labeled_dataset_fast` (`stock_agent/agents/parallel_engine.py:189-341`)
dùng cùng **tham số** (`BacktestConfig.from_rules`, `:221`) nhưng **khác cơ chế**:

| Khía cạnh | Gate / Broker (slow) | Fast label builder | Trạng thái |
|---|---|---|---|
| Phí + slippage | Giá × (1±s), phí tính trên notional đã slip (`period_backtest.py:105-107,125-147`) | Trừ cộng dồn `2c + tax + 2s` khỏi gross % (`backtest.py:136-137`, `parallel_engine.py:318`) | CONFLICTS (sai lệch bậc hai, đổi dấu ở trade gần hòa vốn) |
| Settlement | Shares available `entry + 3` (`period_backtest.py:133`) | Lock 2 phiên (`parallel_engine.py:298-301`) | CONFLICTS |
| Fill entry | Mở cửa vượt limit → skip; yêu cầu `stop < px < target` (`period_backtest.py:244-247`) | Clamp giá vào limit, luôn vào lệnh (`parallel_engine.py:290`) | CONFLICTS (fabricated fill) |
| Fill exit | Không fillable → giữ tiếp, không bịa exit (`:261-262`) | Clamp giá exit (`:316`) | CONFLICTS |
| Gap-down stop | `min(stop, open)` (`:265`) | Fill tại stop (`:302-306`) | CONFLICTS (lạc quan) |
| Exit horizon | `expiry = add_trading_days(entry, hold)` = e+15 (`:256`) | `add_trading_days(entry_date, holding_days)` (`:266`) | MATCHES (e+15 khi rules_mr) |
| Exit policy | `bracket` hoặc `fixed_hold` (`:220-275`) | Chỉ bracket; không có fixed15 label | NOT-COVERED |
| Pending | Lot chưa đáo hạn giữ open, mark-to-close | Row bị bỏ khi `exit_idx` thiếu (`:267-269`); `label_resolved` luôn True (`:330`) | EXTENDS (an toàn nhưng drop im lặng) |
| Double-cost stress | Nhân đôi config (`research_gate.py:166-168`) | Theo rules; nhưng `win_probability.COST = 0.60` hardcode (`win_probability.py:37`) | CONFLICTS ở MR P(win) |

## Bảng đối chiếu claim của audit

| Audit / mục | Claim | Phát hiện R3 + evidence code | Nhãn |
|---|---|---|---|
| research-gate-spec §Deliverables | Fixed15 = entry + 15 sessions; stop chỉ là sizing reference; unmatured giữ open | `period_backtest.py:256,263-271` đúng; không có label builder fixed15 nào | MATCHES (simulator) / NOT-COVERED (label) |
| research-gate-spec §Deliverables | "Every variant shares … costs and execution" | Đúng giữa các variant trong gate; sai giữa gate và training labels (bảng trên) | EXTENDS |
| research-gate-spec §Engineering acceptance | RED/GREEN cho fixed hold, accounting, causality | Không có test parity label↔Broker; `tests/test_mr_exit.py` chỉ test `simulate_mr_exit` (T+2) | NOT-COVERED |
| strategy-reassessment §Risks 1 | "MR UI/canonical exits use a different settlement lock; gap-stop handling differs" | Xác nhận và mở rộng: T+2 trong `mr_exit.py:20-52`, `win_probability.py:40`, `forward_test.py:40`, fast builder; T+3 trong Broker; gap fill khác | MATCHES + EXTENDS |
| strategy-reassessment §Risks 3 | Stop không đảm bảo exit khi limit-down; 6,9% gate dùng adjusted close | Fast builder clamp thay vì đánh dấu unfillable (`parallel_engine.py:290,316`) | EXTENDS |
| strategy-reassessment §Risks 4 | Giá là adjusted vintage hiện tại | Label rows không ghi price vintage hash; `rules_version` (`parallel_engine.py:224`) không gồm cost/calendar | EXTENDS |
| ml-plan-review-features H-P01 | Baseline là neutral-fill theo feature + winsor 1/99, không phải zero-fill | Thêm: regime fill `fillna(0)` cho market return/vol (`feature_engineering_v2.py:127-134`), rs_ratio=1, rs_slope=0, breadth 0,5 (`:175-183`) trộn missing với observed | MATCHES + EXTENDS |
| ml-plan-review-features dòng 117-120 | Registry thay prefix allowlist | Hiện vẫn prefix `rule_`/`feature_` (`calibration.py:99-104`) | MATCHES (chưa sửa) |
| ml-plan-review-features dòng 163-186 | Cùng implementation label/simulator, định nghĩa 15-session | Ba implementation exit khác nhau: `mr_exit`, vòng lặp nhân bản `parallel_engine.py:298-312`, Broker; `mr_exit.py:7` tuyên bố "all four call" nhưng fast builder không gọi | CONFLICTS (với docstring code) / EXTENDS |
| training-remediation §Dataset | Lưu exit thật, exit dự kiến, độ chín, chất lượng nhãn | Có (`parallel_engine.py:326-331`), nhưng `label_available_date = exit_date` bỏ qua T+3 cash/knowledge time | MATCHES + EXTENDS |
| training-remediation §MR | MR chỉ nhận replay `resolved=True` | Đúng (`win_probability.py:148-150`); nhưng stop/target riêng (`STOP_ATR=3`, kijun, `:141-147`) khác risk plan signal_engine | MATCHES + EXTENDS |
| training-remediation §Backtest | Không coi vị thế chưa kết thúc là realized | `run_backtest` đúng (`backtest.py:336-337`); `run_portfolio_backtest` vẫn `_force_close_positions` FORCED_CLOSE (`backtest.py:467,635-651`) và chỉ check stop/target từ planned exit date (`:620-629`) | CONFLICTS ở portfolio path (đã được gate 6 của audit đó nêu "portfolio forced-close chưa xác nhận") |
| training-remediation §Gate 5 | Builder chung chỉ đảm bảo cùng phép tính khi history giống nhau | Momentum features vẫn nhân bản: `momentum_scan._rank` (`:63-72`), `momentum_events.features` (`:42-59`), `momentum_hypotheses` | MATCHES + EXTENDS |
| momentum-events §Locked mechanics | Broker dùng chung research; T+3; double-cost nhân đôi cả tax | `momentum_events.py:129-132` dùng `bt.Broker` | MATCHES |
| momentum-books §Thiết kế | Next-open, lô 100, T+3, PIT | `momentum_books.py:120-143` qua `bt.replay_momentum` | MATCHES |
| momentum-books/momentum-events | (không có momentum label supervised) | Không có builder 21-session relative return/rank nào; `engineer_targets.label_excess_return` dùng return 1 ngày (`feature_engineering_v2.py:290-291`) | NOT-COVERED |
| execution-receipts §Frozen schema | Không suy settlement từ T+3 research; không bịa fill từ daily bar; absence ≠ zero fill | Fast builder clamp/fabricate fill; label chỉ là research estimate, không phải execution evidence | MATCHES (nguyên tắc) / CONFLICTS (fast path) |

## Fast/slow cost và exit parity gaps

1. **Dead `cost_config`:** `build_labeled_t2_dataset` nhận `cost_config` nhưng không
   truyền (`calibration.py:81-96`); `cli.py:138` truyền `BacktestConfig()` vô hiệu.
   Plan §5 đã nghi vấn; xác nhận đúng.
2. **Bốn dạng slippage:** fast additive (`parallel_engine.py:318`); `run_backtest`
   1 tick + phí, không pct slippage (`backtest.py:345-352`) nhưng vẫn báo
   `round_trip_cost_pct` có slippage (`:367`); portfolio tick (`:663-665,734`);
   Broker multiplicative (`period_backtest.py:125-147`).
3. **Settlement T+2 vs T+3** (bảng trên).
4. **Fill realism:** clamp vs reject; gap-down stop; thiếu `stop<px<target` gate.
5. **Exit loop nhân bản** ở fast builder (`parallel_engine.py:298-312`).
6. **Portfolio backtest** chỉ check bracket khi đến planned exit và force-close cuối kỳ.
7. **Hardcoded cost** trong MR P(win) (`win_probability.py:37`) bỏ qua stress.
8. **Stop/target khác nguồn** giữa `win_probability` và signal_engine risk plan.

## Nơi pending/missing có thể thành 0 hoặc thua

| Vị trí | Hành vi | Rủi ro |
|---|---|---|
| `feature_engineering_v2.py:268` (`engineer_targets`) | Return NaN → 0 → `label_binary=0`, `rr_category=2` | Pending thành loss và "small win" cùng lúc; hiện chưa có caller nhưng được export |
| `ensemble_model.py:227,535` | Return NaN → 0 trong sample weight | Weight sai, nhẹ |
| `temporal_validation.py:37-38` | Chỉ lọc unresolved nếu có cột `label_resolved` | Dataset thiếu cột sẽ lọt row chưa chín theo nghĩa fill |
| `parallel_engine.py:317`, `label_resolver.py:137` | `entry_price=0` → gross 0, net = −cost | Dữ liệu hỏng thành loss nhỏ thay vì quarantine |
| `calibration.py:197-198` | Tập rỗng → win rate/return 0.0 | "Không có mẫu" đọc như "0%" |
| `feature_engineering_v2.py:127-134,175-183` | Missing market data → return 0, vol 0, rs 1, breadth 0,5 | Missing trộn observed-neutral (feature side) |
| `parallel_engine.py:267-269` | Row thiếu exit bị bỏ im lặng | Không đếm coverage; cuối dữ liệu bị cắt regime gần nhất |

## Đề xuất layout R3

```text
stock_agent/research/labels/
  contracts.py   # LabelRecord: symbol, signal_date, entry_date, planned_exit_date,
                 # actual_exit_date, label_end, label_available_at, state
                 # {resolved, pending, unfillable_entry, exit_deferred, quarantined},
                 # entry/exit px, gross, cost_vnd/pct, net, cost_config_hash,
                 # calendar_hash, price_vintage_hash, label_contract_version
  costs.py       # MỘT hàm fill/cost dùng chung với period_backtest.Broker
  mr15.py        # fixed15 (e+15 close, T+3, fill gate) và bracket là target riêng
  momentum.py    # 21-session relative return/rank (monthly); 5-session weekly riêng
stock_agent/research/features/
  registry.py    # ID/version/hash/lookback/availability/missing policy/allowed target
  f0_price.py  f1_market.py  f2_liquidity.py  f3_foreign.py
tests/
  test_r3_label_contracts.py   # pending không có numeric label; e+15 index; unfillable state
  test_r3_cost_parity.py       # costs.py == Broker trên cùng synthetic trade (normal/double_cost)
  test_r3_feature_registry.py  # target column bị reject; future mutation không đổi X
```

MR15 label phải gồm cả candidate selected/unselected (plan §5). Momentum label
là diagnostic standalone; danh mục vẫn đánh giá bằng Broker.

## Bước bounded đầu tiên

Viết RED test `test_r3_cost_parity.py`: một synthetic trade (gap-down, limit-up
entry, near-breakeven) qua `period_backtest.Broker` và qua hàm label hiện tại →
mong đợi FAIL, chứng minh skew. Kèm RED test: fixed15 với horizon vượt dữ liệu phải
trả `state=pending`, không có `net`. Phụ thuộc R1: enum missing/state và
`label_available_at` schema; trước khi R1 xong chỉ stub interface, không chốt field.

## MLSys evaluation

Nguồn: skill `machine-learning-system` (Reddi et al., bản 20/10/2025); đọc đủ
`chapters/05-ai-workflow.md`, `06-data-engineering.md`, `13-ml-operations.md`;
`08-ai-training.md` chỉ dùng phần reproducibility. Trích qua
`scripts/read_source.py`, "src L" là dòng trong `assets/book.txt`.

| Phát hiện | Chương / đoạn sách | Quan hệ | Rủi ro bổ sung sách chỉ ra |
|---|---|---|---|
| Thiếu feature registry (`calibration.py:99-104`) | 13 §13.4.1.2 Feature Stores: "feature engineering logic is duplicated, manually reimplemented, or diverges across environments" (src L55995); 06 §6.5: ghi "dataset checksums, transformation parameter values… code version hashes" (src L18920-18922) | REFINES: ủng hộ registry có hash; summary ch.06 nói feature store không bắt buộc mọi dự án → registry nhỏ trong repo | Momentum feature nhân bản 3 nơi đúng là "duplicated" debt |
| Neutral fill trộn missing/zero | 06 §6.7: xử lý missing "using rules that can be applied consistently" (src L19827); vocabulary "persists it with the model" (src L19902) | SUPPORTS fold-fit serialize; REFINES: sách chỉ đòi nhất quán, ngữ nghĩa missing≠zero đến từ plan §4/gate spec | Fill nhất quán vẫn có thể sai ngữ nghĩa; parity test không bắt được |
| Pending → loss/0 | 06 §6.8.1: "track which model versions used which label versions" (src L20391-20393); 06 §6.4: đưa record lỗi vào dead letter queue "rather than discarding them entirely" (src L19749-19751); 13 nhắc "delayed labeling" (src L55805) | SUPPORTS state explicit + quarantine có coverage; sách không có lý thuyết censored label tài chính | Drop row cuối dữ liệu gây bias regime gần nhất |
| Fast/slow cost-exit skew | 06 §6.4: skew khi training "compute features in batch using one set of… logic, while serving… different implementations" (src L18669-18676); 06 §6.7: "Every transformation… must be applied identically in both contexts" (src L19797) | SUPPORTS, mở rộng sang target: label/evaluation/serving dùng 3 hệ settlement-fill | Config hardcode trong code (`COST=0.60`) trái nguyên tắc release = model+features+config (ch.13) |
| Label quality / corporate action | 06: point-in-time "feature values as they existed at specific points in time" (src L21417-21425) | SUPPORTS quarantine + PIT; REFINES: giá adjusted vintage nên label cần price vintage hash | Lineage label thiếu cost/calendar hash |
| Layout + parity test | 05 §5.4: gate giữa stage, "emergent complexity"; summary ch.13: CI/CD phần mềm chưa đủ cho data/model validation, manifest + check nhỏ là đủ khi đáp ứng rủi ro; ch.08: lưu data/config/code version | SUPPORTS manifest + contract tests, không cần platform | Constraint propagation: T+3/limit-lock phải đi ngược vào định nghĩa label → R1 schema phải mang `label_available_at` và fill state |

Số liệu trong sách (ví dụ skew "5-15% accuracy degradation", src L18674) là case
statistic, không là baseline cho hệ này.

## Thay đổi nào cần preregistration research gate

Theo AGENTS.md, đổi labels/exits/costs/data contract phải đăng ký trong
`configs/research/research_gate_v1.json` (version protocol khi đổi thiết kế)
**trước** khi xem kết quả mới:

- **Cần prereg (data-contract hypothesis):** hợp nhất `costs.py` với Broker cho
  training labels; đổi settlement label sang T+3; thay clamp bằng unfillable state;
  gap-down `min(stop, open)` trong `mr_exit`; bỏ hardcode `COST`. Expectation phải
  ghi: replay economics của gate **không đổi** (Broker giữ nguyên), chỉ training
  labels/P(win) đổi; không promotion.
- **Cần prereg (strategy hypothesis):** dataset MR15 meta-label và momentum
  21-session/5-session target (H-MR01, H-MO01, H-MO02); horizon chốt trước.
- **Không cần prereg (pure engineering, không đổi kinh tế):** RED parity test,
  registry skeleton, test pending exclusion, sửa `engineer_targets` fillna thành
  fail-closed khi chưa có caller, ghi coverage row bị drop. Vẫn phải chạy test.
- Sửa `run_portfolio_backtest` forced-close/bracket timing chạm trading/backtest
  path: tách task riêng, cần user chấp thuận rõ.

## Verification / handoff

Read-only: `grep`, `sed`, `wc`, `git rev-parse` và `read_source.py` của skill.
File duy nhất được tạo: tài liệu này. Không test, backtest, training, commit.
Blocker: R1 schema (state enum, `label_available_at`, price vintage). Bước kế
tiếp: coordinator duyệt và quyết định đăng ký data-contract hypothesis cho cost
parity trước khi viết RED test.

## Cập nhật sau merge research_ml (06/10, 4403fc7)

Bối cảnh: R3 đã có trong `stock_agent/research_ml/labels.py` (87 dòng) và `features.py`.
Ledger ghi IMPLEMENTED cho F0/F2/F3 + MR15/mom_5/21/63
(`docs/plans/2026-10-04-data-ml-research-plan.md:399`). Legacy (`parallel_engine.py`,
`calibration.py`, `win_probability.py`, `mr_exit.py`, `backtest.py`, `period_backtest.py`,
`feature_engineering_v2.py`, `ensemble_model.py`, `temporal_validation.py`) không đổi giữa
`4b859cd` và `4403fc7`. research_ml **không** gọi `period_backtest.Broker`.

| ID | Trạng thái | Evidence @ `4403fc7` / ghi chú |
|---|---|---|
| Verdict "R3 chưa thể bắt đầu trước R1 schema" | CHANGED | R3 đã implement trên `research_ml/contracts.py`; `label_row` contract vẫn chờ port R1 |
| Bảng fast/slow: phí + slippage (multiplicative Broker vs additive) | STILL OPEN | `CostConfig` cộng dồn `2c + tax + 2s` = 0,60% (`labels.py:256-264`), net = gross − round_trip (`:309-310`). Test parity chỉ so với `round_trip_cost_pct` của fast path (`test_ml_features_labels.py:103`), không so với Broker. Như vậy research_ml lặp lại skew của fast builder thay vì gỡ nó |
| Settlement T+2 vs T+3 | STILL OPEN (legacy) / N/A cho label fixed-horizon | research_ml không có lock (exit cố định e+h, h ≥ 5). Không mô hình cash settlement: lane weekly vào lot mới ở open của chính phiên lot cũ thoát (`configs/research/ml_protocol_v2_weekly.json:19`). Lane này cần NAV replay R7 trước khi diễn giải |
| Fill entry: clamp vs skip/limit gate | STILL OPEN (một phần) | research_ml chỉ có `entry_unfillable` khi volume ≤ 0 (`labels.py:299-301`), không có gate trần/sàn hay `stop<px<target`. Legacy clamp `parallel_engine.py:290` không đổi |
| Fill exit | RESOLVED một phần | `exit_unfillable` khi volume ≤ 0, không bịa exit (`labels.py:306-308`). Khác Broker (Broker giữ lot và thoát sau); chưa kiểm limit-down |
| Gap-down stop | STILL OPEN (legacy) | research_ml label không có stop |
| Exit horizon e+15 | RESOLVED | `labels.py:240-242,295-305`; test `test_ml_features_labels.py:62` |
| Exit policy: label fixed15 | RESOLVED | `PROTOCOLS["mr_fixed15"]` `labels.py:253` |
| Pending | RESOLVED (research_ml) / STILL OPEN (legacy) | `labels.py:296-304,314-316`; test `:74`; R5 đếm pending (`2026-10-04-r5-weekly-results.md:12`). Legacy vẫn drop im lặng (`parallel_engine.py:267-269`) |
| Double-cost stress | RESOLVED (research_ml, per-lot) / STILL OPEN (`win_probability.py:37` `COST = 0.60`) | `ml_protocol_v2_weekly.json:116`. Mọi excess đều âm hoặc ≈0 khi cost ×2 (`r5-weekly-results.md:16-25`) |
| gate-spec Engineering acceptance: test parity label ↔ Broker | STILL OPEN | Không có test nào |
| reassessment Risks 1 (T+2 ở `mr_exit`/`win_probability`/`forward_test`) | STILL OPEN | Không đổi |
| reassessment Risks 3 (clamp khi limit-down) | STILL OPEN | Như dòng fill entry |
| reassessment Risks 4 (label thiếu hash price vintage/cost/calendar) | STILL OPEN (cả research_ml) | Row label (`labels.py:282-286`) không có hash; bundle chỉ có `manifest_sha256` cấp run (`source_manifest.json`) |
| features H-P01: regime fill | STILL OPEN (legacy) | `feature_engineering_v2.py` không đổi |
| Registry thay prefix allowlist | RESOLVED (research_ml) / STILL OPEN (legacy) | `features.py:358-380`; `calibration.py:99-104` không đổi |
| Ba implementation exit nhân bản | STILL OPEN | research_ml thêm implementation thứ tư (`labels.py:273-317`) |
| remediation: `label_available_date = exit_date` | STILL OPEN (legacy) | `parallel_engine.py:329`. research_ml dùng exit close 16:00 VN (`labels.py:311`) |
| remediation MR: stop/target riêng | STILL OPEN | `win_probability.py` không đổi |
| remediation Backtest: portfolio force-close | STILL OPEN | `backtest.py` không đổi |
| remediation Gate 5: momentum feature nhân bản | STILL OPEN | Thêm bản thứ tư `f0_mom_126_21` (`features.py:390`) |
| Label momentum supervised 21/5 phiên | RESOLVED | `mom_5/21/63` (`labels.py:253`); test `:95` |
| Gap 1: `cost_config` bị bỏ qua | STILL OPEN | `calibration.py:81-96`, được pin bằng characterization test `test_ml_features_labels.py:107` |
| Gaps 2, 4, 5, 6, 7, 8 | STILL OPEN | Legacy không đổi |
| Bảng "pending/missing → 0 hoặc thua" (7 vị trí) | STILL OPEN | Cả 7 vị trí đều là legacy, không đổi |
| Layout `stock_agent/research/labels/...` | SUPERSEDED | Thay bằng `research_ml/labels.py`, `features.py`; không có `costs.py` dùng chung với Broker |
| MR15 giữ cả candidate selected/unselected | RESOLVED | Cột `selected` (`labels.py:284`); test `:74` |
| Bước đầu: RED parity test với Broker | STILL OPEN | Chưa viết |
| Bước đầu: fixed15 vượt dữ liệu → pending | RESOLVED | `test_ml_features_labels.py:74` |
| Preregistration | CHANGED | Thí nghiệm ML đăng ký ở `configs/research/ml_protocol_v1.json` / `ml_protocol_v2_weekly.json`, không phải `research_gate_v1.json`. Giả thuyết data-contract cho cost parity vẫn chưa đăng ký |

Còn actionable:
1. Viết RED test so `research_ml.labels.build_labels` với `period_backtest.Broker` (gap-down, limit-up entry, trade gần hòa vốn), rồi quyết định có chuyển label sang cost multiplicative và gate trần/sàn hay không. Phải đăng ký trước.
2. Legacy: `cost_config` vẫn bị bỏ qua, T+2 vs T+3, clamp fill, `COST = 0.60` hardcode, force-close. Cả nhóm cần task riêng có user cho phép.
3. Kết quả weekly âm sau cost ×2 cho thấy cost là ràng buộc chính. R7 NAV replay với Broker T+3 là bằng chứng còn thiếu.
