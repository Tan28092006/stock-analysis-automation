# Kế hoạch dữ liệu → tiền xử lý → ensemble → bằng chứng

Ngày lập: 04/10/2026. Phạm vi: backend/CLI local, MR/bắt đáy và momentum, VN30
và VN100 khi dữ liệu PIT đủ. Đây là kế hoạch triển khai/nghiên cứu, **không phải
kết quả model mới, cam kết thắng hay giấy phép giao dịch**. Tối ưu onboarding
agent được thực hiện trước; runtime giao dịch, lịch refresh, model và ledger
không thay trong bước lập kế hoạch này.

## 1. Mục tiêu, nền hiện có và giới hạn

Ưu tiên người dùng: lợi nhuận ròng dương, vượt VNINDEX và đủ cơ hội mở mới/mua
thêm sau phí. Ba điều kiện độc lập; thua ít hơn VNINDEX vẫn là lỗ. Không dùng
win rate/AUC cao để thay cho NAV có vốn, khả năng khớp và độ ổn định theo regime.

Inspection baseline: commit `6c3b8f422dd522ac0e4a892f23319e4a431b1ef7`.
Các phát hiện dưới đây là kiểm tra code, chưa phải ablation mới:

| Nền hiện có | Bằng chứng / hệ quả thiết kế |
|---|---|
| Snapshot raw/canonical và manifest hash | `stock_agent/data/reconciliation.py`; tái sử dụng, không dựng lại collector |
| Khối ngoại schema v2, raw vintages, availability, quarantine | `stock_agent/data/foreign_flows.py`; không hồi tố timestamp hoặc cho legacy đủ điều kiện |
| Split theo ngày, purge exit/optional label-availability date | `stock_agent/features/temporal_validation.py`; hiện date granularity, availability field optional; R4 cần mandatory timestamp contract |
| Bốn base: LightGBM, XGBoost, Ridge, CatBoost | `ensemble_model.py::train` hiện là voting/simple mean; tên/docstring stacking không chứng minh meta model đã được fit |
| Preprocessing hiện tại: neutral-fill theo feature → clip quantile 1/99 fit train → estimator transforms | `_prepare_features` / `calibration.py::preprocess_features_robust`; không phải RobustScaler. **Đính chính 06/10:** fill có hiệu lực xảy ra trước ở ba điểm upstream (`indicators.py`, `feature_engineering_v2.py`, `feature_store.py::_safe_float`); MR `win_probability.py:285` là blanket `fillna(0)`; ensemble mặc định 0.0 ngoài 12 prefix — xem [preprocessing 06/10](../audits/2026-10-06-preprocessing-feature-engineering.md) C2–C3 |
| Target mặc định `net_t2_win` / `net_t2_return_pct` | Không tự coi là target phù hợp MR15/momentum; cần label protocol riêng |
| Ridge score đi qua sigmoid; missing base có fallback 0.5 | Không tự gọi là xác suất đã calibrate; thử calibration riêng và fail-closed khi thiếu component |
| Release candidate với hash, `live_approved=false` | `model_release.py`; không bật ML/override hay thay production artifact |
| Pinned fixtures, gate/books/timeslices/VN100/event CI | Giữ toàn bộ; bổ sung ML evidence lane, không thay bằng mỗi training score |

Đọc cùng [gate spec](../audits/2026-09-30-research-gate-spec.md),
[strategy reassessment](../audits/2026-09-30-strategy-reassessment.md),
[system review](../audits/2026-10-04-system-design-review.md) và
[fixture contract](../../tests/fixtures/research/README.md).
Snapshot hiện là giá adjusted ở vintage thu thập, không được chứng nhận lịch
sử corporate actions/PIT toàn phần. Lịch sử đã xem là development, kể cả H1
2026. Boundary 01/10 trong tài liệu cũ không tạo holdout mới cho kế hoạch 04/10;
forward start phải sau khi freeze và trước phiên chưa quan sát thực sự.

## 2. Kiến trúc đề xuất: một backend, ranh giới rõ

```text
Source adapters -> immutable raw vintages + ingestion receipts
               -> validation/quarantine -> canonical as-of tables + catalog
               -> dataset builder (decision-time joins, PIT eligibility)
               -> X/features     |     y/labels + maturity (tách biệt)
               -> date folds -> fold-fitted preprocessing -> base learners
               -> forward OOF -> optional blend/stack -> calibration
               -> frozen predictions -> executable portfolio -> evidence bundle
               -> candidate-only release -> prospective shadow (nếu được yêu cầu)
```

Ưu tiên filesystem raw + canonical hiện có và SQLite catalog/projection có
transaction, unique keys và rebuild từ raw. Chỉ dùng Parquet cho bảng lớn nếu
benchmark chứng minh giảm I/O và dependency được chốt; không bắt buộc thêm ngay.
Không mặc định Kafka, cloud, GPU, microservices hoặc agent tự tối ưu live.
Giới hạn workers/threads cho từng run tránh bốn model cùng dùng mọi core.
Đo wall time theo stage, peak RAM, tốc độ build dataset, thời gian đạt mức chất
lượng cố định; không lấy samples/sec thay chất lượng kiểm định.

## 3. Tổng hợp dữ liệu và source contracts

### 3.1 Grain, identity và thời gian

| Bảng/projection đề xuất | Grain / keys | Yêu cầu |
|---|---|---|
| `ingestion_attempt` | Một attempt ID / receipt | Request, provider/version, fetched/committed UTC, status, retry, references tới raw objects; không lưu secret |
| `raw_object` | Một content SHA256 | Bytes immutable; nhiều attempts có thể tham chiếu cùng content, không gộp observed times/receipts vì trùng hash |
| `price_bar_vintage` | instrument × exchange session × provider × vintage | OHLCV, loại raw/adjusted, unit, session completeness, availability và revision lineage |
| `foreign_flow_vintage` | instrument × session × investor × trade scope × provider × vintage | Buy/sell value VND, volume shares, original epoch, availability; không trộn matched/put-through/unknown |
| `instrument` / `membership_version` | Stable instrument ID; một membership effective interval và knowledge interval | Listing/delisting, symbol changes; cả hiệu lực thực và lúc biết, không hồi tố rổ hiện tại |
| `calendar_version` / `reference_price` / `corporate_action` | Exchange-session; instrument-session; một action version | Phiên, reference/limits, split/dividend/effective/publication; nguồn thiếu thì BLOCKED ở claim liên quan |
| `feature_row` | instrument × decision time × feature-set version × dataset vintage | Input hashes, availability max, missing reasons, eligibility, row lineage |
| `label_row` | Cùng row identity + target/exit/execution protocol | Entry/exit/event end/label available, cost, status; pending không là loss/zero |

Lưu `event_time`, `effective_from/to`, `published_at` nếu nguồn cung cấp,
`observed_at`, `available_at`, `ingested_at` và vintage. Timestamp có timezone,
session date theo lịch Việt Nam. Historical effective date không phải historical
availability; thiếu bằng chứng public-at thì dùng bound bảo thủ hoặc loại khỏi
thử nghiệm PIT. Hai trục valid-time/knowledge-time cho revisions và membership.

Knowledge-time/bitemporal trading schema là mở rộng của dự án: Kimball SCD2
hỗ trợ lịch sử effective/expiry, không tự cung cấp lúc thị trường biết thông
tin. Dùng half-open intervals; tại một knowledge cutoff và valid time chỉ có
một version được chọn cho mỗi business key/source scope. Effective intervals
trùng giữa revisions có knowledge intervals tách biệt không mặc định lỗi.
Freeze source/revision priority và tie-break; reject những version thực sự mơ
hồ cùng cutoff, không blanket reject mọi effective overlap.

Mỗi as-of join phải có `available_at <= decision_time`, interval hợp lệ và
cardinality đã khai báo; reject ambiguous versions-at-cutoff/fanout. Quy đổi unit bằng
source contract rồi kiểm tra trị số; không suy đoán unit/date vì correlation
đẹp. Missing ≠ zero, stale ≠ fresh; observed zero là trạng thái riêng. Không
ghép chắp vá provider/vintage để cứu missing mà thiếu policy và lineage.

### 3.2 Thu thập/cập nhật và completeness

- Giữ collector riêng signal viewing; không sửa lịch đang chạy khi chưa được
  yêu cầu. Khi triển khai, chuẩn hóa receipt `started/succeeded/partial/failed`,
  watermark theo provider/scope và danh sách expected sessions/symbols.
- Fetch pagination có proof coverage; bounded retries/backoff, idempotency trên
  request/content, không overwrite raw. Correction tạo vintage mới và rebuild
  projection idempotently; duplicate không cộng flow hai lần.
- Publication protocol đề xuất: stage immutable generation → persist/validate
  raw, canonical, manifest và hashes → commit generation/readiness/watermark
  trong một SQLite transaction làm visibility boundary duy nhất. Reader chỉ
  dùng generation catalog đã publish và hash-verify files. SQLite không rollback
  filesystem writes; fsync/rename không tạo transaction xuyên hai stores. Crash
  trước catalog commit để lại orphan không visible; crash sau commit gặp missing/
  corrupt files phải fail-closed và reconcile, không trả partial-ready. Test từng
  crash point, concurrent readers/writers và recovery theo fault model thực.
  Đây là thiết kế cần triển khai/test, không guarantee đã có.
  Transport success không bằng dataset ready. Report thiếu phiên, missing,
  late, quarantined, unit errors, duplicate và revision riêng.
- Khối ngoại: giữ quarantine legacy; verified current rows không cứu lịch sử
  thiếu provenance. Rolling N phiên bám exchange-session skeleton, giữ missing
  slots, present/expected coverage và min-support policy đã freeze; không lấy
  N observed rows rồi gọi N trading sessions. Chỉ observations verified đóng
  góp aggregate; partial window không là complete window, missing không là zero.
  Period aggregate chưa kết thúc không được gán về đầu tháng/quý.
- Migration catalog transactional phải có crash/restart/lock/replay tests và
  backup-restore round trip. Giữ raw/manifest cũ; Git không backup runtime data.
  Chốt RPO/RTO/retention và coherent recovery scope raw/catalog/manifest/metadata
  trước operational acceptance, rồi đo restore; các trị số hiện chưa được chốt.
- Tính hash dependency closure của builder/calendar/universe/features/costs,
  không chỉ một hàm scan. Training environment gồm Python, package versions,
  OS/threads/seeds và nondeterminism; broad `requirements.txt` chưa là lock.

### 3.3 Ưu tiên nguồn

P0: OHLCV/VNINDEX, lịch/PIT membership, foreign v2 và audit reference/adjustment.
P1: sector lịch sử, index breadth đủ coverage, official references/corporate
actions. P2: fundamental có publication/revision timestamps; ETF events,
rates/FX, news với archival availability. P2 hiện là research backlog,
không tự thêm sentiment vào technical-only decision path. Nguồn nào thiếu
availability/coverage thì ghi BLOCKED, không tạo feature zero.

## 4. Feature và tiền xử lý: registry trước, fit theo fold

Mỗi feature đăng ký: ID/version, owner, hypothesis, unit/grain, source scope,
formula/code hash, lookback/min history, availability rule, missing policy,
normalization và allowed target/strategy. Không dùng tên cột `feature_` như
allowlist duy nhất; target/future return/exit/P&L không vào X.

| Nhóm | Hướng kiểm định | Ablation / điều kiện |
|---|---|---|
| F0 price/volume | Momentum formation nhiều horizon, short reversal, high proximity, volatility/drawdown, volume anomaly | Baseline vs từng family; cố định horizon trước kết quả, warmup causal |
| F1 market/regime | Relative strength VNINDEX, breadth PIT, trend/vol regime | Cùng universe đủ coverage; regime không dùng tương lai để gán đỉnh/đáy |
| F2 liquidity/execution | ADV/value, gap/limit risk, turnover/cost proxies | Test sau phí/participation stress; OHLCV proxy không được gọi là order-book spread/depth thực |
| F3 foreign | Lagged net value/ADV, buy-sell imbalance, persistence, price-flow interaction | Daily verified availability; without-F3, lag +1/+2; missingness-only control, không chọn dấu từ kết quả |
| F4 sector/fundamental | Relative sector strength/rotation, publication-time quality/value | BLOCKED tới khi có sector/publication history; không dùng current sector hồi tố |

Không mặc định tất cả feature hữu ích; redundant RSI/returns/EMA có thể là cùng
một thông tin. Cluster correlation, conditional permutation và drop-family
được fit/thiết kế ở training development; SHAP chỉ giải thích model, không chứng
minh tín hiệu nhân quả/edge. Permutation phải giữ cấu trúc thời gian và cross-
section phù hợp, không shuffle toàn bộ để tạo null quá dễ.

Preprocessing theo model:

1. Numeric invalid/inf, missing, stale, not-applicable và insufficient warmup
   phải được phân biệt trước imputation. Kiểm thử `0` quan sát vs missing.
2. Linear: train-fold median + missing indicators, scaler; thử robust scaling
   và Yeo–Johnson riêng từng giả thuyết. Tree: baseline hiện có vs native missing
   nếu estimator hỗ trợ và integration test chứng minh cùng semantics.
   Indicators lấy từ original missing mask **trước** imputation. Freeze giữ/bỏ
   winsorization và thứ tự clip/scaler/power của từng trial; không đồng thời đổi
   clipping + imputer + scaler rồi gán gain cho một transform. Outlier thị trường
   có thể hợp lệ: không tự xóa panic tails vì muốn distribution đẹp.
3. Winsorization/imputer/scaler/power/correlation selection/category vocabulary
   **fit training fold**, serialize và chỉ transform validation/test. Không
   refit ở inference; schema mismatch hoặc component thiếu fail-closed.
4. Cross-sectional rank tại decision date chỉ dùng thành viên/giá khả dụng lúc
   đó. Nó khác scaler fit tương lai; coverage/denominator được ghi mỗi phiên.
5. Category unknown/new symbol phải có policy; sector chỉ khi PIT chứng minh.
   Không mặc định symbol identity làm feature; test stock-identity shortcut và
   leave-group diagnostics không thay date holdout.
6. Không SMOTE/shuffle xuyên thời gian. Class/sample weights fit development,
   xem weights theo overlap/concentration; đếm unique days/events chứ không coi
   hàng nghìn stock rows là hàng nghìn quan sát độc lập.

Test preprocessing phải có constant/near-zero-spread columns, negative/zero
numeric domains, missing-mask preservation, unseen levels và batch/single-row
parity. Native-missing/encoder behavior cần official estimator docs đúng version
trước implementation; tên model không là chứng minh mọi loại missing được hỗ trợ.

FEAZ guides có nội dung robust scaling/Yeo–Johnson/missingness/correlation;
nhiều mục time-series và models chỉ là draft/stub. Các quy tắc temporal purge,
financial labels, OOF stacking ở đây là thiết kế dự án + contract code hiện có,
không được gán thành kết luận sách FEAZ. Trước implementation phương pháp cụ
thể, đọc/xác minh chương gốc tương ứng, lưu citation và biến thể đã dùng.
Source review đã kiểm trực tiếp lag/rolling/models là WIP; missing-simple,
missing-indicator, correlation-filter và unseen tuy có TODO/draft vẫn có cơ chế
thực chất. Không đồng nhất draft với empty. Nguồn cụ thể và attribution ở §12.

## 5. Labels và portfolio semantics riêng cho MR / momentum

**MR:** trước tiên conditional meta-label trên đúng trigger rule-only đã freeze;
giữ tập candidate cả selected/unselected để đo selection bias. Primary fixed15:
signal EOD → next eligible open → exit close `entry + 15 trading sessions`,
net theo protocol phí/lot/settlement. Bracket label là target khác, không trộn
với fixed15. Exit không khớp/suspended/limit-locked dùng explicit state, không
fabricate exit. Open/pending labels không train. All-stock MR propensity là
hypothesis riêng; không lấy rare-trigger model làm universal daily classifier.
Counting convention giữ đúng gate hiện có: entry session index `e`, exit close
index `e+15`; không tự đổi thành inclusive 15 sessions (`e+14`). Labels và
simulator dùng cùng calendar/index tests.

**Momentum:** baseline rank rule → supervised relative-return/rank overlay với
21-session target phù hợp monthly lane; weekly lane thử target 5-session như
hypothesis riêng. 63-session target là extension sau, không chọn horizon tốt
nhất trên test. Tách daily candidate ranking khỏi rebalance/add-on order policy.
Net standalone lot labels chỉ diagnostic; tổng danh mục phải tính vốn, giữ
buffer, cash/shares settlement, turnover, participation và concentration.

Dataset builder dùng execution/cost config thực sự truyền xuyên đường fast/slow;
đặc biệt audit API `build_labeled_t2_dataset` hiện nhận `cost_config` nhưng
delegation không truyền tham số đó trực tiếp. Đây là mục cần trace/test, chưa
kết luận mọi label hiện sai. Không đổi định nghĩa nhãn chỉ để tăng win rate.

`label_end` / `label_available_at` là actual path end/knowledge time, không suy
ra một horizon cố định thay giá chưa có. Corporate-action ambiguous rows
quarantine và report coverage bias. Label cost/entry date/exit/share lots phải
khớp simulator dùng đánh giá; MR15 stop sizing reference không là active loss cap.

## 6. Split, model availability và ensemble research

### 6.1 Protocol temporal

- Group toàn bộ instruments cùng signal date vào cùng fold. Expanding/rolling
  date windows preregister; train chỉ gồm labels đã available trước boundary.
  Purge theo actual overlapping label interval, không arbitrary 90-day purge
  vì chỉ báo nhìn quá khứ. Lookback cần warmup; dependence embargo là stress
  riêng được freeze, không thay availability purge.
- Development gồm nested inner folds chọn preprocessing/HPO; outer forward
  folds đánh giá pipeline đã freeze. Calibration có đoạn temporal riêng đã
  mature; test không chọn family/threshold/ensemble weight. Ghi rows/dates
  bị purge và class counts; thiếu hai classes/ít phiên = INCONCLUSIVE/BLOCKED.
  Trước R5 freeze numerical floors cho independent date/event support, từng
  class/fold/calibration và policy insufficient-power; không chọn floor sau
  outer results. Time-to-quality dùng metric/threshold freeze trên development,
  outer test không là stopping/HPO signal. Các trị số ML-specific chưa chốt.
- Historical simulation ghi logical fit cutoff và nguồn có thể biết lúc đó;
  training chạy hôm nay không biến artifact thành model có thật năm 2022.
  Prospective artifact cần actual fitted/calibrated/evaluated timestamps trước
  decision; giữ kiểm tra `model_available_at` và candidate-only policy.
  Helper hiện tại normalize label times về dates, availability metadata có phần
  optional, chỉ kiểm `trained_at` và label-end dates; chưa check riêng actual
  `calibrated_at` / `evaluated_at`. R4 phải version và bắt buộc fit/selection/
  calibration/evaluation/data availability timestamps và required label
  maturity/provenance, không chỉ tái dùng helper rồi gọi contract hoàn chỉnh.
- Snapshot hiện không archival-PIT certified: date-purged research vẫn phải
  gắn limitations, không tuyên bố leakage-free tuyệt đối từ split alone.

### 6.2 Thứ tự thử model, không tìm combo thắng mọi thứ

M0 rule-only/cash/index/PIT equal-weight → M1 logistic/ridge hoặc regression
đơn giản → M2 từng LightGBM/XGBoost/CatBoost → M3 voting mean cố định →
M4 learned blending/OOF logistic stacking → M5 regime-conditional ensemble
nếu đủ independent dates/events. Ranking model chỉ cho momentum với group
theo ngày; classification calibrated dành target win riêng. Không average
rank/regression/probability khác đơn vị mà không có adapter và benchmark.

True stacking cần forward OOF của từng base **và preprocessing/HPO fit chỉ quá
khứ**. Meta learner chỉ train OOF labels available tại cutoff; không train
trên predictions in-sample của bases. Evaluate meta bằng outer forward fold
chưa dùng chọn base/meta; final deployable bases cùng feature/schema/version.
Calibration ở đoạn tách sau meta fit, trước outer test. Các fold đầu không đủ
history được báo mất coverage, không lấp bằng in-sample predictions.

Lưu base OOF error correlation/disagreement, drop-one-base và seed sensitivity.
Model giống nhau không tự tạo diversity. Fixed mean là control bắt buộc;
stacking phải chứng minh tăng lợi ích sau chi phí/độ phức tạp. Component thiếu
không âm thầm điền 0.5 hoặc reweight; reject bundle hoặc dùng fallback đã
preregister riêng. Không warm-start qua các test đã xem, không daily auto HPO.

Đề xuất budget ban đầu: fixed baseline trước; 20 HPO trials/family trên inner
development, seed cố định, cùng wall-time limit đã đo, lưu cả failed/pruned.
Budget này là proposal phải freeze lúc đăng ký. Không mở thêm search sau outer
test rồi tiếp tục gọi test đó untouched. GPU/deep learning chỉ cân nhắc nếu
tabular baseline và sample/latency profile cho thấy cần thiết.

## 7. Backlog giả thuyết và ablation có bằng chứng

Trạng thái tất cả dòng: **PLANNED**, chưa đăng ký executable experiment và
chưa có metric mới. Khi triển khai, đăng ký trong registry trước new results;
đổi thiết kế gate phải version protocol, giữ nguyên v1 để replay lịch sử.

| ID | Câu hỏi / comparison freeze | Primary evidence | Rủi ro / điều kiện dừng |
|---|---|---|---|
| H-D01 | Canonical as-of vs legacy join | Future-mutation invariance, unit/date/cardinality coverage; paired downstream replay | Không dùng legacy leakage làm lợi nhuận baseline hợp lệ |
| H-P01 | Missing-aware vs current feature-specific neutral-fill + train-bound 1/99 clipping | Fixed-fold calibration + net portfolio + missingness-only control | Provider outage shortcut, coverage shift; blanket-zero chỉ là control riêng nếu preregister, không baseline hiện tại. **06/10:** baseline legacy phải freeze lại theo chuỗi fill thực tế (indicators → v2 → snapshot defaults → preprocess); research_ml `FoldTransform` impute trước winsor (`harness.py:110-116`), ngược recipe đề xuất |
| H-P02 | Robust/Yeo–Johnson riêng vs current preprocess | Linear model fixed folds, leave-period diagnostics, transform hashes | Extra transforms có thể không hữu ích với tree; test chọn sau nhìn kết quả invalid |
| H-F01 | Thêm F1 market-relative/breadth | Without/add-F1, exposure-matched control, regime slices | Market exposure giả dạng stock-selection alpha |
| H-F02 | Thêm F2 liquidity/cost-aware | Net P&L/turnover/fill rejection, cost ×2 và delay +1 | OHLCV không chứng minh queue/auction fill |
| H-F03 | Thêm verified F3 foreign | Without/add-F3; lag +1/+2; breadth/missingness controls | Coverage/availability không đủ thì BLOCKED; không flip dấu hậu nghiệm |
| H-MR01 | Rule MR15 vs conditional simple ML filter | Same candidates/sizing/exit, net NAV, missed opportunities, activity | Rare events thiếu power; không giải bằng overfit complex model |
| H-MO01 | Monthly rule rank vs 21-session return/rank overlay | Same monthly execution, date rank IC + net/excess/turnover | Overlap/sector concentration; IC không là lợi nhuận |
| H-MO02 | Weekly continuation/pullback entry lane vs matched weekly baseline | Flat-to-long/add-on/monthly days, net after fees; same sizing | Không so weekly-vs-monthly rồi gán toàn bộ gain cho feature |
| H-E01 | Từng base / fixed mean / OOF stack | Outer predictions, proper scoring, net paired excess, drop-base | Meta leakage/correlated bases; insufficient days stop |
| H-E02 | Separate MR/momentum vs pooled task model | Same feature availability/target contracts, regime and per-task evidence | Pooled rare MR bị momentum lấn; không mix target units |
| H-R01 | Fixed vs causal regime ensemble weights | Outer/panic/rebound, component weights ex ante, cost stress | Regime hindsight/too few panic episodes; exploratory until đủ sample |

Thực hiện stagewise: contract → simple baselines → từng feature family → từng
model → ensemble. One-family/one-rule ablations trước interactions; không full
Cartesian search hay ghép các winner đã xem. Lưu parent/baseline IDs, tất cả
trials và exploratory failures. Multiplicity bao gồm feature/horizon/model/
threshold/HPO selections; registry mới không xóa unknown lịch sử trial count.

Negative controls: future-data mutation không đổi past X/predictions/orders;
target columns rejected; time-shift availability; permutation phù hợp theo
date blocks; missingness-only model. Positive synthetic fixture phải bắt được
vi phạm cố ý. Không coi chạy unit test synthetic là chứng minh profitable edge.

## 8. Khung thời gian và acceptance

Giữ nguyên mọi required crisis/calendar/control/cost block đã đăng ký và hai
pinned fixtures. Required reporting:

- Từng half-year, đặc biệt **01/01–30/06/2026**; latest-six-month cũ
  01/04–29/09 không thay thế H1. Boundary theo manifest, không giả dữ liệu future.
- Causal VNINDEX up/down/transition episodes theo protocol existing; panic/rebound
  2022, shock/recovery 2025, event tháng 3–4/2026 và các blocks khác giữ nguyên.
  Current full fixture chỉ đủ warmup cho Sep2022 onward; spring2022 cần nguồn
  bổ sung valid, không âm thầm cắt rồi gọi full-crisis pass.
- Cash restart mỗi slice và inherited-portfolio attribution riêng. Event windows
  overlap calendar/episodes không cộng thành independent replications.
- VN100 H1 cùng warmup/cost/execution với VN30; history ngắn không tự hỗ trợ
  training 12-1 trên mọi mã. Delisted/suspended/short-history exclusions có lý do.

Metrics: Brier/log loss/reliability/PR-AUC cho classification, rank IC/date-
weighted performance cho ranking, nhưng acceptance chính là net NAV/excess,
drawdown, exposure, turnover, contribution concentration, uncertainty và activity.
Bootstrap paired **date blocks**, existing 20/40-session stress, family-adjusted
inference/minimum days/trades giữ nguyên; stock rows không tăng độc lập thời gian.

Monthly activity tách flat-to-long, add-on, distinct entry days, first-session
construction và zero-new-entry months. **Activity floor đã được người dùng chốt
04/10**: ≥4 flat-to-long entries và ≥3 distinct entry days mỗi tháng
(`configs/research/ml_protocol_v1.json`); validator hiện chỉ kiểm floor đã freeze,
chưa kiểm floor đã đạt (xem §13). Floor phải freeze trước experiment acceptance; không tự đặt quota rồi ép
mua. Sparse-but-profitable chỉ ghi activity fail/inconclusive theo floor, không
thay bằng average fills/half-year hoặc bỏ điều kiện lợi nhuận dương.

Trạng thái result: FAIL / INCONCLUSIVE / BLOCKED / engineering_complete /
research_pass / prospective_ready phải tách. Research pass không live approval.
Missing input/block/failed accounting/hash drift => BLOCKED/FAIL, không skip
green. Phần execution theo [system review](../audits/2026-10-04-system-design-review.md)
(portfolio parity, durable jobs, transaction stores, references/liquidity/fills)
là prerequisite riêng; nhỏ vốn không làm các thiếu hụt này biến mất.

## 9. Evidence bundle và release boundary

Đề xuất directory mới riêng mỗi run, không overwrite:

```text
data/paper/ml_research/<protocol>/<experiment-id>/<run-id>/
  experiment.json         # frozen hypothesis, parent, trials, acceptance, limits
  source_manifest.json    # raw/canonical/catalog hashes, temporal coverage
  environment.json        # exact code/config/dependency closure, seeds, threads
  dataset_manifest.json   # schema/grain, availability, exclusions, labels
  splits.json             # date boundaries, row IDs, purge/maturity reasons
  transforms/             # per-fold fitted params/schema/hash, not test-fit
  predictions/            # base OOF, meta OOF, calibration, outer test + lineage
  metrics.json            # predictive/system metrics with sample counts
  portfolio/              # NAV/cash/lots/receivables/fills/rejections/monthly activity
  regression/             # gate/books/timeslices/VN100/events + source hashes
  decision.json           # pass/fail/blocked reasons, limitations, reviewer status
```

Names là proposed contract, chưa có command hay file generated cho pipeline ML
này. Validator phải kiểm completeness/hash/reference joins, label availability,
finite metrics, sample floor và gate evidence; `--validate` không được tự tạo
profit/pass. Manifest ghi cả actual command/start/end/elapsed/peak RAM, failed
attempts và external-source caveats. Nếu exact rerun nondeterministic thì có
tolerance được freeze và diagnostic, không gọi seed alone là bitwise guarantee.

Unit/contract CI không network/prod state, dùng fixtures synthetic và pinned
public inputs. Thêm ML training fixture nhỏ với full prediction/transform
lineage, không mock hết model rồi gọi ensemble pass. Expensive full ML eval là
lane riêng giữ artifacts; mọi strategy/data change vẫn chạy toàn bộ suite hiện
có. Không tự sửa CI schedules hoặc public-upload private/model/account data.
Provider usage/redistribution rights phải kiểm riêng trước public artifact upload;
source là dữ liệu thị trường nhìn được không tự chứng minh quyền phân phối.

Release bundle gồm preprocessing + feature schema + label/strategy/execution
config + model components + calibration + dependencies + hashes. Candidate
path immutable, rollback cả bundle tương thích, không chỉ pickle. Shadow chỉ
sau yêu cầu triển khai riêng, freeze trước phiên chưa quan sát và ghi data-
ready/missing-session/pending/matured outcomes. Drift theo data/model/system/
business; cảnh báo không tự retrain/promote. Predictions/picks không phải fills,
paper profitability không chứng minh lãi thực; prospective actual receipts
và accounting/risk/execution readiness là lớp bằng chứng khác.

## 10. Work ledger duy nhất cho roadmap này

| Work ID | Trạng thái 04/10 | Phụ thuộc | Deliverable / done criterion |
|---|---|---|---|
| R0 Agent onboarding | IMPLEMENTED documentation/discovery; verification xem audit bên dưới | Không | Root router, ownership/test map, historical-plan warning, current roadmap; verified links/commands, chưa full app boot/fresh-session outcome benchmark |
| R1 Source/dataset contracts | IMPLEMENTED (engineering, synthetic fixtures; `stock_agent/research_ml/contracts.py`, `tests/test_ml_contracts.py`) | R0; prereg contract change | Schemas/grain/as-of/missing/coverage registry + intentional-leak RED/GREEN tests |
| R2 Canonical aggregation/catalog | PARTIAL (price vintages only: `catalog.py`, `tests/test_ml_catalog.py`; foreign-flow projection, pagination proof and RPO/RTO not done) | R1 | Idempotent ingest/projection, pagination/receipts/revision/restore tests; no legacy promotion |
| R3 Feature and strategy label builders | IMPLEMENTED for F0/F2/F3 + MR15/mom_5/21/63 (`features.py`, `labels.py`, `tests/test_ml_features_labels.py`); F1/F4 BLOCKED; legacy `build_labeled_t2_dataset` still drops `cost_config` (characterization test) | R1–R2 | Versioned F0–F3, separate MR15/momentum datasets, fast/slow cost parity, pending exclusion tests |
| R4 Fold/transform/evidence harness | IMPLEMENTED (`harness.py`, `evidence.py`, `tests/test_ml_harness.py`, `tests/test_ml_evidence.py`); not run on real data | R3 | Purged grouped dates, serialized fold fit, OOF identity and availability tests; bundle validator |
| R5 Baselines / feature experiments | RUN 2026-10-04, result **BLOCKED** by sample floors (`scripts/ml_research_r5.py`, `configs/research/ml_protocol_v1.json`, bundle `docs/audits/2026-10-04-r5-bundle/`, record `docs/audits/2026-10-04-r5-results.md`); no M1 or ablation metric exists. Weekly lane (protocol `ml_research_v2_weekly`, same floors) run 2026-10-04: floors met (2 folds, 80 OOS dates), **INCONCLUSIVE**, rank IC ≈0, shuffled-label control beat the model — see `docs/audits/2026-10-04-r5-weekly-results.md` | R4; activity floor agreed | Registered M0/M1 and one-family trials; retained failures and all pinned/regime evidence |
| R6 Ensemble experiments | PLANNED | R5 sufficient samples | Single bases/mean/true OOF stack; calibration, error diversity/drop-base; no automatic promotion |
| R7 Full research regression | PLANNED | R5–R6 | Full gate/books/slices/VN100/events + accounting/causality, cost/delay/multiplicity/activity decisions |
| R8 Prospective readiness | BLOCKED for trading approval | R7 + execution/data blockers | Separate authorization, frozen candidate, complete expected sessions, matured outcomes/actual execution evidence |

Next bounded implementation (cập nhật 06/10): R5 đã chạy (monthly BLOCKED, weekly
INCONCLUSIVE, shuffled control thắng model); không mở R6. Thứ tự việc tiếp theo,
quyết định đang chờ và các gap còn mở sau audit 06/10: xem **§13**. Original R1 note:

**R1** — thiết kế/test source contract và as-of
dataset eligibility trước khi train hay mở rộng số model. Chốt activity floor
trước R5; thiếu floor không ngăn viết contract/tests, nhưng ngăn activity acceptance.
Update trạng thái tại bảng này và link exact run evidence, không tạo tracker
cạnh tranh trong README hoặc skills. Không đánh dấu R1–R8 done từ kế hoạch này.

## 11. Cách bốn skill ảnh hưởng thiết kế

- **ai-agent-build**: repository record/routing/initialization/artifact guides
  ch13/14/16/27 → root router ngắn, ownership map, safe startup, một work ledger,
  handoff bằng evidence. Không agent hóa research loop tự động hoặc đổi runtime.
- **machine-learning-system** (Reddi, guides §5/6/8/12/13): data–algorithm–infra,
  lifecycle, time-to-quality, reproducibility, monitoring và bundle rollback;
  áp dụng cho local tabular, không bê distributed/GPU case vào repo.
- **feature-engineering-az** (Hvitfeldt, numeric/missing/correlated/pattern guides):
  gốc hỗ trợ robust/Yeo–Johnson train-fit, missing/indicator và correlation/
  unknown-category mechanisms; fold nesting và family ablation là tổng hợp
  skill/dự án. Estimator-specific/native-missing, temporal, stacking và financial
  inference không gán cho models/lag/rolling WIP; kiểm official estimator docs
  khi triển khai và nguồn gốc cụ thể ở §12.
- **data-engineer** (FDE ch7–8, Kimball ch2, DDIA index): grain/cardinality,
  ingest completeness, bitemporal/SCD-style revisions, transactional projections,
  provenance/replay/restore. Canonical trade-data contract là thiết kế dự án,
  không lời hứa provider có archival availability.

Reddi ch12 dùng time-to-accuracy; time-to-quality là tổng quát hóa của skill.
Sách hỗ trợ lineage, controlled rollout, rollback và monitoring; full compatible
bundle rollback và no-auto-retrain/promote là lựa chọn guardrail dự án, không
quy tắc literal của tác giả. Ch13 còn thảo luận automatic retraining triggers;
giữ guardrail bảo thủ ở đây không được kể thành lệnh cấm từ sách.

Nguồn skill được đọc từ các thư mục người dùng chỉ định dưới
`C:/Users/acer/.agents/skills/`; không copy sách hoặc skill library vào repo.
Kiểm chứng bước onboarding/planning tại
[audit thực hiện 04/10](../audits/2026-10-04-agent-readiness-and-ml-plan.md).

## 12. Review đối chiếu nguồn gốc sau khi lập kế hoạch

Firstmate dispatch ba scouts riêng; coordinator kiểm harness và kiểm lại các
correction quan trọng. Review input giữ cùng SHA256 để không trộn findings từ
các bản đang đổi. [Run record và tổng hợp](../audits/2026-10-04-ml-plan-review.md)
lưu frozen/corrected hashes, ownership, check và resolution. Các reports bên dưới
giữ line numbers của **bản trước sửa**, không line numbers hiện tại.

| Phần | Nguồn gốc đã đối chiếu / phạm vi hỗ trợ | Claim matrix |
|---|---|---|
| ML lifecycle/benchmark/MLOps | Reddi snapshot 20/10/2025, ch5/6/8/12/13; original extracted source lines và PDF fingerprint verified. Lifecycle/parity/provenance/quality/resource/rollout mechanisms; không prescribe trading horizons/OOF/alpha | [29 nhóm ML + code](../audits/2026-10-04-ml-plan-review-ml.md) |
| Feature/preprocessing | FEAZ original rendered + 14 QMD ở full commit `8f53af54788b3f0917ed442e0e1bdef13c14c560`; [robust scaling](https://feaz-book.com/numeric-robust.html), [Yeo–Johnson](https://feaz-book.com/numeric-yeojohnson.html), [missing indicators](https://feaz-book.com/missing-indicator.html), [correlation filter](https://feaz-book.com/correlated-filter.html); [models](https://feaz-book.com/models.html) WIP. Live deployment không tự khẳng định cùng commit | [30 nhóm feature/code/attribution](../audits/2026-10-04-ml-plan-review-features.md) |
| Data contracts/history/recovery | FDE 1st ed2022 ch7–8; Kimball 3rd ed2013 ch2 grain/null/SCD2; DDIA 1st ed2017 ch7/11 atomicity/dual-writes/versioned joins/idempotence; exact physical/printed pages và verified source hashes | [17 nhóm data + limitations](../audits/2026-10-04-ml-plan-review-data.md) |
| Agent repo/harness | WalkingLab local snapshot `38ddcd2bf8d65271f668b94e7c875ca1d629d622`, original L03/L04/L06 và advanced templates/SOP; năm câu hỏi/routing/evidence mechanisms, không measured agent improvement | [Coordinator S-HAR](../audits/2026-10-04-ml-plan-review.md#s-har--đối-chiếu-trực-tiếp-nguồn-harness-coordinator) |

Kết luận review: dùng được làm **kế hoạch sau correction**, không validation
pipeline hay hiệu quả giao dịch. Financial purge/nested OOF/multiplicity/date-
bootstrap, portfolio execution và H-* signal hypotheses là project protocols;
review này không xác minh toàn bộ original financial/statistical papers cho
chúng. Mỗi experiment vẫn phải có nguồn phương pháp liên quan, preregistration,
test/accounting/statistical evidence và limitations trước khi chấp nhận. Book
mechanism được hỗ trợ không làm H-* đã chứng minh, không gỡ PIT/execution blockers.

## 13. Trạng thái thật và kế hoạch triển khai sau audit 06/10

Ngày: 06/10/2026, Firstmate run `2026-10-06-ml-roadmap`. Revision `4403fc7`
(`codex/data-integrity-audit` đã fast-forward tới `origin/claude/youthful-hawking-hat1sk`
@`7200137` — R1–R4 `stock_agent/research_ml/`, R5 v1/v2 — cộng fix Windows cho
evidence bundle; ml test set 58 passed bằng `anaconda3/python.exe`). Mục này sắp
thứ tự việc; trạng thái chính thức vẫn ở ledger §10. Không cấp trading/deploy approval.

### 13.1 Input

Bảy audit 06/10 viết ở `4b859cd`, mỗi file có mục cuối "Cập nhật sau merge
research_ml (06/10, 4403fc7)" phân loại RESOLVED / STILL OPEN / SUPERSEDED / CHANGED:
[R1](../audits/2026-10-06-r1-source-dataset-contracts.md),
[R2](../audits/2026-10-06-r2-canonical-catalog.md),
[R3 labels/cost](../audits/2026-10-06-r3-labels-cost-parity.md),
[ML](../audits/2026-10-06-ml-system-verification.md),
[preprocessing](../audits/2026-10-06-preprocessing-feature-engineering.md),
[compute](../audits/2026-10-06-compute-acceleration.md),
[EOD flow](../audits/2026-10-06-data-update-flow.md).
Legacy code (parallel_engine, calibration, ensemble, win_probability, backtest,
EOD) không đổi giữa `4b859cd` và `4403fc7`, nên mọi finding legacy vẫn STILL OPEN.

### 13.2 Quyết định đã có / đang chờ

| ID | Nội dung | Trạng thái |
|---|---|---|
| D1 | Lấy `research_ml` làm gốc R1–R4; R1 song song (`firstmate/r1-source-contracts`) không merge, chỉ port gap | **Đã quyết 06/10** (người dùng) |
| D6 | Activity floor ≥4 entries & ≥3 ngày/tháng; sample floors ≥120 train / ≥40 test dates, ≥50 events/class | **Đã chốt 04/10** (`ml_protocol_v1.json`) |
| D2 | Preregistration cho contract change | Thực tế đang theo protocol ML riêng; `research_gate_v1` chưa sửa — cần người dùng xác nhận |
| D3 | Price-vintage: R5 dùng "provider-adjusted as retrieved" nhưng không có mode có version | Chờ: đề xuất mode có version mang `claims_blocked` |
| D4 | Khôi phục writer cho `prices_hist`/dashboard cache/`forward_test` (đứng từ 21/09) | Chờ; đụng lịch chạy |
| D5 | Đổi logic portfolio backtest (forced-close, stop/target muộn, T+3 cash) | Người dùng **có ý định**; chờ kết quả VeriHarness backtest rồi duyệt thiết kế |
| D8 | Hướng R5 tiếp theo: dừng ML trên universe này vs v3 exposure-matched control | Chờ (R5 weekly: control thắng model) |

### 13.3 Gói việc STILL OPEN, theo thứ tự

| WP | Nội dung bounded | Phụ thuộc | Gate |
|---|---|---|---|
| WP1 | **DONE 06/10** (`0f299f4`..`46a1300`, merged ff, 127 tests): legacy quarantine, `MissingState.INVALID` + `validate_feature_states`, foreign-flow timing/net checks, `membership_as_of` VN30/VN100 với coverage BLOCKED, `validate_label_outcomes`. Còn mở: chưa wire vào R5 runner/`build_foreign_feature`; bug `labels.py:78` (pending giữ `net_return_pct` khi có `data_end`); xác nhận `exit_unfillable`→pending | D1 | Wire-in đổi runtime → quyết định riêng |
| WP2 | R2: foreign-flow vintages + `ingest_issue`/legacy inventory trong `research_ml/catalog.py`, migrate manifest read-only + parity `load_flows`; pagination proof, `revision_event`, `catalog_meta`, incremental verify `read_prices`; hòa giải catalog-boundary vs manifest-authority | WP1 | Test catalog; không đổi collector/lịch |
| WP3 | R3: Broker-parity RED test cho `research_ml/labels.py` (cost nhân, limit gate, exit trì hoãn, T+3); hash price-vintage/cost/calendar trên label row | Preregistration (D2) trước khi sửa semantics | Không xem kết quả mới trước đăng ký |
| WP4 | Legacy labels/backtest: `cost_config` chết, T+2 vs T+3, clamp fill, `COST=0.60`, forced-close, `label_available_date=exit_date`, pending bị drop; mâu thuẫn docstring `mr_exit.py:7` | D5 + VeriHarness backtest | Task riêng có approval; đổi label → research gate |
| WP5 | Preprocessing: winsor trước impute, F0 trên session skeleton, chuẩn hóa `f2_adv_value_20`, persist transform thật | WP3 | Ablation H-P* chỉ khi có lane mà signal thắng control |
| WP6 | R4 hardening: embargo, row-ID hash/purge reasons/class counts trong `splits.json`, persist OOF, `component_matrix` fail khi trùng, validator từ chối placeholder và kiểm activity floor **đạt**; ledger ngày test đã xem qua v1/v2/v3 | Không | Test harness/evidence |
| WP7 | Legacy ML: wire contract v2 vào `model_release`/`ml_models`; ensemble release bypass, evaluate serve threshold, holdout re-cut, calibration; giữ ML override off | WP6 | Không đổi live behavior khi chưa approval |
| WP8 | Compute: profile `mr_signal_bank` (O(T²)), `asof_join`, catalog verify; ghi device/threads vào `environment.json`; GPU chỉ theo tiêu chí audit compute | Không | Profiler không đổi hành vi |
| WP9 | EOD (blocker R8): sửa DEPLOY/RENDER docs, thủ tục quarantine run dir/lock, báo missed session, guard lịch 2027, manifest snapshot atomic + retention, lock/silent swallow ở writer legacy | (docs, guard lịch) không; phần runtime cần xác nhận | Test foreign/paper/reconciliation |
| WP10 | R7 NAV replay (Broker T+3, cost ×2, delay, activity) cho lane nào còn được giữ | D8 | Research gate đầy đủ |

### 13.4 Việc đang chạy ngoài ledger

- VeriHarness (scratchpad): challenge backtest thực tế, T+2, nghiên cứu thời điểm
  thoát lệnh, giải mâu thuẫn `mr_exit.py:7`, khuyến nghị thiết kế portfolio backtest → input cho D5/WP4.
- Swing desk (T10): nghiên cứu bộ lọc trước khi vào lệnh (PASS/WARN/BLOCK), tín hiệu
  DB, sentiment, quy trình 4 agent; chỉ là công cụ tư vấn, không đặt lệnh, phải log
  đề xuất prospective để kiểm chứng. Thiết kế tại `docs/audits/2026-10-06-swing-desk-signal-research.md` khi xong.

### 13.5 Môi trường

- Test: `C:\Users\acer\anaconda3\python.exe -m pytest ...` (msys `python` không có
  pytest; `py` 3.13 thiếu lightgbm/optuna). Không chạy full suite song song (paging file).
- Ổ D: còn ~5 GB (từng về 0 trong phiên 06/10); dọn worktree đã merge trước khi tạo thêm.

### 13.6 Cập nhật 07/10/2026 (cuối phiên Firstmate)

Đã merge local vào `codex/data-integrity-audit` (chưa push), HEAD `72b93de`:
R1 port (`46a1300`), refactor `mr_exit` single path (`cb4601e`), swing desk + Codex skill
(`740b26d`, lô lẻ `08dc727`), kho lịch sử 10 năm (`9c13d17`, store `data/history/20261006T161307Z_6ea2ce75`,
103/107 mã), WP9 EOD hardening (`aa0463f`), bottomfish mode + NAV journal (`a45da17`),
hash độc lập line-ending (`6f7e324`), desk dùng kho 10 năm (`72b93de`).
Kiểm tra pipeline độc lập: [2026-10-07-pipeline-check](../audits/2026-10-07-pipeline-check.md) — PASS có điều kiện.

Còn mở (bounded, chưa làm):
- F1: `review` chưa dùng dữ liệu `refresh --held` cho mã ngoài VN30 (PNJ/TPB hiện n/a).
- F2: các dòng frozen 2026-10-07 mang `desk_code_hash` lai (working tree trộn CRLF/LF trước fix `6f7e324`);
  ghi vào `configs/swing_desk/protocol_amendments.md`, không sửa dòng đã đóng băng.
- O1: `history_store status` nêu store aborted làm headline; O2: README/SKILL còn ghi "lot 100".
- D4 (khôi phục writer `prices_hist`/dashboard), quyết định nới cửa sổ indicator để dùng 10 năm (cần protocol version mới),
  dùng kho 10 năm cho research_ml/R5 (preregistration), danh sách VN30 lịch sử trước 2022 (survivorship).
