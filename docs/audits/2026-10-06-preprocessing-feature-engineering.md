# T6: Tiền xử lý / feature engineering, đối chiếu với các audit trước

- Ngày: 06/10/2026.
- Reviewer: Firstmate run `2026-10-06-ml-roadmap` / T6.
- Source revision: `4b859cd` (branch `codex/data-integrity-audit`). Dirty tree lúc đọc chỉ có `data/pipeline/local_server.log` và `data/pipeline/swing_scan_cache.json` (untracked); hai file này không liên quan.
- R1 target contract được đọc ở worktree riêng `D:/Chungkhoan-r1` (branch `firstmate/r1-source-contracts`, `stock_agent/data/contracts.py`). Đây là target, chưa phải code ở `4b859cd`.
- Phạm vi:
  - Đọc code bằng Read/Grep/git read-only và skill `feature-engineering-az`, kèm các chương gốc feaz-book.com.
  - **Không chạy test, code, training hay backtest. Không commit.** Chỉ tạo file này.
- Verdict: đây là **phát hiện thiết kế/code fact** và đề xuất. Báo cáo không chứng minh edge, lợi nhuận hay readiness, và mọi H-P* bên dưới vẫn ở trạng thái PLANNED.

Giới hạn nguồn FEAZ: các chương gốc `models.html`, `order.html` và `summary.html` chỉ có "WIP". Toàn bộ method page time-series (lag, rolling, expanding…) là stub. Vì vậy báo cáo không gán cho sách bất kỳ quy tắc nào về model-specific preprocessing hay thứ tự transform. Những phần như vậy được ghi là project design hoặc general expertise, khớp kết luận của [S-FE](2026-10-04-ml-plan-review-features.md).

## 1. Audit đã đọc

| Audit | Phần đọc |
|---|---|
| [2026-09-22-architecture-and-leakage.md](2026-09-22-architecture-and-leakage.md) | Toàn bộ |
| [2026-09-22-training-remediation.md](2026-09-22-training-remediation.md) | Toàn bộ |
| [2026-10-04-ml-plan-review.md](2026-10-04-ml-plan-review.md) | Toàn bộ |
| [2026-10-04-ml-plan-review-features.md](2026-10-04-ml-plan-review-features.md) | Toàn bộ |
| [2026-10-04-ml-plan-review-ml.md](2026-10-04-ml-plan-review-ml.md) | Toàn bộ; các dòng liên quan preprocessing là ML04, ML05, ML12, ML14, ML22 |

Nhãn so sánh:
- **MATCHES**: code `4b859cd` xác nhận claim.
- **CONFLICTS**: code đọc được mâu thuẫn với claim, hoặc với cách claim được diễn giải.
- **EXTENDS**: claim đúng nhưng thiếu phần quan trọng.
- **NOT-COVERED**: chưa audit nào nêu.

## 2. Bảng đối chiếu

| # | Claim của audit trước (file / phần) | Nhãn | Bằng chứng code `4b859cd` và giải thích |
|---|---|---|---|
| C1 | Baseline: `preprocess_features_robust` forward-fill trên panel không group theo mã (architecture §2 "P0 — Tiền xử lý và split") | MATCHES (đã sửa) | Không còn ffill. Fill là row-local tại `calibration.py:422-444`. Remediation đúng ở điểm này. |
| C2 | "Dùng giá trị trung lập độc lập từng dòng… Batch và single-row dùng cùng chính sách" (remediation §Hợp đồng mới, dòng Preprocessing) | EXTENDS, kèm một CONFLICT về diễn giải | Parity batch/single-row có thật: training đi qua `parallel_engine.py:347-358`, serving qua `ml_models.py:584-593`, cả hai lấy `signal.features`. **Nhưng imputer thực tế không phải neutral map** ở `calibration.py:407-420`. NaN đã bị điền ở **ba điểm upstream** trước khi tới preprocess (xem F1). Ví dụ: `feature_store.py:59` điền `volume_ratio_20` = 0.0, trong khi neutral map ghi 1.0 (`calibration.py:409`). |
| C3 | H-P01 baseline thực tế là "RSI=50, volume ratio=1, rank/breadth=.5, others 0, then train 1/99 clipping" (S-FE matrix dòng "245", mục Required integration 1; ml-plan-review §Final integration, dòng "Real preprocessing baseline") | **CONFLICTS** | Mô tả này chỉ đúng với nội dung hàm `preprocess_features_robust`, không đúng với dữ liệu mà model nhận. Fill có hiệu lực xảy ra trước đó, ở 3 nơi: `indicators.py:23,102,146,188`; `feature_engineering_v2.py:60-70,127-134,175,177,183`; `feature_store.py:12-15,36-123`. Có giá trị 0, có giá trị `close`, có giá trị 50; ví dụ `vwap20` mặc định `close` ở `feature_store.py:54`. Hàng không finite bị bỏ ở `parallel_engine.py:348,355`. **Baseline H-P01 trong plan §7 cần freeze lại theo chuỗi fill thực tế**, nếu không ablation sẽ so với một baseline không tồn tại. |
| C4 | Imputation là "row-local constants/neutral defaults, **không** train-fit median imputer" (S-ML ML04) | MATCHES + EXTENDS | Đúng. Bổ sung: điểm điền chính nằm ở snapshot, và các upstream fill phá hủy missing reason trước khi R1 `feature_row` có thể ghi nhận. |
| C5 | Winsor bounds fit train-only theo fold (S-ML ML04; S-FE dòng "24"; remediation dòng HPO/Ensemble) | MATCHES | `calibration.py:341-343`; `ensemble_model.py:262-266,277,652-669`; HPO ở `:569-573`. Bounds được fit **sau fill**, nên giá trị fill làm lệch quantile (`calibration.py:446-467`). |
| C6 | Registry ensemble là "49 feature (19 rule + 30 V1)"; chưa chắc artifact chứa V2 (architecture §1 bảng thành phần; §2 "P0 — Feature V2") | MATCHES + EXTENDS | `feature_columns` (`calibration.py:99-104`) trả `rule_*` + `MODEL_NUMERIC_FEATURES` khi có cột ưu tiên, và `daily_runner.py:171,176` dùng đúng danh sách này. **Các cột V2 cross_/regime_/temporal được copy vào row (`parallel_engine.py:352-356`) nhưng không vào X.** Vì vậy các mục neutral cho `feature_cross_*`/`feature_regime_*` (`calibration.py:410-416`) là dead code trên đường mặc định. |
| C7 | "Một số rule có thể tiêu thụ các cột bổ sung" (architecture §2 "P0 — Feature V2") | MATCHES (xác nhận) | `regime_trend` và `rs_slope_5` được rule engine dùng: `signal_engine.py:261-285,359,367`. Default khi thiếu là `regime_trend`=1.0 (bull) và `rs_slope_5`=0.01 (pass). Ngoài ra `np.select` gán 0 (sideways) khi EMA còn NaN (`feature_engineering_v2.py:154-161`). V2 vì thế đi **gián tiếp** vào X qua `rule_*` và `feature_score`. |
| C8 | V2 regime quantile chỉ từ quá khứ, lag một phiên (remediation, dòng V2 regime) | MATCHES | `feature_engineering_v2.py:137-138` dùng expanding quantile và `shift(1)`. Còn lại: `days_since_regime_change` phụ thuộc ngày bắt đầu chuỗi (`:186-189`), chưa audit nào nêu. |
| C9 | MR "Imputation dùng chung ở cả bốn bước" (remediation, dòng MR) | MATCHES + EXTENDS | `win_probability.py:280-285` fillna(0) dùng ở fit, calibration, test và serve. **Chưa audit nào nêu default regime = 1.0 (bull)** khi thiếu ngày VNINDEX, ở cả training (`win_probability.py:184`) lẫn serving (`mr_scan.py:180`). Tương tự `attach_market_regime` fail-open +1 (`market_regime.py:72-77`). |
| C10 | Native-missing cho tree là PROPOSAL, cần docs estimator (S-FE dòng "141–142"; plan §4.2) | EXTENDS | Code fact: LGBM, XGB và CatBoost **không bao giờ thấy NaN**, vì nhận cùng ma trận đã fill và clip (`ensemble_model.py:223-232`, `win_probability.py:226`). Mọi so sánh native-missing là một thay đổi so với hiện trạng. |
| C11 | Ridge sigmoid không phải calibrated probability; base thiếu thì điền 0.5 (S-ML ML05, ML22) | MATCHES | `ensemble_model.py:672-693`. |
| C12 | Sample weights | **NOT-COVERED** | `ensemble_model.py:228` và `:536` dùng `weights = where(y==1, 1, 1+|net_t2_return_pct|)`, tức chỉ upweight lệnh thua theo độ lớn %. Hệ quả: đẩy xác suất xuống, tương tác với `probability_threshold` cố định và với việc thiếu calibration. |
| C13 | OBV/ADL scale | **NOT-COVERED** | `feature_obv_slope_5` và `feature_adl_slope_5` nằm trong `MODEL_NUMERIC_FEATURES` (`calibration.py:40-41`). Đây là `diff(5)` của chuỗi cumsum theo đơn vị cổ phiếu (`indicators.py:66-75,212-215`), nên lệch nhiều bậc độ lớn giữa các mã và trôi theo thời gian. Đó là scale/identity leak; clip 1/99 pooled không chữa được. |
| C14 | Prefix-invariance base indicators; rolling/EWM/positive-shift nhân quả (architecture §3) | MATCHES + EXTENDS | Nhân quả theo **dòng** là đúng. Nhưng rolling tính trên row index, không phải exchange-session skeleton, nên phiên vắng mặt làm cửa sổ "20 phiên" dài hơn. Plan §3 (S-DE) đã yêu cầu skeleton; code chưa có. Zero-volume làm `rolling_vwap` ra NaN (`indicators.py:60-63`), snapshot thay bằng `close`, và gap = 0 trông như "không lệch". |
| C15 | Universe cố định, không PIT (architecture §2 P1 Universe; remediation gate 2) | MATCHES | `parallel_engine.py:364`. Cross-sectional rank và breadth (`feature_engineering_v2.py:78-93`, `win_probability.py:117-125`) cũng dùng basket cố định. Ngoài ra rank được tính **sau** default fill (RSI 50, ADX 20) tại `:60-70`. |
| C16 | Registry: explicit feature registry, không dùng prefix làm allowlist (S-FE dòng "117–120"; S-ML ML12) | MATCHES; bản này EXTENDS | Thiết kế chi tiết ở §5, gắn với R1 `MISSING_REASONS` (`contracts.py:46,168-171,476-491`). |
| C17 | Correlation clustering phải fit trong development fold (S-FE dòng "130–131") | MATCHES + EXTENDS | `audit_indicator_correlation` dùng zero-fill trên toàn dataset (`calibration.py:148,253`). Hiện chỉ để chẩn đoán, nhưng `_suggested_weights` đọc logistic coefficients (`calibration.py:369-371`), và các hệ số này không ổn định khi feature collinear. |
| C18 | `engineer_targets` | NOT-COVERED (dead code) | `feature_engineering_v2.py:268` fillna(0) trên return, nên outcome thiếu bị gán nhãn **thua**. Điều này trái `label_row` R1 ("never loss or zero"). Grep không thấy caller. Nên xóa hoặc guard. |

**Remediation bị đọc ngược?** Không có claim "đã sửa" nào sai về cơ chế: ffill đã bỏ, bounds train-only, regime quantile causal. Chỗ mâu thuẫn thực chất là **C3**: baseline được mô tả trong plan/H-P01 (sau S-FE) là neutral map, trong khi chính sách hiệu lực là snapshot defaults ở upstream.

## 3. Phát hiện

**F1. Ba điểm fill upstream phá missing reason.** Thứ tự xử lý hiện tại:
1. `indicators.py`: RSI→50 (`:23`), ADX/DI→0 (`:102`), bb_kc_ratio→1 (`:146`), spread_ratio→1 (`:188`).
2. V2 `_safe` và fillna (`feature_engineering_v2.py:60-70,127-183`).
3. Snapshot `_safe_float` (`feature_store.py:12-15`).
4. Bỏ giá trị không finite (`parallel_engine.py:348,355`).
5. Neutral map, rồi clip 1/99 (`calibration.py:395-469`).
6. Model.

Với `min_history_rows`=90 (`configs/rules_mr.json:5`, `rules_t2.json:4`), warm-up phần lớn bị loại vì mọi lookback ≤ 78 dòng. Nguồn missing thật còn lại là zero-volume, chia cho 0 và phiên vắng mặt. Có một rủi ro tiềm ẩn: nếu `sma50` NaN, snapshot gán 0 và `_add_normalized_features` cho ra gap = 1.0 (`calibration.py:111-137`). Cách VN suspension xuất hiện trong cache (dòng volume 0 hay vắng dòng) là **INFERRED, cần kiểm tra**.

FEAZ, [missing.html](https://feaz-book.com/missing.html): MNAR cần "careful exploratory data analysis and domain expertise". [missing-simple.html](https://feaz-book.com/missing-simple.html) cảnh báo simple imputation "doesn't preserve relationship between predictors".

**F2. Informative missingness.** Provider outage và suspension biến thành giá trị trông bình thường: gap 0, volume ratio 0, regime bull. [missing-indicator.html](https://feaz-book.com/missing-indicator.html) nói indicator phù hợp với non-MCAR, "No performance harm when added to variables with no missing data". Cách R1 có thể ghi từng trường hợp:

| R1 reason | Trường hợp hiện tại |
|---|---|
| `insufficient_warmup` | Trước lookback, hoặc trước EMA `min_periods` (`indicators.py:8`) |
| `stale` | Cửa sổ vắt qua phiên vắng; VNINDEX hoặc foreign cũ hơn decision |
| `not_applicable` | `close <= 0` (`calibration.py:109`); rule không áp dụng cho mode |
| `invalid` | inf, std = 0, volume = 0 làm mẫu số |
| `missing` | Thiếu bar, thiếu dòng foreign, thiếu mẫu breadth |

`scripts/foreign_flow_research.py:107-121` là mẫu đúng: bỏ duplicate, reindex theo calendar, `min_periods`, `dropna` theo ngày, members theo PIT (`:119`) và rank trong ngày (`:130`).

**F3. Scaling và outliers.**
- Clip 1/99 gần như vô hại với tree (general expertise). [outliers.html](https://feaz-book.com/outliers.html) xác nhận "tree-based models" bền với outlier, và [outliers-identify.html](https://feaz-book.com/outliers-identify.html) nói natural outliers "should generally be retained".
- Bounds pooled trộn nhiều regime, nên panic tail bị cắt theo ngưỡng của giai đoạn yên tĩnh.
- Linear cần scale: [numeric.html](https://feaz-book.com/numeric.html) viết "Lasso and ridge regression also need to be scaled". Pipeline hiện có làm việc này (`ensemble_model.py:192-196`, `calibration.py:335-340`).
- Feature price-level (sma/ema/close_lag) đã bị loại khỏi X, điều này tốt. Ngoại lệ là OBV/ADL (C13).

**F4. Time-series.**
- Lag, rolling và skip-month đều nhân quả: `momentum_events.py:55-57`, `momentum_scan.py:63-77`.
- Sách chỉ có [patterns.md "Temporal feature contract"] (skill) và [time-series.html](https://feaz-book.com/time-series.html) (overview draft): "special care needs to be taken with time series data".
- Calendar sin/cos (`feature_engineering_v2.py:214-220`) khớp [Periodic-trig.html](https://feaz-book.com/Periodic-trig.html) ("using both sin and cos… you don't have to worry about the offset"), nhưng hiện không vào X. `days_to_expiry` là feature kiểu reference-point theo [datetime-advanced.html](https://feaz-book.com/datetime-advanced.html), tính theo ngày lịch, fallback 15 (`:243`).

**F5. Correlated features.**
- Các nhóm trùng thông tin: gap ema9/21/50, sma5/20/50, vwap; return_1d với log_return_1d; rolling_std_20, atr_pct và bb_width; bb room/buffer với high/low room.
- [correlated.html](https://feaz-book.com/correlated.html) nói có thể thay "with just one predictor with minimal loss of information". [correlated-filter.html](https://feaz-book.com/correlated-filter.html) cảnh báo "Risk of losing signal".
- Tree tự xử lý được. Linear cần chọn theo family ngay trong fold.

**F6. Categorical.**
- Hiện không có symbol/sector feature, như vậy là tốt.
- [categorical-target.html](https://feaz-book.com/categorical-target.html): target encoding thô "Prone to overfitting". [categorical-catboost.html](https://feaz-book.com/categorical-catboost.html): "row order matters".
- Nếu dùng sau này: OOF, sắp theo thời gian, có smoothing. Sector chỉ khi có PIT (F4 BLOCKED).

**F7. Imbalance và weights.**
- Weights bất đối xứng (C12) cộng với không có calibration ở ensemble làm `probability_threshold` cố định chọn một tỷ lệ lệnh không kiểm soát được.
- [imbalanced.html](https://feaz-book.com/imbalanced.html) đặt metric, weights và threshold trước sampling. Skill `patterns.md` ghi sampling "may affect probability calibration".
- MR P(win) đã có isotonic tách riêng (`win_probability.py:223-228`), nên giữ.

## 4. Recipe theo model, fit theo fold

Sách không có cơ sở cho phần này vì `models.html` là WIP. Đây là project design và general expertise.

**Giai đoạn untrained (feature build, trước split, nhân quả):**
- Tính trên exchange-session skeleton.
- Mỗi feature xuất ra value, hoặc `None` kèm R1 reason.
- Biểu diễn tương đối theo `close`/ADV.
- Cross-sectional rank theo ngày trên PIT members, có ghi coverage.
- **Không fill ở `indicators.py`, `feature_engineering_v2.py`, `feature_store.py` trên research path.** Serving path phải đổi đồng thời để giữ parity.

FEAZ [introduction.html](https://feaz-book.com/introduction.html): "Untrained methods can be applied before the data-splitting procedure… Trained methods have to be performed after."

**Giai đoạn fit trên training của từng inner fold, sau đó freeze cho outer fold:**
1. inf → `invalid`.
2. Lập missing mask từ reason **trước** imputation, rồi tạo indicator cho các family có tỷ lệ missing không tầm thường. Ngưỡng chọn trên train.
3. Winsorize theo quantile của train (bật/tắt theo trial; thứ tự đã freeze).
4. Impute:
   - linear: train median;
   - tree: native NaN (chỉ sau khi có official docs đúng version và integration test semantics, theo plan §4).
5. Chỉ cho linear: StandardScaler hoặc RobustScaler; Yeo–Johnson tùy trial.
6. Correlation-cluster selection trong fold (chủ yếu cho linear).
7. Serialize kèm hash. Schema mismatch thì fail closed. Không refit ở inference.

Meta/stack chỉ dùng OOF (plan §6.2). Calibration làm ở đoạn temporal riêng.

## 5. Feature registry (căn theo plan §4 và R1)

| Field | Nội dung |
|---|---|
| `feature_id`, `version` | Ví dụ `F0.ema21_gap_pct@1` |
| `family` | F0–F4 (plan §4) |
| `formula_sha` | Hash code/công thức |
| `unit`, `scale_basis` | ratio / pct / rank; mẫu số (close, ADV) |
| `lookback_sessions`, `warmup_sessions` | Theo session skeleton |
| `availability` | Ví dụ "EOD t, dùng cho open t+1"; `source_contracts` R1 |
| `missing_policy` | Hành vi cho từng `MISSING_REASONS`; observed zero là giá trị |
| `cs_scope` | PIT members, chính sách tie/denominator, coverage tối thiểu |
| `transforms` | Theo model family; fit per fold; serialize |
| `allowed_targets`, `allowed_strategies` | MR15 / momentum21 / … |
| `status`, `owner`, `hypothesis_id` | PLANNED / ACTIVE / RETIRED |

## 6. Ablation một yếu tố

**Mọi dòng dưới đây cần preregistration trong `configs/research/research_gate_v1.json` trước khi xem kết quả** (AGENTS.md: thay đổi selection hoặc data contract của MR/momentum). Chỉ contract test và parity test (missing-mask preservation, batch/single-row, constant/near-zero/negative-domain fixtures) là kỹ thuật thuần, không thay đổi output.

| Ưu tiên | ID | Một thay đổi | Đo | Rủi ro / điều kiện dừng |
|---|---|---|---|---|
| 1 | H-P01 (freeze lại baseline theo C3) | Missing-aware (bỏ upstream fill, giữ reason, thêm indicator) so với chuỗi fill hiện tại | Brier/log loss trên fixed folds, reliability, net portfolio, activity, missingness-only control | Provider-outage shortcut, coverage shift |
| 2 | H-P01b | Tree native NaN so với fill hiện tại (chỉ tree) | Như trên, theo từng base | Semantics native khác nhau giữa các version |
| 3 | H-P03 | OBV/ADL chuẩn hóa theo ADV, hoặc rank theo ngày, so với raw slope | Kết quả theo family; identity-shortcut diagnostic | Mất tín hiệu nếu chuẩn hóa sai |
| 4 | H-P04 | Clip 1/99 bật/tắt, đo riêng tree và linear | Từng model; các lát panic/rebound | Gán gain cho clip khi đã đổi imputer cùng lúc |
| 5 | H-P02 | Robust/Yeo–Johnson cho linear | Linear trên fixed folds, transform hashes | Đã đăng ký trong plan |
| 6 | H-P05 | Rank/z-score theo ngày cho F0 so với raw theo mã | Lát theo regime; rank IC theo ngày | Universe không PIT |
| 7 | H-P06 | Bỏ sample weights so với weights hiện tại; threshold chọn trong inner folds | Calibration, độ ổn định selection rate, net | Weights che giấu miscalibration |
| 8 | H-P07 | Correlation-cluster selection trong fold so với toàn bộ feature | Độ ổn định hệ số, metric | Mất tín hiệu |
| 9 | H-F01 | Đưa V2/F1 vào X (hiện đang bị bỏ) | Như plan | Market exposure trá hình alpha |

Phải báo các phiên/sự kiện độc lập, không đếm stock-rows (plan §4.6). Không ghép các winner đã xem.

## 7. Nơi lời khuyên của sách không áp dụng được

- Quy tắc trained/untrained giả định split IID. Với panel tài chính, mọi thống kê theo thời gian còn phải causal (expanding + shift), không chỉ "sau split".
- Rank cross-sectional chỉ hợp lệ khi membership là PIT. Survivorship do basket cố định (`parallel_engine.py:364`) nằm ngoài phạm vi sách.
- Skill `patterns.md` cho phép fit final "on the full training set after model selection". Dự án cấm tái dùng test sau refit (`ensemble_model.py:247-249`, plan §6.1).
- Số dòng không phải số quan sát độc lập. Nhãn chồng lấn phải purge (`temporal_validation.py:66-100`).
- Không SMOTE hay shuffle xuyên thời gian (plan §4.6). Đây là chính sách dự án, sách không cấm tường minh (S-FE dòng "151").
- Panic tail là dữ liệu hợp lệ. Sách cũng nói giữ natural outliers, nhưng chính sách ở đây là của dự án (plan §4.2).
- Kết quả theo sách không thay thế research gate. Engineering pass không phê duyệt giao dịch.

## 8. Sources

**Skill** (`C:/Users/acer/.claude/skills/feature-engineering-az`):
- `SKILL.md`, `cheatsheet.md`, `patterns.md`, `glossary.md`
- `chapters/{introduction, models, order, numeric, missing, outliers, correlated, too-many, time-series, datetime, periodic, imbalanced, summary, categorical, index}.md`

**Chương gốc** (feaz-book.com, đọc qua web fetch; không gắn với bitwise snapshot `8f53af5`):
- Overview và cấu trúc: introduction; **models (WIP)**, **order (WIP)**, **summary (WIP)**
- Numeric: numeric, numeric-normalization, numeric-robust, numeric-percentile
- Missing: missing, missing-indicator, missing-simple
- Outliers: outliers, outliers-identify, outliers-imputation, outliers-indicate
- Correlated / too many: correlated, correlated-filter, too-many
- Thời gian: time-series, datetime, datetime-advanced, Periodic-trig, Periodic-indicators
- Imbalanced: imbalanced
- Categorical: categorical, categorical-target, categorical-catboost

**Repo:**
- Audits: 5 file ở §1; plan `docs/plans/2026-10-04-data-ml-research-plan.md` §4, §6, §7.
- Code đã trích: `stock_agent/features/{calibration, feature_engineering_v2, indicators, feature_store, ensemble_model, win_probability, market_regime, momentum_scan, temporal_validation, signal_engine, ml_models, mr_scan}.py`, `stock_agent/agents/parallel_engine.py`, `stock_agent/pipeline/daily_runner.py`, `scripts/{momentum_events, foreign_flow_research}.py`, `configs/rules_{mr,t2}.json`.
- R1: `D:/Chungkhoan-r1/stock_agent/data/contracts.py`.

**Chưa đọc:** `docs/audits/2026-09-30-research-gate-spec.md` và `2026-09-30-strategy-reassessment.md`. Báo cáo không diễn giải kết quả backtest nào.

## 9. Handoff

- Path thay đổi: chỉ file này.
- Không có lệnh verify runtime, không test, không commit.
- Bước tiếp theo nên làm: coordinator xác nhận C3 với code, rồi sửa H-P01 baseline trong plan §7 trước khi đăng ký executable experiment.

## Cập nhật sau merge research_ml (06/10, 4403fc7)

Bối cảnh: research path mới gồm `stock_agent/research_ml/features.py` (registry F0/F2/F3, NaN
kèm cột `_state`, không fill upstream) và `harness.py` (`FoldTransform`: median impute fit
trên train, missing indicator lấy từ mask trước imputation, winsor, robust/standard scaler,
serialize kèm hash). R5 dùng recipe này (`configs/research/ml_protocol_v1.json:31`). Đường
legacy (`indicators.py`, `feature_engineering_v2.py`, `feature_store.py`, `calibration.py`,
`ensemble_model.py`, `win_probability.py`, `parallel_engine.py`, `signal_engine.py`,
`market_regime.py`, `mr_scan.py`) không đổi giữa `4b859cd` và `4403fc7`.

| ID | Trạng thái | Evidence @ `4403fc7` / ghi chú |
|---|---|---|
| C1 | N/A (đã sửa từ trước) | Không đổi |
| C2, C3, F1: ba điểm fill upstream; H-P01 baseline thực tế | STILL OPEN (legacy) / CHANGED (research path) | Legacy không đổi. research_ml không fill upstream: NaN + `_state` (`features.py:404-413`), median fit theo fold (`harness.py:108-128`). Mô tả H-P01 ở plan §7 vẫn chưa sửa (`plan:291`) |
| C4 imputation row-local | STILL OPEN (legacy) | research_ml dùng median train-fit (`harness.py:110`) |
| C5 winsor bounds fit sau khi fill | STILL OPEN (cả research_ml) | `FoldTransform.fit` impute median (`harness.py:110-112`) rồi mới lấy quantile (`:114-116`). Recipe §4 đặt winsor trước impute. Lệch ít hơn zero-fill nhưng thứ tự vẫn khác |
| C6 V2 không vào X | STILL OPEN | Legacy không đổi |
| C7 default bull khi thiếu `regime_trend`/`rs_slope_5` | STILL OPEN | Không đổi |
| C8 `days_since_regime_change` | STILL OPEN | Không đổi |
| C9 MR `fillna(0)`, regime mặc định 1.0 | STILL OPEN | `win_probability.py` không đổi |
| C10 tree không bao giờ thấy NaN | STILL OPEN (cả research_ml) | `FoldTransform.transform` luôn impute (`harness.py:138`); chưa có native-missing |
| C11 sigmoid Ridge, fallback 0.5 | STILL OPEN (legacy) / RESOLVED (research path) | Thiếu component thì raise (`harness.py:191-199,215-220`) |
| C12 sample weights bất đối xứng | STILL OPEN (legacy) | research_ml không dùng weights |
| C13 OBV/ADL scale | STILL OPEN (legacy) | Ghi chú mới: `f2_adv_value_20` của research_ml là level VND (`features.py:367,395`). Robust scaler pooled (`harness.py:119-120`) không chuẩn hóa theo mã, nên có cùng loại rủi ro scale/identity |
| C14 rolling theo row index | RESOLVED (F3) / STILL OPEN (F0 research_ml + legacy) | F3 dùng session skeleton (`features.py:426-437`); F0 vẫn rolling theo dòng bar (`features.py:383-397`) |
| C15 universe cố định | RESOLVED (R5 research runs) / STILL OPEN (legacy) | PIT VN30 (`ml_protocol_v1.json:16`; `2026-10-04-r5-results.md:5-6`); `parallel_engine.py:364` không đổi |
| C16, §5 feature registry | RESOLVED một phần | `FeatureSpec` + `registry_hash` + allowlist guard (`features.py:347-380`, `contracts.py:187-194`). Còn thiếu version từng feature, `formula_sha`, availability/source_contracts, `cs_scope`, `allowed_targets`, status/owner. Liên kết với `MISSING_REASONS` chờ port R1 |
| C17 correlation clustering trong fold | STILL OPEN | Không có trên cả hai đường |
| C18 `engineer_targets` fillna(0) | STILL OPEN | `feature_engineering_v2.py:268` không đổi |
| F2 informative missingness / indicator | RESOLVED một phần (research_ml) | Mask trước impute + indicator (`harness.py:137,142-143`); test `test_ml_harness.py:51`. Indicator chỉ là isna nhị phân; lý do trong `_state` không vào X; F0 không tách `invalid`/`stale` |
| F3 scaling | RESOLVED (research path) | Scaler fit theo fold (`harness.py:117-123`) |
| F4, F5, F7 | STILL OPEN (legacy) | F6 N/A (không có categorical) |
| §4.1 inf → `invalid` | RESOLVED một phần | inf → missing (`harness.py:104-106`), chưa thành reason `invalid` |
| §4.2 mask trước imputation | RESOLVED | `harness.py:137` |
| §4.3 winsor theo quantile train | RESOLVED (nhưng sau impute, xem C5) | `harness.py:114-116`; test `test_ml_harness.py:79` |
| §4.4 impute: linear median / tree native NaN | RESOLVED / STILL OPEN | `harness.py:110` |
| §4.5 scaler / Yeo–Johnson | RESOLVED / STILL OPEN | `harness.py:117-123` |
| §4.6 correlation-cluster trong fold | STILL OPEN | |
| §4.7 serialize + hash + fail-closed | RESOLVED (code) | `harness.py:134-135,146-157`; test `:65`. Bundle R5 chưa lưu transform: `transforms/` chỉ có README placeholder |
| §6 ablation H-P01…H-F01 | STILL OPEN (chưa chạy) / CHANGED (ưu tiên) | R5 chỉ chạy drop-family (`ml_protocol_v1.json:35-36`). Lane weekly: rank IC ≈ 0, shuffled-label control thắng (`2026-10-04-r5-weekly-results.md:27-32`). Chưa có lane nào có tín hiệu để ablate preprocessing |
| §6 nơi preregister | CHANGED | Thí nghiệm ML đăng ký ở `configs/research/ml_protocol_*.json` |
| Handoff: sửa H-P01 baseline ở plan §7 | STILL OPEN | `plan:291` không đổi |

Còn actionable:
1. Sửa wording H-P01 (`plan:291`): legacy baseline là chuỗi upstream fill; research baseline là `FoldTransform` của `harness.py`.
2. research_ml: (a) chuyển winsor lên trước impute, hoặc ghi thứ tự hiện tại vào protocol; (b) F0 rolling theo session skeleton; (c) chuẩn hóa `f2_adv_value_20` (log/rank theo ngày); (d) lưu transform JSON thật vào bundle.
3. Ablation H-P* chỉ nên chạy sau khi có một lane có tín hiệu vượt control (ví dụ v3 với exposure-matched control).
