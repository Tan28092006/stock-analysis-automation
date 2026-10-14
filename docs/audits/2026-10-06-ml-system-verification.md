# Kiểm chứng hệ thống ML so với các audit trước

Ngày: 06/10/2026. Reviewer: Firstmate run `2026-10-06-ml-roadmap` / T2.
Source revision: `4b859cd` (branch `codex/data-integrity-audit`); working tree
có sẵn hai file dirty `data/pipeline/local_server.log`,
`data/pipeline/swing_scan_cache.json`, không liên quan và không bị sửa.

Phạm vi: review read-only code ML (`stock_agent/features/ensemble_model.py`,
`calibration.py`, `temporal_validation.py`, `model_release.py`,
`win_probability.py`, `ml_models.py`, `stock_agent/pipeline/daily_runner.py`)
và tests liên quan; đối chiếu với §1/§6 của
[roadmap](../plans/2026-10-04-data-ml-research-plan.md) và năm audit:
[S-ML](2026-10-04-ml-plan-review-ml.md),
[review tổng hợp](2026-10-04-ml-plan-review.md),
[agent readiness](2026-10-04-agent-readiness-and-ml-plan.md),
[remediation](2026-09-22-training-remediation.md),
[baseline leakage](2026-09-22-architecture-and-leakage.md).
Chỉ file này được tạo. Không training, backtest, data refresh, network fetch,
commit hay push. Đây là code inspection, **không** phải research pass, model
approval hay trading approval.

## 1. Kết luận

**PARTIAL — các remediation chính của 22/09 vẫn đứng vững; ba lỗ hổng mới
chưa audit nào phủ.**

- Các sửa P0 của baseline (fit-row leakage của ensemble, winsor fit trước split,
  row-level `TimeSeriesSplit`, chọn model bằng test, MR refit sau calibration)
  khớp với code ở `4b859cd`. Không thấy audit nào tuyên bố "đã sửa" mà code
  đảo ngược trực tiếp.
- **Conflict quan trọng nhất:** roadmap §1 L24 và H-P01 L291 nói baseline
  "không blanket zero-fill". Nhưng MR `WinProbModel` dùng `fillna(0.0)` toàn
  bộ (`win_probability.py:285`). Ensemble cũng chỉ có 12 prefix có giá trị
  trung lập; mọi feature khác mặc định 0.0 (`calibration.py:407-420, 430, 439`).
- **Ba rủi ro mới HIGH/MED-HIGH chưa audit nào phủ:**
  (a) rule threshold khi serve chưa từng được evaluate;
  (b) holdout bị cắt lại ở mỗi lần retrain/HPO mà không ghi lại;
  (c) ensemble đi vòng qua release gating của `model_release.py`.

## 2. Bảng đối chiếu claim của audit với code

Verdict: **MATCHES** = code xác nhận; **CONFLICTS** = code trái với claim;
**EXTENDS** = claim đúng nhưng thiếu điều kiện hoặc phạm vi;
**NOT-COVERED** = phát hiện của T2 mà các audit trước không đề cập.

| # | Claim (nguồn audit §) | Verdict | Bằng chứng code `4b859cd` |
|---|---|---|---|
| C1 | Ensemble chấm test trên chính các dòng đã fit (baseline §2 P0) → đã sửa: holdout 20% ngày, model lưu là model fit development (remediation §Hợp đồng, dòng "Ensemble") | MATCHES (đã sửa) | `ensemble_model.py:240-308`: `purged_time_split(.8)` → `_fit_bases(development)` → `predict(test)`; test `test_training_integrity.py:81` |
| C2 | Winsor fit trên toàn dataset; forward-fill xuyên mã (baseline §2 P0) → bounds chỉ fit trên train, impute độc lập từng dòng (remediation, dòng "Preprocessing") | MATCHES | `ensemble_model.py:652-669`; `calibration.py:395-469`; tests `test_training_integrity.py:24, 90, 315` |
| C3 | `TimeSeriesSplit` chia theo dòng, không purge (baseline §2 P0) → chia theo ngày + purge | MATCHES | `temporal_validation.py:66-85, 88-100`; test `test_training_integrity.py:41` |
| C4 | `_select_model` chọn model bằng test (baseline §2 P0) → chọn bằng validation (remediation, dòng "Model suite legacy") | MATCHES | test `test_training_integrity.py:108` |
| C5 | MR refit classifier sau isotonic (baseline §2 P1) → fit một lần, isotonic fit trên calibration, chấm trên test (remediation, dòng "MR") | MATCHES | `win_probability.py:219-232`; test `test_training_integrity.py:162` |
| C6 | `backtest.py` override lịch sử mà không kiểm trained-at (baseline §2 P1) → availability gate (remediation, dòng "Model availability") | MATCHES | `backtest.py:303-304, 719-720` → `ml_models.py:232-233, 301-304` gọi `model_available_at` |
| C7 | HPO chỉ thấy development, "cùng outer split với train" (remediation, dòng "HPO") | EXTENDS | Đúng trong một lần chạy (`ensemble_model.py:531-533`; test `:187`). Nhưng split được cắt lại theo as-of của từng lần chạy, nên chỉ cùng split khi cùng dataset. Xem R2 |
| C8 | Daily update fit mới, không save khi thất bại (remediation, dòng "Daily update"; S-ML ML22) | MATCHES + EXTENDS | `ensemble_model.py:310-316`; `daily_runner.py:184-187`. Không có gate quality/promotion; save ghi đè registry |
| C9 | Imputation row-local constants/neutral defaults (S-ML ML04; review tổng hợp "Real preprocessing baseline") | EXTENDS | Đúng là row-local, nhưng chỉ 12 prefix có trị trung lập (`calibration.py:407-420`); còn lại 0.0 (`:430, :439`) với adx14, DI, atr_pct, rolling_std. Bounds tính sau khi fill (`:459-460`) |
| C10 | "không blanket zero-fill", blanket-zero chỉ là control (roadmap §1 L24, H-P01 L291) | **CONFLICTS** | MR: `win_probability.py:285` `fillna(0.0)` toàn bộ (remediation gọi đây là "Imputation dùng chung"). Ensemble: phần lớn feature thực chất là zero-fill |
| C11 | Ensemble là voting mean, không fitted stack (S-ML ML03; agent readiness §Kết quả) | MATCHES | `ensemble_model.py:237` (`meta_model=None`), `:682-684`, `:289-290`. Docstring `:1, :142`, registry `type:"ensemble_stacking"` `:469` và `_build_meta` `:210-221` vẫn ghi sai là stacking |
| C12 | Mặc định T+2; Ridge qua sigmoid; thiếu base thì 0.5 (S-ML ML05) | MATCHES + EXTENDS | `:244, :310, :522`; sigmoid `:686-693`; 0.5 fallback âm thầm `:352-355, :673-676`. Sample weights luôn lấy `net_t2_return_pct` dù `label_col` khác (`:227-228`) — NOT-COVERED |
| C13 | Candidate-only, `live_approved=false` (S-ML ML06; roadmap §1 L27) | EXTENDS | Chỉ đúng cho MR: `model_release.py:40-43, 126-150, 161-162`. Ensemble `save()` (`ensemble_model.py:445-487`) là pickle, không hash/code provenance/snapshot binding, ghi đè registry mutable |
| C14 | Artifact lưu code SHA, config+feature+data hashes, dependencies (baseline §4 mục 6) | **CONFLICTS** (với ensemble) | Ensemble chỉ có dataset/feature hash (`temporal_validation.py:103-109`), không code SHA/dependencies. MR có `code_provenance` (`model_release.py:126-150`), nhưng không hash `calibration.py`/`ensemble_model.py` |
| C15 | Calibration/threshold không dùng final test (baseline §4 mục 4) | MATCHES (MR) / NOT-COVERED (ensemble) | MR: `win_probability.py:228, 234` (threshold cố định). Ensemble: không có calibration segment; threshold lúc serve là heuristic (R1) |
| C16 | Helper availability chỉ kiểm trained_at + label-end dates, chưa có calibrated_at/evaluated_at (S-ML ML19; roadmap §6.1) | MATCHES | `temporal_validation.py:47-63`; `calibration_label_end` mặc định = `fit_label_end` (`:58`) |
| C17 | Purge theo `label_available_date` (remediation, sơ đồ luồng) | EXTENDS | Trường này optional; thiếu thì dùng `exit_date` (`temporal_validation.py:25-29`); granularity là ngày (`:21-22`) |
| C18 | Cần lưu OOF prediction của từng fold (baseline §2 P0 "Khắc phục") | NOT-COVERED bởi remediation | `train()` chỉ lưu fold metrics tổng hợp (`ensemble_model.py:263-275`), không lưu row-level OOF. Đây là gap R4 |
| C19 | HPO mặc định 50 trials, objective profit-factor×log(count) (S-ML ML23) | MATCHES | `ensemble_model.py:519-617` |
| C20 | 123 passed (agent readiness §Verification); 193 passed (remediation §Kiểm chứng) | Không đối chiếu được | Môi trường hiện tại khác: xem §6 |

## 3. Rủi ro mới, xếp theo mức độ

1. **HIGH — R1: rule quyết định khi serve chưa từng được evaluate.**
   `ml_models.py:318-351` dựng threshold live như sau:
   - lấy threshold từ rules;
   - override theo `configs/ml_symbol_thresholds.json` (repo không có generator cho file này);
   - cộng hiệu chỉnh volatility/regime;
   - clamp vào 0.45–0.75;
   - rồi so với mean probability chưa calibrate.

   Cả `train()` lẫn HPO đều không đo rule này. Các audit trước chỉ xét gate
   availability (`:301-304`), không xét bước threshold sau đó.
2. **HIGH — R2: holdout bị cắt lại mà không có record.**
   - Mỗi lần `train()`/`daily_update` cắt lại 20% ngày cuối (`ensemble_model.py:256`). `run_hpo` cắt split theo as-of riêng (`:533`).
   - Khi dữ liệu tăng, các ngày test cũ trượt vào development/HPO. Không có ledger nào ghi các ngày test đã bị xem.
   - HPO mutate `self.config` (`:621-643`), nhưng `daily_runner.py:175,181` ghi đè config từ rules, nên lineage của hyperparameter không rõ.
   - `detect_drift` so với `test_win_rate` của holdout bị dùng lại (`:390`).
3. **MED-HIGH — R3: ensemble đi vòng qua release gating.** Không có candidate
   stage, artifact hash, code provenance, snapshot binding hay quality gate
   (C13, C14). `daily_runner.py:186,194` save thẳng vào registry production.
   Lớp bảo vệ duy nhất là `model_available_at`.
4. **MED — R4: calibration và đơn vị score.**
   - Mean của probability từ tree trộn với sigmoid(margin Ridge), không có calibration segment.
   - Ensemble không có ECE/Brier.
   - Accuracy đo ở ngưỡng 0.5 trong khi selection dùng threshold khác (`ensemble_model.py:286-287`).
   - Component thiếu bị thay âm thầm bằng 0.5.
5. **MED — R5: fill làm méo bounds.** Quantile tính sau khi fill 0.0
   (`calibration.py:430, 459-460`), nên tỷ lệ missing cao kéo bounds về 0.
   Không có missing indicator.
6. **MED — R6: sample weight không đi theo label.** Weight luôn lấy từ
   `net_t2_return_pct` (`ensemble_model.py:227-228, 535`). Nếu đổi label (ví dụ
   MR15), code vẫn âm thầm giữ weight của T+2.
7. **LOW.**
   - HPO dùng 3 splits trong khi `train()` dùng `config.n_splits` (`:569` so với `:262`).
   - Model deploy loại bỏ 20% ngày gần nhất nên luôn stale.
   - MR gates chấm trên test mà không giới hạn số lần retry (`win_probability.py:236`; `model_release.py:27-46`).

Đã kiểm và không thấy vấn đề:
- scaler fit trong từng fold (Pipeline, `ensemble_model.py:192-196`);
- không có early stopping;
- MR isotonic không fit trên test;
- HPO bị giới hạn trong development (test `:187`).

## 4. R4 gap: "purged grouped dates, serialized fold fit, OOF identity và availability tests; bundle validator"

Những phần còn thiếu:

1. **Không có fold spec serialize được.** Folds được tính lại mỗi lần chạy. Chưa có:
   - boundaries;
   - hash row-ID (`symbol`+`signal_date`);
   - danh sách dòng/ngày bị purge kèm lý do;
   - class counts từng fold;
   - floor số phiên tối thiểu / bắt buộc đủ hai class, trả BLOCKED khi không đạt.
2. **Chưa có embargo optional** dưới dạng stress parameter freeze trước; chưa có lựa chọn expanding vs rolling.
3. **Chưa có timestamp contract bắt buộc.**
   - Chưa có giờ availability intraday (signal 16:00 ICT, label available).
   - Chưa có kiểm tra availability của feature.
   - `label_available_date` vẫn optional.
4. **Chưa có OOF store và test identity.** Cần lưu theo (row_id, fold_id, base, fit_cutoff, artifact hash). Cần test:
   - `fit_label_end < signal_date` cho mọi dòng OOF;
   - mỗi dòng eligible có đúng một prediction;
   - dòng không được cover thì báo coverage loss, không lấp bằng in-sample.
5. **Chưa có nested inner folds** cho HPO và preprocessing, và chưa serialize fit từng fold (bounds, params, seed, preprocessing hash).
6. **`model_available_at` cần protocol mới có version**, bắt buộc `calibrated_at`/`evaluated_at`/`selected_at` cùng label-end tương ứng.
7. **Chưa có bundle validator** cho các file §9 của roadmap (`source_manifest`, `splits`, OOF, metrics, preprocessing, schema hashes) dùng chung cho ensemble và MR. Ensemble cần candidate path giống MR (sửa R3).

Tests hiện có phủ những gì:

| Test file | Phủ |
|---|---|
| `test_training_integrity.py` | date-disjoint + purge (`:41`), thiếu exit (`:49`), fit/predict disjoint (`:81`), bounds không thấy holdout (`:90`), HPO isolation (`:187`), MR calibration freeze (`:162`), temporal edge cases (`:333`), model vintage (`:352`) |
| `test_model_release.py` | MR gates, immutability, tamper, không có live mode |
| `test_ensemble.py` | Smoke test train/predict/save/drift |
| `test_calibration.py` | Label causality, threshold optimizer |
| `test_backtest_leakage.py` | Causality của indicator/decision/entry/exit |

Chưa test nào phủ: fold serialization, OOF identity, embargo, class/session
floors, calibrated/evaluated timestamps, bundle schema, đường threshold khi serve.

**Bước R4 bounded đầu tiên đề xuất:**
- Thêm hàm thuần `build_date_folds(frame, spec) -> FoldPlan` cùng `assert_oof_identity(plan, oof_frame)` vào một module mới cạnh `temporal_validation.py`.
- Không đổi model, pipeline hay serving.
- `FoldPlan` serialize ra JSON xác định (deterministic) gồm boundaries, row-ID sha, purged rows có lý do, class/session counts và status BLOCKED.
- Viết test RED trước trên panel tổng hợp:
  - determinism và round-trip;
  - không train label nào vượt boundary;
  - mỗi dòng OOF có đúng một prediction với fit cutoff < signal;
  - fold một class hoặc quá ít phiên phải trả BLOCKED.
- Wiring vào ensemble/release là bước sau.

## 5. MLSys evaluation

Nguồn: skill `machine-learning-system` (Reddi, *Machine Learning Systems*,
snapshot 20/10/2025). Guide chương trong
`C:/Users/acer/.claude/skills/machine-learning-system/chapters/`; dòng nguồn là
dòng của `assets/book.txt` (không phải trang PDF), đọc bằng
`scripts/read_source.py`. Gán nguồn theo nguyên tắc của
[S-ML](2026-10-04-ml-plan-review-ml.md) §2: sách hỗ trợ cơ chế; các protocol
tài chính là PROJECT.

| Chủ đề | Chương / nguồn | Bằng chứng repo |
|---|---|---|
| Leakage, transform parity | `06-data-engineering.md` §Four Pillars: "Training-serving consistency… Fit transforms bằng training data"; §Pipeline decisions mục 1 "Xác định prediction time và dữ liệu nào thật sự available". Nguồn dòng 20181: pipelines chuẩn bị dữ liệu "consistently… across training and inference stages, reducing the risk of data leakage" | Bounds truyền đúng (`ensemble_model.py:652-669`), nhưng default 0.0 (`calibration.py:430`) và thiếu availability timestamp (`temporal_validation.py:25-29`) chưa đạt nguyên tắc "available at prediction time" (C9, C10, C17) |
| Validation protocol | `12-benchmarking-ai.md` §Bẫy: "test-set tuning phá đánh giá". Nguồn dòng 53522: "reduces the risk of models being overfitted to a single test set". `05-ai-workflow.md` §Lỗi cần tránh: "Development score không thay production validation" | R1 (`ml_models.py:318-351`), R2 (`ensemble_model.py:256, 533`) |
| Calibration | `13-ml-operations.md` §Monitoring bốn chiều: calibration thuộc model performance. `16-robust-ai.md` §Thiết kế kiểm chứng mục 3: "Kiểm tra confidence/calibration trên điều kiện đổi". Nguồn dòng 70665: label shift "requiring recalibration of diagnostic thresholds" | Ensemble không calibrate, threshold heuristic (`ensemble_model.py:673-693`; `ml_models.py:342-351`); MR làm đúng (`win_probability.py:222-232`) |
| Reproducibility | `08-ai-training.md` §Reproducibility: lưu "data/config/code/runtime versions, seeds… và evaluation". `13-ml-operations.md` §Bẫy: "Experiment tracking không bảo đảm tái lập" | Ensemble thiếu code provenance/hash (`ensemble_model.py:445-487`); folds không serialize (gap 1) |
| Release / drift | Nguồn ch13 dòng 56031-56035: registering "trained models as immutable artifacts… promoting, deploying, and rolling back". `13-ml-operations.md` §Feedback loop: "Không auto-retrain chỉ vì drift alert" (tổng hợp của skill / PROJECT policy, xem S-ML §1 mục 1) | `daily_runner.py:174-195` retrain + ghi đè hằng ngày, không gate (R3); `detect_drift` (`ensemble_model.py:373-409`) dùng test win-rate bị tái sử dụng |
| Benchmarking | `12-benchmarking-ai.md` §Hợp đồng benchmark: task/splits/threshold/slices/variance; §Quy trình mục 4 "Thay một yếu tố" | Ensemble không có fixed-mean control, per-fold variance hay slice/regime metrics trong bundle; ủng hộ thứ tự M0→M5 của roadmap §6.2 và bundle validator (gap 7) |

Guides đã đọc đầy đủ: 05, 06, 08, 12, 13, 16. Các truy vấn nguồn đã chạy:
- ch06 `leakage`
- ch12 `test set`
- ch13 `training-serving skew`, `model registry`, `retrain`
- ch16 `calibrat`
- ch05 `validation`
- ch08 `reproducib`

## 6. Giới hạn môi trường test

- `python` (msys64 mingw) không có pytest. Dùng `py` (Python launcher).
- `py -m pytest -q tests/test_training_integrity.py`: **38 passed, 2 failed**. Cả hai failure là `ModuleNotFoundError`: `lightgbm` (test `:163`) và `optuna` (test `:188`). Đây là lỗi môi trường, không phải bằng chứng code hỏng.
- Chạy chung 5 file thì ba file lỗi khi collection: "paging file is too small" / MemoryError lúc import scipy. Vì vậy `test_ensemble.py`, `test_calibration.py`, `test_model_release.py`, `test_backtest_leakage.py` **chưa được chạy** trong môi trường này.
- Không tuyên bố số test mới. Các con số 123/193 passed của audit trước là bằng chứng của môi trường khác (Python 3.12.4 có đủ dependencies), không được tái xác nhận ở đây.
- Mọi kết luận ở trên là code inspection tại `4b859cd`. Chưa đo hiệu quả, calibration thực, lợi nhuận hay activity.

## 7. Handoff

- Path thay đổi: chỉ file này.
- Blockers chưa giải quyết:
  - R1–R3 cần quyết định của owner trước khi cho phép bất kỳ ML override nào;
  - cần sửa wording của C10 trong roadmap §1 L24/H-P01 L291 (coordinator làm, không phải T2);
  - cần cài môi trường có `lightgbm`/`optuna` và đủ bộ nhớ để chạy lại đủ 5 file test.
- Next bounded action: bước R4 đầu tiên ở §4, theo TDD, chạy full regression trước khi coi hoàn tất.

## Cập nhật sau merge research_ml (06/10, 4403fc7)

Bối cảnh: research_ml cung cấp harness R4 **riêng** (`stock_agent/research_ml/harness.py`,
`evidence.py`), dùng cho R5 qua `scripts/ml_research_r5.py`, `scripts/ml_research_r5_weekly.py`.
Đường legacy (`ensemble_model.py`, `calibration.py`, `temporal_validation.py`,
`model_release.py`, `win_probability.py`, `ml_models.py`, `daily_runner.py`) không đổi giữa
`4b859cd` và `4403fc7`. Vì vậy ba rủi ro mới (threshold khi serve, holdout bị cắt lại, ensemble
đi vòng release gating) vẫn còn nguyên trên legacy path. Contract timestamp v2 chưa được nối
vào `model_release.py` (`docs/audits/2026-10-04-r1-r4-implementation.md:18`).

| ID | Trạng thái | Evidence @ `4403fc7` / ghi chú |
|---|---|---|
| C1–C6, C11, C19 | N/A (MATCHES, legacy không đổi) | Không cần hành động mới |
| C7 HPO split bị cắt lại theo as-of | STILL OPEN | `ensemble_model.py:533` |
| C8 daily update không có quality/promotion gate | STILL OPEN | `daily_runner.py:184-187` |
| C9, C10 zero-fill thực tế vs wording "không blanket zero-fill" | STILL OPEN (legacy + plan) | `win_probability.py:285`, `calibration.py:407-439`; `plan` L24/L291 chưa sửa |
| C12 fallback 0.5, sigmoid, weights theo `net_t2` | STILL OPEN (legacy) / RESOLVED (research path) | research_ml raise khi thiếu component (`harness.py:191-199,215-220`), không dùng weights |
| C13, C14 ensemble: pickle, ghi đè registry, thiếu code provenance | STILL OPEN | `ensemble_model.py:445-487` |
| C15 calibration/threshold của ensemble | STILL OPEN | R6 (calibration) chưa chạy |
| C16 availability thiếu `calibrated_at`/`evaluated_at` | RESOLVED (validator) / STILL OPEN (wiring) | `validate_model_contract` + `REQUIRED_TIMESTAMPS` (`harness.py:259-291`); test `test_ml_harness.py:154`. `temporal_validation.py`/`model_release.py` không đổi |
| C17 `label_available_date` optional, độ phân giải ngày | RESOLVED (research_ml) / STILL OPEN (legacy) | Bắt buộc + tz-aware (`harness.py:48-50`), 16:00 VN (`labels.py:311`); `temporal_validation.py:25` không đổi |
| C18 lưu OOF row-level | RESOLVED (in-memory research_ml) / STILL OPEN (persist + legacy) | `forward_oof` (`harness.py:170-188`). Bundle weekly chỉ có `predictions/README.json` = "OOF summarized in metrics.json only" |
| C20 số test | CHANGED | 51 passed cho `tests/test_ml_*.py` + 2 guard, chạy trong môi trường không có LightGBM/XGBoost/CatBoost (`r1-r4-implementation.md:12,22`); suite legacy chưa chạy lại |
| Rủi ro R1: threshold khi serve chưa được evaluate | STILL OPEN | `ml_models.py:318-351` |
| Rủi ro R2: holdout cắt lại không có record | STILL OPEN (legacy) / RESOLVED một phần (research) | research_ml dùng fixed geometry đã đăng ký (`harness.py:55-64`; `ml_protocol_v2_weekly.json:54`). Nhưng không có ledger chung các ngày test đã xem: v1 và v2 dùng cùng cửa sổ 2022-09..2026-09, cả hai chỉ ghi là "development data" |
| Rủi ro R3: ensemble đi vòng release gating | STILL OPEN | `daily_runner.py:186,194` |
| Rủi ro R4: calibration, đơn vị score | STILL OPEN | R6 chưa chạy |
| Rủi ro R5: fill làm méo bounds | STILL OPEN (legacy) / RESOLVED một phần (research) | Research có indicator nhưng vẫn winsor sau median impute (`harness.py:110-116`) |
| Rủi ro R6: weight không theo label | STILL OPEN | `ensemble_model.py:227-228,535` |
| Rủi ro R7 (LOW) | STILL OPEN | Không đổi |
| §4 gap 1: FoldPlan serialize được | RESOLVED một phần | `Fold` có boundary, purged_rows, train/test dates, class_counts, problems (`harness.py:22-36`). Fold có problems → BLOCKED (`evidence.py:401-404`). Còn thiếu row-ID sha và danh sách row bị purge kèm lý do. `splits.json` đã commit không chứa boundary/class_counts (weekly: chỉ index, problems, purged_rows, dates) |
| §4 gap 2: embargo optional, expanding vs rolling | STILL OPEN | Chỉ expanding + purge theo `label_available_at` (`harness.py:39-90`); không có tham số embargo |
| §4 gap 3: timestamp contract bắt buộc | RESOLVED (research_ml) | `labels.py:311`, `features.py:414`, `contracts.py:134-180`, `harness.py:48-50` |
| §4 gap 4: OOF store + identity test | RESOLVED một phần | Row (row, fold, component, pred); row không được cover thì không lấp (`harness.py:170-188`; test `test_ml_harness.py:91`). Còn thiếu fit_cutoff/artifact hash theo từng row OOF và persist trong bundle. `component_matrix` dùng `aggfunc="first"` (`harness.py:192`), nên prediction trùng bị gộp im lặng thay vì fail |
| §4 gap 5: nested inner folds, serialize fit theo fold | RESOLVED một phần | `nested_stack_eval` (`harness.py:223-249`), `FoldTransform.to_json/sha256` (`:146-157`); không có inner folds cho HPO |
| §4 gap 6: `model_available_at` có version | RESOLVED (validator) / STILL OPEN (wiring) | Như C16 |
| §4 gap 7: bundle validator | RESOLVED (research bundles) / STILL OPEN (legacy MR/ensemble; độ chặt) | `evidence.py:353-419`; test `tests/test_ml_evidence.py`. Lineage check chỉ yêu cầu thư mục không rỗng (`evidence.py:384-385`), nên README placeholder vẫn qua. Activity floor chỉ được kiểm là đã freeze, không kiểm là đạt (`:399-400`) |
| §4 bước đầu `build_date_folds`/`FoldPlan`/`assert_oof_identity` cạnh `temporal_validation` | SUPERSEDED | Thay bằng `research_ml/harness.py` (`purged_grouped_folds`, `Fold`, `forward_oof`) |
| §6 giới hạn môi trường | CHANGED một phần | Có môi trường chạy được `test_ml_*`; vẫn thiếu lightgbm/optuna cho 5 file legacy |
| Handoff: quyết định owner cho R1–R3 trước khi bật ML override | STILL OPEN | Không đổi |

Còn actionable:
1. Legacy: threshold khi serve chưa được evaluate, holdout bị cắt lại không có ledger, ensemble đi vòng `model_release`. Nên giữ ML override tắt cho tới khi wire contract v2 và có candidate path cho ensemble.
2. research_ml: thêm embargo, row-ID hash, row bị purge kèm lý do, persist OOF/transform thật vào bundle; `component_matrix` phải fail khi có prediction trùng; validator phải từ chối README placeholder.
3. Lập ledger các ngày test đã xem giữa các protocol (v1/v2/v3), để kết quả lặp lại trên cùng cửa sổ không bị đọc như holdout.
