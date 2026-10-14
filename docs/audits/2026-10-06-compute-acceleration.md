# Compute acceleration: GPU readiness và bottleneck thực tế — 06/10/2026

## Header

- Ngày: 06/10/2026.
- Reviewer: Firstmate run `2026-10-06-ml-roadmap` / T5 (scout read-only).
- Scope: đánh giá repo có thể dùng GPU để tăng tốc compute không và nếu có thì
  ở stage nào; đối chiếu với các audit 04/10 và plan §2. Không sửa code/config/
  data, không cài/gỡ package, không chạy training, backtest, refresh hay network
  fetch, không commit. File này là thay đổi duy nhất.
- Source revision: `4b859cd` (branch `codex/data-integrity-audit`). Dirty tree
  lúc inspection: chỉ `data/pipeline/local_server.log`,
  `data/pipeline/swing_scan_cache.json` (untracked, không đụng tới).
- Hardware (coordinator quan sát, `nvidia-smi` xác nhận lại): NVIDIA GeForce RTX
  3050 Ti Laptop GPU 4 GB VRAM, driver model WDDM, driver 610.62 / CUDA 13.3;
  Intel Iris Xe iGPU; CPU i7-12700H 14 cores / 20 threads; RAM 15,7 GB. Windows
  từng báo `MemoryError` "paging file too small" khi chạy pytest.
- Phương pháp: skill `machine-learning-system` (Reddi, *Machine Learning
  Systems*, snapshot 20/10/2025); đọc guide 08/09/11/12 và trích nguồn gốc bằng
  `scripts/read_source.py`. Source line dưới đây là dòng `assets/book.txt`,
  không phải số trang PDF.

## 1. Kết luận

**Chưa nên chuyển sang GPU.** Môi trường test (Anaconda) đã có XGBoost và
CatBoost build hỗ trợ CUDA, nên về kỹ thuật có thể bật GPU mà không cài thêm.
Tuy vậy, workload hiện tại là tabular daily nhỏ, khoảng 220k rows cho 100
symbols. Thời gian chủ yếu nằm ở các vòng Python per-bar: mỗi bar lại copy toàn
bộ prefix lịch sử, nên chi phí tăng theo O(T²) mỗi symbol. Ngoài ra là backtest
replay và HPO chạy tuần tự, không phải ở bước fit tree. Nên ưu tiên tối ưu CPU
và profiling stage. GPU chỉ được cân nhắc khi profile chứng minh boosting fit
chiếm phần lớn wall time. Khi đó GPU phải được xem là một evidence lineage mới,
vì kết quả theo device khác nhau.

## 2. Bảng đối chiếu với audit và plan hiện có

Nhãn: MATCHES (khớp), CONFLICTS (mâu thuẫn), EXTENDS (bổ sung bằng chứng hoặc
cơ chế mới), NOT-COVERED (audit cũ không đề cập).

| # | Finding T5 | Audit / plan liên quan | Nhãn | Evidence (code file:line, đo đạc) |
|---|---|---|---|---|
| C1 | Không mặc định dùng GPU; profile trước | Plan §2 L55, §6.2 L279–280, §11 L417–419; `ml-plan-review-ml.md` §3 ML09, ML23 | MATCHES | Synthetic benchmark §4 cho thấy fit chỉ là một phần nhỏ; bottleneck nằm ở loop (§3) |
| C2 | Library readiness: XGBoost 3.2.0 `USE_CUDA=True`, CatBoost 1.2.10 GPU chạy được, LightGBM 4.6.0 không có CUDA learner (chỉ OpenCL `device="gpu"`), torch 2.13.0+cpu, không có cupy/cuDF/RAPIDS | Không audit nào kiểm tra package build | NOT-COVERED | Probe trong `gpu_bench.py`, `gpu_bench2.py` (scratchpad, §4); LightGBM báo "CUDA Tree Learner was not enabled in this build" |
| C3 | Hot path O(T²): copy prefix per bar | `system-design-review.md` §Các vấn đề P2 "Full-history validation sẽ tăng cost theo archive" (cùng loại vấn đề, nhưng ở `foreign_flows`) | EXTENDS | `stock_agent/agents/parallel_engine.py:272-273` (`validate_ohlcv` trên `data.iloc[:idx+1]`), `:284` (`full_features.iloc[:idx+1].reset_index`), `:286`; `scripts/research_gate.py:134-139`; `scripts/period_backtest.py:65-67`, `:301`, `:320`; `scripts/momentum_books.py:167`; `stock_agent/features/win_probability.py:180-190` |
| C4 | "Parallel engine" nhưng label builder chạy tuần tự | `agent-readiness-and-ml-plan.md`: ownership map, không bàn hiệu năng | NOT-COVERED | `parallel_engine.py:1` docstring nói ProcessPool, nhưng thực tế chỉ có `ThreadPoolExecutor` cho I/O (`:84`) và report (`:168`); `build_labeled_dataset_fast` là loop đơn (`:247-286`) |
| C5 | HPO tuần tự, không thread cap toàn cục | Plan §2 L56 (giới hạn workers/threads); `ml-plan-review-ml.md` ML23 (HPO mặc định 50 trials) | MATCHES (vấn đề), EXTENDS (vị trí) | `stock_agent/features/ensemble_model.py:523` `n_trials=50`; `:569` 3 folds; `:581-584` mỗi trial `n_jobs=1`/`thread_count=1`; `:620` `study.optimize` không có `n_jobs`; full fit `n_jobs=-1` tại `:173`, `:188`, `:207`; không file nào set `OMP_NUM_THREADS`/threadpoolctl |
| C6 | Optuna sampler không seed trong ensemble HPO | Plan §6.2 L276–277 yêu cầu "seed cố định" | CONFLICTS (code hiện tại khác target của plan) | `ensemble_model.py:619` `optuna.create_study(direction="maximize")` không có sampler seed; đối chứng `stock_agent/features/optimizer.py:369` có `TPESampler(seed=cfg.seed)` |
| C7 | GPU nhanh hơn 3–5× mỗi fit ở 60k–220k rows; hòa ở khoảng 5k rows | Không audit nào có số đo | NOT-COVERED | §4 bảng SYNTHETIC |
| C8 | Device nondeterminism: cùng device lặp lại thì giống hệt; CPU so với CUDA lệch tới max\|Δp\| = 0,13 | Plan §3.2 L125–126, §9 L371–373 (ghi threads/nondeterminism, seed không phải bitwise guarantee); `ml-plan-review-ml.md` ML11 | MATCHES (nguyên tắc), EXTENDS (đo được: device phải là một trường của environment) | §5 |
| C9 | Đo wall time theo stage, peak RAM, time-to-quality | Plan §2 L57–58, §9 L371; ML10 | MATCHES | Hiện chưa có instrumentation stage trong code; `research_gate.py:280` chỉ in return |
| C10 | Ưu tiên tối ưu CPU (bỏ copy prefix, song song theo symbol/variant) | Plan không nêu cụ thể | EXTENDS | §6 |
| C11 | Tiêu chí chuyển sang GPU | Plan §6.2 L279–280: "chỉ cân nhắc nếu … profile cho thấy cần thiết", chưa có ngưỡng | EXTENDS | §8 đưa ra ngưỡng định lượng đề xuất |
| C12 | RAM 15,7 GB và lỗi paging file giới hạn số process song song | `system-design-review.md` ghi 2,8 GB artifact `research_gate`, không bàn RAM | NOT-COVERED | Mỗi worker copy frames; cần đo peak RSS trước khi tăng workers |

## 3. Bottleneck theo code

Quy mô dữ liệu: `data/raw/prices_hist` có 101 CSV (11 MB), khoảng 2.199
rows/symbol. Tổng khoảng 220k rows cho VN100 và 66k rows cho VN30. Win-prob dùng
20 features (`win_probability.py:29-34`), v2 khoảng 60. Ma trận float32 khoảng
53 MB, nên VRAM 4 GB không phải giới hạn.

1. **Label builder** (`parallel_engine.py:257-286`): mỗi bar lại validate cả
   prefix, copy cả prefix features và gọi `score_symbol`. Chi phí O(T²)/symbol,
   chạy tuần tự.
2. **MR signal bank / period backtest** (`research_gate.py:130-139`,
   `period_backtest.py:65-67`, `momentum_books.py:167`): score mỗi bar trên
   `prepared.iloc[:i+1]`.
3. **Momentum replay** (`period_backtest.py:301`, `:320`): copy prefix của mọi
   symbol ở mỗi rebalance, và ở mọi ngày execution khi có `entry_gate`.
4. **Suite loop** (`research_gate.py:247-276`): blocks × variants × stresses
   chạy tuần tự. Bootstrap (`:51-68`) đã vectorized numpy, không phải điểm nóng.
5. **HPO** (`ensemble_model.py:519-620`): 50 trials × 3 folds × 3 boosters, tuần
   tự, 1 thread mỗi fit.
6. **SHAP top-5** có loop Python per row (`ensemble_model.py:429-437`). Bước này
   phụ thuộc CPU dù model chạy trên GPU.

GPU chỉ tác động tới mục 5 và một phần fit đơn lẻ. Mục 1–4 là logic Python/pandas
có nhiều rẽ nhánh. Theo Amdahl, nếu fit chỉ chiếm phần nhỏ thì speedup
end-to-end bị chặn chặt (xem MLSys evaluation).

## 4. Micro-benchmark — SYNTHETIC

**Toàn bộ số dưới đây dùng dữ liệu ngẫu nhiên (random float32, 40 features), không
dùng dữ liệu repo.** Mỗi ô là min của 2 lần. Cấu hình: 300 trees, depth 4,
lr 0,05; XGBoost `tree_method="hist"`. Python Anaconda
`C:/Users/acer/anaconda3/python.exe`. Scripts:
`C:/Users/acer/AppData/Local/Temp/claude/d--Chungkhoan/e65ae674-4bcd-4501-996a-7c1a23ac2c88/scratchpad/`
`gpu_bench.py`, `gpu_bench2.py`, `bench3.py`. Đây là scratchpad tạm ngoài repo,
có thể bị xóa.

| Rows | XGB CPU 20 thread | XGB CPU 1 thread | XGB CUDA | Ghi chú |
|---|---|---|---|---|
| 5.000 | 0,97 s | 1,18 s | 0,75 s | Gần hòa; overhead chiếm ưu thế |
| 60.000 (≈VN30) | 2,96 s | 4,81 s | 0,92 s | ≈3,2× so với CPU 20 thread |
| 220.000 (≈VN100) | 7,61 s | 14,36 s | 1,40 s | ≈5,4× |
| 1.000.000 | 21,7 s | 67,1 s | 4,1 s | Ngoài quy mô hiện tại |

| 220k rows, khác | Kết quả |
|---|---|
| LightGBM CPU / `device="gpu"` (OpenCL) / `device="cuda"` | 6,54 s / 6,09 s / lỗi build |
| CatBoost CPU / `task_type="GPU"` | 9,70 s / 3,22 s |
| 4 fit XGB: tuần tự `n_jobs=20` so với 4 process × `n_jobs=5` | 41,5 s so với 19,0 s (≈2,2× throughput, đã gồm spawn và tạo data) |
| Copy prefix `iloc[:i+1].reset_index` trên frame 2.200×60, mỗi symbol | 0,455 s (×100 symbols ≈ 45 s), so với 0,04 s khi truy cập trực tiếp theo index |

Giới hạn: đây là số micro-benchmark, không phải wall time của stage thật. Máy
laptop có thể throttle nhiệt, và CPU còn có tác vụ nền. GPU timing đã gồm
`predict_proba` (có đồng bộ do trả kết quả về host). Chưa đo time-to-quality.

## 5. Reproducibility và hash-pinned evidence

- Cùng device, cùng seed, 2 lần fit: max\|Δp\| = 0,0 (CPU và CUDA) trong các lần
  đo. Chỉ 2 lần nên chưa chứng minh mọi build/driver đều deterministic. GPU
  histogram dùng atomic reduction, và thứ tự cộng float có thể khác nhau.
- CPU so với CUDA, XGB `subsample=0.8`, cùng `random_state=42`: max\|Δp\| = 0,132
  trên 5.000 rows; top-100 theo xác suất chỉ trùng 86/100. Nguyên nhân hợp lý
  là RNG sampling và thứ tự reduction khác nhau giữa các device. Đây là suy
  luận, chưa tách riêng từng nguồn.
- Hệ quả: đổi device sẽ đổi predictions, thứ hạng signal, và từ đó đổi NAV và
  hash của artifact. Evidence chạy trên GPU **không so sánh trực tiếp** được với
  evidence CPU đã pin. `environment.json` (plan §9 L357) phải ghi device,
  library build, driver và threads. GPU run được xem là một lineage mới, cần
  chạy lại đầy đủ gate/books/timeslices theo AGENTS.md. Tolerance cho rerun
  phải được freeze trước (plan §9 L372–373).
- Seed của Optuna trong ensemble HPO cần được cố định trước khi bàn tới
  determinism của device (C6).

## 6. Phương án CPU có hiệu quả cao hơn

1. **Bỏ copy prefix per bar**: truyền index cùng các cột đã precompute, validate
   incremental hoặc vectorized một lần cho mỗi symbol. Ước tính ≥10× ở mục 1–3.
   Đây là ước tính, chưa đo end-to-end. Vì đụng tới research/evidence path,
   cần causality/accounting tests và full pinned suite theo AGENTS.md trước khi
   dùng kết quả.
2. **Song song theo symbol, block và variant** bằng `ProcessPoolExecutor` với số
   worker khai báo rõ. Kết quả phải độc lập với thứ tự thực thi (sort keys khi
   gộp).
3. **HPO**: seed sampler, chạy trial song song với workers × threads/fit ≤ 20.
   Đặt `OMP_NUM_THREADS` hoặc threadpoolctl để tránh oversubscription. Theo
   synthetic, cách này cho khoảng 2,2× throughput.
4. **Cache** prepared frames theo snapshot hash. Dùng `numba` (0.65 đã có) cho
   vòng exit-simulation (`parallel_engine.py:298`).
5. Số worker phải bị chặn bởi RAM (15,7 GB, đã có tiền lệ lỗi paging file). Đo
   peak RSS mỗi worker trước khi tăng.

## 7. Khuyến nghị và bước đầu có giới hạn

Bước đầu chỉ là **profiling, không đổi hành vi**:

- Bọc từng stage: load/verify snapshot, `precompute_signal_frames`, label loop,
  fit theo family, mỗi HPO trial, `mr_signal_bank`, replay theo block/variant,
  timeslices. Ghi `time.perf_counter`, peak RSS (`psutil`, `tracemalloc` cho
  Python heap), rows/symbols, threads/workers, device, versions.
- Chạy một lần trên một verified manifest đã pin, ghi kết quả vào evidence
  bundle (plan §9 L371). Xếp hạng các stage theo % wall time.
- Sau đó tối ưu stage lớn nhất trên CPU (mục 6.1 trước), đo lại cùng manifest,
  và chứng minh outputs không đổi (hash signals/NAV) trước khi tính speedup.

Việc này cần một task riêng được user cho phép, vì nó chạm vào research
pipeline. Bước profiling không tự cho phép thay đổi strategy hay schedule.

## 8. Tiêu chí chuyển sang GPU (đề xuất, chưa freeze)

Chỉ chuyển khi **tất cả** điều kiện sau đúng, sau khi đã tối ưu CPU:

1. Profile thật cho thấy boosting fit cộng HPO ≥ 40% wall time end-to-end.
2. Số rows mỗi fit ≥ 50k (dưới mức này overhead GPU làm mất lợi thế).
3. XGB hoặc CatBoost CUDA đạt cùng quality level trên inner development nhanh
   hơn CPU đã tối ưu, tính theo **time-to-quality**, không theo samples/s. Lặp
   ít nhất 2 lần và báo variance.
4. Model và data vừa VRAM 4 GB cùng các process khác. Chỉ chạy một GPU worker,
   vì nhiều process tranh VRAM dưới WDDM.
5. Device/build/driver được ghi trong manifest; GPU run được xem là lineage mới
   và chạy lại gate đầy đủ.

LightGBM CUDA cần build từ source (không thực tế trên Windows). torch CUDA wheel
chỉ cần nếu sau này có deep learning; RAPIDS/cuDF không hỗ trợ Windows native,
chỉ chạy qua WSL2, nên không phải lựa chọn cho repo này.

## 9. MLSys evaluation

| Kết luận | Nguồn gốc (chapter file + section/src line) |
|---|---|
| Profile để xác định bottleneck, rồi tối ưu đúng stage đó | `chapters/08-ai-training.md`; book §8.5 và §8.5.1, src 30734–30765 |
| Ưu tiên optimization impact cao và complexity thấp; cân nhắc reproducibility khi chọn kỹ thuật | §8.5, src 30793–30807 |
| Amdahl: phần không tăng tốc được giới hạn speedup (ví dụ 80% data movement thì tối đa 5×) | ch01, src 2818–2823 |
| Host và device memory tách biệt; transfer PCIe và synchronization gây stall | `chapters/11-ai-acceleration.md`; §11.4, src 45471–45493 |
| Batch nhỏ không tận dụng hết accelerator (giải thích vì sao hòa ở 5k rows) | ch08, src 29001–29007, 29385–29388 |
| Accelerator hợp với dense math có pattern đều; CPU hợp với irregular control flow (vòng per-bar) | §11.2.4, src 43626–43631, 43655–43658 |
| Hiệu quả cần phối hợp algorithmic, compute và data; không chỉ thêm hardware | `chapters/09-efficient-ai.md`; §9.4 và §9.4.1, src 34766–34785; src 34125–34131 |
| Utilization thấp có thể do data movement/I/O; tăng compute không giúp nếu bottleneck ở chỗ khác | `chapters/12-benchmarking-ai.md`; §12.7.2.3, src 51949–51970 |
| Đánh giá training bằng time-to-accuracy và tìm bottleneck ở data loading | ch12, src 51665–51672, 51830–51834 |
| Hiệu năng là của cả workflow, không của một component | ch12, src 50186–50189 |
| Kết quả biến động theo hardware/software; reproducibility cần chuẩn hóa protocol | §12.7.2.6, src 52004–52016 |

"Time-to-quality" là cách skill khái quát hóa từ time-to-accuracy của sách, như
`ml-plan-review-ml.md` ML10 đã ghi. Các ngưỡng 40% / 50k rows trong §8 là đề
xuất của dự án, không phải quy tắc của sách.

## 10. Giới hạn và handoff

- Không đo wall time thật của stage nào trong repo (cấm chạy pipeline code trên
  repo data). Mọi ước tính speedup end-to-end đều UNVERIFIED cho tới khi chạy
  bước profiling ở §7.
- Changed paths: chỉ file này. Không chạy test; không có test count mới.
- Next bounded action: user quyết định có mở task profiling stage (chỉ
  instrumentation, đo trên verified manifest) hay không. Không suy ra quyền
  cài package, đổi device mặc định, push hay promote từ audit này.

## Cập nhật sau merge research_ml (06/10, 4403fc7)

Bối cảnh: merge research_ml không chạm vào hot path của audit này (`parallel_engine.py`,
`research_gate.py`, `period_backtest.py`, `momentum_books.py`, `ensemble_model.py`,
`win_probability.py` không đổi giữa `4b859cd` và `4403fc7`). R5 chạy trên research_ml với
sklearn ridge/logistic, không HPO, không boosting
(`configs/research/ml_protocol_v1.json:29`, `ml_protocol_v2_weekly.json`).

| ID | Trạng thái | Evidence @ `4403fc7` / ghi chú |
|---|---|---|
| Kết luận "chưa nên chuyển GPU" | CHANGED (củng cố) | R5 weekly: 6.060 lot labels, ridge; rank IC ≈ 0, shuffled control thắng (`docs/audits/2026-10-04-r5-weekly-results.md:12,27-32`); R6 ensemble không có cơ sở. Workload và nhu cầu boosting đều nhỏ hơn giả định của §8 |
| C1 profile trước, không mặc định GPU | N/A (MATCHES, không đổi) | |
| C2 library readiness | STILL OPEN / ghi chú | Môi trường chạy `test_ml_*` không có LightGBM/XGBoost/CatBoost (`2026-10-04-r1-r4-implementation.md:12`); không có probe mới |
| C3 copy prefix O(T²) | STILL OPEN (legacy) | research_ml F0/F2 vectorized theo mã (`features.py:383-415`), nên tránh được trên research path. Nhưng R5 MR vẫn lấy candidate từ `research_gate.mr_signal_bank` (`scripts/ml_research_r5.py:207`), tức vẫn đi qua loop O(T²) |
| C4 "parallel engine" chạy tuần tự | STILL OPEN | Không đổi |
| C5 HPO tuần tự, không cap thread | STILL OPEN (legacy) | research_ml không có HPO |
| C6 Optuna không seed | STILL OPEN (legacy) / RESOLVED (research path) | Seed cố định `20261004` (`ml_protocol_v1.json:29`) |
| C7 micro-benchmark synthetic | N/A | Không đo lại |
| C8 device/build/threads trong `environment.json` | STILL OPEN | `environment.json` của R5 chỉ có python/pkgs/seed (`scripts/ml_research_r5.py:219`, `scripts/ml_research_r5_weekly.py:140`; `docs/audits/2026-10-04-r5-weekly-bundle/environment.json`) |
| C9 đo wall time/RSS theo stage | STILL OPEN | Runner R5 không có instrumentation |
| C10 tối ưu CPU | STILL OPEN | Không đổi |
| C11 tiêu chí chuyển GPU | STILL OPEN (chưa freeze) | |
| C12 giới hạn RAM/paging file | STILL OPEN | Ngày 06/10, ổ D: từng về 0 byte trống khi ghi file. Cần tính cả dung lượng đĩa (evidence bundle, catalog raw) |
| Mới: `asof_join` là loop Python theo dòng | STILL OPEN | `contracts.py:156-177`; `r1-r4-implementation.md:17` ("benchmark before real-size use") |
| Mới: `Catalog.read_prices` rehash mọi raw mỗi lần đọc | STILL OPEN | `catalog.py:160-169`; chi phí tăng theo archive |
| Mới: `build_foreign_feature` lọc `net` theo từng mã | STILL OPEN (nhẹ) | `features.py:432-434`, O(symbols × rows) |
| §7 bước đầu: profiling | STILL OPEN | Chưa có task |
| §8 tiêu chí GPU | STILL OPEN | Cần freeze lại theo quy mô thực tế của R5 |

Còn actionable:
1. Bước đầu vẫn là profiling không đổi hành vi. Nên ưu tiên `mr_signal_bank` (O(T²)), `asof_join` và `Catalog.verify` trước khi đưa research_ml lên dữ liệu thật quy mô lớn.
2. Thêm device/threads/build vào `environment.json` của runner R5 trước run kế tiếp.
3. GPU không có lý do cho tới khi một lane vượt control và cần boosting/HPO.
