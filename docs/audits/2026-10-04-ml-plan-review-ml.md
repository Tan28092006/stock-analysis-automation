# S-ML — đối chiếu kế hoạch với nguồn gốc ML Systems

Ngày: 04/10/2026. Worker: `/root/review_ml_sources`; coordinator: `/root`.
Review read-only đối với plan/code/config/data; chỉ báo cáo này được tạo.

## 1. Kết luận và phạm vi bằng chứng

**SUPPORTED WITH QUALIFICATIONS**, không phải research pass hay trading approval.
Kế hoạch phân biệt khá tốt code hiện hữu, thiết kế đề xuất và hiệu quả chưa đo.
Các luận điểm lifecycle, benchmark chất lượng cùng tài nguyên, provenance,
training-serving consistency, rollout và monitoring có hỗ trợ trực tiếp từ sách.
Nguồn đã được kiểm tra là bản gốc trích xuất, không chỉ guide của skill.

Ba điểm cần sửa/ghi rõ trước implementation:

1. **P2 — giới hạn attribution:** no-auto-retrain/promote là guardrail dự án và
   tổng hợp của skill, không lời cấm literal từ Reddi. Nguồn còn thảo luận các
   triggers tự động trên drift/performance. Bundle rollback toàn dependency là
   suy luận kỹ thuật từ lineage + rollback, không checklist được sách đưa nguyên
   văn. Không đổi guardrail; đổi cách gán nguồn.
2. **P2 — current contract yếu hơn target contract:** availability hiện theo
   **ngày** và trường label availability tùy chọn. Helper kiểm fitted timestamp,
   không có riêng actual calibrated/evaluated timestamps. R4 phải thêm contract
   thời gian rõ ràng; “giữ helper” không tự đáp ứng target requirement.
3. **P2 — freeze tiêu chí ML:** plan nhận biết ít phiên/hai classes nhưng chưa
   chốt numerical floors cho ML folds/events/classes/calibration và quality
   threshold của time-to-quality. Để ở PLANNED là hợp lệ; phải đưa các trị số và
   quy tắc INCONCLUSIVE vào preregistration trước R5, không chọn sau test.

Không phát hiện plan tuyên bố đã có true stacking hoặc có lợi nhuận mới: code
`train()` hiện là bốn base voting mean, và plan nói chính xác như vậy.
Không có training/backtest, cài dependency, broker action, push hoặc deployment.

## 2. Input, source integrity và ranh giới

- Frozen plan: `docs/plans/2026-10-04-data-ml-research-plan.md`.
- SHA256 trước và sau đọc: `13985318dcd8b889c9a01e1725c1fc24282122aae29d6167d1d96f99d1ed5bb7`.
- Base code HEAD: `6c3b8f422dd522ac0e4a892f23319e4a431b1ef7` từ brief;
  đây là inspection revision, không release approval.
- Root AGENTS và shared Firstmate brief đã đọc đầy đủ. Skill
  `C:/Users/acer/.agents/skills/machine-learning-system/SKILL.md` đọc đầy đủ;
  guides 05/06/08/12/13 và source-provenance đã đọc đầy đủ.
- Original: Vijay Janapa Reddi và contributors, *Machine Learning Systems:
  Principles and Practices of Engineering Artificially Intelligent Systems*,
  PDF snapshot 20/10/2025, 2.620 trang.
- PDF local `D:/data sciiiiii/HAVARD ML.pdf` có thật, hash thực đo
  `a4b42e5e75aac76789dd28f763bd2e1c17890b52f0c464f278f346eb442e8bf6`,
  trùng provenance skill.
- Extracted original `assets/book.txt` hash thực đo
  `ca2cfc358caa54e2f1932319382bb239b644af2cf4c29e5a648feb75f1cfe443`;
  `assets/index.json` hash
  `2a5722d639f8e683adaf2a6afed4fc5a0b0f9539715d775a1eb2550a6e178584`.
- Hash xác nhận identity/integrity, không chứng minh toàn bộ OCR hoàn hảo. Các
  claim đã đọc là prose, không phụ thuộc phương trình/figure chưa render.
- Source-line dưới đây là dòng `book.txt`, **không phải PDF page**. Chapter page
  ranges: ch1 printed 1–52/PDF61–112; ch5 273–322/333–382; ch6
  323–428/383–488; ch8 543–662/603–722; ch12 997–1100/1057–1160; ch13
  1101–1198/1161–1258. Nêu precise line là locator chính.

Support levels: DIRECT = nguồn nói cơ chế đó; SYNTHESIS = tổng hợp hợp lý nhưng
không literal author rule; PROJECT = quyết định dự án, cần test; CODE = kiểm tra
code, chưa chạy behavioral test; UNVERIFIED = thiếu empirical/source evidence.
Không chuyển DIRECT thành chứng minh edge hay hiệu quả giao dịch.

## 3. Claim matrix — toàn bộ owned ML families

| ID / plan lines | Assertion và origin | Primary locator thực sự đọc | Support / limitation / correction |
|---|---|---|---|
| ML01 §1 L11–13 | Metric phải gắn mục tiêu; AUC/win-rate không thay net NAV. Project + systems principle | Reddi §5.4 lines15829–15854; §13.5.3.3 lines56985–57004 | DIRECT về mục tiêu/constraints và business impact; PROJECT về net/excess/activity cụ thể. Không có bảo đảm lợi nhuận từ sách. |
| ML02 §1 L22 | Split theo ngày/purge availability đã có. Code fact | `temporal_validation.py`17–30,66–100; Reddi ch6 lines21416–21425 hỗ trợ PIT feature history | CODE đúng nhưng date-level, availability field optional. Sửa wording current-vs-R4; sách không quy định financial purge implementation. |
| ML03 §1 L23 | Bốn base, hiện voting mean, chưa fitted stack. Code fact | `ensemble_model.py`223–238,240–308,672–683 | CODE supported. `_fit_bases` đặt `meta_model=None`; metrics khai `simple_average`; class/docstrings không là implementation. Không cần sửa. |
| ML04 §1 L24 | Preprocessing được fit train trong đường train/HPO. Code fact | `ensemble_model.py`225,652–670; HPO553–578; `calibration.py`395–476 | CODE supported cho winsor bounds. Imputation hiện row-local constants/neutral defaults, **không** train-fit median imputer; robust name không là robust scaler. Future median/native-missing là đề xuất. |
| ML05 §1 L25–26 | T+2 defaults; sigmoid Ridge không chứng minh calibrated probability; thiếu base có0.5. Code fact | `ensemble_model.py`244,227,673–700; `calibration.py`81–99 | CODE supported. Không coi transformed scores cùng calibrated unit mặc nhiên; efficacy calibration chưa đo. |
| ML06 §1 L27, §9 L335–343 | Candidate-only, chưa live; shadow tách authority. Project + code | `model_release.py`27–46,109–171; Reddi §13.5.1 lines56395–56419 | CODE/DIRECT hỗ trợ candidate boundary và incremental rollout. Financial execution/authority là PROJECT; sách không approve candidate này. |
| ML07 §1 L34–37, §6 L201–206 | Split không đủ chứng minh inputs PIT/leakage-free; today's fit không artifact năm2022. Project correctness | Reddi ch6 lines21416–21425; `model_available_at`47–63 | DIRECT về truy cập lịch sử đúng thời điểm; PROJECT về historical-fit semantics. Snapshot historical adjustment còn UNVERIFIED; cần archival evidence riêng. |
| ML08 §2 L41–50, §10 | Lifecycle/data/features/learning/evaluation/release/monitoring và dependency stages. Book mechanism + project workflow | Reddi §5.4 lines15787–15864 | DIRECT về sáu stages và feedback; diagram/tasks là PROJECT. Không ép waterfall: failure có thể quay về source/feature contract. |
| ML09 §2 L52–56 | Local tabular trước, không mặc định GPU/services, cap threads. Project adaptation | Reddi ch8 lines30793–30807; ch12 lines51949–51970 | SYNTHESIS: profile/bottleneck/complexity trade-off hỗ trợ minimal stack; SQLite/Parquet/thread budget không sách bắt buộc. Need benchmark trước migration. |
| ML10 §2 L57–58 | Stage wall time/RAM/I/O và time-to-quality, không throughput-only. Book + skill adaptation | Reddi ch12 lines51830–51834,51909–51917,51949–51970,52018–52026 | DIRECT dùng **time-to-accuracy**; time-to-quality là skill mở rộng cho non-accuracy metric. Chốt metric/threshold ở inner development trước đo; không repeatedly look at outer test để dừng search. |
| ML11 §3 L102–104, §9 L324–326 | Data/config/code/runtime dependency closure/seeds/nondeterminism. Book mechanism + project implementation | Reddi ch6 lines18916–18924; ch8 lines33534–33541; ch12 lines52004–52016; §13.4.1.3 lines56019–56049 | DIRECT tracking/versioning; full hash closure và frozen tolerances là SYNTHESIS/PROJECT. Seed alone không bitwise guarantee. Broad requirements còn chưa lock. |
| ML12 §4 L117–120,143–145 | Feature schema/availability allowlist, train-fit transforms, serialize, parity. Project + general book support | Reddi §6.7.5 lines20177–20189; §13.4.1.3 lines56019–56049 | DIRECT pipeline consistency/lineage, PROJECT feature registry/fit-per-fold/fail-closed. Không viện Reddi như nguồn của mỗi statistical transformation. |
| ML13 §4 L130–134 | Redundancy/error interpretation, SHAP không chứng minh causal edge. Project statistical caution | Reddi ch13 lines56407–56415 nói attribution/slices để diagnose | SYNTHESIS cho diagnostic; causal-null/permutation cụ thể không được các đoạn này xác nhận. Cần prereg controls; không chạy result mới trong review. |
| ML14 §4 L136–153 | Model-specific missing/imputer/scaling/category/overlap controls. Project + FE skill | Reddi §6.7.5 lines20177–20189 supports parity only; actual preprocess395–476 | PROJECT, không chứng minh lựa chọn median/native missing sẽ tốt hơn. FEAZ scout sở hữu method citations. Overlap-independent-days là financial inference rule, không literal Reddi. |
| ML15 §5 L163–176 | MR15 conditional labels, momentum21/5/63, rank-overlay, portfolio costs. Project proposal | `EnsembleTrainer.train`244 xác nhận target hiện khác; Reddi §5.4 supports task-fit objectives only | PROJECT/UNVERIFIED về efficacy. Cần freeze candidate population, label endpoint counting và execution before results. 15/21/5/63 không do Reddi prescribe. |
| ML16 §5 L178–181 | cost_config không trực tiếp delegation; trace/test trước kết luận. Code fact | `calibration.py`81–99 | CODE supported: signature accepts cost_config but call passes symbols/rules/price_dir/start/end. Plan giữ thận trọng đúng; chưa chứng minh downstream rules có/không cùng costs. |
| ML17 §5 L183–186 | Label actual end/availability/pending/quarantine/portfolio parity. Project contract | `temporal_validation.py`17–44; Reddi ch6 lines21416–21425; ch13 lines55923–55928 | CODE partially existing; PROJECT đầy đủ timestamp/provenance/execution. Missing resolved/quality cols hiện không reject; R1/R4 phải declare required metadata. |
| ML18 §6.1 L192–200 | Group date, actual label purge, nested inner/outer, separate matured calibration. Project research protocol | `purged_time_split`66–85/`purged_walk_forward`88–100; Reddi §13.4.2.3 lines56213–56224 | DIRECT holdout unused train/validation; CODE date-group/purge only. Nested HPO/calibration/embargo specifics là PROJECT, không chứng minh bởi Reddi guides. Numerical day/event/class/calibration floors còn freeze-needed. |
| ML19 §6.1 L201–204 | Actual fitted/calibrated/evaluated timestamps prospective. Project requirement vs code | `model_available_at`47–63; Reddi ch13 lines56019–56049 | CODE chỉ trained_at và fit/calibration/evaluation label-end dates; **không** separate calibrated_at/evaluated_at. Sửa plan để yêu cầu R4 tracking/check rõ ràng, giữ legacy helper là necessary but insufficient. |
| ML20 §6.2 L210–215 | Simple baselines→singlebases→mean→learnedstack→regime; không mix score units. Project experimental ladder | Reddi §5.4 lines15844–15854 hỗ trợ techniques task-fit/eval | PROJECT. Không có source inspected prescribe model lineup/ordering hay rank-vs-prob adapter; đúng khi gọi đề xuất cần benchmark. |
| ML21 §6.2 L217–222 | Forward OOF past-only preprocessing/HPO, matured meta labels, outer evaluation, separate calibration. Project design | Reddi §13.4.2.3 lines56213–56224 hỗ trợ untouched holdout general; `train()`240–308 không lưu/train meta OOF | PROJECT, chưa implemented. Query stacking ch12 trả0, calibration ch13 trả0: **không** evidence cả sách không có, chỉ không corroborated những chapter scopes. Không gán OOF method cho Reddi. |
| ML22 §6.2 L224–228 | Diversity/drop-base/seeds/fixed-mean control/fail-closed/missing0.5/no warm-start. Project + code baseline | `ensemble_model.py`310–316,673–683; Reddi ch12 lines52004–52016 supporting variability | PROJECT test design; diversity improvement UNVERIFIED. Current missing0.5 vẫn code fact, proposed reject chưa runtime. Daily update already fresh purged retrain but not prospective promotion permission. |
| ML23 §6.2 L230–234 |20trials/family/fixedwallbudget; no outer retune; no GPU until profile. Project | Reddi ch8 lines30793–30807; ch12 lines51949–51970,52004–52016 | PROJECT numerical budget; no claim20 là optimum. Pipeline current HPO default50, objective profit-factor×log-count on independent lot returns; future net NAV acceptance phải dùng portfolio. |
| ML24 §7 L238–273 | Registry/stagewise/negativecontrols/retained failures/multiplicity | Reddi ch8 lines33534–33541; ch13 lines56019–56049 | DIRECT experiment tracking; particular prereg/family correction/null tests are PROJECT. No new source establishes power/profit of listed theories. |
| ML25 §8 L291–316 | Calibration/ranking metrics supplement netprofit/activity; dateblock uncertainty; separate statuses | Reddi ch12 lines51053–51060; §13.5.3.3 lines56985–57004 | DIRECT multi-dim/business-aligned benchmark; financial bootstrap/Brier/PR-AUC/statuses PROJECT. Thresholds/sample feasibility not established by literature. |
| ML26 §9 L319–332 | New evidence bundle per run, validator, full lineage, training fixture, no mock-only ensemble pass | Reddi ch6 lines18916–18924; ch13 lines56019–56049,56058–56067 | DIRECT provenance+CI/ML validation; proposed paths/artifact schema/tests PROJECT not implemented commands. No prior tests certify future ensemble pipeline. |
| ML27 §9 L335–337 | Full compatible release bundle/rollback, not pickle-only. Skill synthesis + project adaptation | Reddi §13.4.1.3 lines56019–56049; §13.5.1 lines56403–56419 | SYNTHESIS supported by tracked components + model rollback; original excerpt says previous stable model version, **not** exact full-bundle inventory. Preserve design, qualify attribution. |
| ML28 §9 L339–343 | Data/model/system/business monitoring; drift alert no automatic retrain/promote; paper≠actualfills | Reddi lines56756–56842,56985–57004; contrary automatic-trigger discussion55249–55254,56068–56091 | DIRECT monitoring distinctions/business impact; four-way grouping SYNTHESIS. No-auto-retrain is PROJECT conservative policy, not direct author's ban. Paper/fill/receipts distinction PROJECT execution requirement. |
| ML29 §11 L369–371 | Skill contributes triangle/lifecycle/time-to-quality/repro/monitor/rollback | Reddi ch1 lines2429–2434; §5.4; ch12 time-to-accuracy; ch13 lineage/rollout refs above | Supported with wording refinement: clarify primary-source principles vs skill generalization vs repo financial protocol. Add precise source locators rather than only guides. |

## 4. Proposed bounded corrections for coordinator

No shared files edited by scout. Suggested replacements/additions:

- L22 table: “Grouped signal dates; purge theo exit date hoặc optional
  `label_available_date` ở date granularity. R4 bổ sung mandatory availability
  timestamps/provenance; current split không chứng minh PIT.”
- After L204: “Helper hiện có chỉ kiểm timezone của trained_at và các label-end
  dates; chưa lưu/check riêng calibrated_at/evaluated_at. R4 phải version contract,
  bắt buộc timestamps của toàn bộ fit/selection/calibration/evaluation và data
  availability trước decision; không chỉ tái sử dụng helper rồi claim hoàn tất.”
- Near L200/230: “Trước R5 freeze minimum independent dates/events và per-class
  counts ở mỗi fold/calibration cùng target quality threshold, bootstrap/power
  diagnostics và INCONCLUSIVE policy. Time-to-quality chỉ dùng threshold fixed
  trên development; outer test không là stopping/HPO signal.”
- §11 attribution: “Reddi ch12 dùng time-to-accuracy; time-to-quality là tổng hợp
  skill. Sách hỗ trợ lineage/controlled rollout/monitoring; full-bundle rollback
  và no-auto-retrain/promote là thiết kế dự án suy ra theo rủi ro, không literal
  rule của sách.” Add source locator table linking this audit.
- Optional clarification L165: specify entry-day index convention, e.g.
  `entry index=e; exit index=e+15`, or inclusive 15-session hold=`e+14`;
  pick one before labels. Current wording is formula but colloquial “15 ngày”
  can be read inclusively. This scout does not choose financial convention.

## 5. Actual read/check commands and errors

All commands local/read-only; `python` is existing environment, reader inspected
first and only prints bounded source text. Commands are reproducible examples
actually executed here, not suggested training steps:

```powershell
Get-Content -Raw AGENTS.md
Get-Content -Raw docs/audits/2026-10-04-ml-plan-review.md
Get-Content -Raw C:/Users/acer/.agents/skills/machine-learning-system/SKILL.md
Get-FileHash docs/plans/2026-10-04-data-ml-research-plan.md -Algorithm SHA256
Get-FileHash 'D:/data sciiiiii/HAVARD ML.pdf' -Algorithm SHA256
Get-FileHash C:/Users/acer/.agents/skills/machine-learning-system/assets/book.txt -Algorithm SHA256
Get-FileHash C:/Users/acer/.agents/skills/machine-learning-system/assets/index.json -Algorithm SHA256
python C:/Users/acer/.agents/skills/machine-learning-system/scripts/read_source.py --section 5.4 --max-lines 85
python C:/Users/acer/.agents/skills/machine-learning-system/scripts/read_source.py --section 6.2 --max-lines 70
python C:/Users/acer/.agents/skills/machine-learning-system/scripts/read_source.py --chapter 1 --query triangle --matches 3 --context 6 --max-lines 50
python C:/Users/acer/.agents/skills/machine-learning-system/scripts/read_source.py --chapter 6 --query leakage --matches 5 --context 8 --max-lines 100
python C:/Users/acer/.agents/skills/machine-learning-system/scripts/read_source.py --chapter 6 --query training-serving --matches 5 --context 8 --max-lines 100
python C:/Users/acer/.agents/skills/machine-learning-system/scripts/read_source.py --chapter 6 --query point-in-time --matches 5 --context 8 --max-lines 75
python C:/Users/acer/.agents/skills/machine-learning-system/scripts/read_source.py --chapter 8 --query reproducib --matches 6 --context 8 --max-lines 100
python C:/Users/acer/.agents/skills/machine-learning-system/scripts/read_source.py --chapter 12 --query time-to --matches 8 --context 7 --max-lines 140
python C:/Users/acer/.agents/skills/machine-learning-system/scripts/read_source.py --chapter 12 --line 51935 --max-lines 70
python C:/Users/acer/.agents/skills/machine-learning-system/scripts/read_source.py --chapter 12 --line 52004 --max-lines 24
python C:/Users/acer/.agents/skills/machine-learning-system/scripts/read_source.py --chapter 12 --query 'three dimensions' --matches 3 --context 6 --max-lines 50
python C:/Users/acer/.agents/skills/machine-learning-system/scripts/read_source.py --chapter 13 --query rollback --matches 6 --context 8 --max-lines 100
python C:/Users/acer/.agents/skills/machine-learning-system/scripts/read_source.py --chapter 13 --query drift --matches 8 --context 6 --max-lines 100
python C:/Users/acer/.agents/skills/machine-learning-system/scripts/read_source.py --chapter 13 --query retraining --matches 16 --context 5 --max-lines 170
python C:/Users/acer/.agents/skills/machine-learning-system/scripts/read_source.py --chapter 13 --line 56010 --max-lines 60
python C:/Users/acer/.agents/skills/machine-learning-system/scripts/read_source.py --chapter 13 --line 56200 --max-lines 63
python C:/Users/acer/.agents/skills/machine-learning-system/scripts/read_source.py --chapter 13 --line 56700 --max-lines 150
python C:/Users/acer/.agents/skills/machine-learning-system/scripts/read_source.py --chapter 13 --query business --matches 12 --context 3 --max-lines 85
Get-Content -Raw stock_agent/features/temporal_validation.py
Get-Content stock_agent/features/ensemble_model.py | Select-Object -Skip 220 -First 190
Get-Content stock_agent/features/ensemble_model.py | Select-Object -Skip 650 -First 51
Get-Content stock_agent/features/ensemble_model.py | Select-Object -Skip 520 -First 98
Get-Content stock_agent/features/calibration.py | Select-Object -Skip 394 -First 91
rg -n 'def build_labeled_t2_dataset|cost_config' stock_agent/features/calibration.py
rg -n 'live_approved|calibrat|fitted_at|training_metadata|stage|candidate' stock_agent/features/model_release.py
git status --short
```

Zero-match searches actually performed: ch12 `stacking`; ch13 `calibration`;
ch12 `data leakage`; ch13 `monitoring dimensions`. They establish only absence
of literal match in bounded chapter scope, not whole-book absence. Additional
reads ch13 query `version`/section list/`13.5.5`, line56485, and numbered full
temporal-validation supplied code/source locations. One erroneous `rg` path
`stock_agent/features/dataset_builder.py` returned nonexistent-file error;
recovered by searching `features` and reading actual `calibration.py` delegation.
Some multi-command tool outputs truncated; required guides/claims were reread
in smaller batches. No truncated excerpt is treated as full chapter review.

## 6. Unverified/blockers and handoff

- No empirical profit, independent event count, power, calibrated probabilities,
  score diversity, model speed or superiority measured. Literature support does
  not turn any H-* hypothesis into acceptance.
- No archival adjustment/PIT source certification established by ML source review.
- Financial purged OOF, nested selection, costs/execution and block bootstrap are
  project protocols; original financial/statistical method papers are not verified
  in this family review. Plan is allowed to propose them, must not attribute them
  as exact Reddi methods or verified efficacy.
- Review completeness: lifecycle/data consistency/benchmark/resources/repro/
  release/monitoring plus ensemble/preprocessing/label/availability code claims
  covered; FEAZ method fidelity and DDIA/FDE/Kimball are other scouts' ownership.
- Dirty tree at inspection includes coordinator onboarding edits plus existing
  `data/pipeline/local_server.log`, `data/pipeline/swing_scan_cache.json` and
  system review. Scout did not modify these or shared plan. Final hash remained
  frozen before writing owned report.
- Changed path: only this audit. Verification is content/source inspection;
  **no new test counts claimed**, no runtime behavior changed.
- Next bounded action: coordinator independently recheck proposed attribution and
  temporal-contract edits, integrate shared plan after all scouts finish, then
  update corrected-plan hash and routing/link verification.
