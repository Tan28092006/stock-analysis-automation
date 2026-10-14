# S-FE: đối chiếu feature / preprocessing với FEAZ gốc

Ngày kiểm tra: 04/10/2026. Worker: `/root/review_feature_sources`.
Phạm vi: review nguồn và kế hoạch; không chạy experiment, training hay backtest.

## Input cố định và verdict

- Plan: `docs/plans/2026-10-04-data-ml-research-plan.md`, SHA256
  `13985318dcd8b889c9a01e1725c1fc24282122aae29d6167d1d96f99d1ed5bb7`.
- Code inspection baseline: `6c3b8f422dd522ac0e4a892f23319e4a431b1ef7`.
- Line spans dưới đây thuộc **plan trước integration**, không phải line numbers
  của bản được coordinator sửa sau review.
- Verdict: **SUPPORTED WITH CORRECTIONS**, chỉ về nguồn/lập luận. Không chứng
  minh feature, ensemble hay phương pháp giao dịch có lợi nhuận.

Hai sửa đổi có ý nghĩa trước khi triển khai:

1. H-P01 gọi control là zero-fill, nhưng preprocess hiện tại dùng neutral values
   theo feature rồi winsorize 1/99. Freeze baseline thực tế riêng với artificial
   blanket-zero control; không gọi hai pipeline này tương đương.
2. §11 gán model-specific preprocessing cho FEAZ quá rộng. Chương models gốc
   vẫn WIP. Cần ghi estimator-specific/native-missing choices, temporal rules,
   stacking và financial inference là project design/skill synthesis; sách chỉ
   hỗ trợ các cơ chế thực sự được đối chiếu dưới đây.

## Reader, phiên bản và source coverage

Đã đọc đầy đủ `feature-engineering-az/SKILL.md` tại thư mục người dùng chỉ định,
và các guides `numeric`, `missing`, `correlated`, `categorical`, `imbalanced`,
`time-series`, `models`, `patterns`. Đã đọc root AGENTS, onboarding và shared
Firstmate brief. Không coi các guides/patterns là tài liệu gốc.

Theo routing `agent-reach/SKILL.md` và `references/web.md`, reader thực tế:

```text
curl.exe -s --max-time 35 "https://r.jina.ai/https://feaz-book.com/<chapter>.html"
curl.exe -s --max-time 30 "https://r.jina.ai/https://raw.githubusercontent.com/EmilHvitfeldt/feature-engineering-az/8f53af54788b3f0917ed442e0e1bdef13c14c560/<chapter>.qmd"
```

Reader trả nonempty body/title/URL, exit code 0 cho các URL trong bảng. Không
download corpus/sách, không upload source hay ghi raw book content vào repo.
Một số combined tool outputs bị truncate bởi navigation/rendered sklearn HTML;
đã đọc lại body windows và **toàn bộ pinned QMD** của các chương cần dùng.
Không dùng đoạn bị truncate làm chứng minh thiếu nội dung. `models.html` tail
và pinned `models.qmd` đều có WIP; lag/rolling body cũng được đọc riêng.

Skill pinned ở commit `8f53af5` (full hash ở command), còn website live không
có deployment commit được reader chứng nhận. Vì thế không gọi website live là
bitwise snapshot đó. Pinned original QMD đã được đối chiếu thêm cho **cả 14
chương dưới đây**. Các phép đọc không lấy response-byte SHA256; content identity
ở đây là original repo full commit + filename, không phải hash của Jina rendering.

| Source ID | Original rendered chapter / locator | Pinned QMD đã đọc; nội dung thực tế |
|---|---|---|
| FE01 | [Robust scaling](https://feaz-book.com/numeric-robust.html), opening / Pros and Cons / Python Examples | `numeric-robust.qmd`: median/IQR learned on training, apply to new data; outlier, zero-spread and sparse-data cautions; RobustScaler example |
| FE02 | [Yeo–Johnson](https://feaz-book.com/numeric-yeojohnson.html), opening / trained preprocessing / Pros and Cons | `numeric-yeojohnson.qmd`: positive and negative domains, lambda estimated on training to avoid leakage; not universal normalization; formula inspected in QMD because rendering drops math |
| FE03 | [Missing overview](https://feaz-book.com/missing.html), mechanisms / sentinels / Imputation / Indication | `missing.qmd`: column/row missingness, MCAR/MAR/MNAR, domain investigation, distinct sentinels; does not identify this provider's mechanism |
| FE04 | [Simple imputation](https://feaz-book.com/missing-simple.html), opening / drawbacks / Python Examples | `missing-simple.qmd`: median/mean/mode/manual fill, impossible feature combinations, distribution change; substantive despite TODO examples |
| FE05 | [Missing indicators](https://feaz-book.com/missing-indicator.html), §45.1–45.2 / Python Examples | `missing-indicator.qmd`: Boolean missing flags may add signal or noise; zero variance and dimensionality concerns; substantive despite TODO |
| FE06 | [Correlation overview](https://feaz-book.com/correlated.html), opening | `correlated.qmd`: possible redundancy and model-dependent issue; not proof RSI/EMA are useless |
| FE07 | [Correlation filter](https://feaz-book.com/correlated-filter.html), iterative/clustering methods / Pros and Cons | `correlated-filter.qmd`: learned selection, threshold tuning and signal loss; substantive draft, TODO includes outcome in example; no ready sklearn implementation supplied |
| FE08 | [Unseen categories](https://feaz-book.com/categorical-unseen.html), opening / R Examples / Python Examples | `categorical-unseen.qmd`: save train vocabulary, policy for new levels; R example, TODO diagrams, no Python solution supplied in chapter |
| FE09 | [Imbalance overview](https://feaz-book.com/imbalanced.html), metrics/weights/thresholds / Changes to Data Distribution | `imbalanced.qmd`: accuracy warning, weights/thresholds/sampling alternatives, sampling changes distribution |
| FE10 | [SMOTE](https://feaz-book.com/imbalanced-smote.html), mechanism / limitations | `imbalanced-smote.qmd`: minority-neighbor interpolation, numeric/nonmissing base case, outlier/high-dimensional/quality limitations; no temporal protocol |
| FE11 | [Time-series overview](https://feaz-book.com/time-series.html), §114.1 | `time-series.qmd`: dependence and special precautions acknowledged, assumes precautions already known, TODO external resource; not financial validation handbook |
| FE12 | [Lag features](https://feaz-book.com/time-series-lag.html), §121.1–121.4 | `time-series-lag.qmd`: WIP + empty example/pros/cons scaffolding |
| FE13 | [Rolling features](https://feaz-book.com/time-series-rolling-window.html), §122.1–122.4 | `time-series-rolling-window.qmd`: WIP + empty scaffolding |
| FE14 | [Model-specific input](https://feaz-book.com/models.html), chapter body | `models.qmd`: WIP only |

Pinned locators resolve under [original source commit](https://github.com/EmilHvitfeldt/feature-engineering-az/tree/8f53af54788b3f0917ed442e0e1bdef13c14c560).
Do not reinterpret `draft` as `empty`: FE04/05/07/08 have usable mechanisms but
unfinished examples/sections. Do not upgrade FE12/13/14 to substantive chapters
merely because the skill patterns contain sensible guidance.

## Claim–evidence matrix

Support labels: DIRECT = primary text supports mechanism; SYNTHESIS = reasonable
skill/project adaptation, not author's exact prescription; PROPOSAL = testable
project choice without efficacy evidence; CORRECTION = material wording/control
problem; UNVERIFIED = missing implementation/statistical evidence.

| Plan span / claim | Origin / primary locator inspected | Support and limitation | Correction / required evidence |
|---|---|---|---|
| 24: `_prepare_features` fits preprocessing on train | Code fact; `ensemble_model.py:652–669`, `calibration.py:397–469` | DIRECT code: fit computes winsor bounds, transform reuses them. Does not certify every call path or missing semantics | Preserve wording as bounded observation, trace HPO/calibration/fold callers before R4 pass |
| 117–120: explicit feature registry excludes targets/future outcomes | Project contract; FE02 supports general train/test leakage concern | SYNTHESIS; exact registry/allowlist not FEAZ recipe | Intentional target injection + future mutation + schema tests; use causal data contract, not name prefix alone |
| 124: F0 horizons/reversal/high proximity/volatility/volume | Financial hypotheses; FE11–13 | PROPOSAL; temporal catalog does not substantiate these stock signals or horizon values | Freeze individual formulas/windows, availability/warmup, compare rule baseline and one-family removal; no book-backed alpha claim |
| 125: F1 relative strength/breadth/regime | Financial hypotheses; FE11 body | PROPOSAL; FEAZ mentions time dependence, not VNINDEX causal regime construction | PIT coverage and future-mutation tests; regime evidence from project protocol rather than FEAZ |
| 126: F2 cost/liquidity proxies not true spread/depth | Domain/protocol proposal, not FEAZ | PROPOSAL; sensible measurement boundary; no liquidity efficacy evidence in inspected FE sources | Label proxies explicitly; empirical fee/fill/participation stress and reference data required |
| 127, 249: F3 foreign-flow interaction, lag controls | Domain hypothesis, not FEAZ | PROPOSAL; FE03/05 support investigating missing information, not trading-sign direction | Source availability/coverage and missingness-only/lag controls; positive edge unverified |
| 128: F4 historical sector/fundamental eligibility | Project source contract, not FEAZ | PROPOSAL; blocker is correct; new symbol policy alone cannot certify PIT sectors | Archived membership/sector/publication provenance before experiment |
| 130–131: redundancy and correlation clustering | FE06/FE07 opening + clustering paragraphs | DIRECT mechanism, not proof current indicators duplicate; source warns filtering may lose signal | Learn clusters only in development fold; freeze selection/tuning; paired performance/coverage/runtime evidence |
| 131–134: conditional permutation, SHAP not causal edge, structured null | Project/statistical synthesis; not found in FE06/07 | SYNTHESIS, financial null/permutation efficacy UNVERIFIED by FEAZ | Specify exchangeability/unit/blocking and diagnostic purpose; separate model explanation from causal identification |
| 138–139: distinct invalid/inf/stale/not-applicable/warmup/zero | FE03 sentinels + domain project contract | DIRECT for distinguish missing mechanisms/sentinels; exact statuses are project extension | Provider dictionary + fixture observed-zero tests; mechanism cannot be inferred solely from observed missing pattern |
| 140: median + indicator linear baseline | FE04 opening/drawbacks, FE05 §45.1 | DIRECT mechanisms; SYNTHESIS for choosing linear baseline. Neither establishes best preprocessing or profit | Freeze before fit; indicators constructed from original missing mask, not after imputation; record variance/dimensionality/coverage |
| 140–141, 246: Robust / Yeo–Johnson alternative | FE01/FE02 trained-transform passages / formula / pros/cons | DIRECT mechanisms; H-P02 is properly hypothetical. Not required normal predictors; no universal improvement | Freeze feature eligibility/order; finite/constant/near-zero-spread/negative-domain fixtures; do not silently combine with existing clipping |
| 141–142: native missing tree comparison | FE04 opening notes native-missing model possibility; FE14 WIP | PROPOSAL, not verified per LightGBM/XGBoost/CatBoost version | Read official estimator docs and test missing/inf/unknown/all-missing behaviors per release; no blanket native support assertion |
| 143–145: every learned transform/vocabulary fold-fit, serialize, fail closed | FE01/02 train-fit; FE07 learned selection; FE08 saved train vocabulary; skill patterns | DIRECT training/new-data principle; SYNTHESIS extending to all nested folds and artifact fail-closed | Fit-set row IDs/timestamps and no-validation-fit tests; serialization/schema inference parity; no refit at serving |
| 146–147: same-date cross-sectional rank availability/denominator | Financial feature contract; FE12/13 do not cover it | PROPOSAL; distinct from learned temporal scaler but can still leak via PIT universe/data timing | Permute input order/mutate future dates; record eligible contemporaneous sample and tie/rank denominator policy |
| 148–150: unseen symbol/category policy, identity shortcut, leave-group diagnostic | FE08 opening/train vocabulary; project identity/generalization policy | DIRECT unknown-level mechanism; SYNTHESIS for stock identity/sector/group diagnostics | Declare Python encoder unknown/missing policy; held-out-symbol diagnostics secondary, not substitute for forward dates |
| 151: no SMOTE/shuffle across time | FE09 distribution changes, FE10 numeric/interpolation limitations; FE11 dependence | SYNTHESIS conservative finance policy, not FEAZ explicit chronological prohibition | If later tested, prereg train-fold-only temporally valid design; source does not justify universal SMOTE ban in every time-series task |
| 151–153, 287–288: weights overlap/concentration, days/events not stock rows | FE09 weights supports mechanism; project statistical inference contract | PROPOSAL for dependence correction. FE11 acknowledges dependence but gives no effective-sample-size/bootstrap protocol | Specify sample unit and dependence assumptions; dates/events still not automatically independent; cluster/block sensitivity and power required |
| 155–159: FEAZ source boundary/draft-stub caution | FE01–14 actual bodies + local guides | DIRECT: models/lag/rolling genuinely WIP; time-series overview has short conceptual text, not empty | Cite originals; mention substantive draft missing/simple/filter/unseen rather than treating all drafts as unsupported |
| 163–186: financial MR15/momentum labels and execution/accounting | Project proposal/code, not FEAZ | Explicit project attribution correct; horizons/label accounting/efficacy not sourced from FEAZ | Preserve strategy-specific tests/gate; define 15-session inclusive/exclusive convention and same implementation in labels/simulator |
| 192–206: grouped dates, label-interval purge, calibration/availability | Existing temporal contract + new project design; FE11 only broad caution | FEAZ-specific support UNVERIFIED; attribution already separated in §4 | Do not cite FEAZ for purge length/nested OOF/embargo; exact split/maturity/code-source evidence belongs separate protocol |
| 210–228, 253–255: rank model, voting/stacking/regime ensemble and diversity | Project ML research proposal; FE14 WIP | PROPOSAL; no FEAZ endorsement or improvement proof | Consistent units, proper forward OOF/meta/calibration boundaries and executable net performance; compare fixed mean and drop-base |
| 245: H-P01 missing-aware vs zero-fill baseline | Code baseline vs FE03–05 | CORRECTION: current fills are RSI=50, volume ratio=1, rank/breadth=.5, others 0, then train 1/99 clipping | Rename to current feature-specific neutral-fill baseline; optional blanket-zero is separate registered control |
| 246: extra transforms may not help trees | Project hypothesis / FE14 WIP | PROPOSAL, appropriately not claimed universal tree benefit | Include no-transform/model-specific controls; do not imply all tree methods mathematically invariant to every numeric transform |
| 257–266: stagewise family ablations, multiplicity, leak controls | Skill experiment synthesis + project statistical protocol | SYNTHESIS; sensible scope, no exact FEAZ multiple-testing family or block-null proof | Record total attempts, frozen family/decision logic and retained failures; intentional-leak tests not alpha evidence |
| 284–288: Brier/log loss/PR-AUC/rank IC vs net acceptance | FE09 supports appropriate metrics/Brier vs accuracy; other metrics and financial acceptance project-specific | DIRECT limited classification metric principle; ranking/profit/multiplicity protocol not FEAZ | Metric definitions/weights/units/min samples and portfolio accounting before interpretation |
| 313–315, 335–337: fold transforms/prediction lineage/release bundle | Project MLOps design; FE01/02/08 apply saved training parameters | SYNTHESIS beyond FEAZ scope | Manifest IDs/hashes fit provenance and serving equivalence; no claim implemented from directory proposal |
| 372–374: FEAZ influence includes model-specific preprocessing | Skill synthesis plus FE01–14 | CORRECTION attribution too broad since FE14 WIP; chapter text supports transform/missing/correlation mechanisms, not entire model choice policy | Rewrite supported mechanisms vs project/estimator-specific policy explicitly; add original citations and this matrix |

## Required integration changes / open evidence

1. Update H-P01 control wording and explicitly name current preprocessing order:
   neutral imputation → training-bound winsorization → estimator preprocessing.
   Alternative missing-mask indicators must be generated **before** imputation.
   If RobustScaler/Yeo–Johnson are compared, freeze whether winsorization remains
   enabled; otherwise multiple simultaneous changes defeat one-rule attribution.
2. Add FEAZ primary links and scope distinction in §4/§11. Mention that `draft`
   may contain mechanisms, whereas the verified models/lag/rolling WIP pages do
   not substantiate those methods. Do not overclaim all time-series chapters were
   examined: lag/rolling/models were directly verified; other catalog stubs are
   only skill-index classification in this review.
3. Register numeric domain/constant/near-zero spread and missing-mask/unknown-
   level invariance tests in R3/R4. Robust scaling is not winsorization and should
   not automatically delete panic-tail observations. Extreme market values may
   be valid, not data errors; efficacy decided by preregistered slices, not EDA
   aesthetics or distribution normalization alone.
4. All H-F/H-E/financial claims remain PLANNED. Required source coverage,
   preprocessing implementation, data availability, statistical power and
   executable net-return evidence are absent for those future experiments.
   Literature mechanism support cannot remove these blockers.

## Verification / handoff

Performed read-only `Get-FileHash`, `rg` and bounded `Get-Content` inspections;
reader commands above; `apply_patch` creates only this report. No tests, training,
backtest or gate execution: no runtime behavior changed by this worker.
Report is uncommitted alongside the parent's docs changes; frozen plan hash
rechecked before handoff. Parent owns final corrected hash and integration.
Next bounded action: coordinator independently checks material baseline/source
attribution corrections, then updates shared plan after all scouts finish.
