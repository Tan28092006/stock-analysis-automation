# Agent readiness và kế hoạch data/ML — 04/10/2026

## Phạm vi / revision

User yêu cầu tối ưu repo cho agent trước, sau đó lập kế hoạch tổng hợp dữ liệu,
tiền xử lý, ensemble và research/testing có evidence theo bốn skill chỉ định.
Base Git HEAD: `6c3b8f422dd522ac0e4a892f23319e4a431b1ef7`, branch
`codex/data-integrity-audit`. Các thay đổi onboarding/plan trong audit này là
local chưa commit; không có claim chúng đã nằm trong CI của HEAD trên.

Dirty files đã có trước và không sửa: `data/pipeline/local_server.log`,
`data/pipeline/swing_scan_cache.json`, `docs/audits/2026-10-04-system-design-review.md`.
Không chỉnh strategy, data contract, model, ledger, schedules hoặc broker.
Không chạy experiment/backtest mới; do đó không đăng ký trial hoặc tự diễn giải
protocol validation thành strategy pass.

## Onboarding changes và fresh-session acceptance

| Câu hỏi | Before: điểm thiếu trong docs được kiểm tra | After: route xác định |
|---|---|---|
| System / scope? | Root có invariants nhưng README còn nhiều legacy claims | Root giữ invariants; guide nói EOD/local, research/paper không là broker execution |
| Where / ownership? | Packaging liệt kê adapters, không có code ownership/test map | `docs/agent-workflow.md` map data/features/research/paper/packaging |
| How to start? | Không có startup route chung tách khỏi refresh/scan/train | Discovery Git/Python/CLI help không yêu cầu các tác vụ đó |
| How to verify? | Root gate rõ nhưng thiếu task-specific route | Test file routes + full pinned workflow; `--validate` không là backtest |
| Progress / next action? | Scrum plan cũ T+2/UI và hướng purge cố định có thể gây nhầm | Plan hiện hành duy nhất R0–R8, next R1; legacy notice và evidence links |

Đây là kiểm tra khả năng tìm được năm câu trả lời bằng route/path đã ghi,
**không** benchmark task success/latency của agent mới. Không tuyên bố giảm
token/thời gian X% hoặc tăng chất lượng giao dịch từ thay đổi tài liệu.

## Verification

Trên Python 3.12.4, working tree cùng base revision trên:

```text
python -m pytest tests/test_pytest_collection_safety.py tests/test_training_integrity.py tests/test_ensemble.py tests/test_model_release.py tests/test_reconciliation.py tests/test_foreign_flows_v2.py -q --tb=short
```

Kết quả: **123 passed, 19 warnings, 34.89s**, không failed/skipped. Warnings
liên quan pyparsing/matplotlib, Jupyter paths, invalid-date fixture pandas và
SHAP output; không sửa dependencies ngoài phạm vi. Các model fit trong isolated
test fixtures không phải training/promotion production.

- Startup smoke: `python -B -m scripts.research_gate --help`,
  `python -B -m scripts.momentum_books --help`,
  `python -B -m scripts.research_timeslices --help` đều exit 0, đúng options
  manifest/results/output đã ghi. Git status/branch/HEAD và Python version
  commands hoạt động; không chạy refresh/scan/schedule.
- `python -B -m scripts.research_gate --validate`: exit 0, explicitly
  `Protocol valid. This is NOT a full data/research pass.`
- Markdown local-link check: kiểm tra relative targets của các link trên bảy
  file onboarding/planning thay đổi; 26 links tồn tại. Kiểm tra ownership ban
  đầu phát hiện hai path sai (historical universe và foreign refresh), đã sửa
  thành `scripts/historical_universe.py` và
  `stock_agent/pipeline/foreign_refresh.py` rồi kiểm tra lại. Final check:
  **26 local Markdown links, 58 inline code/test routes, zero missing**.
  Method: PowerShell regex extract Markdown link targets relative to each file,
  inline repository paths relative to root, and `test_*.py` relative to `tests/`;
  every target checked with `Test-Path -LiteralPath`. This checks existence,
  not semantic correctness of every historical document.
- `git diff --check`: không whitespace errors; Git cảnh báo LF→CRLF do config
  Windows, không đổi line-ending configuration.

Mọi check ở đây chỉ xác minh onboarding/commands và regression engineering của
code hiện có, không chứng nhận pipeline ML đề xuất đã triển khai. Không chạy
full pinned replay/books/timeslices trong bước documentation-only này; khi R1
thay data contract, chúng vẫn là gate bắt buộc, không được dùng 123 tests này
thay cho full research evidence.

Retained document fingerprints, measured with `Get-FileHash -Algorithm SHA256`
on exact local bytes (before Git line-ending normalization):

| Artifact | SHA256 |
|---|---|
| `docs/agent-workflow.md` | `35d7236c9ede67a86724e05e6c4bf5941d4a67c56ff0d8624caf179249de0c41` |
| `docs/plans/2026-10-04-data-ml-research-plan.md` | `13985318dcd8b889c9a01e1725c1fc24282122aae29d6167d1d96f99d1ed5bb7` |

## Kết quả lập kế hoạch và giới hạn

[Roadmap](../plans/2026-10-04-data-ml-research-plan.md) tách source/grain/as-of,
canonical aggregation, fold-only preprocessing, MR15/momentum labels,
simple bases/voting/true OOF stacking, hypothesis ablations, crisis/half-year/
causal-regime regression, evidence bundle và prospective release boundary.

Code inspection xác nhận `EnsembleTrainer.train()` hiện voting mean, không fit
meta-learner mặc dù module mang tên stacking; target mặc định T+2. Điều này là
đầu vào kế hoạch, không thay bằng kết quả model mới. FEAZ time-series/model
guides có stub được ghi rõ, không viện làm bằng chứng của protocol tài chính.

Activity floor số học chưa được user chốt; archival corporate actions/PIT source
vintage, execution parity và prospective evidence vẫn còn giới hạn. Không
approve trading, retrain, promote, push hoặc sửa lịch trong bước này.

## Sau Firstmate source review

[Review tổng hợp](2026-10-04-ml-plan-review.md) và ba reports ML/FE/data giữ
claim–source matrices, original locators, frozen-input hashes và correction
verdicts. Các fingerprints/check counts ở trên là bằng chứng **bản onboarding/
plan trước source-review corrections**, không byte identity hiện tại.

Sau integration: workflow bổ sung on-demand operator route và chỉ rõ preview
`--check` vẫn writes latest.json; R0 chỉ documentation/discovery. Plan làm rõ
neutral-fill baseline, fold/indicator/transform order, actual timestamps,
receipt vs content grain, bitemporal cutoffs, single-store publication boundary,
rolling calendar coverage và book/project attribution. Code không đổi.

Final plan SHA256:
`df0f173cdabb460f0e5e482acf69d045003b08a3ba2cd07ae115a8450d00af50`.
Final workflow SHA256:
`5fca4bf3e63878091a4341514be9dd236658e032c9428fe8bacceae8c5879161`.
Final checks/resolutions ghi ở review tổng hợp; 123 focused tests trước đó là
engineering evidence của code không đổi, không phải model/backtest kết quả mới.
