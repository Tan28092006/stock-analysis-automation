# Firstmate: đối chiếu kế hoạch với tài liệu gốc

## Run / scope

- Run ID: `ml-plan-source-review-20261004`.
- Dispatched: 2026-10-04 15:10 Asia/Saigon (UTC+07); current observation applies
  only to this active session.
- Coordinator: `/root`, project `D:/Chungkhoan`, native Codex collaboration,
  concurrency limit 4 including coordinator; no app threads or extra runtime.
- Request: after planning, use subagents to assess the Markdown's claims against
  original sources used by the four skills. Delivery: local review reports and
  evidence-backed corrections to the requested plan/onboarding docs; no push,
  training, promotion, schedule changes or broker actions.
- Base code revision: `6c3b8f422dd522ac0e4a892f23319e4a431b1ef7`; preserve all
  existing dirty files described in the onboarding audit.
- Frozen input plan: `docs/plans/2026-10-04-data-ml-research-plan.md`, SHA256
  `13985318dcd8b889c9a01e1725c1fc24282122aae29d6167d1d96f99d1ed5bb7`.
- Frozen input onboarding: `docs/agent-workflow.md`, SHA256
  `35d7236c9ede67a86724e05e6c4bf5941d4a67c56ff0d8624caf179249de0c41`.
- Supervision: review delivered locally; no continuing daemon/monitor promised.

## Tasks and ownership

| ID | Kind / objective | Dependencies | Owned report; all other paths read-only | Worker ID | Observed state |
|---|---|---|---|---|---|
| S-ML | Scout: ML lifecycle/benchmark/ensemble/availability claims vs Reddi original text and current code | Frozen plan | `docs/audits/2026-10-04-ml-plan-review-ml.md` | `/root/review_ml_sources` | done; initial report + corrected-input recheck accepted |
| S-FE | Scout: feature/preprocess/missing/stub claims vs original FEAZ chapters; financial protocol attribution boundary | Frozen plan | `docs/audits/2026-10-04-ml-plan-review-features.md` | `/root/review_feature_sources` | done after resume; material corrections resolved; one minor wording corrected |
| S-DE | Scout: grain/ingestion/revisions/PIT/transaction/replay claims vs original FDE/Kimball/DDIA | Frozen plan | `docs/audits/2026-10-04-ml-plan-review-data.md` | `/root/review_data_sources` | done after resume; corrected data sections accepted |
| S-HAR | Coordinator scout/integration: onboarding/routing/evidence claims vs original WalkingLab/AI Agents sources | Frozen onboarding/plan; worker reports before integration | This run record + shared plan/onboarding corrections after scouts finish | `/root` | done; primary-source rechecks, integration and document validation |

## Brief / acceptance shared by scouts

Read root AGENTS and your skill instructions, then original sources using the
skill's bounded verified reader or original public FEAZ chapters. Do not treat
skill synthesis as the primary source. Bind findings to the frozen hash above;
if it differs, notify coordinator. Do not change the plan, onboarding, shared
record, runtime, configs, data, models or other reports. Do not spawn a second
supervisor or publish/download books/install dependencies. Return here.

Produce a claim matrix: exact plan section/line span and paraphrased assertion,
origin kind (book / skill synthesis / project proposal / code fact), primary
source edition/locator/page/line or original URL, what was actually inspected,
support level, limitation/failure and proposed correction. Separate supported
mechanism from unverified efficacy and financial-specific protocol. Keep short
quotes only when necessary; no copied source corpus. Record actual source
reader commands/errors and source availability/hash checks. Cover all owned
claim families, not only easy examples. Missing source = unverified, not pass.
Return report path, material findings, source coverage and any blocked claims.

## Validation and integration

Coordinator must inspect worker reports and independently recheck each material
correction against source/code. Shared documents stay frozen during scout reads.
After integration, validate paths/links and applicable checks, update artifact
hashes and distinguish original verdict from corrected verdict. Do not reuse
old hashes as proof of corrected content. Existing 123 focused tests concern
unchanged code, not efficacy of the proposed ML research pipeline.

### Resume after usage-limited recheck

User requested continuing with subagents. Native status reconciled: S-FE and
S-DE were errored, not active writers; S-ML completed corrected review. Reuse
same workers/task identities for bounded read-only correction rechecks, not a
new literature survey. No change to original reports or shared plan during
reads. Coordinator records returned acceptance/remaining defects here.

Corrected input frozen for resume:

- Plan SHA256 `2cd050d4cd52236d916dc511cd90679c3e34917b6087a4a11aaa4883bad70e34`.
- Onboarding SHA256 `5fca4bf3e63878091a4341514be9dd236658e032c9428fe8bacceae8c5879161`.
- S-FE acceptance: check actual neutral-fill baseline, missing-mask-before-fill,
  transform ordering/clipping, draft vs WIP and source attribution/citations.
- S-DE acceptance: check attempt/content grains, knowledge-cutoff version
  selection, single catalog generation publication/crash boundary, calendar
  skeleton rolling, recovery targets and original-source attribution.
- Both return checked hash and concise resolution verdict through native results;
  write ownership: none. No train/replay/refresh/schedule/push actions allowed.

## Decisions / resume

Numerical activity floor remains a user decision before R5 experiment
acceptance; source availability/archival PIT and execution readiness remain
project prerequisites. No authority expansion from literature support.
If interrupted, inspect this record and preserved artifacts; old worker states
are historical. Source-review task is complete, not R1–R8 implementation.
Next bounded implementation in the roadmap remains R1 source/dataset contracts,
requiring its own preregistration/tests/full regression for behavior changes.

## S-HAR — đối chiếu trực tiếp nguồn harness (coordinator)

Đọc guides ch13/14/16/27 đầy đủ, sau đó đọc nguồn WalkingLab local bằng
`C:/Users/acer/.agents/skills/ai-agent-build/scripts/read_source.py`. Reader
kiểm raw source fingerprints; selected reads exit 0, không update clone.
Snapshot nguồn: `38ddcd2bf8d65271f668b94e7c875ca1d629d622`, local
`D:/data sciiiiii/learn-harness-engineering`. Đây là course/template của
WalkingLab, không gán thành phát biểu nguyên văn của Bojie Li hoặc thực nghiệm
chạy tại repo chứng khoán.

| Claim / frozen target | Nguồn gốc thực đã đọc | Kết luận / limitation / correction |
|---|---|---|
| Workflow §Five fresh-session answers, L8–24 | WalkingLab `l03`, original lecture03/index.md L39–53 | Supported five-question mechanism. Current check is route existence, chưa phải fresh-context agent outcome benchmark; audit đã ghi giới hạn này |
| Root router, progressive disclosure, plan §11 | `l04`, lecture04/index.md L60–80; original advanced repo template `AGENTS.md` L1–35 | Supported routing mechanism; local root giữ research invariants, không áp số dòng hoặc template FastAPI như mandatory |
| One work ledger / update next bounded action | `l03` L69–101; advanced template L47–55; original encode-knowledge SOP L18–42 | Supported conceptual design; cụ thể R0–R8 và layout repo là project adaptation, không course yêu cầu chính xác filenames đó |
| Evidence before done / current revision / linked artifacts | Advanced template `AGENTS.md` L28–45; original L06 L118–126 | Supported; document existence/CLI help không chứng minh app boot hay ML pipeline hoạt động. Cần gọi R0 documentation/discovery-only, không full startup readiness |
| Safe discovery has no refresh/scan/train | Workflow L26–44 vs original L06 L114–126 and template L7–15 | Project safety adaptation: nguồn minh họa setup/commit/bootstrap có effects; không khẳng định nguồn bắt startup luôn read-only, không chạy install/commit vì sách có ví dụ |
| Current run route for user signals | Workflow Five-answer Q3 L14–17; actual `paper_runner.py::main` | Minor discovery gap: nói documented lane nhưng chưa link operator runbook/explicit command. Propose add on-demand route, effects label; `--check` vẫn writes latest.json, không gọi read-only |
| 123 tests / 26 links / 58 routes | Onboarding audit actual commands, results and frozen document fingerprints | Code engineering evidence supported locally; không proof statistical edge, prospective readiness hoặc measured agent improvement |
| Permission/scoped ownership | Source L03 ACID analogy L94–101; current root/host/user instructions | Plan correctly treats ownership/checkpoint as adaptation, không Git ACID guarantee; source stash/commit và repo-authority rhetoric không vượt host/user authority |

Selected primary reads (absolute script path above, from project root):

```text
python -B -X utf8 <reader> --corpus harness --source l03 --line 26 --count 28
python -B -X utf8 <reader> --corpus harness --source l04 --line 58 --count 23
python -B -X utf8 <reader> --corpus harness --source l06 --line 114 --count 18
python -B -X utf8 <reader> --corpus harness --source resources-openai-advanced-repo-template-agents --count 70
python -B -X utf8 <reader> --corpus harness --source resources-openai-advanced-sops-encode-knowledge-into-repo --count 75
```

Additional l03 L58+ / l04 L30+ / l06 L65+ reads had bounded/truncated output;
only actually displayed sections are used above. Precise fresh-session and
router ranges were reread narrowly without truncation. No source copy added
to repository; matrix paraphrases mechanisms and records locators.

## Final integration and corrected-plan acceptance

Three scout reports cover 76 claim groups (ML29 + FE30 + DE17), with coordinator
S-HAR covering the harness family. This is coverage of scoped claim groups,
not a percentage of all book content or all financial/statistical literature.
Original reports retain the original frozen plan's line numbers and hashes.

| Finding | Resolution in corrected plan | Returned corrected-review verdict |
|---|---|---|
| Voting vs stacking and T+2 vs strategy-specific labels | Current code facts kept explicit; OOF stack/MR15/momentum builders remain PLANNED | S-ML accepted; no new capability/efficacy claim |
| Date-level current helper vs required timestamps | §1/6.1 distinguish optional dates/current trained_at helper from mandatory R4 fit/selection/calibration/evaluation availability | S-ML accepted |
| Quality/sample floors and Reddi attribution | Freeze before R5; time-to-accuracy vs time-to-quality and project no-auto-retrain/full-bundle policy qualified | S-ML accepted |
| Real preprocessing baseline and order | §1/H-P01 name neutral-fill + train-bound1/99; indicators before fill; clip/scaler/power ordering frozen and numeric fixtures required | S-FE material corrections resolved |
| FEAZ draft/WIP attribution | §4/11/12 distinguish substantive drafts, true WIP, live pages vs pinned QMD, book mechanism vs financial/estimator project design | S-FE accepted with minor H-P02 wording note |
| Receipt/content grains | §3 separates attempts from deduplicated raw bytes; receipt observations retained | S-DE accepted |
| Valid-time vs knowledge-time overlap | §3 one selected version per key/source/valid-time at knowledge cutoff; legitimate revision overlaps not blanket rejected | S-DE accepted |
| File/DB publication atomicity | §3 single catalog generation visibility boundary, staged immutable files, failure reconciliation/reader checks; no SQLite rollback of filesystem | S-DE accepted |
| Calendar and recovery contracts | §3 rolling exchange-session skeleton + support policy; RPO/RTO/retention freeze and coherent restore required | S-DE accepted |
| Startup/on-demand effects | Workflow links actual operator route; `--check` writes preview, startup remains help/discovery; R0 documentation-only, not app-boot benchmark | Coordinator code/help check accepted |

All three corrected reviews bind plan SHA256
`2cd050d4cd52236d916dc511cd90679c3e34917b6087a4a11aaa4883bad70e34`.
On resume S-FE/S-DE first-turn rechecks had errored due usage limits; same native
workers were resumed after user request, with no competing writer, and returned
acceptance. These failures did not invalidate their already-retained original
source reports and were not silently counted as acceptance.

S-FE's sole remaining minor note was H-P02's unconditional wording. Coordinator
changed only `Extra transforms không hữu ích với tree` to
`Extra transforms có thể không hữu ích với tree` after all reviewers finished.
It weakens an untested efficacy statement to a hypothesis; no method, code,
experiment or acceptance rule changed. Final hash is recorded below; review
verdicts above do not pretend the earlier hash is the final byte identity.

Coordinator independently rechecked material corrections against originals:
Reddi ch13 source lines56068–56091 (automatic triggers),56403–56422 (rollback);
DDIA physical pages475/498 (dual writes/historical join); Kimball physical79/90
(periodic grain/SCD2), all source readers verified fingerprint. FEAZ pinned
`models.qmd` and `missing-indicator.qmd` fetched via agent-reach/Jina confirm
WIP versus substantive content. Actual neutral-fill/clip and paper preview
effects checked in current code. Detailed locators/commands remain in scouts'
reports and S-HAR rather than duplicating book text here.

Conclusion: corrected documentation is suitable as a bounded implementation
roadmap. No remaining material **source-attribution/design-wording** defect
in the reviewed families; not proof of full literature completeness, archival
PIT, statistical power, implementation readiness, profitable backtest or money
trading. No ML training, full replay, schedule/data mutation, orders, commit or
push performed in this review. Activity floor and ML-specific numerical floors,
recovery targets, financial-method source checks and execution blockers remain
explicit prerequisites for later work.

### Final validation — 2026-10-04 16:49 Asia/Saigon

All documents remain local/uncommitted; base source HEAD unchanged. Final byte
identities from `Get-FileHash -Algorithm SHA256`:

| Artifact | SHA256 |
|---|---|
| Corrected plan after minor H-P02 wording | `df0f173cdabb460f0e5e482acf69d045003b08a3ba2cd07ae115a8450d00af50` |
| Corrected agent workflow | `5fca4bf3e63878091a4341514be9dd236658e032c9428fe8bacceae8c5879161` |
| Original S-ML report, unchanged | `07b758d87b38c17d0f1aedcf27708d72a44f0334885e7e1cfc888824deb1ef51` |
| Original S-FE report, unchanged | `9c731624928efe0c1b107d55301c1a4824f428d6fbe1c4df6567799f11ccf6d8` |
| Original S-DE report, unchanged | `7d399dce75af85860c7321c00096cf8cfd1a5431b068a5fb5b0bfeafb398ca2d` |

From `D:/Chungkhoan`:

- `python -B -m stock_agent.pipeline.paper_runner --help`: exit0, flags match
  added onboarding route; no scan/refresh or output publication.
- `python -m pytest tests/test_pytest_collection_safety.py -q --tb=short`:
  **1 passed in 2.13s**; rerun validates the onboarding test route, not app boot.
- PowerShell local Markdown target validation on 11 changed onboarding/plan/
  review documents: **33 local links, zero missing**. Strip anchor before
  `Test-Path -LiteralPath`, resolve relative to each document. This is existence
  validation, not a check of external URL status or every heading fragment.
- Trailing-whitespace scan on the same 11 documents (including untracked
  reports): zero issues; `git diff --check` exit0 with normal Windows LF→CRLF
  warnings. No line-ending config modified.
- Prior 123-test evidence remains scoped to unchanged code and its original
  run; not presented as a fresh suite rerun. Full pinned suite/books/timeslices
  not rerun because this review changed documentation only, not data/strategy
  runtime or contracts. R1 implementation will still require full research gates.

Delivery is the local corrected plan plus source-review artifacts. Crew work
for this review is complete; no pending reviewer, publish or merge is implied.
