# S-DE: đối chiếu kế hoạch dữ liệu với tài liệu gốc

Ngày: 04/10/2026. Reviewer: `/root/review_data_sources`, Firstmate run
`ml-plan-source-review-20261004`. Phạm vi: data claims §§2–5/9/11; không đánh giá
profitability hoặc cấp trading approval. Không sửa runtime/config/data/shared plan.

## Input và verdict

- Frozen plan: `docs/plans/2026-10-04-data-ml-research-plan.md`, SHA256
  `13985318dcd8b889c9a01e1725c1fc24282122aae29d6167d1d96f99d1ed5bb7`.
- Source revision: `6c3b8f422dd522ac0e4a892f23319e4a431b1ef7`; working tree có
  onboarding/planning edits và user-owned log/cache. Review này không dùng dirty
  files như bằng chứng capability mới. Các line numbers là bản frozen, không bản
  sau integration.
- **REQUEST CLARIFICATIONS trước implementation**, không bác kiến trúc tổng thể.
  Các cơ chế grain, history, replay, idempotence, audit và transaction có hỗ trợ
  trực tiếp từ sách gốc. Thiết kế SQLite, hai trục thời gian, trading availability,
  naming/schema và bộ evidence cụ thể là adaptation/proposal của dự án, không
  được suy thành yêu cầu hoặc bảo đảm của tác giả.
- Ba clarification quan trọng: (1) publication/recovery giữa filesystem và SQLite;
  (2) rolling windows không nén missing sessions; (3) valid-time khác knowledge-time,
  và không áp blanket reject effective overlap cho mọi revision bitemporal.
- Không còn primary-source availability blocker sau khi tìm được Kimball bản
  relocated và reader xác minh đúng hash. Tài liệu không chứng minh provider có
  archival availability hay pipeline đã leakage-free; các claim đó vẫn chưa có
  bằng chứng thực nghiệm.

## Tài liệu thực sự đọc và fingerprint

Đọc toàn bộ root `AGENTS.md`, full review brief, skill `data-engineer` tại
`C:/Users/acer/.agents/skills/designing-data-application/SKILL.md`, source indices
FDE/Kimball/DDIA, provenance/coverage và các guides FDE7/FDE8/Kimball2/DDIA7/DDIA11.
Các guides là synthesis, **không thay source reads** bên dưới. Lần output combined
đầu bị truncate phần Kimball2; đã đọc lại guide đó riêng đầy đủ.

Reader `python -B -X utf8 scripts/read_pdf.py` chạy từ thư mục skill; mọi successful
page read hiển thị `Verified SHA-256` và reader kiểm page count trước extraction.
Không download/install/OCR/bỏ qua fingerprint; chỉ text extraction, không tuyên bố
đã visually audit toàn bộ diagrams. Không sao chép sách vào repo.

| ID | Original source / edition | Fingerprint; physical pages | Locator convention |
|---|---|---|---|
| F | Reis & Housley, *Fundamentals of Data Engineering*, 1st ed., July 2022 / first release 2022-06-22 | `7c7b266be2d7aadc2136945ba7519cbb922264f1ae5f80cc5709e47243dc1e05`; 445 PDF pages | Printed page = physical −25 |
| K | Kimball & Ross, *The Data Warehouse Toolkit*, 3rd ed., copyright 2013 | `aa7a4d2ebbabdc88769b2fbf2c6b6fb64e617588a721c9da89d195c8d0972f1a`; 601 PDF pages | Printed page = physical −36 |
| D | Kleppmann, *Designing Data-Intensive Applications*, 1st ed., March 2017 | `3480da3ae9a353067c574924e296bee55b63adfb4af43cbb852a98fdd91e0394`; 613 PDF pages | Printed page = physical −22 |

Actual reader invocations, all with `--max-lines 250 --max-chars 16000`, no query:

```powershell
python -B -X utf8 scripts/read_pdf.py --sources
# Each (page,count) pair below was a separate actual bounded invocation:
# --source fde --page P --count C:
# (260,3), (265,3), (274,3), (282,3), (278,3), (301,3), (314,3), (290,3)
# --source ddia --page P --count C:
# (474,3), (481,3), (497,3), (245,3), (248,2), (500,2)
# --source kimball --pdf "D:/data sciiiiii/The Data Warehouse Toolkit, 3rd Edition.pdf"
# --page P --count C:
# (75,3), (89,3), (78,3), (102,2), (85,3), (96,3), (99,3)
```

Failures/recovery recorded rather than hidden:

- Kimball `--page 75 --count 3` and `--page 86 --count 3` without `--pdf` failed
  exit 1 because configured `C:/Users/acer/Downloads/The Data Warehouse Toolkit,
  3rd Edition.pdf` no longer exists. Read-only `rg --files -g '*.pdf'` in known
  source/download roots, filtered for warehouse/Kimball/FDE/DDIA, found the
  relocated copy under `D:/data sciiiiii`. Supported `--pdf` override passed the
  same catalog fingerprint; no edition substitution.
- One Kimball96 command accidentally ran from `D:/Chungkhoan` and failed exit 2
  because `scripts/read_pdf.py` is a skill reader, not a repo script. Reran from
  skill directory successfully; no repo reader was created.
- Initial combined file-discovery/read output was truncated by command-output
  budget. Subsequent bounded source reads were independent and complete for all
  claim locators below; no truncated passage is treated as sufficient evidence.

## Claim matrix — mechanism, attribution và limitations

Origin: **B** = book-supported mechanism, **A** = project adaptation/proposal,
**C** = inspected current code fact. Supported mechanism ≠ implemented capability
≠ measured statistical edge. Source locators name the original passage actually
read, with physical PDF page followed by printed page.

| ID / frozen plan lines | Assertion / origin | Original passage inspected | Support, limits, correction |
|---|---|---|---|
| DE01 §2 L42–45; §3 L91–93 | Immutable raw → derived canonical → as-of dataset; corrections append vintages. B+A | D ch11, PDF481–483 / pp459–461, State/Streams/Immutability and deriving views; F ch7 PDF278 / p253, intermediate balance history loss | Supported separation/history/rebuild principle. Raw API snapshots record observed states, not every provider event. Do not call them full event sourcing or complete CDC history. Keep the provider limitations already at L34–37. |
| DE02 §2 L52–55 | Filesystem + SQLite transactional catalog; Parquet only after benchmark. A | F ch7 PDF283 / p258, file formats and single-instance ingestion; D ch7 PDF245–247 / pp223–225, transaction guarantees depend on invariants | Reasonable local proposal, not a book-prescribed stack. No SQLite version/config/locking/backup behavior verified here, no Parquet speed measurement. Retain explicit proposed status; implementation must inspect current official engine docs and run workload/concurrency tests. |
| DE03 §3 L64–72 | Declare grain/identity before building tables, separate feature and label identity. B+A | K ch2 PDF75–76 / pp39–40, grain as row contract; F ch8 PDF314–315 / pp289–290, conceptual/logical/physical and preserving detail | Supported principle. Concrete instrument/session/strategy/version keys are own design. L66 groups attempt and content into one table row in the prose: clarify two grains, run/attempt ID vs raw hash and many-to-many association, so repeated identical payload does not erase separate observations. |
| DE04 §3 L67–70; L80–84 | Units/provider/scope/grain explicit; prevent fanout and accidental flow mixing. B+A | F ch8 PDF301 / p276, many-to-many row explosion; K ch2 PDF78–79 / pp42–43, conformed facts; PDF96–97 / pp60–61, approved conversions/units; PDF97 / p61, fact-to-fact fanout | Supported business semantics and cardinality checks. Books do not establish VCI/Vietstock units or matched/put-through meaning. Financial source contract must provide original units, factors, revision/proof; cannot infer units from correlation. L68 VND/shares remains a proposed canonical convention. |
| DE05 §3 L69; L74–81 | Stable instrument ID, effective interval and knowledge interval; available-at <= decision time. B+A | K ch2 PDF90 / p54, Type2 new row/surrogate/effective/expiry; PDF98 / p62, late facts; D ch11 PDF497–498 / pp475–476, time-dependent joins and retained versions | Direct support for valid-time/versioned historical join, **not a complete bitemporal trading-knowledge schema**. Knowledge intervals/archival publication and prediction-time bound are own adaptation. Explicitly separate valid and knowledge predicates, use half-open boundaries and deterministic revision/source resolution; at a given knowledge cutoff allow at most one active valid version. Effective overlap between revisions with disjoint knowledge intervals is legitimate, not blanket corruption. |
| DE06 §3 L74–78; §5 L183–185 | Historical effective date is not knowledge/publication availability; unknown source history conservative/excluded. A, with B motivation | F ch7 PDF291 / p266, multiple time aspects and stale model risk; D ch11 PDF498 / p476, current vs historical tax rate join | Correct proposed causality contract; original sources do not prove archival public-at. A conservative bound must be grounded in actual receipt/commit or independently documented publication evidence, not invented EOD timestamp. Already-preserved legacy quarantine is essential. |
| DE07 §3 L83–84; §4 L138–139 | Missing, zero, stale, invalid and not-applicable must not collapse. B+A | K ch2 PDF78 / p42, null facts and default unknown dimensions; PDF79 / p43, periodic snapshots may use zero or null | Supported distinct null/unknown semantics. Text is stronger project ML policy, not literal Kimball rule prohibiting all zeros. Expected no-activity certified zero differs from missing provider row; maintain reasons and eligibility. Books do not choose imputation estimator here. |
| DE08 §3 L88–90, L94–96 | Receipts, watermarks, expected sessions/symbols, readiness beyond transport success. B+A | F ch7 PDF260 / p235, ingestion use-case/source-quality questions; PDF265 / p240, reliability/durability trade-offs; PDF291–292 / pp266–267, time/volume/data-quality monitoring | Supported observability and separate data quality. Status vocabulary/coverage proof/market expected-session grid are own contract. Watermark alone is not completeness proof. Define expected population incl listing/suspension/provider limitations, source terminal paging evidence and denominator. |
| DE09 §3 L91–93; §9 L323–327 | Bounded retry, idempotent request/content/projection, duplicate not summed twice. B+A | D ch11 PDF500–501 / pp478–479, idempotence stable metadata and state recovery; F ch7 PDF274–275 / pp249–250, replay retention and DLQ | Supported mechanism with conditions, not automatic from content hash. Hash dedups byte objects; business identity/version controls semantic duplicates, retries and corrections. Every attempt retains receipt even identical raw; applied-source-version ledger/projection checksum and ordered rebuild must be tested. |
| DE10 §3 L94, L100–101 | Publish after raw/canonical/catalog durable commit. B+A, **needs boundary clarification** | D ch11 PDF474–476 / pp452–454, dual-write anomaly; PDF499 / p477, checkpoint boundary vs external side effects; D ch7 PDF246–249 / pp224–227, atomicity/durability scope | Ordering alone is not multi-store atomicity; SQLite transaction does not include ordinary file writes. Proposal: stage immutable generation, hash/validate durable files, commit catalog generation/publication pointer atomically within DB, expose only completed generations and reconcile orphan/staged files on restart. Choose one authoritative visibility boundary, not independently mutable filesystem and DB current pointers. Test crash after each step and during concurrent readers. This mechanism is adaptation, not proven existing code. |
| DE11 §3 L97–99 | Legacy quarantine, rolling verified sessions only, no future period aggregates assigned early. A | F ch7 PDF278 / p253, source-state history limits; K ch2 PDF79 / p43, periodic grain; D ch11 PDF498 / p476, point-in-time joins | Correct not to invent flow. **Clarify N-session rolling windows:** retain exchange calendar, report present/expected coverage/min observations; missing day stays missing, do not compress last N observed rows into last N trading sessions. A partial aggregate may exist with explicit observed-period/support and availability but is not a complete period value. No book proves predictive benefit of foreign data. |
| DE12 §3 L100–101; §9 L337 | Crash/restart/lock/replay/restore tests; Git is not runtime backup. B+A | D ch7 PDF248–249 / pp226–227, durability faults/backups; D ch11 PDF500–501 / pp478–479, state rebuild; F ch7 PDF265 / p240, cost/impact-based reliability | Supported recovery need, own test recipe. Recovery must restore coherent raw/catalog/manifest/metadata snapshot, verify hashes and rebuild outputs; define measured RPO/RTO and retention/storage failure budget before acceptance. Merely backup job green or Git branch preserved is insufficient. No restore test executed in this review. |
| DE13 §3 L102–104; §9 L310–327, L335–337 | Complete builder/calendar/universe/feature/cost dependency provenance, environment and whole bundle. B+A | K ch2 PDF102–103 / pp66–67, audit metadata including ETL code versions/timestamps; D ch11 PDF483 / p461, buggy transformation recovery and multiple views | Supports provenance/audit/versioned derived output. Exact closure hashes, label/strategy bundle and environment JSON are project/ML adaptation. Hash correctness alone does not establish temporal data truth; git HEAD alone omits dirty changes, config data, packages. Record raw originals, transformed values and dependency edges for deterministic rebuild. ML rollback details owned S-ML. |
| DE14 §3 L108–113; §4 L117–129, L146–150 | Source priority/coverage, registry, PIT cross-sectional denominator and blocked historical sector/fundamental. A | F ch7 PDF260 / p235, use-case and source fitness; K ch2 PDF87 / p51, common definitions/conformance; D ch11 PDF498 / p476, historical dimension versions | Sensible project policy, not author-ranked feature/source priority. Source existence/current sector is not historical membership/sector evidence. Feature/target allowlist and decision-time registry are own ML contract; market alpha and source sufficiency not demonstrated. |
| DE15 §5 L163–186 | Separate MR15/momentum targets, pending not zero, actual end/availability, cost and corporate-action consistency. A | K ch2 PDF75–76 / pp39–40, measurements consistent with grain; PDF96 / p60, cost allocation to revenue grain; PDF98 / p62, late-event dimensions | Books support grain/cost/history discipline, not these strategy horizons/triggers/lot/settlement rules. Do not source those financial choices to FDE/Kimball/DDIA. Label_end is economic event end while label_available_at is knowledge time; both needed. Ambiguous corporate actions and unavailable exits properly excluded/reported; timing convention needs project test, not literary approval. |
| DE16 §9 L329–333, L338–342 | Synthetic/pinned CI separate from prod/private state; prospective receipts/fills distinct from predictions. B+A | F ch7 PDF290 / p265, touchless test and scoped production access; PDF291–292 / pp266–267, ingestion/data health | Supports safe test/data monitoring separation. Fixture redistribution rights/public eligibility are not proven merely because market data observable; verify provider license before public artifact upload. No broker/live permission follows from CI or books. |
| DE17 §11 L375–378 | data-engineer influence: FDE7–8/Kimball2/DDIA index → bitemporal/replay/restore/provenance. Skill synthesis+A | Original F/K/D passages above, not index alone | Mostly defensible influence summary; change to exact original chapters/pages with link to this audit. Distinguish SCD2/history author mechanisms from added knowledge-time trading adaptation, and SQLite/publication/restore tests from author prescriptions. Index is navigation, not primary evidence. |

## Minimal suggested shared-plan corrections

1. L66: explicitly split `ingestion_attempt(attempt_id)` from
   `raw_object(content_sha256)` with receipt-to-object references. Repeated content
   can have different observed times/statuses; hash dedup must not merge receipts.
2. After L78–81: valid-time and knowledge-time are project bitemporal extension;
   specify half-open intervals, one valid version per key **at a knowledge cutoff**,
   tie-break policy and tests for retroactive corrections. Reject actual ambiguous
   rectangles, not all overlapping effective ranges across vintages.
3. L94: stage immutable generations and choose a single authoritative published
   pointer; SQLite commit cannot roll back file writes. Crash/reconciliation tests
   must show readers never see incomplete generation; do not promise absolute
   durability from fsync/rename alone.
4. L98–99: rolling lookback bound to exchange-session grid, with observed/expected
   coverage and min-support; missing not zero and not calendar compression.
5. L100–101: freeze RPO/RTO/retention and recovery consistency scope, then measure
   restore round trip. Current proposal does not yet state numbers; recording
   UNKNOWN/proposed is sufficient during planning, not during operational approval.
6. L375–378: cite original FDE7/8, Kimball2 and DDIA7/11 plus exact audit locators;
   explicitly label financial/bitemporal/schema/engine choices as own adaptation.

## Current-code spot checks (read-only, not fresh performance tests)

- `stock_agent/data/reconciliation.py:92–138`: unique new directory, preserved raw,
  canonical CSV and hashes; explicitly limits verification to source/hash/OHLCV/
  completed freshness and notes retrieved adjusted/current-fixed universe. Writes
  use filesystem methods and terminal manifest, not multi-store transaction. Plan
  L20 and L34–37 are grounded; no archival-PIT certification inferred.
- `stock_agent/data/foreign_flows.py:235–262`: `available_at` bounded by completed
  receipt, expected sessions/gaps/latest coverage, partial/blocked states and raw
  records manifest. `:341–358`: as-of filter before vintage selection, per-source/
  scope dedup, internal numeric conversion to billion-VND panel. New canonical
  contract must distinguish panel units from VND source records, not silently reuse
  names. Supports L21 but not blanket history readiness.
- `stock_agent/features/feature_store.py:12`: numeric fallback defaults to zero;
  supports need to audit missing semantics, not evidence all paths mishandle missing.
- No collectors, scans, training, strategy backtests, migrations, schedule changes,
  broker orders, installs, network source fetches or pushes were performed.

## Handoff / unresolved proof

Only changed path: this owned audit. Fresh frozen SHA check matched before review.
Final read-only `Get-FileHash` recheck matched the same frozen plan hash;
`git diff --check` returned exit 0 (only existing LF/CRLF warnings for shared
onboarding files). No runtime tests were needed/run for this source-review-only
report; these checks are not a data or strategy regression pass.
Primary-source reads all succeeded after recorded recovery; no blocked original
source remains for the owned mechanism claims. Adoption/efficacy/execution remain
unverified because plan R1–R8 are not implemented; archival PIT provider evidence,
calendar/availability resolution, restoration targets and numerical activity floor
are still prerequisites. Next bounded action for coordinator: independently recheck
DE10/DE11/DE05, then integrate only these evidence-backed clarifications into shared
plan after all scouts finish and record a new plan hash.
