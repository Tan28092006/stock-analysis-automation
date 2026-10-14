# Luồng cập nhật dữ liệu EOD hiện tại — đối chiếu với các audit trước — 06/10/2026

## Phạm vi / revision

- Ngày: 06/10/2026.
- Reviewer: Firstmate run 2026-10-06-ml-roadmap / T7.
- Source revision: `4b859cd` (branch `codex/data-integrity-audit`). Working tree
  có sẵn hai file untracked `data/pipeline/local_server.log`,
  `data/pipeline/swing_scan_cache.json`, không sửa.
- Phạm vi: **flow và orchestration** của việc cập nhật dữ liệu (trigger, provider,
  validation, writes, consumers, timing, failure, concurrency, observability,
  freshness). Không thiết kế lại storage.
- Phương pháp: chỉ đọc code, docs, git history (`git log/show/ls-files`), danh sách
  thư mục `data/` và vài dòng cuối của `data/pipeline/eod_update.log` cùng hai file
  CSV. **Không** chạy test/script/Python, không refresh dữ liệu, không gọi mạng,
  không truy vấn Task Scheduler, không commit. Cấu hình Task Scheduler
  (`IgnoreNew`, `StartWhenAvailable=true`, `WakeToRun=false`) lấy từ các audit
  trước, chưa kiểm chứng lại trong review này.
- Các mục đánh dấu **INFERRED** là suy luận từ code/log, chưa tái hiện.

Audit đã đối chiếu: `2026-10-04-system-design-review.md`,
`2026-09-30-foreign-data-rollout.md`, `2026-09-30-foreign-data-contract.md`,
`2026-09-22-market-readiness-plan.md`, `2026-09-26-market-readiness.md`,
`2026-09-30-paper-requested-window.md`, `2026-09-30-execution-receipts.md`,
cùng `docs/FOREIGN-FLOWS.md`, `docs/DEPLOY.md` (và `docs/RENDER-FREE.md` vì
DEPLOY trỏ tới).

## Kết luận ngắn

Đường chạy theo lịch (17:05) hiện chỉ gồm **foreign flow collector + paper
runner**. Cả hai fail closed và khớp với contract trong các audit 26/09 và 30/09.
Vấn đề chính nằm ngoài đường đó. Commit `7d788cb` (22/09) cố ý bỏ
`eod_update` và `git push` khỏi BAT. Từ đó `data/raw/prices_hist`, các dashboard
scan cache và `forward_test.jsonl` **đứng yên ở 21/09/2026**. `DEPLOY.md` và
`RENDER-FREE.md` vẫn mô tả luồng cũ.

Ba rủi ro vận hành chưa được audit nào nêu đầy đủ:

- Một run flow bị ngắt để lại lock hoặc thư mục run chưa commit. Runbook hiện tại
  bảo giữ nguyên thư mục đó, nhưng chính thư mục đó làm `health` báo partial
  **vĩnh viễn**.
- Một phiên bị mất hẳn nếu job không chạy xong trước 09:00 VN của phiên kế tiếp.
- Bảng ngày nghỉ hết hiệu lực sau 2026.

## Luồng hiện có (ASCII)

```text
Windows Task Scheduler "StockAgent_EOD_Update" 17:05 T2-T6 (run_eod_update.bat:4)
 └─ run_eod_update.bat
     ├─(1) foreign_refresh --retry-stale            (bat:13)
     │     collect_flows: .collect-lock -> Vietstock foreign+proprietary, 105 mã
     │       -> foreign_flows_v2/runs/<UTC>_<id>/{raw/*,records.jsonl,quarantine.json,manifest.json}
     │       -> foreign_flows_v2/latest_status.json
     │     health() trên mọi vintage; retry mã thiếu sau 60/300/900 s
     ├─(2) paper_runner --refresh --run             (bat:15)  [chạy cả khi (1) lỗi]
     │     build_market_snapshot: VCI gap-chart, 31 mã (VN30 + VNINDEX), ~650 ngày
     │       -> paper/snapshots/<UTC>/{raw/*,prices/*,manifest.json}
     │     assess_market_data -> MR + momentum _compute (pure) -> re-verify hash
     │       -> paper/runs/<session>/paper.json (.write-lock, window 16:00 -> 09:00 VN)
     │       -> paper/latest.json -> paper_scoring -> paper/scores/latest.json
     └─ log: data/pipeline/eod_update.log ; exit = paper trước, flow sau (bat:17-29)
     [git add/commit/push: ĐÃ BỎ ở 7d788cb]

Đường legacy (không có lịch; chạy tay / HTTP):
  python -m stock_agent.pipeline.eod_update  ─┐
  POST /api/data/update (app.py:334-337)     ─┤-> vnstock VCI -> raw/prices_hist/*.csv (101 file)
                                              └-> pipeline/{mr,momentum,swing}_scan_cache.json
                                                  -> GET /api/{mr,momentum,swing}/scan (scan_guard)
                                              (chỉ eod_update) -> pipeline/forward_test.jsonl, collect_flows
  cli daily-run (run_pipeline.bat) -> collect_flows + label/train/scan -> pipeline/*.jsonl, models
  cli refresh-data -> optimizer.refresh_price_data (Yahoo/vnstock) -> raw/prices/*.csv
  GitHub Actions eod-update.yml: schedule bị comment, chỉ workflow_dispatch -> POST app từ xa
```

Lịch sử: 31 commit "data: EOD update" trên `origin/main` (commit cuối `086b328`,
21/09 18:48, tác giả Tan28092006) do BAT cũ tạo bằng `git add/commit/pull
--rebase/push` (xem `git show 7d788cb -- run_eod_update.bat`).

## Writer → reader

| Writer | Đường ghi | Reader | Ghi chú |
|---|---|---|---|
| `foreign_flows.collect_flows` (foreign_refresh, eod_update, DailyRunner) | `data/foreign_flows_v2/runs/*`, `latest_status.json` | `health`, `verified_records`/`load_flows`, `scripts/foreign_flow_research.py` | Có lock; manifest ghi sau cùng, atomic |
| `reconciliation.build_market_snapshot` | `data/paper/snapshots/<UTC>/` | `paper_runner._source`, `paper_scoring`, research_gate (manifest pinned) | Thư mục mới mỗi lần; manifest `write_text` không atomic (`reconciliation.py:138`) |
| `paper_runner.commit_paper_run` | `data/paper/runs/<session>/paper.json` | `paper_scoring` | `.write-lock` theo session, atomic + fsync |
| `paper_runner.main` | `data/paper/latest.json`, `scores/latest.json` | Người vận hành | Atomic |
| `eod_update.refresh_prices` (CLI + dashboard thread) | `data/raw/prices_hist/*.csv` | dashboard scans, `position_manager.check_positions`, `forward_test` resolution, backtest/robustness | **Không lock**; `.csv.tmp` cố định (`eod_update.py:88`) |
| `mr_scan`/`momentum_scan`/`swing_scan` | `data/pipeline/*_scan_cache.json` | dashboard API, `eod_update.log_forward_test` | `write_text` không atomic, lỗi bị nuốt |
| `forward_test.log_recommendations` | `data/pipeline/forward_test.jsonl` | forward-test scoring | Check-then-append, không lock |
| `optimizer.refresh_price_data` | `data/raw/prices/*.csv` | optimizer, DailyRunner, `LocalCsvProvider` | Cache mutable legacy |
| BAT | `data/pipeline/eod_update.log` | Người vận hành | Append, không rotate |

## Bảng đối chiếu với audit trước

Ký hiệu: MATCHES = code hiện tại khớp claim; CONFLICTS = claim khác trạng thái
hiện tại; EXTENDS = claim đúng nhưng thiếu hệ quả quan trọng; NOT-COVERED = chưa
audit nào nêu.

| # | Chủ đề / claim trong audit | Nguồn audit (section) | Trạng thái | Evidence code / dữ liệu |
|---|---|---|---|---|
| 1 | "EOD data update: `python -m stock_agent.pipeline.eod_update`, daily 17:05" | `DEPLOY.md` §1 (dòng 20), §6 Linux cron (dòng 156) | **CONFLICTS** | BAT chỉ chạy `foreign_refresh` + `paper_runner` (`run_eod_update.bat:13,15`); `eod_update` đã bị bỏ từ `7d788cb` |
| 2 | "Dashboard reads the price history and caches that the EOD job writes" | `DEPLOY.md` §1 (dòng 23) | **CONFLICTS** | Job theo lịch không ghi `prices_hist`/cache; `VNINDEX.csv`, `ACB.csv` kết thúc 2026-09-21; cache và `forward_test.jsonl` có mtime 21/09 18:48 |
| 3 | "Free tier: refresh via local `run_eod_update.bat` + `git push`"; "File bat giờ có thêm bước git push" | `DEPLOY.md` §4 (dòng 96-97); `RENDER-FREE.md` (dòng 5, 41-42) | **CONFLICTS** | BAT ghi rõ "No ... Git publishing" (`bat:5,20`); commit EOD cuối trên origin/main là 21/09 |
| 4 | "Safe launcher must not automatically commit/rebase/push"; "Legacy dashboard caches, swing scan, daily-run ... separate paths, not migrated" | `2026-09-22-market-readiness-plan.md` (Fallback); `2026-09-26-market-readiness.md` (Architecture) | **EXTENDS** | Cố ý bỏ push và không migrate là đúng. Audit chưa ghi hệ quả: không còn writer theo lịch nào cho `prices_hist`, dashboard và `forward_test` đứng ở 21/09 |
| 5 | Luồng hiện có: "Legacy CLI/HTTP -> mutable price caches" | `2026-10-04-system-design-review.md` (Luồng hiện có) | **EXTENDS** | Đúng về cấu trúc; chưa nêu dữ liệu đã cũ 11 phiên và dashboard bị chặn (xem #13) |
| 6 | `.collect-lock` orphan: "chỉ dọn sau điều tra; thư mục run chưa commit phải được giữ để kiểm tra" | `FOREIGN-FLOWS.md` (Xử lý sự cố) | **CONFLICTS** (vận hành) | `health()` coi mọi `runs/*` thiếu `manifest.json` là `incomplete_runs`, khiến `ready=False` (`foreign_flows.py:327-328`); retry coi là nonretryable (`foreign_refresh.py:48-50,64-66`). Giữ thư mục tại chỗ thì **mọi ngày sau đều partial, exit 2**. Docs không có thủ tục di chuyển/quarantine |
| 7 | "Unique temp + fsync + replace ...; stale lock chặn retry; thiếu hướng dẫn xử lý lock có kiểm chứng" | system-design-review (bảng, 04-DD1) | **EXTENDS** | Lock mkdir không kiểm tra tuổi (`foreign_flows.py:186-190`). Khi bị hard kill, lock còn lại và chặn **mọi** collector (foreign_refresh, eod_update, DailyRunner). Ctrl-C thì `finally` xoá lock (`:266-269`) nhưng để lại run dir chưa commit → #6 |
| 8 | "Snapshot manifest dùng write_text trực tiếp" | system-design-review (04-DD1) | **MATCHES** | `reconciliation.py:138`; `verify_snapshot` chặn JSON dở (`:142-147`); không có retention, thư mục orphan tích lũy |
| 9 | Phiên 01/10 thiếu record; "không được backfill" | system-design-review (Git và bằng chứng vận hành) | **MATCHES / EXTENDS** | Log dòng 4630-4631; không có `paper/runs/2026-10-01` hay flow run 01/10. Bổ sung: window đóng 09:00 VN phiên kế tiếp (`paper_runner.py:39-42`). Nhiều lần chạy bắt đầu trễ (23/09 20:47, 24/09 20:16, 25/09 20:37, 28/09 19:45, 01/10 20:03). **INFERRED**: máy tắt/ngủ lúc 17:05 rồi `StartWhenAvailable` kích hoạt. Không có cảnh báo khi một phiên mất hẳn |
| 10 | "Window opens 16:00 VN ... closes 09:00 next session; exit 3 preview only" | `2026-09-26-market-readiness.md` (Operator runbook) | **MATCHES** | `paper_runner.py:39-42,192,251` |
| 11 | "Calendar maintained through 2026 and requires updating before 2027" | 2026-09-26-market-readiness (Data gap corrections) | **MATCHES** (chưa xử lý) | `HOSE_HOLIDAYS` kết thúc `date(2026, 9, 2)` (`exchange_calendar.py:22-82`). Không có guard cảnh báo khi vượt năm cuối. Từ 01/01/2027, ngày nghỉ sẽ bị coi là ngày giao dịch và readiness chặn (fail closed, không giải thích nguyên nhân) |
| 12 | Exit codes: "0 success; 2 invalid ... scan/score conflict; 3 outside window"; "BAT không báo SUCCESS nếu một bước lỗi"; "không suy diễn một nguyên nhân duy nhất cho mã thoát" | 2026-09-26-market-readiness (runbook); foreign-data-rollout (Tự động cập nhật; Theo dõi phiên) | **MATCHES / EXTENDS** | `paper_runner.py:240-251`, `bat:17-29`. Bổ sung: scoring blocked trả 2 dù `paper.json` **đã ghi** (`:242-250`; log 28-30/09 in "FAILED" sau dòng `"status": "recorded"`). Không có status JSON theo từng bước |
| 13 | Dashboard input fail-closed (`scan_guard`) | (không audit nào mô tả kết quả thực tế) | **NOT-COVERED** (INFERRED) | `readiness` gọi `assess_market_data` với universe VN30 (`scan_guard.py:9-17`). `prices_hist` có 101 CSV (VN100) nên mọi file ngoài rổ bị báo "unexpected symbols" (`paper_runner.py:52-55`). Kết hợp với dữ liệu cũ, MR/momentum/swing trên dashboard trả `blocked_data` kể cả sau khi bấm cập nhật |
| 14 | `_DATA_UPDATE.running` check/start không lock; "IgnoreNew chỉ bảo vệ task đó, không khóa các entrypoints khác" | system-design-review (P1 thiếu session; bảng 13-TĐ4) | **MATCHES / EXTENDS** | `app.py:335-336`. Bổ sung: `eod_update` CLI và dashboard thread cùng ghi `prices_hist` với cùng `.csv.tmp` (`eod_update.py:88-90`); cache và `forward_test.jsonl` không lock |
| 15 | "Collector có khóa chống chạy đồng thời, status được thay thế nguyên tử" | FOREIGN-FLOWS (Hợp đồng lưu trữ); foreign-data-contract (Storage) | **MATCHES** | `foreign_flows.py:44-55,186-190,264` |
| 16 | Silent swallow trong đường legacy | (không có) | **NOT-COVERED** | Lỗi swing warm-up bị nuốt (`eod_update.py:140-141`). Đọc/ghi cache `except Exception: pass` (`mr_scan.py:305-316`, `swing_scan.py:101,108`, `momentum_scan.py:161,168`). `eod_update.main` luôn exit 0 (`:169-190`). Collector và paper runner **không** nuốt lỗi; lỗi được ghi vào manifest/report |
| 17 | VCI: "Never retry rate limits/auth failures"; pacing 3,1 s | system-design-review (13-TĐ2); code comment | **EXTENDS** | `reconciliation.py:80-84,113-115`. Một HTTP 429 ở 1/31 mã làm cả snapshot blocked (`:136`), và không có backoff cho 429. Đúng với nguyên tắc không né rate limit, nhưng chưa được ghi thành failure mode |
| 18 | "EOD job resilient to vnstock rate-limit: retries 4×, back off 20 s" | DEPLOY §6 (dòng 159-160) | **MATCHES** (chỉ đường legacy) | `eod_update.py:65-101`; không áp dụng cho job theo lịch |
| 19 | Retry-stale 60/300/900 s, tối đa ~21 phút; abort khi đổi phiên/đồng hồ lùi; không retry lỗi schema/gap/run dở | foreign-data-contract (Collection readiness extension); FOREIGN-FLOWS (Cập nhật tự động) | **MATCHES** | `foreign_refresh.py:18-76`; log 02/10 và 05/10 `ready` |
| 20 | "Verified collection rollout": "Lần chạy tự động kế tiếp chưa diễn ra"; trạng thái 30/09 `health=ready` | foreign-data-rollout (Đã tích hợp bộ thu thập) | **EXTENDS** (đã có quan sát mới) | Sau đó: 01/10 mất (bị ngắt); 02/10 và 05/10 flow `ready` và paper `recorded` (log 4631-4642; `foreign_flows_v2/runs/20261002*`, `20261005*`). Claim "ready" tại 30/09 vẫn đúng cho thời điểm đó, không chứng minh tính liên tục |
| 21 | "Lần gần nhất 02/10 ... LastTaskResult=0; 5 thư mục phiên: 25, 28, 29, 30/09 và 02/10" | system-design-review (Git và bằng chứng) | **EXTENDS** | Đã có thêm `paper/runs/2026-10-05` (log 4637-4642, SUCCESS 17:13) |
| 22 | `/api/health` luôn ok; chưa có freshness SLO/expected-session coverage | system-design-review (05-TĐ1/05-DD1) | **MATCHES** | `app.py:149`; không có endpoint/CLI nào gộp trạng thái phiên hôm nay cho cả flow, paper và legacy |
| 23 | Guard `dashboard_data_guard`/`evidence_guard`/`integrity_blockers` | chỉ xuất hiện là tên test trong `2026-09-2x-*tests.xml` | **NOT-COVERED** | Không có module code cùng tên; guard thực tế là `scan_guard.readiness` và `assess_market_data` |
| 24 | Paper scoring chặn khi có revision; volume-only revision được báo riêng | `2026-09-30-paper-requested-window.md` | **MATCHES** | Log 02/10 và 05/10 `ok_with_revisions`; 28-30/09 blocked |
| 25 | Execution receipts: "No ... collector schedule change" | `2026-09-30-execution-receipts.md` (Outcome and scope) | **MATCHES** (không liên quan flow) | Không có writer dữ liệu thị trường trong lane này |

## Failure modes

| Tình huống | Hành vi hiện tại | Fail | Evidence |
|---|---|---|---|
| VCI timeout/5xx | 3 lần, backoff 3,1·n s; sau đó symbol fail, snapshot blocked, exit 2 | closed | `reconciliation.py:71-84,136`; log 25/09 |
| VCI 429/4xx | Không retry; snapshot blocked | closed | `reconciliation.py:80-84` |
| Vietstock lỗi một phần | 3 lần/cặp, backoff 2-4 s; circuit sau 5 lỗi liên tiếp; status partial/blocked, exit 2 | closed | `foreign_flows.py:196-252`; `foreign_refresh.py:107` |
| Nguồn công bố muộn | Retry mã thiếu sau 60/300/900 s rồi `retry_exhausted` | closed | `foreign_refresh.py:61-76`; log 30/09 |
| Hard kill khi đang thu flow | `.collect-lock` còn lại; mọi collector sau raise | closed, **dính vĩnh viễn** | `foreign_flows.py:186-190` |
| Ctrl-C / exception khi đang thu flow | Lock được xoá; run dir thiếu manifest làm health không bao giờ ready | closed, **dính vĩnh viễn** | `foreign_flows.py:266-269,327-328` |
| Kill khi đang snapshot | Manifest thiếu/dở; `verify_snapshot` chặn; lần sau tạo thư mục mới | closed | `reconciliation.py:103,138,142-147` |
| Kill khi đang commit paper | `.write-lock` còn lại, chặn session đó | closed | `paper_runner.py:112-116` |
| Chạy lại cùng ngày | Snapshot mới; cùng identity → `already_recorded`; provider sửa bar → conflict, exit 2 | closed | `paper_runner.py:118-125` |
| Chạy trước 16:00 VN | Session = phiên trước; ngoài window → preview, exit 3 | closed | `exchange_calendar.py:8-19`; `paper_runner.py:192,251` |
| Máy tắt lúc 17:05 | Catch-up chỉ ghi được nếu chạy trước 09:00 VN hôm sau; nếu không, phiên mất | closed, **mất dữ liệu** | `paper_runner.py:39-42`; 01/10 |
| Mất mạng giữa run | Như timeout/lỗi ở trên; BAT vẫn chạy paper | closed | `bat:13-18` |
| Scoring gặp revision | Exit 2 dù record đã ghi | closed, **mơ hồ** | `paper_runner.py:242-250` |
| `eod_update` (legacy) provider lỗi | Bắt `BaseException`, retry 4× ×20 s; các bước sau trả error dict; exit 0 | **open** | `eod_update.py:97-101,129,149,165,201` |
| Đọc/ghi cache lỗi | Nuốt im lặng | **open** | `mr_scan.py:305-316` và tương tự |
| Vị thế MR trên dữ liệu cũ | `check_positions` không kiểm tra readiness, trả HOLD trên bar cũ | **open** (hiện 0 vị thế) | `position_manager.py:138-175` |
| Năm 2027 | Ngày nghỉ bị coi là ngày giao dịch; readiness chặn | closed, không giải thích | `exchange_calendar.py:22-82` |

## Concurrency

- Có lock: `.collect-lock` (mọi caller của `collect_flows`) và
  `paper/runs/<session>/.write-lock`. Task Scheduler `IgnoreNew` chỉ chặn chính task.
- Không lock: `prices_hist` (CLI và dashboard thread, cùng tên `.csv.tmp`),
  `*_scan_cache.json`, `forward_test.jsonl`, `raw/prices` legacy,
  `repository.write_json` (`.tmp` cố định), check/start của `_DATA_UPDATE`
  (`app.py:335`).
- Scheduled job và đường legacy chỉ chạm nhau ở `collect_flows`. Nếu chạy chồng,
  bên đến sau raise và báo blocked (fail closed). Snapshot của paper nằm trong thư
  mục riêng nên không bị writer legacy làm hỏng.

## Observability và freshness

- Hiện có: `eod_update.log` (append, không rotate), `foreign_flows_v2/latest_status.json`
  (chỉ mô tả batch cuối), `foreign_refresh --status`, `paper/latest.json`,
  `paper/scores/latest.json`, `paper_runner --check`, `LastTaskResult`.
- Thiếu:
  - Một chỉ báo duy nhất cho "phiên hôm nay đã đủ".
  - Dòng kết thúc khi job bị ngắt (log 4609, 4631 chỉ có `^C`).
  - Cảnh báo phiên bị mất.
  - Cảnh báo `prices_hist`/dashboard cũ.
  - Health endpoint có freshness.
- Freshness contract:
  - Paper: nghiêm ngặt (manifest hash, `as_of` = session, readiness, window).
  - Dashboard: fail closed qua `scan_guard`.
- Chỗ dữ liệu cũ có thể bị dùng im lặng:
  - `check_positions`.
  - `log_recommendations` (nhận mọi payload).
  - `/api/scan/latest` (đọc `LATEST_SCAN_PATH` nguyên trạng).
  - Training của `DailyRunner` từ cache legacy.
- Wall-clock:
  - `eod.completed_bars` và `scan_input_snapshot` dùng giờ thật (`eod.py:14,36`).
  - `date.today()` theo giờ máy ở `daily_runner.py:56` và `optimizer.py:122`
    (đã lọc bằng `completed_bars`).
  - Phần chuyển sang giờ VN trong `completed_session_date` không phụ thuộc
    timezone của máy.

## Findings xếp hạng

1. **HIGH — Đường legacy không còn được cập nhật, docs lỗi thời.**
   `prices_hist`, cache và `forward_test` cũ từ 21/09. Render free không nhận data
   từ 22/09. Docs sai: DEPLOY §1/§4/§6, RENDER-FREE (#1-#4). Kịch bản: người dùng
   mở dashboard tin rằng 17:05 đã cập nhật, nhưng thấy `blocked_data` hoặc số liệu cũ.
2. **HIGH — Flow block "dính" sau khi bị ngắt** (#6, #7). Một lần Ctrl-C hoặc mất
   điện khi đang thu flow làm mọi ngày sau báo partial/blocked cho tới khi người
   vận hành can thiệp. Runbook hiện tại bảo giữ thư mục tại chỗ, tức là giữ luôn lỗi.
3. **HIGH — Mất phiên vĩnh viễn** khi job không xong trước 09:00 VN (#9). Đã xảy ra
   01/10. Lịch sử log cho thấy máy thường bắt đầu chạy sau 19:45 (INFERRED: tắt/ngủ).
4. **MEDIUM — Bảng ngày nghỉ chỉ đến 2026** (#11). Không có cảnh báo trước.
5. **MEDIUM — Exit code mơ hồ** (#12). Record đã ghi vẫn hiện "FAILED"; flow lỗi
   che việc paper đã thành công trong `LastTaskResult`.
6. **MEDIUM — Snapshot manifest không atomic, không retention** (#8).
7. **MEDIUM — Writer không lock ở đường legacy** (#14).
8. **LOW — Silent swallow và exit 0 ở `eod_update`/cache** (#16).
9. **LOW (INFERRED) — Dashboard `blocked_data` vĩnh viễn** vì VN100 files không
   khớp universe VN30 (#13).
10. **LOW — Không có backoff cho 429 của VCI**. Một mã bị throttle làm hỏng cả ngày (#17).

## Đề xuất có giới hạn (không đổi lịch, không đổi strategy/data contract)

1. Cập nhật `DEPLOY.md` §1/§4/§6 và `RENDER-FREE.md` theo luồng thực tế: BAT chỉ
   chạy flow + paper, không push, `eod_update` chỉ chạy tay. Ghi rõ dashboard
   không được cập nhật theo lịch.
2. Bổ sung `FOREIGN-FLOWS.md` (Xử lý sự cố): một run dir thiếu manifest sẽ chặn
   `health`. Thêm thủ tục có kiểm chứng: xác nhận không còn process, rồi **di
   chuyển** (không xoá) run dir sang một khu quarantine có ghi lý do. Có thể kèm
   một lệnh chỉ đọc `--status` liệt kê lock/orphan cùng tuổi của chúng.
3. Thêm báo cáo "missed session" chỉ đọc vào `paper/latest.json` hoặc một lệnh
   check: so các phiên giao dịch gần nhất với `paper/runs/*` và báo phiên đã đóng
   window mà không có record. Không backfill.
4. Thêm guard cảnh báo khi `completed_session_date()` vượt năm cuối của
   `HOSE_HOLIDAYS`. Cập nhật lịch 2027 khi Sở công bố, theo quy trình nguồn trong
   `2026-09-30-research-sources.md`.
5. Ghi một status JSON theo bước (flow/snapshot/record/score) cho mỗi lần BAT
   chạy, tách "record đã ghi" khỏi "scoring blocked". Giữ nguyên exit code hiện
   tại để không phá contract của launcher.
6. Ghi snapshot manifest bằng temp + fsync + replace như `_atomic_json`. Đặt
   chính sách giữ lại cho snapshot **không** được record nào tham chiếu. Thay đổi
   này chạm tới data contract nên cần preregister và chạy test theo AGENTS.md.
7. Đường legacy: thêm lock cho `refresh_prices` và đặt tên tmp duy nhất; ghi log
   thay cho `except: pass`; cho `eod_update.main` trả exit khác 0 khi có bước lỗi.
8. Muốn dashboard hữu ích trở lại thì cần quyết định của user: hoặc nạp từ
   snapshot đã verify, hoặc tách universe. Không tự bật lại `eod_update` hay `git push`.

## Giới hạn của review

- Không truy vấn Task Scheduler. Cấu hình task lấy từ các audit 30/09 và 04/10.
- Không chạy test. Các failure mode "dính" được suy ra từ code đọc được, chưa tái
  hiện bằng fault injection.
- Không mở file dữ liệu lớn ngoài vài dòng cuối log/CSV và danh sách thư mục.
- Review này không phải chấp thuận giao dịch hay triển khai.

## Cập nhật sau merge research_ml (06/10, 4403fc7)

Bối cảnh: `git diff --stat 4b859cd 4403fc7` không chạm `run_eod_update.bat`,
`stock_agent/data/`, `stock_agent/pipeline/`, `app.py`, `docs/DEPLOY.md`,
`docs/RENDER-FREE.md`, `docs/FOREIGN-FLOWS.md`. Thay đổi chỉ nằm ở `stock_agent/research_ml/`,
`scripts/ml_research_r5*.py`, tests, configs, docs/audits và `.gitattributes`. research_ml
không được nối vào luồng EOD. R5 đọc snapshot pinned `research-snapshot-v1`
(`as_of` 2026-09-29; `configs/research/ml_protocol_v1.json:15`), nên các lỗi EOD dưới đây
không ảnh hưởng evidence R5. Chúng vẫn là blocker cho R8 (prospective readiness).

| ID | Trạng thái | Evidence @ `4403fc7` / ghi chú |
|---|---|---|
| Bảng đối chiếu #1–#3 (DEPLOY/RENDER-FREE lỗi thời) | STILL OPEN | Docs không đổi |
| #4, #5 (không còn writer theo lịch cho `prices_hist`, dashboard, `forward_test`) | STILL OPEN | Không đổi |
| #6, #7 (run dir/lock "dính" sau khi bị ngắt) | STILL OPEN | `foreign_flows.py:186-190,266-269,327-328` không đổi |
| #8 snapshot manifest không atomic, không retention | STILL OPEN | `reconciliation.py:138` không đổi. `research_ml/catalog.py:123-155` có mẫu publication một transaction nhưng chưa nối vào collector |
| #9 mất phiên khi job không xong trước 09:00 VN | STILL OPEN | `paper_runner.py:39-42` không đổi |
| #11 lịch giao dịch chỉ tới 2026 | STILL OPEN | `exchange_calendar.py` không đổi. Port R1 (`firstmate/r1-port-into-research-ml`) chỉ đưa calendar coverage → `blocked` vào research path, không thêm cảnh báo cho EOD |
| #12 exit code mơ hồ | STILL OPEN | `paper_runner.py:240-251` không đổi |
| #13 dashboard `blocked_data` (VN100 vs VN30) | STILL OPEN (INFERRED) | Không đổi |
| #14 writer không lock ở legacy | STILL OPEN | `eod_update.py:88-90`, `app.py:335-336` không đổi |
| #16 lỗi bị nuốt im lặng, exit 0 | STILL OPEN | Không đổi |
| #17 không backoff cho VCI 429 | STILL OPEN | Không đổi |
| #22 thiếu health/freshness SLO | STILL OPEN | `app.py:149` không đổi |
| #10, #15, #18, #19, #24, #25 | N/A (MATCHES, không đổi) | |
| #20, #21, #23 | N/A (quan sát/ghi chú, không có evidence vận hành mới trong review này) | |
| Findings 1–10 | STILL OPEN | Như các hàng trên |
| Đề xuất 1–8 | STILL OPEN | Chưa đề xuất nào được thực hiện. Đề xuất 6 (atomic manifest + retention) có thể tái dùng mẫu của `research_ml/catalog.py`, nhưng vẫn cần preregister theo AGENTS.md |

Còn actionable:
1. Ưu tiên giữ nguyên: sửa docs DEPLOY/RENDER-FREE (đề xuất 1), thủ tục quarantine run dir dở (đề xuất 2), báo cáo missed session (đề xuất 3) và cảnh báo lịch 2027 (đề xuất 4).
2. Muốn nối catalog research_ml vào luồng thu thập thì cần task riêng có user cho phép; không đổi lịch 17:05 hay BAT.
