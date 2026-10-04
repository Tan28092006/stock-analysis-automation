# Đánh giá kiến trúc backend theo Alex Xu / Sahn Lam — 04/10/2026

## Phạm vi và kết luận

Review code tại commit `6c3b8f422dd522ac0e4a892f23319e4a431b1ef7`, trạng thái local
và remote, Task Scheduler và các artifact đã có. Không chạy scan mới, thay lịch,
gọi broker, sửa chiến lược hoặc chạy một thử nghiệm backtest mới.

Hệ thống phù hợp làm công cụ nghiên cứu EOD và ghi nhận quan sát paper ở quy mô
cá nhân. Phần snapshot/hash, chặn dữ liệu không đủ và CI replay có cơ chế đáng
kiểm chứng. Hệ thống chưa có một đường danh mục thống nhất nối tín hiệu đã kiểm
định với tiền mặt, cổ phiếu chờ về, lệnh và khớp thực tế. Độ tin cậy vận hành còn
phụ thuộc máy local và nhiều kho JSON với guarantee khác nhau.

Quy mô hiện tại không tạo lý do để thêm Kafka, microservices hoặc distributed
matching engine. Một backend local với transaction boundary rõ và sổ sự kiện
bền vững là phương án nên kiểm chứng trước.

## Git và bằng chứng vận hành

- Origin: `https://github.com/Tan28092006/stock-analysis-automation.git`.
- Nhánh local/remote: `codex/data-integrity-audit`, cùng SHA `6c3b8f4` theo
  `git ls-remote`, không chỉ theo remote-tracking cache.
- PR #1: OPEN, draft, base `main`; push lên nhánh chưa phải merge vào main.
- CI `36719949209`: completed/success, đúng SHA trên; đây là bằng chứng run
  đã có, không phải chạy lại toàn bộ suite trong review này.
- Windows task `StockAgent_EOD_Update`: `Ready`, chạy launcher local, 17:05
  thứ Hai–Sáu, `MultipleInstances=IgnoreNew`. Lần gần nhất 02/10/2026 17:05,
  LastTaskResult=0; lần tiếp theo 05/10/2026 17:05, giờ Việt Nam.
- Log 02/10: bắt đầu 17:05:01, hoàn tất 17:17:01; khoảng 12 phút là một lần
  đo end-to-end, không phải SLO hoặc p99. Giá 31 symbols gồm VNINDEX;
  flow job báo ready trên toàn universe yêu cầu, không có missing/gaps/errors.
- `data/paper/latest.json`: session 02/10, recorded, data_ready/source_verified
  đều true. Có 5 thư mục phiên: 25, 28, 29, 30/09 và 02/10.
- Score 02/10: 50 pending, 0 resolved, 0 not_entered; 4 volume revision rows.
  `assumptions.actual_execution_verified=false`, `portfolio_pnl=false`.
- Phiên 01/10 thiếu paper record. Log dòng 4630–4631 ghi job bắt đầu lúc
  20:03:45 rồi có `^C`, không có terminal success/failure tương ứng. Điều này
  phù hợp một lần chạy bị ngắt; chưa xác định ai/nguyên nhân gây ngắt. Không
  được backfill thành record ex-ante.
- Các artifact runtime và source snapshots nằm trong `.gitignore`; push code
  không sao lưu chúng. Có khoảng 2,8 GB artifact nghiên cứu dưới
  `data/paper/research_gate`; đây là dung lượng hiện đo, chưa phải dự báo tăng trưởng.

## Luồng hiện có

```text
Windows Task Scheduler -> run_eod_update.bat
  -> foreign_refresh --retry-stale -> raw vintages + canonical rows + manifests
  -> paper_runner --refresh --run -> snapshot VCI 31 symbols
       -> readiness/hash -> MR + momentum scan -> immutable session paper.json
       -> paper_scoring -> scores/latest.json

Pinned fixtures -> research_gate + books + timeslices + universe + events -> CI artifacts
Legacy CLI/HTTP -> mutable price caches + position stores + daily exit checks
Supplied order plan/receipts -> execution_receipts -> single-plan audit
```

Paper runner dùng hàm scan thuần và không đọc vị thế. Momentum picks là ranking
top-N được ghi mỗi phiên, chưa phải danh mục có vốn được tái cân bằng theo tháng.
Chương Stock Exchange được áp dụng cho sequencing/replay và risk state; hệ thống
không vận hành matching engine nên price-time priority, multicast, 1 tỷ lệnh/ngày
và SLA của case trong sách không phải requirement ở đây.

## Đối chiếu câu hỏi với implementation

Nguồn: skill người dùng chỉ định tại
`C:/Users/acer/.agents/skills/system-design-alex-xu/SKILL.md`, checklist và
guides 04, 05, 06, 12, 13. Trang dưới đây là trang PDF vật lý ghi trong guide.
TĐ là câu trao đổi của case; DD là câu chuyển thể từ deep dive. Con số trong
case chỉ xác lập phạm vi của sách; đánh giá dùng requirements EOD/local thực tế.

| Câu hỏi | Hệ thống trả lời thế nào | Tiêu chí sách và trang PDF | Evidence | Kết luận / khoảng thiếu |
|---|---|---|---|---|
| 13-TĐ1: Instruments, thao tác và phiên nào? | Cổ phiếu VN30, scan EOD; research VN100 H1; receipt contract chỉ BUY/SELL LO 100 cổ phiếu. | 381–382: chốt instrument, order types, session và exclusions trước thiết kế. | `execution_receipts.validate_plan`, `exchange_calendar`, `paper_runner._window`, universe configs. | Có contract ở từng lane; chưa có một executable strategy contract chung. |
| 13-TĐ2 / 04-TĐ1: Workload và output thực là gì? | Một tài khoản sử dụng local; 31 price series mỗi paper refresh; flow collection rộng hơn; 10 momentum candidates/phiên. | 381–382, 98–99: tách outputs/workload; ưu tiên throughput/latency theo use case. | Launcher, config, log 02/10; `build_market_snapshot` pacing 3,1 giây giữa request. | Phù hợp batch EOD. Chưa có percentile latency, giới hạn disk hoặc deadline được đo; khoảng 93 giây pacing riêng cho 31 symbols chỉ là estimate. |
| 04-TĐ2: Có replay và retention đủ dài? | Raw snapshots + hashes, pinned fixtures; mỗi session record lưu riêng. | 98–99: log giữ lại để consumers replay; hai tuần là giả định riêng của case. | `reconciliation.verify_snapshot`, `paper_runner.commit_paper_run`, workflow CI. | Có replay đầu vào trong phạm vi tests. Chưa thấy retention, backup độc lập, restore drill hoặc RPO/RTO. |
| 04-TĐ3: Ordering/delivery và duplicate? | Một paper record/session; rerun cùng identity trả already_recorded, khác identity bị chặn; collector dùng directory lock. | 98–99: order domain rõ; at-least-once cần duplicate handling. | `commit_paper_run`, `collect_flows`; paper/flow tests. | Đã kiểm chứng idempotency và lock trong phạm vi test. Chưa có transaction cho toàn job flow→snapshot→record→score. |
| 04-DD1: Crash sau ghi hoặc trước commit? | Unique temp + fsync + replace cho paper/flow JSON; stale lock chặn retry; snapshot manifest là mốc commit. | 105–129: output/progress boundary và recovery phải rõ. | `paper_runner._atomic_json`, `foreign_flows.atomic_json`, manifests. | Có cơ chế tránh overwrite một phần. Thiếu kill-process/disk-full/restore drills và hướng dẫn xử lý lock có kiểm chứng. Snapshot manifest dùng write_text trực tiếp: crash có thể để JSON dở dang và verification phải chặn. |
| 06-TĐ1: Fields, identity, event time? | Flow có session/source timestamp/observed_at/available_at/raw hash; giá lưu raw request và fetched/completed times. | 163–164: event identity/timestamp là điều kiện để aggregate/replay đúng. | `normalize_chart`, `verified_records`, `reconciliation`. | Đã kiểm chứng schema/as-of trong tests; adjustment/PIT lịch sử chưa được nguồn chứng nhận. |
| 06-TĐ3 / 06-DD1: Dữ liệu trễ, trùng và revisions? | Bounded retry 60/300/900 giây; giữ vintages; as-of lọc theo available_at; legacy quarantine; positive-volume revision được báo riêng. | 163–164, 176–194: event time khác processing time; dedupe/correction/reconciliation riêng. | `refresh_with_retries`, `load_flows`, `_check_history`; tests late/duplicate/tamper. | Kiểm chứng trong các failure cases dẫn. 4 revision rows hiện có là cảnh báo; không chứng minh inputs không đổi về ý nghĩa chiến lược. |
| 12-TĐ2 / 13-DD1: Có tái dựng cùng quyết định/fills? | Research dùng immutable snapshot, config/source hashes và replay; paper ghi quyết định + code hash. | 342–343, 386–387, 399–411: cùng ordered input, rules, initial state phải tạo cùng output. | research CI; `paper_runner` sources/IDENTITY_KEYS; `period_backtest`. | Research replay có bằng chứng. Paper code_hash chưa bao phủ `config.py`, `data/eod.py`, `data/reconciliation.py`, constants, môi trường/dependencies; không có self-contained archived runtime bundle. |
| 13-TĐ3: Tiền/cổ phiếu đang pending được giữ thế nào? | Research broker có cash, FIFO lots, phí, settlement; receipt audit kiểm tra resource một plan. Store manual chỉ lưu vị thế. | 381–382: available funds khác total balance; pending cần reservation. | `period_backtest.Broker`, `execution_receipts.validate_plan`, `PositionStore`, `PortfolioStore`. | Chưa đáp ứng quản lý vốn toàn tài khoản qua nhiều plans. Hai plans riêng có thể cùng khai báo cùng cash. Cần một transaction boundary và reservation chung trước khi coi plan có thể cấp vốn. |
| 12-DD1: Ghi đồng thời hoặc store hỏng có làm mất state? | MR store có threading.Lock một process; `_load` trả [] nếu JSON lỗi; PortfolioStore read-modify-write không lock; shared writer dùng temp name cố định. | 348–358, 363–374: authoritative durable state, concurrency/recovery phải bảo toàn invariant. | `position_manager:84–97,113–127`, `portfolio/pnl:27–38`, `repository:24–29`. | Chưa đáp ứng multi-process và fail-closed corruption. Test concurrent adds hiện chỉ kiểm tra threads; không suy rộng sang CLI + server. |
| 05-TĐ1 / 05-TĐ4 / 05-DD1: Theo dõi gì và khi mất data thì báo thế nào? | Readiness JSON, exit code, log; /api/health luôn ok; Task Scheduler giữ kết quả gần nhất. | 135–136, 142–156: missing data khác normal; alert state và delivery cần evidence. | `app.py:137`, launcher, flow health; thiếu session 01/10. | Batch failures có tín hiệu tại máy local. Chưa thấy freshness SLO/expected-session coverage, durable incident state hoặc kênh cảnh báo đã kiểm chứng. |
| 05-TĐ2 / 05-TĐ3: Tăng trưởng, giữ data và query cost? | `verified_records` đọc và kiểm raw của mọi committed run; health/load lặp toàn lịch sử. | 135–136: series cardinality, rate, retention/rollup phải định lượng. | `foreign_flows:282–329`, dung lượng artifact hiện có. | Chưa đủ performance evidence khi archive tăng. Index/read projection đã xác minh có thể giảm cost; raw phải giữ để audit và rebuild. |
| 13-TĐ4: Latency/recovery/availability và bảo vệ? | Local scheduler IgnoreNew; public bind không password bị lockdown; dữ liệu state trên một disk. | 381–382: SLA phải đo, recovery/failure scope rõ. | Scheduler settings, `app.main`, tests auth; job 02/10 ~12 phút. | Đáp ứng cơ chế auth trong tests, chưa có RTO/RPO/availability claim. IgnoreNew chỉ bảo vệ task đó, không khóa các entrypoints khác. |

## Các vấn đề cần ưu tiên

**P1 — Hợp đồng danh mục chưa chung giữa research và sử dụng local.**
`paper_scoring` dùng settle_lock=2 và cost_pct=0,6 cho MR single-trade outcomes;
momentum dùng close-to-close 21 phiên không tính phí để theo dõi thông tin.
`period_backtest` dùng broker vốn hữu hạn, phí từng phía, T+3 bảo thủ và cadence
tháng/tuần. `momentum_scan` tạo top-N ngày và kiểm tra buffer hàng ngày cho vị thế
manual. Chạy những lane này không đảm bảo người dùng thực hiện đúng đường đã
backtest. Cần một strategy contract versioned và một portfolio reducer dùng chung
cho simulation và kế hoạch on-demand, với cash/receivables/lots/fees/cadence.

**P1 — Store legacy không bảo vệ đầy đủ authoritative state.**
File MR JSON lỗi bị coi là danh sách rỗng, lần add sau có thể thay thế toàn bộ
state. Thread lock không bảo vệ hai process; PortfolioStore còn thiếu lock trong
cùng process. Phương án vừa đủ là SQLite transaction + unique event/idempotency
keys + immutable imported receipts, hoặc một writer độc quyền được mọi entrypoint
tuân thủ. Không tự migrate sổ thật trong một review.

**P1 — Thiếu session và job completion chưa được giám sát đầu cuối.**
01/10 thiếu record dù phiên tiếp theo thành công. HTTP update trả 202 ngay khi
khởi chạy; workflow remote chấp nhận 200/202 là success. `_DATA_UPDATE.running`
được set bên trong worker, check/start bên ngoài không có lock: hai requests có
thể cùng khởi chạy. Đây là đường legacy riêng với scheduler hiện tại. Cần durable
job identity/state/result, expected-session coverage và trạng thái interrupted;
202 chỉ xác nhận nhận việc. Khôi phục được log/job không đồng nghĩa được tạo
quyết định ex-ante cho phiên đã bỏ lỡ.

**P2 — Lineage runtime, backup và phục hồi chưa khép kín.**
Mở rộng identity sang code/config/dependency closure hoặc lưu runtime bundle
versioned. Backup độc lập snapshots, paper records và receipts; thử restore và
so sánh hashes/NAV. Đặt RPO/RTO theo nhu cầu local thực tế. Git push không thay
backup cho các thư mục ignored. Chưa có chứng cứ để nói dữ liệu đã mất hoặc
không tồn tại backup bên ngoài repo; review chỉ chưa tìm thấy cơ chế/drill trong
phạm vi đã kiểm tra.

**P2 — Full-history validation sẽ tăng cost theo archive.**
Giữ raw immutable, thêm verified manifest catalog/projection có checkpoint để
query as-of nhanh, có rebuild/parity tests. Đo p50/p95/max với archive lớn và cold
cache trước khi chọn tối ưu; không suy throughput từ request pacing.

## Trình tự kiểm chứng đề xuất

1. Đóng execution/portfolio contract; test parity decision, target weights,
   cash, settled shares, fills, fees, exits và monthly activity giữa hai lanes.
   Mọi thay đổi rule/execution/data contract phải preregister theo AGENTS.md và
   chạy complete pinned suite + book suite + timeslices, kể cả H1/2026.
2. Test corruption, hai process cùng mutation, crash sau durable event/trước
   projection, replay từ opening balance, duplicate/conflicting receipts và
   reservation cho hai plans tranh cùng cash.
3. Test mất cả phiên, process bị ngắt, stale lock, disk full, provider timeout,
   failed notification và restore từ backup; đặt freshness/deadline từ nhu cầu
   thực. Việc này không mặc nhiên cho phép thay lịch cập nhật.
4. Thu thập prospective data/execution evidence với version đã khóa; báo cáo
   net profit, excess VNINDEX và activity từng tháng. Record thiếu và ngày đã
   được xem không phải holdout; CI pass không phải bằng chứng có lãi tiền thật.

## Kiểm chứng thực hiện trong review

Chạy read-only inspection và unit/integration tests dùng fixtures/tmp_path:

```text
python -m pytest tests/test_paper_runner.py tests/test_paper_scoring.py tests/test_foreign_flows_v2.py tests/test_foreign_refresh_retry.py tests/test_execution_receipts.py tests/test_hardening.py tests/test_market_runtime_hardening.py -q --tb=short
```

Kết quả: **171 passed in 38.13s** trên Python local Anaconda. Không load test
production; không fault-inject vào data thật; không chạy lại full research gate vì
review không đổi hành vi/config/data contract. Các finding concurrency/crash ngoài
test hiện có được ghi là thiếu cơ chế hoặc thiếu evidence, không giả thành lỗi
đã tái hiện bằng stress test. Chỉ bổ sung tài liệu review; không sửa pipeline.
