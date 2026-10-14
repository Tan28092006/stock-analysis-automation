# VN Stock Agent — swing desk, bắt đáy và nghiên cứu MR/momentum (HOSE)

Nền tảng nghiên cứu và **công cụ kỷ luật trước khi vào lệnh** cho cổ phiếu HOSE (mặc định VN100,
chọn được VN30). Chạy theo dữ liệu cuối phiên (EOD), backend/CLI là chính, **không tự đặt lệnh**.

> [!IMPORTANT]
> Công cụ cá nhân, **không phải lời khuyên đầu tư**. Chưa có chiến lược nào được chứng nhận có lợi thế
> bằng tiền thật. Lịch sử đã xem là dữ liệu phát triển, không phải holdout. Một lần chạy "xanh" không
> phê duyệt giao dịch hay triển khai — xem `AGENTS.md`.

Agent bắt đầu tại [`AGENTS.md`](AGENTS.md) và [`docs/agent-workflow.md`](docs/agent-workflow.md).

---

## 1. Swing desk (dùng hằng ngày)

Bộ lọc **trước khi mua** để chống mua theo cảm xúc, kèm quy trình nhiều agent cho ý tưởng và review
danh mục. Skill: `.claude/skills/swing-desk/SKILL.md` (Claude Code), `.codex/skills/swing-desk/SKILL.md` (Codex).

| Chế độ | Lệnh | Kết quả |
|---|---|---|
| Kiểm tra trước khi mua | `python -m stock_agent.swing_desk check TCB --price 32450 --amount 1000000` | PASS / WARN / BLOCK, luật trượt, giờ bán sớm nhất (chiều T+2), stop 3% và −8%, khối lượng lô lẻ, VND rủi ro |
| Bắt đáy | `... check VRE --mode bottomfish --profile P1` | phân loại cú giảm (a)/(b)/(c), gate BF-G0..G12, cờ hành vi, checklist 10 câu |
| Quét | `... scan` / `... scan --mode bottomfish` | bảng gate toàn universe |
| Review danh mục | `... review` | vị thế, rủi ro mở, kết quả theo gate và nhãn cảm xúc |
| Dữ liệu | `... status`, `... refresh [--held] [--history] [--with-flows]` | độ mới snapshot, kho lịch sử, khối ngoại |
| Đánh giá tiến cứu | `... freeze`, `... score`, `... verify-log` | log hash-chain đóng băng trước 09:00, so desk / baseline / control / 1 analyst |

- Journal riêng: `swing_desk/portfolio.txt` (git-ignored; mẫu `swing_desk/portfolio.example.txt`).
- Rủi ro mặc định 1% NAV ở lỗ hiệu dụng (stop 3% + 1 phiên gap + phí ≈ 4,6%), lô lẻ, trần rủi ro mở 4% NAV.
- Bắt đáy: mặc định **P0 = không bắt đáy**; P1 hồi ngắn sau hoảng loạn, P2 mua 3 đợt có kế hoạch, P3 tình
  huống đặc biệt (chưa có bằng chứng lợi thế, chỉ để đo).
- Nguồn tin cho agent: [`swing_desk/prompts/NEWS_SOURCES.md`](swing_desk/prompts/NEWS_SOURCES.md).
- Thay đổi luật/dữ liệu sau khi đăng ký: [`configs/swing_desk/protocol_amendments.md`](configs/swing_desk/protocol_amendments.md).

## 2. Dữ liệu

| Kho | Nội dung | Ghi chú |
|---|---|---|
| `data/paper/snapshots/` | Giá VCI VN30 + VNINDEX, 650 ngày, snapshot verified mỗi phiên | do job 17:05 ghi |
| `swing_desk/snapshots/` | Giá VN100 ngoài VN30 và mã đang giữ, tải theo yêu cầu | `refresh` / `refresh --held` |
| `data/history/` | Lịch sử ~10 năm (VN30, VN100, VNINDEX), bất biến, có `verify` | [`docs/HISTORY-STORE.md`](docs/HISTORY-STORE.md) |
| `data/foreign_flows_v2/` | Khối ngoại / tự doanh (Vietstock), sạch từ 30/09/2026 | [`docs/FOREIGN-FLOWS.md`](docs/FOREIGN-FLOWS.md) |
| `data/flows_history/` | Khối ngoại lịch sử từ API HOSE (đang xây) | [nguồn đã khảo sát](docs/audits/2026-10-08-foreign-flow-history-sources.md) |

Giá là giá đã điều chỉnh theo thời điểm tải; danh sách thành phần đúng theo thời điểm chỉ có từ 2022
(VN30) và 08/2025 (VN100). `data/raw/foreign` là dữ liệu legacy đang bị quarantine.

**Job tự động 17:05 (T2–T6, `run_eod_update.bat`)**: `foreign_refresh --retry-stale` rồi
`paper_runner --refresh --run`. Không còn chạy `eod_update` và không git push. Kiểm tra sức khỏe:
`python -m stock_agent.pipeline.paper_runner --status`, `python -m stock_agent.pipeline.foreign_refresh --status`
(chi tiết trong [`docs/DEPLOY.md`](docs/DEPLOY.md)). Máy phải bật lúc 17:05 và job phải xong trước 09:00
phiên sau, nếu không phiên đó bị mất.

## 3. Nghiên cứu (không phải chiến lược đã duyệt)

- **Research gate MR/momentum**: `configs/research/research_gate_v1.json`, đọc
  [gate spec](docs/audits/2026-09-30-research-gate-spec.md) và
  [strategy reassessment](docs/audits/2026-09-30-strategy-reassessment.md) trước khi diễn giải kết quả.
- **ML (R1–R5)**: `stock_agent/research_ml/` (contracts, catalog, features, labels, harness, evidence).
  R5 tháng: BLOCKED vì thiếu mẫu; R5 tuần: INCONCLUSIVE, control xáo trộn nhãn thắng model.
  Kế hoạch và ledger: [`docs/plans/2026-10-04-data-ml-research-plan.md`](docs/plans/2026-10-04-data-ml-research-plan.md).
- **Kiểm chứng gần nhất**: [pipeline](docs/audits/2026-10-07-pipeline-check.md),
  [chỉ số nâng cao](docs/audits/2026-10-08-advanced-indicator-check.md),
  [backtest thực tế và T+2](docs/audits/2026-10-06-backtest-realism-challenge.md),
  [nghiên cứu swing desk](docs/audits/2026-10-06-swing-desk-signal-research.md),
  [nghiên cứu bắt đáy](docs/audits/2026-10-06-bottom-fishing-research.md).
- Dashboard cũ (`stock_agent/app.py`, `web/index.html`) và các engine momentum/MR legacy vẫn còn, nhưng
  dữ liệu dashboard (`data/raw/prices_hist`) đứng từ 21/09/2026 và số backtest cũ bị survivorship.

## 4. Cài đặt và test

```bash
pip install -r requirements.txt
# Test dùng interpreter có đủ thư viện (trên máy này: anaconda)
C:/Users/acer/anaconda3/python.exe -m pytest tests/test_swing_desk*.py tests/test_history_store.py -q -p no:cacheprovider
```

Không chạy toàn bộ suite song song với job khác (giới hạn paging file trên Windows).

## 5. Lưu ý trung thực

- **T+2,5**: cổ phiếu mua phiên T bán được từ chiều T+2; stop 3% không thực thi được trước đó, giá có thể
  gap qua stop. Phí + thuế khứ hồi ~0,6%.
- **"T+2" trong code legacy** (`net_t2_*`) thực chất là nhãn 20 phiên có stop/target.
- **Survivorship**: rổ hiện tại áp ngược làm thổi phồng backtest (momentum VN30 +190% so với ~+33%
  point-in-time).
- **LLM / VeriHarness** chưa được chứng minh làm tăng tỉ lệ thắng; chỉ log tiến cứu mới kiểm được.
- **Bắt đáy** loại (b) (tin xấu riêng doanh nghiệp) có bằng chứng tiếp tục trôi giảm.
