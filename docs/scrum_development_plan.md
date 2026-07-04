# Scrum Development Plan - VN30 T+2 Stock Agent

## 1. Product Goal

Phát triển VN30 T+2 Stock Agent thành nền tảng nghiên cứu và gợi ý giao dịch T+2 có kiểm chứng, auditable, technical-only, không dùng news/RSS/sentiment trong quyết định. Hệ thống phải ưu tiên dữ liệu OHLCV đáng tin cậy, backtest tái lập được, risk guard rõ ràng, và dashboard đủ minh bạch để người dùng hiểu vì sao một mã được BUY_SETUP, WATCH hoặc REJECT.

Mục tiêu sản phẩm trong 4 sprint tới:

- Biến các lần refresh/backtest/optimizer/robustness thành pipeline tái lập được bằng report có schema ổn định.
- Tìm hướng cải thiện strategy sau khi optimizer 3 năm chưa đạt target.
- Giữ ML ở vai trò advisory, không cho ML hoặc LLM vượt rule guard.
- Chuẩn hóa dashboard/CLI/API để theo dõi run dài, lỗi provider, và kết quả nghiên cứu.

## 2. Current State

Nguồn số liệu chính được lấy từ các report verification 3 năm trong `data/reports`.

| Hạng mục | Trạng thái hiện tại |
|---|---|
| Dữ liệu 3 năm | `refresh_3y.json`: OK, 29/30 mã lưu được, `VPL` bị loại vì `260 < 540` dòng |
| Khoảng dữ liệu | 2023-05-18 đến 2026-05-27 |
| Baseline full backtest | `-48.818%` total return, `37.248%` win rate, `894` trades, expectancy `-0.3776%`, max drawdown `48.8564%` |
| Optimizer 200 trials | `target_not_reached`, target `10% / 60%` không đạt, apply bị `skipped` |
| Best candidate final test | `-0.714%` return, `33.835%` win rate, expectancy `-0.042%`, max drawdown `2.6988%` |
| Best rules full backtest | `-14.4693%` return, `34.552%` win rate, `547` trades, expectancy `-0.2629%`, max drawdown `16.2935%` |
| Robustness best rules | Monte Carlo p50 `-78.9022%`, max drawdown p95 `92.3599%` |
| Test suite | Full unit suite từng chạy xanh: `55 tests OK` |
| News/RSS | Không còn trong config live; chỉ còn test guard và `_strip_news` |

Kết luận hiện tại: optimizer đã giảm drawdown và loss so với baseline, nhưng strategy vẫn âm và chưa đủ điều kiện apply. Ưu tiên tiếp theo là nghiên cứu nguyên nhân chất lượng tín hiệu, tăng tính tái lập của experiment, và thêm guard chống overfitting trước khi mở rộng vận hành.

## 3. Scrum Operating Model

### Roles

| Role | Trách nhiệm |
|---|---|
| Product Owner | Ưu tiên backlog theo giá trị nghiên cứu, quyết định target acceptance, duyệt increment cuối sprint |
| Scrum Master | Giữ cadence 2 tuần, gỡ blocker về dữ liệu/provider/runtime, đảm bảo retro có action rõ |
| Development Team | Thiết kế, code, test, chạy experiment, ghi report, cập nhật dashboard/CLI/API |
| Quant/Research Reviewer | Review assumption trading, metric, leakage, overfitting, corporate action, và risk guard |

### Cadence

- Sprint length: 2 tuần.
- Sprint Planning: chọn sprint goal, scope, acceptance criteria, và run/report cần tạo.
- Daily Scrum: cập nhật blocker theo 3 nhóm: data, strategy, engineering.
- Sprint Review: demo increment bằng CLI/API/dashboard và report JSON/Markdown.
- Sprint Retrospective: ghi 1-3 cải tiến cụ thể cho sprint sau.

### Definition of Ready

Một backlog item chỉ được kéo vào sprint khi có đủ:

- Mục tiêu người dùng hoặc research goal rõ.
- Input/output dự kiến rõ, gồm file/report/API nếu có.
- Acceptance criteria đo được.
- Dữ liệu hoặc fixture tối thiểu sẵn có.
- Rủi ro leakage/overfitting/provider được nêu nếu liên quan.

### Definition of Done

Một backlog item được xem là done khi:

- Code hoặc tài | PB-01 [3/3] | Reproducible Research | Là researcher, tôi muốn mỗi run refresh/backtest/optimize/robustness có manifest để so sánh giữa các lần chạy. | P0 | 5 | Có manifest gồm command, rules hash, data range, symbols used/excluded, report paths; test đọc manifest pass |
| PB-02 | Data Quality | Là researcher, tôi muốn audit provider theo symbol để biết mã nào bị stale, thiếu rows hoặc có corporate action flag. | P0 | 5 | Refresh report có summary bảng; dashboard/CLI hiển thị excluded reason; `VPL` được báo rõ |
| PB-03 | Strategy Diagnostics | Là quant reviewer, tôi muốn phân rã loss theo indicator, sector, symbol và exit reason. | P0 | 8 | Có report diagnostics cho baseline và best candidate; chỉ ra top negative symbols/exits; không thay đổi signal logic |
| PB-04 | Backtest Safety | Là developer, tôi muốn kiểm tra leakage và execution timing để backtest không dùng dữ liệu tương lai. | P0 | 8 | Thêm tests cho signal date, entry date, exit date, indicator precompute; existing tests vẫn pass |
| PB-05 | Experiment Registry | Là researcher, tôi muốn so sánh nhiều rule set mà không apply config chính. | P1 | 5 | Có registry report-only cho candidate params, metrics, status, notes; apply vẫn qua gate |
| PB-06 | Optimizer Runtime | Là developer, tôi muốn optimizer chạy nhanh và có thể resume/prune. | P1 | 8 | 200 trials có checkpoint hoặc study storage; timeout không mất kết quả; log không spam |
| PB-07 | Search Space Refinement | Là quant reviewer, tôi muốn search space phản ánh T+2 thực tế hơn thay vì chỉ tối ưu threshold hiện tại. | P1 | 8 | Có proposal và experiment cho filter mới như market regime, quant fundamentals (ROIIC/GAAP gap), volatility, liquidity, exit rule; không apply nếu OOS âm |
| PB-08 | Acceptance Gate | Là Product Owner, tôi muốn rule apply chỉ xảy ra khi final test và robustness đều pass. | P0 | 5 | Gate kiểm tra return, win rate, expectancy, drawdown, min trades, MC stress; test pass/fail rõ |
| PB-09 | ML Label Quality | Là ML reviewer, tôi muốn label T+2 không bị lệch calendar hoặc corporate action. | P1 | 8 | Label builder có audit skipped rows; tests weekend/holiday/corporate action; dataset summary report |
| PB-10 | ML Advisory Calibration | Là trader, tôi muốn ML probability được calibrate và chỉ làm advisory. | P1 | 8 | Calibration report có reliability bins, precision/recall by threshold; ML không override REJECT nếu rule guard fail |
| PB-11 | Dashboard Research Tab | Là người dùng, tôi muốn xem optimizer/backtest/robustness summary mà không đọc JSON thô. | P2 | 5 | Dashboard có cards cho status, metrics, warnings, report path; lỗi API hiển thị rõ |
| PB-12 | Async Job Control | Là người dùng, tôi muốn chạy optimizer dài mà không treo request HTTP. | P2 | 8 | API tạo job id, poll status, cancel/view result; dashboard không block |
| PB-13 | Provider Monitoring | Là operator, tôi muốn biết provider nào đang lỗi hoặc bị rate-limit. | P2 | 5 | Health report cho Yahoo/VnStock/local CSV; cảnh báo khi nhiều symbol fallback |
| PB-14 | Documentation | Là engineer mới, tôi muốn hiểu workflow research bằng một tài liệu ngắn. | P2 | 3 | README hoặc docs link tới Scrum plan, runbook, và report examples |
| PB-15 | AI Chatbot | Là người dùng, tôi muốn hỏi LLM chatbot về từng mã cổ phiếu để xem phân tích chi tiết. | P0 | 5 | Có API `/api/chat` nhận `message` và `symbol`, tổng hợp đầy đủ context kỹ thuật & giá |
| PB-16 | AI Chatbot | Là developer, tôi muốn chatbot tự động chọn local Ollama hoặc cloud Nvidia NIM làm client. | P1 | 3 | Tích hợp factory chọn client, bổ sung method `chat_text` cho Nvidia client |
| PB-17 | AI Chatbot | Là người dùng, tôi muốn có khung Chat Assistant trực quan trong Dashboard với các gợi ý nhanh. | P0 | 5 | UI hỗ trợ hiển thị lịch sử chat, context mã đang xem, các gợi ý nhanh (như phân tích điểm mua) |

## 4.1. Bổ sung chi tiết kỹ thuật định lượng & Hiệu năng (Quantitative & Performance Details)

Dưới đây là các chi tiết kỹ thuật định lượng và tối ưu hóa hiệu năng được bổ sung chi tiết cho các Backlog Items trên:

### PB-01 (Manifests)
- **Hệ thống chi phí & Slippage thực tế (HOSE):**
  - Thuế & Phí: Phí mua 0.15% (theo cài đặt mặc định), Phí bán 0.25% (đã bao gồm phí giao dịch bán 0.15% và thuế thu nhập cá nhân PIT 0.1% khi bán).
  - Slippage khớp lệnh theo bước giá (Tick-size-aware): Giá khớp lệnh mua/bán phải trượt theo bước giá tối thiểu của sàn HOSE thay vì phần trăm cố định (dưới 10k VND: bước 10 VND; từ 10k - 49.95k VND: bước 50 VND; từ 50k VND trở lên: bước 100 VND).
- **Tính toàn vẹn dữ liệu:** Tích hợp mã băm SHA-256 của file dữ liệu nguồn vào manifest để phát hiện thay đổi dữ liệu lịch sử giữa các lần chạy.

### PB-03 (Strategy Diagnostics)
- **Hệ thống Metric định lượng chuẩn mực:**
  - Bổ sung chỉ số **Sortino Ratio** (sử dụng downside deviation thay vì standard deviation để chỉ phạt các biến động âm).
  - Bổ sung chỉ số **Calmar Ratio** (tỷ lệ giữa lợi nhuận trung bình năm và Max Drawdown).
  - Bổ sung đo lường **Max Drawdown Duration (MDD)** (khoảng thời gian dài nhất tài khoản bị sụt giảm từ đỉnh cũ chưa về bờ).
  - Bổ sung đo lường **Maximum Consecutive Losses (MCL)** (số lần thua lỗ liên tiếp tối đa).

### PB-06 (Optimizer Runtime)
- **Kiến trúc chạy song song & Tránh mất trạng thái:**
  - Sử dụng SQLite làm RDB storage backend cho Optuna thay vì in-memory để hỗ trợ lưu vết và khôi phục khi bị treo/timeout.
  - Sử dụng song song đa nhân (`n_jobs=-1`) chạy bất đồng bộ đa luồng an toàn.
- **Ngăn ngừa rò rỉ dữ liệu (Data Leakage):**
  - Thực hiện **Purging (lọc bỏ)** ít nhất 90 ngày giao dịch ở biên phân tách Train-Validation để triệt tiêu ảnh hưởng của các chỉ báo tính toán có lookback dài (như EMA50, SMA20).
  - Thực hiện **Embargoing (cấm vận)** 20 ngày giao dịch ngay sau tập Validation trước khi bắt đầu tập Test để triệt tiêu ảnh hưởng của thời gian nắm giữ cổ phiếu (T+2 cho đến tối đa 20 ngày).
  - Vô hiệu hóa Optuna Pruning ngoại trừ trường hợp backtest được tái cấu trúc chạy theo từng chu kỳ ngắn (ví dụ: chia nhỏ từng tháng/quý) và báo cáo chỉ số trung gian liên tục.

### PB-07 (Search Space Refinement)
- **Tinh chỉnh không gian tham số & Quản lý vốn:**
  - **Bộ lọc thị trường chung:** Thêm chỉ báo EMA 50 ngày của VNIndex làm bộ lọc xu hướng (chỉ kích hoạt BUY_SETUP khi VNIndex nằm trên EMA50).
  - **Tránh tối ưu hóa cục bộ (Parameter Cliffs):** Thiết lập lấy mẫu tham số tương đối (Relative Offset Sampling) thay vì lấy mẫu tuyệt đối rồi clip cứng (ví dụ: lấy mẫu `watch_offset` từ [5, 20] và tính `watch = buy_setup - watch_offset`).
  - **Quy mô vị thế động:** Xác định quy mô vị thế (position sizing) tỷ lệ nghịch với độ biến động lịch sử ATR (Average True Range).
  - **Bộ lọc cơ bản định lượng (Quantitative Fundamental / Capital Allocation Filters):** Sàng lọc danh sách VN30 dựa trên các chỉ tiêu tài chính định lượng có tính hệ thống thay vì phân tích định tính:
    - Loại bỏ các mã có chất lượng lợi nhuận thấp (khoảng cách GAAP gap giữa GAAP earnings và Pro Forma earnings quá lớn).
    - Ưu tiên các mã có hiệu quả sử dụng vốn tăng thêm tốt (chỉ số ROIIC - Return on Incremental Invested Capital cải thiện hoặc dương).
    - Phân nhóm doanh nghiệp theo cấu trúc sử dụng vốn (Capital Allocation Profiles: capex-heavy, dividend/buyback-heavy, hoặc R&D-heavy) để áp dụng các trọng số định giá (EV/EBITDA vs Free Cash Flow yield) phù hợp.

## 5. Sprint Roadmap

### Sprint 1 - Chuẩn hóa baseline và reproducible reports

Sprint goal: mọi experiment quan trọng đều tái lập được, có manifest và current-state report dễ đọc.

Sprint backlog:

- [x] PB-01 [3/3]: thêm experiment manifest cho refresh/backtest/optimize/robustness.
- PB-02: chuẩn hóa data-quality summary và excluded-symbol reporting.
- [x] PB-04: bổ sung leakage/execution timing tests cho backtest.
- PB-14: viết runbook ngắn cho quy trình verification 3 năm.

Increment:

- Một command/runbook có thể tái tạo baseline 3 năm và sinh đầy đủ report paths.
- Test suite pass và có thêm guard chống dùng dữ liệu tương lai.

Sprint Review:

- Demo refresh 3 năm và baseline report.
- So sánh manifest với report JSON.
- Xác nhận `VPL` hoặc mã thiếu dữ liệu được báo rõ, không âm thầm bỏ qua.

Retrospective focus:

- Runtime run dài.
- Độ rõ của report.
- Blocker provider/data.

### Sprint 2 - Strategy research và technical rule experiments

Sprint goal: hiểu vì sao baseline lỗ và tạo ít nhất 2 experiment technical-only có OOS diagnostics rõ.

Sprint backlog:

- PB-03: loss attribution theo symbol, indicator, exit reason, drawdown period.
- PB-05: experiment registry report-only cho candidate rule set.
- PB-07: thử search space mới cho market regime/liquidity/exit rules.
- PB-08: nâng apply gate để xét robustness trước khi ghi config.

Increment:

- Có diagnostics report giải thích nguồn lỗ chính.
- Có ít nhất 2 candidate strategy experiments với final test, full backtest, robustness.
- Không candidate nào được apply nếu expectancy hoặc robustness fail.

Sprint Review:

- Trình bày top loss drivers và hypothesis tiếp theo.
- So sánh candidate vs baseline bằng cùng data window.
- Quyết định có mở rộng search space hay thu hẹp universe không.

Retrospective focus:

- Experiment nào hữu ích nhất.
- Metric nào đang đánh lừa optimizer.
- Có cần thêm dữ liệu regime/index không.

### Sprint 3 - ML advisory, labeling và drift checks

Sprint goal: làm sạch pipeline ML advisory để probability hữu ích nhưng không override rule guard.

Sprint backlog:

- PB-09: audit label T+2, skipped rows, holiday, corporate action.
- PB-10: calibration report và threshold diagnostics.
- PB-05: lưu model experiment vào registry để so với rule-only.
- PB-08: thêm ML-specific guard vào apply/decision path.

Increment:

- Dataset labels có audit rõ và tests chống leakage.
- ML output có calibration metrics.
- Dashboard/CLI cho biết ML đang advisory, không phải decision authority.

Sprint Review:

- Demo train/calibration report.
- Kiểm tra các case ML disagree với rule engine.
- Quyết định threshold policy cho sprint sau.

Retrospective focus:

- Dữ liệu training có đủ hay chưa.
- Model family nào đáng giữ.
- Chi phí runtime training.

### Sprint 4 - Dashboard, async jobs và release readiness

Sprint goal: biến workflow research thành trải nghiệm vận hành được trên dashboard/API.

Sprint backlog:

- PB-11: nâng Research tab cho optimizer/backtest/robustness.
- PB-12: async job control cho optimizer dài.
- PB-13: provider health monitoring.
- PB-14: cập nhật docs release/runbook.

Increment:

- Dashboard chạy optimizer/backtest/robustness không block UI.
- Người dùng xem được status, warnings, report path và apply result.
- Provider health rõ ràng trước khi chạy scan hoặc refresh.

Sprint Review:

- Demo job lifecycle: create, poll, result, failed/cancelled.
- Demo dashboard summary thay vì JSON thô.
- Chốt release checklist cho bản research MVP tiếp theo.

Retrospective focus:

- Observability còn thiếu gì.
- UI có đủ để debug không.
- Sprint sau ưu tiên strategy hay ops.

### Sprint 5 - Tích hợp Chatbot LLM hỗ trợ phân tích mã

Sprint goal: Hoàn thành tích hợp chatbot LLM vào dashboard để trả lời các câu hỏi kỹ thuật về từng mã chứng khoán dựa trên data bundle.

Sprint backlog:

- [x] PB-15: Backend Chat: API endpoint `/api/chat` và prompt context compiler.
- [x] PB-16: Chatbot LLM Client Unified Factory: tích hợp chọn client & Nvidia `chat_text`.
- [x] PB-17: Frontend UI Chat Assistant Panel: khung chat Assistant trên giao diện web.

Increment:
- API `/api/chat` hoạt động ổn định, cung cấp đầy đủ thông tin kỹ thuật (indicators, rules, ML) cho LLM.
- Khung chat Assistant trên giao diện web hỗ trợ hỏi đáp theo thời gian thực về bất kỳ mã nào, có context ticker đang chọn.

Sprint Review:
- Demo hỏi đáp về các mã cổ phiếu BUY_SETUP, WATCH và REJECT trong scan gần nhất.
- Đảm bảo LLM chỉ sử dụng thông tin kỹ thuật trong prompt, không bịa đặt số liệu hay dùng tin tức bên ngoài.

Retrospective focus:
- Tốc độ phản hồi của LLM (local Ollama vs Nvidia cloud).
- Chất lượng câu trả lời và mức độ bám sát dữ liệu thực tế.

## 6. Metrics và Acceptance Gates

### Product Metrics

| Metric | Target trước khi cân nhắc apply |
|---|---|
| Final-test total return | `>= 10%` |
| Final-test win rate | `>= 60%` |
| Final-test total trades | `>= 30` |
| Final-test expectancy | `> 0` |
| Final-test max drawdown | `<= 20%` |
| Robustness Monte Carlo p50 | `> 0%` hoặc được Product Owner duyệt bằng văn bản |
| Provider freshness | Latest date không quá `max_price_age_days` |
| Test suite | Pass trước khi merge hoặc release |

### Engineering Metrics

- Backtest full 3 năm phải có runtime được ghi trong manifest.
- Optimizer timeout không được làm mất toàn bộ kết quả.
- Mỗi public API/CLI mới phải có smoke test hoặc unit test.
- Mọi report quan trọng phải có `rules_hash`, `symbols_used`, `symbols_excluded`, `created_at`.

## 7. Risks và Mitigation

| Risk | Impact | Mitigation |
|---|---|---|
| Provider instability | Refresh fail hoặc dữ liệu lệch | Provider health report, local CSV cache, excluded-symbol summary |
| Overfitting optimizer | Candidate đẹp ở validation nhưng fail OOS | Walk-forward, final-test acceptance, robustness gate, report-only registry |
| Corporate actions | Giá nhảy làm label/backtest sai | Corporate action flags, skipped rows audit, manual review list |
| Stale or short-history symbols | Backtest universe lệch | Min rows/freshness gate, explicit exclusions |
| Long optimizer runtime | Mất kết quả do timeout | Study storage, checkpoint, lower log verbosity, async jobs |
| ML override sai | Decision path mất kiểm soát | ML advisory-only, deterministic rule guard luôn thắng |
| Dashboard che giấu lỗi | Người dùng tin nhầm output | Hiển thị warnings, provider status, report path, apply result |

## 8. Working Agreements

- Technical-only remains the default: không thêm news/RSS/sentiment vào signal, optimizer, backtest, hoặc LLM decision path.
- Không apply `configs/rules_t2.json` nếu final test hoặc robustness fail.
- Mọi experiment lớn phải lưu report, không chỉ in console.
- Tài liệu và code phải dùng thuật ngữ nhất quán: baseline, validation, final test, full backtest, robustness, apply gate.
- Khi target không đạt, report phải nói rõ `target_not_reached`, không diễn giải như thành công.
