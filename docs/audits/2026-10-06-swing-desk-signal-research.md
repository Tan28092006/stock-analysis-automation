# Swing desk VN30 — tín hiệu, sentiment, kỷ luật trước lệnh và đánh giá tiến cứu — 06/10/2026

## Phạm vi / revision

- Ngày: 06/10/2026. Reviewer: Firstmate run `2026-10-06-ml-roadmap` / T10 (scout + design).
- Source revision đã đọc: `4403fc7` (branch `codex/data-integrity-audit`). Working tree có sẵn file
  untracked (2 file runtime trong `data/pipeline/`, 7 audit ngày 06/10); không sửa.
- Loại tài liệu: **nghiên cứu + thiết kế**. Không viết code production, không đổi lịch, không đổi rule
  MR/momentum, không đặt lệnh. Builder spec tách riêng:
  `.firstmate-native/runs/2026-10-06-ml-roadmap/reports/T10-swing-desk-design.md`.
- Phương pháp: đọc code/data local (chỉ đọc), hai script mô tả chạy trong scratchpad trên dữ liệu VN30
  (không phải backtest chiến lược, không chấp thuận gì), khảo sát tài liệu theo kỷ luật trích dẫn của
  skill `deep-research`: mỗi nguồn được **mở** (WebFetch/đọc PDF) bởi tôi hoặc bởi 5 scout subagent
  chạy song song; mục nào chỉ thấy trong snippet tìm kiếm ghi **UNVERIFIED** và không dùng làm căn cứ.
  Bảng nguồn ở cuối ghi rõ grade và người mở.
- Bối cảnh người dùng (coordinator chuyển lại, 06/10): người dùng **thường mua theo cảm xúc**, không
  phân tích tín hiệu, không có bộ lọc cảnh báo. Vì vậy **nhiệm vụ chính của desk là kỷ luật trước lệnh
  (pre-trade discipline filter)**; chọn mã là việc thứ hai.

## Kết luận ngắn

1. Mục tiêu "lướt sóng, stop 3%, rủi ro vừa, ROI cao, xác suất thắng cao mỗi lệnh" **mâu thuẫn nội tại**
   trên VN30 với T+2.5, biên độ ±7% và phí ≈0,4–0,6%. Với dữ liệu VN30 2018–2026: ATR14 trung vị 2,6%
   nên stop 3% ≈ 1,1×ATR (vùng nhiễu); 22% lệnh mua ngẫu nhiên đã xuyên −3% **trước khi được phép bán**;
   lệnh dừng lỗ thực tế trung bình −3,9% chứ không phải −3%; nới target thì tỷ lệ thắng giảm từ ~51% xuống
   ~40%. Chi tiết §1.
2. Bằng chứng trong repo không ủng hộ "ML/LLM nâng tỷ lệ thắng": R5 weekly không có ranking skill (đối
   chứng shuffled-label thắng model); P(win) dự báo 55–79% nhưng thực tế 10% trong đợt 07/2026. Văn liệu
   LLM-trading: khung multi-agent được thử 3–6 tháng trên 3–5 mã Mỹ; benchmark sạch (StockBench) cho thấy
   đa số agent không thắng buy-and-hold; nghiên cứu sentiment tin tức Việt Nam duy nhất có kiểm định giá
   thấy phản ứng giá không có ý nghĩa thống kê. Niềm tin "LLM + VeriHarness tăng mạnh win rate" là
   **giả thuyết**; §13 thiết kế cách kiểm.
3. Thứ desk có thể làm tốt ngay: **chặn các lệnh có đặc điểm mà văn liệu hành vi gắn với hiệu quả kém**
   (mua mã tăng trần/tin nóng, đuổi gap, gỡ lỗ, bình quân giá xuống, tăng size, giao dịch quá nhiều) và
   **buộc lệnh có kế hoạch** (stop, size, thời điểm được bán). Đây là bộ lọc kỷ luật, không phải alpha.
4. BASIC GATE (§4) là deterministic, chạy offline trong vài giây: G0 dữ liệu tươi → G1 thuộc VN30 → G2 thị
   trường (VNINDEX > EMA50) → G3 mã > SMA50 → G4/G5 không đuổi giá → G6 stop so với biến động → G7 thanh
   khoản → G8 R:R ≥ 1,5 → G9 ngân sách rủi ro → G10 sự kiện → G11 có setup; cộng cờ hành vi B1–B9 (§5).
5. Sentiment (§9) chỉ được **hạ** một kết quả kỹ thuật (veto), không bao giờ nâng WARN/BLOCK lên PASS.
   Tin được chấm theo loại bằng chứng (FACT_PRIMARY → RUMOR), giọng điệu (khách quan/lạc quan/bi quan),
   tác động −2..+2, suy giảm theo thời gian, và phải trích dẫn nguồn đã mở.
6. Luồng 4 agent đề xuất: **4 analyst rollout cùng đề + các phase verifier riêng của VeriHarness**, với
   bước làm tươi dữ liệu deterministic chạy trước (phase 0) và verifier "thách thức đồng thuận" kiểm lại
   độ tươi + tính lại chỉ báo (§12). Không dùng 1 trong 4 agent làm data-verifier.

## 1. Thách thức mục tiêu (goal challenge)

### 1.1 Quy tắc thị trường (đã kiểm nguồn)

| Quy tắc | Nội dung | Nguồn (grade) |
|---|---|---|
| Thanh toán T+2, bán được chiều T+2 | Từ 29/8/2022: "nhà đầu tư có thể bán chứng khoán mua ngày T+0 ngay trong phiên giao dịch chiều ngày T+2" (QĐ 109, 110/QĐ-VSD) | [R1] VERIFIED; [R2] VERIFIED |
| Không có T0 / bán chứng khoán chờ về cho NĐT cá nhân trong nước (10/2026) | KRX vận hành 5/5/2025; VSDC 24/10/2025 nói đang nâng cấp để "bán chứng khoán chờ về, giao dịch trong ngày", triển khai dần; CCP dự kiến đầu 2027. Không tìm thấy bằng chứng đã triển khai | [R3] PARTIAL (vắng bằng chứng, không phải chứng minh) — **hỏi CTCK để xác nhận** |
| Biên độ ±7% HOSE | "Biên độ dao động giá… ± 7% (so với giá tham chiếu)"; ±20% ngày đầu niêm yết/sau tạm ngừng >25 phiên | [R4] VERIFIED (tài liệu CTCK trích quy chế HSX sau KRX) |
| Phiên, lô | ATO 9:00–9:15; liên tục 9:15–11:30 và 13:00–14:30; ATC 14:30–14:45; lô chẵn 100 cp | [R4] VERIFIED |
| Giá tham chiếu ngày GDKHQ | Giá đóng cửa điều chỉnh theo cổ tức/quyền | [R4] VERIFIED (công thức không có trong tài liệu) |
| Thuế bán | 0,1% trên giá chuyển nhượng từng lần (giữ nguyên trong Luật Thuế TNCN mới) | [R5] VERIFIED; ngày hiệu lực luật mới UNVERIFIED |
| Phí môi giới | Ví dụ Shinhan 27/3/2026: online 0,15%; tại quầy 0,15–0,3% | [R6] VERIFIED (một CTCK); repo dùng mua 0,15% + bán 0,25% (gồm thuế) = 0,4% |

### 1.2 Hệ quả đo trên dữ liệu VN30 (mô tả, không phải kiểm định chiến lược)

Script scratchpad `stop_feasibility.py`, `barrier.py` (không commit). Mẫu: 28–30 mã VN30 **hiện tại**
(survivorship: rổ hôm nay áp ngược), mỗi phiên là một lệnh mua giả định ở giá mở cửa phiên kế tiếp — tức
**không có tín hiệu**, chỉ đo cấu trúc rủi ro. Nguồn: `data/raw/prices_hist` 2018-01→2026-09-21 (57.550
lệnh) và snapshot VCI verified `20261006T100836844041Z` 2025-01→2026-10-06 (12.366 lệnh).

| Đại lượng | prices_hist 2018–2026 | snapshot 2025–2026 |
|---|---|---|
| ATR14/giá: trung vị (p25–p75) | 2,63% (2,02–3,50%) | 2,68% (2,07–3,51%) |
| Stop 3% tính theo ATR (trung vị) | 1,14×ATR | 1,12×ATR |
| Tỷ lệ phiên-mã có \|ret\| ≥ 6,5% (gần trần/sàn) | 2,94% (giảm ≥6,5%: 1,41%) | 3,72% (1,43%) |
| Tỷ lệ phiên mở cửa gap ≤ −3% | 1,40% | 1,35% |
| Lệnh có giá thấp nhất ≤ −3% **trong lúc chưa được bán** (phiên T và T+1) | **22,0%** | **22,2%** |
| …trong đó đóng cửa T+2 vẫn ≤ −3% / đã hồi > giá mua | 47,6% / 17,7% | 44,8% / 18,2% |
| Lệnh chạm stop −3% trong 10 phiên (tính từ đóng cửa T+2) | 50,4% | 49,4% |
| Giá thoát khi stop: trung bình / p10 | **−3,88% / −6,01%** | −3,83% / −5,91% |
| …thoát tại đóng cửa T+2 / gap xuyên stop lúc mở cửa / đúng giá stop | 29% / 9% / 62% | 29% / 7% / 64% |

Diễn giải:

- Stop 3% ≈ 1,1×ATR nằm trong vùng dao động thường ngày. Repo đã đo trên MR (memory
  `strategy-edge-findings`): stop 1,5×ATR "mostly pays whipsaw"; rủi ro thực được quản lý bởi time-stop +
  cổng vào lệnh, stop chỉ là bảo hiểm thảm họa. Kaminski & Lo [A26] cho thấy stop-loss chỉ cải thiện kỳ
  vọng trong một số điều kiện (tần suất lấy mẫu dài, có momentum); luận văn Lo–Remorov [A27] trên cổ phiếu
  Mỹ: "tight stop-loss strategies tend to underperform the buy-and-hold policy due to excessive trading
  costs".
- Với T+2.5, **không thể thực thi stop 3% trong hai phiên đầu**; khoảng 1/5 lệnh đã vượt ngưỡng trong lúc
  bị khóa. Khi được bán, ~29% lần dừng lỗ xảy ra ở đóng cửa T+2 với mức lỗ đã sâu hơn, ~7–9% bị gap xuyên.
  Vì vậy desk dùng **"lỗ hiệu dụng" ≈ 3% + 1 điểm % gap allowance + 0,6% phí = 4,6%** để tính size.
- Ngày giảm sàn có thể "trắng bên mua" — không khớp được lệnh bán; scorer coi phiên giảm ≥ 6,5% là không
  khớp (§13).

### 1.3 Đánh đổi tỷ lệ thắng – lợi nhuận (cùng mẫu, stop 3%, giữ tối đa 10 phiên, khóa T+2.5)

| Target | Phí | Win (net>0) | Lãi TB/lệnh | Lãi TB khi thắng | Lỗ TB khi thua |
|---|---|---|---|---|---|
| +3% | 0,4% | 50,8% | −0,18% | +3,51% | −4,00% |
| +5% | 0,4% | 44,5% | −0,11% | +4,54% | −3,84% |
| +8% | 0,4% | 41,0% | −0,03% | +5,37% | −3,78% |
| +12% | 0,4% | 39,8% | +0,04% | +5,80% | −3,76% |
| +3% | 0,6% | 50,3% | −0,38% | +3,35% | −4,16% |
| +12% | 0,6% | 38,7% | −0,16% | +5,75% | −3,90% |

Không có tín hiệu thì kỳ vọng ≈ 0 hoặc âm sau phí ở mọi target (giai đoạn 2018–2026 là thị trường tăng,
nên đây còn là trường hợp thuận lợi). Target gần → win rate cao nhưng lãi khi thắng nhỏ hơn lỗ khi thua;
target xa → ngược lại. Công thức kỳ vọng `E = win% × lãi TB − (1 − win%) × lỗ TB` [A35, practitioner].
Muốn **vừa** win rate cao **vừa** ROI cao thì tín hiệu phải có edge thật đủ lớn để bù cả hai — repo hiện chỉ
có một cấu hình như vậy ở mức mỏng: RSI(2)<5 & close>SMA50, +0,52%/lệnh ở phí 0,4%, win 64%, **với stop
thảm họa −8%**, không phải 3% (memory `swing-rsi2-edge`; survivorship còn nguyên). Đổi sang stop 3% là
một thay đổi chiến lược → phải đăng ký theo research gate trước khi xem kết quả (AGENTS.md).

### 1.4 Giả thuyết "LLM đọc nhiều văn bản tài chính + VeriHarness sẽ tăng mạnh win rate"

Bằng chứng hiện có (chi tiết ở §15):

- Trong repo: R5 weekly — rank IC ≈ 0, đối chứng xáo nhãn cho excess +0,45%/tuần, cao hơn model
  (`docs/audits/2026-10-04-r5-weekly-results.md`). P(win) meta-label: AUC gộp OOS 0,496; đợt 07/2026 dự báo
  55–79%, thực tế 10% (memory `crash-response-by-shape`).
- LLM trên tin tức: Lopez-Lira & Tang [S7] — ~90% hit rate là cho **phản ứng ban đầu không giao dịch được**;
  "Strategy returns decline as LLM adoption rises". Glasserman & Lin [S12]: backtest LLM bị lệch nếu trùng
  giai đoạn huấn luyện; hiệu ứng "distraction" mạnh nhất ở **công ty lớn** (VN30 toàn công ty lớn).
  Lopez-Lira, Tang & Zhu [S14]: "LLMs cannot be trusted for economic forecasts during periods covered by
  their training data… masking fails". Chen et al. [S23]: LLM ngoại suy xu hướng trong khi lợi suất cổ phiếu
  đơn lẻ có xu hướng đảo chiều, khoảng tin cậy quá hẹp.
- Agent giao dịch: TradingAgents [S18] thử 3 tháng trên 3 mã Mỹ, Sharpe 5,6–8,2 mà chính tác giả nói vượt
  vùng kỳ vọng; FinMem [S19] chọn cấu hình theo kết quả giai đoạn test; StockBench [S20] (cửa sổ không
  nhiễm dữ liệu, 82 phiên): "most models struggle to outperform the simple buy-and-hold baseline".
  Harris 2026 [S22]: agent tốt nhất khớp nhãn chuyên gia "only 52.4%" về tin nào làm giá chạy.
- Việt Nam: Vu et al. 2023 [S32] (PhoBERT, ~40 nghìn bài): "insignificant difference in the stock price as
  a response to the news", chỉ phương sai thay đổi với tin tiêu cực.
- VeriHarness: bằng chứng của bài báo là trên benchmark tác vụ văn phòng/bảng tính/code có bộ chấm (skill
  `veriharness`, mục "Phạm vi bằng chứng"). Nó giảm lỗi **kiểm chứng được** (số liệu sai, trích dẫn bịa, vi
  phạm luật lệnh), không có bằng chứng làm tăng độ chính xác dự báo giá.

Kết luận: desk dùng LLM để **thu thập và kiểm chứng thông tin, áp luật nhất quán, ghi nhật ký**, còn câu hỏi
"có tăng win rate không" được trả lời bằng §13, không bằng niềm tin.

## 2. Dữ liệu có sẵn (kiểm 06/10/2026)

| Nguồn | Đường dẫn | Nội dung / độ tươi | Dùng cho desk |
|---|---|---|---|
| Snapshot VCI verified | `data/paper/snapshots/<UTC>/` | 30 mã VN30 + VNINDEX, OHLCV từ 2024-12-25 (~440 phiên), giá đã điều chỉnh của provider; mới nhất `as_of` 2026-10-06 | Nguồn giá chính |
| Foreign/tự doanh v2 | `data/foreign_flows_v2/runs/*/records.jsonl` | 105 mã × ~20 phiên × {foreign, proprietary}, có `available_at`; run mới nhất 06/10 `ready`; lịch sử chỉ từ ~09/09/2026 | Chỉ báo dòng tiền, gắn cờ thiếu lịch sử |
| Giá legacy | `data/raw/prices_hist` | VN100 2017→**21/09/2026 (đứng yên)** | Không dùng cho desk live; chỉ hiệu chỉnh offline có nhãn |
| Foreign legacy | `data/raw/foreign/*` | Không provenance | **Cách ly** theo AGENTS.md |
| Universe | `configs/universe_vn30.json` | Rổ hiệu lực 03/08/2026 (MCH, TCX thay PLX, TPB) | G1 |
| Lịch | `stock_agent/data/exchange_calendar.py` | Ngày nghỉ đến hết 2026 | T+2, phiên; cảnh báo khi vượt 2026 |

Không có trong DB local (agent phải tìm trên web và trích dẫn): lịch GDKHQ/cổ tức/phát hành, ĐHĐCĐ, ngày
công bố KQKD, giao dịch nội bộ, tin doanh nghiệp, chính sách, vĩ mô, giá realtime.

## 3. Chế độ pre-trade check (nhiệm vụ chính)

Người dùng gõ mã muốn mua ngay (kèm giá đang thấy và số tiền định mua, nếu có). Desk trả về trong vài
giây, **không cần LLM**:

```text
HPG — WARN   (dữ liệu EOD tới 06/10/2026, snapshot sha 3f…)
Luật không đạt: G8 R:R 1,2 < 1,5 (kháng cự high_20 cách +4,1%); B2 giá nhập +3,4% so với đóng cửa hôm qua
Setup: không có (G11) — bạn đang mua không có tín hiệu
Bán được sớm nhất: chiều 09/10/2026 (T+2). Stop 3% KHÔNG thực thi được trước thời điểm này.
Stop 3%: 26.530 | stop đã kiểm định (RSI2): 25.160 (−8%)
Size gợi ý (rủi ro 1% NAV, lỗ hiệu dụng 4,6%): 7.900 cp ≈ 216 tr
Số tiền bạn nhập 300 tr: rủi ro danh nghĩa 9,0 tr (0,9% NAV); hiệu dụng 13,8 tr (1,38% NAV) > ngân sách
Cần: viết 1 dòng THESIS (setup, stop, target, điều kiện sai) trước khi xác nhận; hoặc chờ phiên sau.
```

PASS nghĩa là "không có cờ cấu trúc/hành vi và có setup đã biết", **không** có nghĩa "sẽ thắng". BLOCK
là khuyến nghị dứt khoát; desk không thể và không được chặn lệnh thật ở CTCK.

## 4. BASIC GATE (luôn phải chạy; hard → BLOCK, soft → WARN, thiếu dữ liệu → UNVERIFIED = WARN)

Ký hiệu: C, O, H, L, V = đóng, mở, cao, thấp, khối lượng; t = phiên hoàn tất cuối cùng.

| Id | Luật & công thức | Tham số | Lý do | Nguồn | Data field |
|---|---|---|---|---|---|
| G0 | Snapshot `status=verified`, `as_of = completed_session_date(now)`, sha256 khớp, đủ 31 file | — | Không phân tích trên dữ liệu cũ/hỏng; T7 cho thấy `prices_hist` đứng từ 21/09 | `docs/audits/2026-10-06-data-update-flow.md`; `paper_runner.assess_market_data` | `manifest.json` |
| G1 | Mã ∈ rổ VN30 hiện hành | universe file | Phạm vi; thanh khoản | `configs/universe_vn30.json` | symbols |
| G2 | VNINDEX C > EMA50 (regime ≠ PANIC_BEAR); GRIND = soft | EMA 50 | Bộ lọc "chống đu đỉnh" rẻ nhất trong repo (2022: momentum −10,5% → +8,9%); momentum crash xảy ra ở trạng thái "panic" sau khi thị trường giảm | memory `strategy-edge-findings`; Daniel & Moskowitz [A6] | `market.json.regime` |
| G3 | C > SMA50 của mã | SMA 50 | Bộ lọc giúp RSI(2) dương mọi năm 2018–2026 và đứng ngoài downtrend; bắt đáy bằng MR là "coin flip" | memory `swing-rsi2-edge`, `crash-response-by-shape`; Connors [A17–A19] | `sma50` |
| G4 | Không đuổi giá: ret_1d < +6,5% (hard); ret_1d ≥ +4% hoặc ret_3d ≥ +8% hoặc RSI2 ≥ 90 (soft) | 6,5/4/8%, 90 | Ở T+2 các đặc trưng breakout/ret_5d có IC **âm** trong repo; mã chạm trần thu hút NĐT chú ý rồi đảo chiều trong tuần sau | memory `strategy-edge-findings`; Seasholes & Wu [B7]; Barber & Odean 2008 [B3] | `ret_1d`, `ret_3d`, `rsi2` |
| G5 | Đuổi trong phiên (khi người dùng nhập giá): P/C_t − 1 ≥ +5% hard, +3..5% soft | 3/5% | Như G4; tránh mua đúng lúc tin nóng | như G4 | `--price`, `close` |
| G6 | Stop so với nhiễu: ATR% ≤ 3% pass; 3–4% soft; > 4% hard; soft nếu MAE khóa p80 < −3% | 3/4% | Stop 3% < 1×ATR là vùng nhiễu; MAE cho biết lệnh "thường" đi ngược bao xa trước khi được bán | §1.2; Sweeney [A21]; Wilder ATR [A12] | `atr_pct`, `locked_mae_p80` |
| G7 | Thanh khoản: GTGD TB 20 phiên ≥ 20 tỷ và lệnh ≤ 5% GTGD TB | 20 tỷ, 5% | Tránh trượt giá; VN30 thường đạt | thiết kế | `avg_value_20_bn` |
| G8 | R:R: (target − entry − phí) ≥ 1,5 × (entry − stop + phí); target = high_20 (hoặc SMA20 cho setup MR) | 1,5 | Đòn bẩy duy nhất còn sống sót qua 7 đợt sập là RR ≥ 1,5 | memory `crash-response-by-shape`; §1.3 | `dist_high_20_pct`, `sma20` |
| G9 | Ngân sách rủi ro: lỗ ở stop hiệu dụng ≤ 1% NAV; tổng rủi ro mở ≤ 4% NAV; ≤ 4 vị thế; tỷ trọng ≤ 25%; ≤ 2 mã ngân hàng | 1%/4%/4/25%/2 | §8; VN30 nặng ngân hàng (~37% phần cổ phiếu của ETF VN30, PARTIAL) | §8; [M3] | journal + `size` |
| G10 | Sự kiện: không có GDKHQ/phát hành/KQKD/ĐHĐCĐ trong 3 phiên tới; không phải ngày đáo hạn VN30F; lịch không rõ → UNVERIFIED | 3 phiên | Giá tham chiếu điều chỉnh ngày GDKHQ làm stop giả bị kích hoạt; ngày đáo hạn có biến động ATC | [R4]; [R7]; §10 | agent cung cấp + `calendar_events` |
| G11 | Có setup S1 hoặc S2 (soft) | — | Vấn đề gốc của người dùng là mua không có tín hiệu | — | `setups` |

Setup (nhãn thông tin, không phải lệnh):

- **S1 — swing RSI(2)**: RSI(2) < 5 và C > SMA50; thoát khi C > SMA5 hoặc sau 8 phiên. Cấu hình duy nhất
  dương sau phí trong repo (+0,52%/lệnh @0,4%, win 64%, 2018–2026, với stop −8%; edge mỏng, nhạy phí,
  survivorship). Văn liệu nền: đảo chiều ngắn hạn Jegadeesh [A1], Lehmann [A2]; trên HOSE, Nguyễn Thị Yến
  [A30] thấy lợi suất 4 tuần có xu hướng đảo chiều tuần kế tiếp. Lưu ý: Avramov et al. [A3] cho rằng lợi
  nhuận contrarian "smaller than the likely transactions costs" — mâu thuẫn với Lehmann; repo đo thấy edge
  chỉ còn +0,3%/lệnh ở phí 0,6%.
- **S2 — VN30 MR gate** (`mr_scan`, RSI14<35, dải ×1,02, nến đảo chiều + volume): hiếm (~vài lệnh/năm/rổ),
  win 58%, +1,08%/lệnh trong development data.

## 5. Cờ hành vi (behavioural red flags)

| Id | Cờ | Phát hiện | Mức | Căn cứ |
|---|---|---|---|---|
| B1 | FOMO / mua mã tăng trần | ret_1d ≥ +5% hoặc chạm trần ở t hoặc t−1, hoặc giá nhập ≥ trần − 1% | hard | Seasholes & Wu [B7]: mã chạm trần trên, có tin và volume cao, thu hút NĐT cá nhân mua mã chưa từng sở hữu, sau đó "statistically significant price mean reversion over the following week". Barber & Odean 2008 [B3]: NĐT cá nhân là người mua ròng các mã "with extreme one-day returns" |
| B2 | Đuổi gap | giá nhập ≥ C_t × 1,03, hoặc ≥ giá tham chiếu setup × 1,015 | soft | như B1; IC âm của breakout ở T+2 trong repo |
| B3 | Gỡ lỗ (revenge) | có lệnh lỗ đã đóng trong 2 phiên gần nhất; ≥ 2 lệnh lỗ liên tiếp → hard đến phiên sau | soft/hard | Coval & Shumway [B5]: trader lỗ buổi sáng chấp nhận rủi ro cao hơn buổi chiều để gỡ |
| B4 | Bình quân giá xuống | mua thêm mã đang lỗ so với giá vốn | hard, trừ khi kế hoạch scale-in đã ghi trước lệnh đầu | Odean 1998 [B2] mục IV: mua thêm mã đang lỗ nhiều hơn mã đang lãi (0,135 so với 0,094, t = 19), gắn với risk seeking khi lỗ [B12]; repo: scale-in vào đợt sập chưa hồi vẫn −6% (memory `swing-rsi2-edge`) |
| B5 | Tăng size dần | rủi ro yêu cầu > 1,25 × trung vị 10 lệnh gần nhất, hoặc tỷ trọng > 25% | soft (vượt trần → G9 hard) | Overconfidence → giao dịch quá mức, hiệu quả kém [B1, B9]; house money: chấp nhận rủi ro hơn sau khi lãi [B11] |
| B6 | Hype diễn đàn/tin | lý do ∈ {forum, tip, kol, group, news} mà không có setup; hoặc vol_ratio_20 ≥ 3 và ret_1d ≥ +4% | soft | Barber et al. 2022 [B4]: mua dồn dập do chú ý dự báo lợi suất âm, "−4.7%" 20 ngày cho các mã được mua nhiều nhất; Barber & Odean 2008 [B3]; herding trên HOSE 2005–2015 [B17], tăng sau COVID [B18] |
| B7 | Giao dịch quá nhiều | > 3 lệnh mới trong 5 phiên hoặc > 8 trong 21 phiên | soft | Barber & Odean 2000 [B1]: nhóm giao dịch nhiều nhất lãi 11,4%/năm so với thị trường 17,9%; Odean 1999 [B8]: mã mua vào thua mã bán ra 3,3%/năm; Đài Loan: "Less than 1%" day trader có lãi bất thường ròng ổn định [B10] |
| B8 | Không tôn trọng stop (chỉ khi review) | đóng lệnh ≥ 1 phiên sau khi C ≤ stop trong lúc đã bán được | review | Odean 1998 [B2] |
| B9 | Hiệu ứng disposition (review) | trung vị ngày giữ lệnh lỗ > 1,5 × lệnh lãi, ≥ 10 lệnh đã đóng | review | Shefrin & Statman 1985 [B6] (META); Odean 1998 [B2]: hiện thực hóa 0,148 lãi so với 0,098 lỗ; VN: 1.201 tài khoản giữ lỗ, bán lãi [B19]; 27.591 lệnh, mạnh hơn ở tài khoản giao dịch nhiều [B20] |

Cooling-off (tùy chọn, người dùng quyết định bật):

- WARN → phải viết một dòng `THESIS` (setup, stop, target, điều kiện sai) vào `portfolio.txt` trước khi
  xác nhận.
- BLOCK do cờ hành vi → chỉ được kiểm lại từ ATO phiên sau.
- Sau 2 lần dừng lỗ liên tiếp → nghỉ 1 phiên.

Căn cứ của các luật này nằm ngoài tài chính và nói về **hành vi**, không về lợi nhuận:

- Người ta tự đặt deadline tốn kém để kiểm soát bản thân, nhưng đặt không tối ưu (Ariely & Wertenbroch
  [B13]); vì vậy luật nên do desk đặt sẵn.
- Thời gian chờ bắt buộc giảm hành vi bốc đồng, ví dụ súng đạn giảm ~17% (Luca et al. [B14], causal, ngoài
  tài chính).
- Checklist giảm sai sót trong phẫu thuật (Haynes et al. [B15], before–after, qua báo cáo).
- Nhật ký giao dịch là công cụ tự huấn luyện (Steenbarger [B16], practitioner).

Không có nghiên cứu nào chứng minh cooling-off làm tăng lợi nhuận giao dịch; H4 ở §13 sẽ đo. Desk chỉ ghi
nhận, không thể chặn lệnh ở CTCK.

Về giả định "mua mã tăng trần thì đảo chiều" ở VN: bằng chứng trực tiếp là Seasholes & Wu (Thượng Hải)
[B7]. Ở Việt Nam chỉ có:

- Pham, Nguyen & Tô (2007) [B21]: 2001–2005, giá đảo chiều ~3 ngày sau biến động một ngày lớn, nhưng lợi
  nhuận không bù được phí.
- Nghiên cứu cũ về tác động của biên độ giá [B22].

Không tìm thấy nghiên cứu magnet-effect ở giá trần HOSE. B1 vì vậy là luật **phòng thủ hành vi**: ở horizon
T+2.5, mua sau ngày trần là mua đúng lúc NĐT cá nhân bị thu hút.

## 6. ADVANCED signals (chú thích để kết luận sâu hơn; không bao giờ biến WARN thành PASS)

| Id | Tín hiệu & công thức | Tham số | Lý do / hướng | Nguồn | Data |
|---|---|---|---|---|---|
| A1 | Volume bất thường: vol_ratio_20 = V_t/TB20(V); z-score | ≥ 2 | Khối lượng cao bất thường → tăng giá tháng sau (visibility) [A7]; winners volume cao đảo chiều nhanh hơn [A8]; giao dịch chia sẻ rủi ro đảo chiều, giao dịch thông tin tiếp diễn [A9]. Trên HOSE, Duong & Bertrand [A31]: momentum đến từ overreaction đo bằng volume | [A7–A9], [A31] | snapshot |
| A2 | VSA stopping volume / no-supply | như `add_vsa_features` | Xác nhận thay thế cho nến đảo chiều trong MR hybrid (repo: tăng win rate) | memory `strategy-edge-findings` | snapshot |
| A3 | Kijun (Ichimoku) làm target hồi | 26 | Target Kijun +27% full-sample so với EMA21 trong MR hybrid (development data) | memory | snapshot |
| A4 | Foreign net 5/20 phiên, % GTGD | — | Mua ròng NN tác động dương lên thanh khoản, âm lên biến động **ngắn hạn**, không có quan hệ dài hạn (VN30 2012–2022) [M5]; lan tỏa NN → tự doanh "very low" [M6]. Lịch sử v2 mới ~20 phiên → INSUFFICIENT_HISTORY | [M5], [M6] | flows v2 |
| A5 | Tự doanh net 5 phiên | — | HOSE công bố tự doanh cuối ngày từ 17/05/2022 [R8]; chưa có bằng chứng dự báo | [R8], [M6] | flows v2 |
| A6 | Khoảng cách tới đỉnh 52 tuần | 250 phiên | Gần đỉnh 52 tuần liên quan momentum trung hạn (George & Hwang [A16], PAYWALL — chỉ dùng như ngữ cảnh) — không hợp horizon 2–10 phiên | [A16] | snapshot |
| A7 | Bối cảnh đảo chiều vs momentum theo horizon | — | Đảo chiều tuần [A1, A2]; momentum 3–12 tháng [A5]; trong repo IC đổi dấu theo horizon (T+2 MR, T+10 momentum) | [A1–A5]; memory | snapshot |
| A8 | Regime biến động: ATR% VNINDEX, ATR% mã | — | Giảm rủi ro khi biến động cao [A11]; mã biến động riêng cao có lợi suất thấp [A10]; lợi nhuận đảo chiều (cung cấp thanh khoản) tăng khi thị trường hỗn loạn [A4] | [A4], [A10], [A11] | snapshot |
| A9 | Breadth: % VN30 trên SMA20 | 40% | Dùng làm bối cảnh regime; Brown & Cliff: ít bằng chứng sentiment dự báo ngắn hạn (PARTIAL) [M9]; định nghĩa [M8] | [M8], [M9] | snapshot |
| A10 | Phân phối MAE/MFE của mã và của chính setup | 250 phiên | Đặt stop dựa trên excursion đo được thay cho cảm giác [A21–A23] | [A21–A23] | snapshot + reclog |
| A11 | Tin có/không có: biến động mạnh không kèm tin dễ đảo chiều; tin xấu cơ bản trôi tiếp | — | Chan 2003 [S4] (PARTIAL: chỉ citation + snippet); Tetlock et al. 2008 [S2] giá phản ứng thiếu với từ tiêu cực; Tetlock 2011 [S29] tin cũ → đảo chiều tuần sau, mạnh ở mã nhiều NĐT cá nhân | [S2], [S4], [S29] | agent |
| A12 | Giao dịch nội bộ (mua) | — | Mua nội bộ "opportunistic" có giá trị thông tin; bán nội bộ hầu như không dự báo [S25, S26] | [S25], [S26] | agent (công bố trước ≥ 3 ngày làm việc [R9]) |

Lưu ý data snooping: Sullivan, Timmermann & White [A14] thử ~8.000 tham số rule kỹ thuật; ngoài mẫu
1987–1996 "scant evidence that technical trading rules were of any economic value". Mỗi tín hiệu advanced
chỉ được nâng thành rule sau khi qua research gate.

## 7. Vào lệnh / thoát lệnh / stop phù hợp T+2.5 và ±7%

- **Thời điểm quyết định**: sau EOD (≥ 16:00 VN, khi `completed_session_date` = phiên vừa xong). Khuyến nghị
  phải đóng băng trước 09:00 phiên kế tiếp.
- **Vào lệnh**: lệnh giới hạn ≤ entry_ref × 1,005 trong phiên liên tục, hoặc ATO; không mua nếu giá mở cửa >
  C_t × 1,02 (B2). Setup S1 là mua dip, nên đuổi giá là trái setup.
- **Khóa T+2.5**: không có hành động trong phiên T, T+1 và sáng T+2. Size phải chịu được kịch bản xấu
  (§1.2: p10 khi dừng lỗ ≈ −6%).
- **Stop**: hai lựa chọn, desk in cả hai.
  - (a) Stop người dùng 3%, đánh giá theo giá đóng cửa từ chiều T+2. Thoát phiên kế tiếp nếu C ≤ stop.
  - (b) Stop thảm họa −8% đã dùng trong kiểm định RSI(2); size nhỏ hơn tương ứng.
  - Lệnh dừng điều kiện của CTCK (nếu có) khớp theo giá thị trường, có thể gap xuyên [A29]. Tính năng lệnh
    điều kiện cụ thể của từng CTCK: UNVERIFIED.
- **Thoát có lãi**:
  - S1: C > SMA5, rồi bán phiên kế tiếp.
  - S2: target Kijun/EMA21.
  - Chung: target = high_20 nếu R:R ≥ 1,5.
  - Time-stop 8 phiên.
  - Trong repo, trailing/partial **không** cải thiện (memory `portfolio-power-verdict`), nên giữ exit cố định.
- **Ngày GDKHQ**: giá tham chiếu bị điều chỉnh [R4]. Stop/target tính trên giá chưa điều chỉnh phải trừ
  giá trị cổ tức/quyền, nếu không sẽ kích hoạt giả. Snapshot VCI là giá đã điều chỉnh, có thể khác giá bảng
  điện: desk ghi rõ.
- **Giảm sàn**: có thể không bán được ("trắng bên mua"). Size là phòng thủ duy nhất.

## 8. Position sizing cho "rủi ro vừa"

- **Rủi ro mỗi lệnh**: 1% NAV, tính trên **lỗ hiệu dụng**, không phải lỗ danh nghĩa 3%.
  - Repo MR dùng 1,5%. "Vừa" là lựa chọn của người dùng: 0,75 / 1 / 1,5%.
  - Rủi ro theo R-multiple và expectancy theo Tharp [A24, A25].
  - Công thức: `qty = floor(1% × NAV / (entry × 4,6%) / 100) × 100`, rồi áp trần 25% NAV và trần exposure 60%.
  - Tái dùng `position_manager.suggest_size`.
- **Ví dụ** NAV 1 tỷ:
  - Rủi ro 10 tr.
  - Vị thế tối đa ≈ 217 tr với stop 3% (lỗ hiệu dụng 4,6%), hoặc ≈ 116 tr với stop 8% (lỗ hiệu dụng 8,6% + gap).
- **Danh mục**:
  - ≤ 4 vị thế.
  - Tổng rủi ro mở ≤ 4% NAV.
  - ≤ 2 mã ngân hàng.
  - ≤ 2 mã họ Vingroup (VIC, VHM, VRE, VPL ~17,7% phần cổ phiếu ETF VN30, PARTIAL [M3]).
- **Giới hạn lỗ tháng** (tùy chọn): lỗ thực hiện trong tháng ≥ 4% NAV thì dừng mở mới đến hết tháng.
  - Đây là luật hành vi, không có bằng chứng alpha.
  - Repo đã bác bỏ circuit-breaker theo chỉ số cho momentum vì whipsaw; luật này áp lên P&L cá nhân, mục đích khác.

## 9. Khung sentiment cho agent

### 9.1 Loại tin và nguồn

| Loại | Ví dụ | Nguồn ưu tiên |
|---|---|---|
| MACRO | Lãi suất điều hành, tỷ giá, tín dụng, FDI, Fed | SBV, Tạp chí Thị trường Tài chính Tiền tệ, VnExpress/VnEconomy/Reuters |
| POLICY | Luật, nghị định, thông tư; thuế; biện pháp phòng vệ thương mại | Văn bản gốc (vbpl, Bộ ngành), báo Bộ Tài chính |
| SECTOR | Tín dụng/NIM ngân hàng; giá thép, giá dầu; tiêu dùng | Báo uy tín + báo cáo CTCK (sell-side opinion) |
| COMPANY_MAJOR | KQKD, phát hành/cổ tức, M&A, thay đổi lãnh đạo, kiện tụng, rổ VN30 | CBTT HOSE/website DN; lịch sự kiện Vietstock/CafeF [M10] |
| COMPANY_MINOR | Hợp đồng nhỏ, giao dịch nội bộ, tin vận hành | CBTT, Vietstock/CafeF |
| MARKET_FLOW | NN/tự doanh, margin, ETF, FTSE/MSCI | flows v2 local + báo |
| FORUM | F247, nhóm Zalo/Facebook | **Chỉ là ý kiến người chơi, không phải bằng chứng**; dùng để phát hiện hype (B6) |

### 9.2 Rubric chấm từng tin

- `evidence_type`, kèm trọng số:
  - FACT_PRIMARY = 1,0: văn bản gốc, CBTT, số liệu thống kê.
  - FACT_REPORTED = 0,7: báo uy tín đưa tin sự kiện.
  - OPINION = 0,3: CTCK, chuyên gia, bình luận.
  - RUMOR = 0: forum, mạng xã hội, chưa xác nhận. Chỉ ghi nhận để gắn cờ hype.
- `tone`: lạc quan / **khách quan** / **bi quan** của văn bản. Tách khỏi tác động: Tetlock 2007 [S1] cho thấy
  bi quan trên truyền thông dự báo áp lực giảm tạm thời rồi **hồi về cơ bản**, tức tone ≠ thông tin.
- `impact` ∈ {−2, −1, 0, +1, +2}: tác động lên **mã này trong 2–10 phiên**, chấm sau khi viết lại sự kiện
  bằng lời trung tính. Glasserman & Lin [S12] khuyên ẩn tên công ty để giảm "distraction effect". Áp dụng
  tùy chọn: chấm hai lần, có và không có tên, và ghi chênh lệch.
- `novelty`: tin mới hoặc tin đăng lại. Tin đăng lại có trọng số 0; theo Tetlock 2011 [S29], lợi suất ngày tin
  cũ dự báo đảo chiều tuần sau.
- Suy giảm theo thời gian: `w = 0,5^(tuổi_phiên / h)`. Đây là tham số thiết kế, không phải ước lượng:
  - h = 1 phiên cho COMPANY_MINOR và bình luận thị trường. Heston & Sinha [S30]: tin theo ngày dự báo 1–2 ngày.
  - h = 5 cho COMPANY_MAJOR.
  - h = 10–20 cho POLICY/MACRO.
- Điểm mã: `score = clip(Σ w × trọng_số × impact, −3, +3)`. Gán nhãn:
  - ≤ −1,5: tiêu cực mạnh.
  - −1,5..−0,5: tiêu cực.
  - ±0,5: trung tính.
  - … tương tự cho chiều dương.
- Luật dùng điểm:
  - Có FACT tiêu cực (impact ≤ −1, trọng số ≥ 0,7) → **veto/WARN**. Lý do: tin xấu cơ bản trôi tiếp
    [S2, S4 PARTIAL].
  - Chỉ có tone bi quan hoặc OPINION tiêu cực, không có FACT → chỉ ghi chú. Có thể chính là điều kiện cho
    setup đảo chiều S1 [S1].
  - Điểm dương lớn + ret_1d ≥ 4% → cờ hype B6.
  - Điểm dương **không bao giờ** nâng WARN/BLOCK lên PASS: bằng chứng LLM-sentiment cho VN large-cap yếu
    [S12, S14, S32].

### 9.3 Cách agent trích dẫn

Mỗi tin là một dòng `S-xx`:

- URL, nhà xuất bản, thời điểm đăng (giờ VN), thời điểm truy cập.
- Câu trích nguyên văn ≤ 2 câu.
- Các trường của rubric 9.2.
- `opened: yes/no`. Không mở được → UNVERIFIED, trọng số 0.

Cấm:

- Bịa tin.
- Dùng trí nhớ của model thay cho nguồn. Lý do: memorization [S14]; chỉ tin **sau** ngày cutoff của model mới
  không bị nhiễm.
- Trích nguồn mà không có thời điểm đăng.

Verifier "thách thức đồng thuận" mở lại mọi URL làm căn cứ cho một khuyến nghị.

### 9.4 Bản đồ vĩ mô/chính sách → nhóm ngành VN30 (trạng thái kiểm 06/10/2026)

| Yếu tố | Trạng thái gần nhất (nguồn) | Nhóm chịu tác động | Ghi chú |
|---|---|---|---|
| Lãi suất điều hành SBV | Tái cấp vốn 4,5%, tái chiết khấu 3,0% từ 19/6/2023; giữ nguyên tới ít nhất 5/2026 [M1a, M1b]; lãi huy động tăng lại đầu 2026 [M1b] | Ngân hàng (NIM), BĐS, CK | Chưa mở nguồn nào cho 6–10/2026 |
| Tăng trưởng tín dụng | Mục tiêu 2026: 15%; 2025: 19,1%; tín dụng BĐS không vượt mức chung [M1c] | Ngân hàng, BĐS | — |
| Tỷ giá | Tỷ giá trung tâm kỷ lục 25.306 (28/7/2026), biên ±5% [M1d] | NN bán ròng; DN nợ USD; xuất khẩu | Văn liệu không thống nhất về tác động FX [M2a–c] |
| Lãi suất → VN-Index | Lãi suất tăng tác động âm (NARDL; ARDL); Fed funds tác động tắt dần ~5–6 tháng [M2a–c] | Toàn thị trường | Kết quả tháng/dài hạn, không phải horizon swing |
| Thuế VAT giảm 2% | 1/7/2025 → **31/12/2026** [M4f] | Bán lẻ MWG, MSN | Sự kiện hết hạn trong cửa sổ swing cuối năm |
| Giá xăng dầu điều hành | Chu kỳ 7 ngày, thứ Năm [M4e] | PLX/GAS/BSR, VJC | — |
| Phòng vệ thép HRC | QĐ 612/QĐ-BCT 02/4/2026 chống lẩn tránh HRC TQ [M4d] | HPG | — |
| Brent / Trung Đông | Q2/2026 Brent ~97 USD; BSR hưởng crack spread [M4e] | BSR, GAS, PLX, VJC | Tin báo dẫn dự báo (PARTIAL) |
| Margin | ~435 nghìn tỷ cuối Q2/2026, kỷ lục; margin/free-float 14,3% [M7a, M7b] | CK (SSI), toàn thị trường | 22/07/2026 VN-Index −62 điểm với call margin lan rộng [M7c PARTIAL] |
| Dòng NN | 2025 bán ròng kỷ lục 135.329 tỷ [M5c] | VIC, VHM, FPT bị bán nhiều | — |
| FTSE nâng hạng | Hiệu lực 21/09/2026, phân kỳ 10% (9/2026) → 20% (3/2027) → 35% (6/2027) → 35% (9/2027) [R10] | Mã lớn thuộc FTSE index | MSCI chưa đưa vào watchlist 6/2026 [R10] |
| Đầu tư công | Giải ngân 51,8% kế hoạch tới 03/09/2026 [M4i PARTIAL] | Thép, xây dựng | Số tuyệt đối không nhất quán trong nguồn |

## 10. Yếu tố khác ảnh hưởng mua/bán

- **Đáo hạn VN30F**: thứ Năm thứ ba của tháng đáo hạn. Nếu trùng ngày nghỉ thì lùi về ngày giao dịch trước
  đó. Giá thanh toán cuối cùng là trung bình 30 phút cuối, loại 3 giá trị cao nhất và 3 thấp nhất, không phải
  giá đóng cửa [R7]. Bằng chứng biến động ngày đáo hạn chỉ là tin báo → G10 soft, không phải tín hiệu.
- **GDKHQ, cổ tức, phát hành thêm**: tham chiếu điều chỉnh [R4]. Phát hành thêm gắn với lợi suất dài hạn
  kém (Loughran & Ritter [S27]: dài hạn, không phải phản ứng ngày công bố). Cổ tức: tháng chia cổ tức có
  lợi suất bất thường dương, tăng trước ngày GDKHQ rồi đảo chiều (Hartzmark & Solomon [S28]).
- **Dòng tiền NN/tự doanh**: xem A4/A5. Room NN: Nghị định 155/2020 không còn mặc định 49% cho mọi ngành
  (PARTIAL [R11]).
- **Margin**: tỷ lệ ký quỹ ban đầu ≥ 50%, duy trì ≥ 30% theo QĐ 87/QĐ-UBCK (2017). Chưa xác nhận còn hiệu
  lực 2026 (PARTIAL [R12]). Margin kỷ lục làm tăng rủi ro bán giải chấp dây chuyền; dùng làm bối cảnh regime.
- **Regime VNINDEX**: RISK_ON / GRIND / PANIC_BEAR (`mr_scan._market_state`). Banner phản ứng nhanh với
  breadth, nhưng có lỗ hổng với **flash crash trong bull** (01/2021, 07/2021). Exit vị thế không có tầng phản
  ứng nhanh (memory `crash-response-by-shape`).
- **Cơ cấu VN30**:
  - Xét định kỳ bán niên, tháng 1 và tháng 7; trần tỷ trọng 10%; công bố thay đổi thứ Tư tuần thứ ba
    tháng 1/7 [R13].
  - Cập nhật free-float theo quý; trần nhóm ngành 40% (rulebook v4.0, PARTIAL).
  - Đợt 7/2026: MCH, TCX thay PLX, TPB, hiệu lực 3/8. ETF VN30 cơ cấu 27–31/7 [R13].
  - Mã sắp ra/vào rổ chịu áp lực dòng ETF.
- **Thanh khoản**: G7. Cổ phiếu trần/sàn có thể không khớp.
- **Giao dịch nội bộ**: đăng ký trước ≥ 3 ngày làm việc, giao dịch ≤ 30 ngày, báo kết quả trong 5 ngày làm
  việc [R9]. Mua nội bộ có giá trị thông tin hơn bán [S25, S26].
- **Cơ cấu NĐT**: cá nhân chiếm ~80–85% giá trị giao dịch [M11]. Các hiệu ứng chú ý/hành vi của văn liệu
  bán lẻ là phù hợp bối cảnh VN.

## 11. Phương pháp review danh mục của người dùng

Đầu vào: `swing_desk/portfolio.txt` (định dạng trong builder spec §6) + snapshot verified.

1. **Vị thế mở**:
   - Giá vốn FIFO, lãi/lỗ chưa thực hiện (% và R so với stop ban đầu).
   - Số phiên đã giữ; đã bán được chưa (T+2.5).
   - Kết quả gate **hôm nay** của mã. C < SMA50 hoặc regime PANIC → "luận điểm hỏng".
   - Vi phạm stop (C ≤ stop → SELL), vượt time-stop, đạt target / C > SMA5 với S1.
   - GDKHQ sắp tới; dòng NN 5 phiên (nếu đủ lịch sử).
2. **Rủi ro danh mục**: tổng rủi ro mở so với 4% NAV, tỷ trọng từng mã, mức tập trung ngân hàng/Vingroup,
   exposure.
3. **Lệnh đã đóng**:
   - P&L ròng sau phí 0,4%/0,6%, win rate, lãi/lỗ TB, expectancy, R TB.
   - MAE/MFE của từng lệnh từ nến ngày (Sweeney [A21]): sau ≥ 20 lệnh, so phân phối MAE của lệnh thắng với
     stop 3% → có bằng chứng về độ rộng stop **của chính người dùng**.
4. **Theo nhãn**: kết quả nhóm theo `gate` (PASS/WARN/BLOCK/NONE) và `emotion`. Chỉ mô tả cho tới khi mỗi
   nhóm ≥ 30 lệnh (§13).
5. **Hành vi**:
   - B7: số lệnh/tháng, chi phí giao dịch/NAV/tháng.
   - B8: stop không tôn trọng.
   - B9: disposition.
   - Số lệnh mua tháng sau tháng lỗ.
6. **Hoạt động (AGENTS.md)**: số lệnh mở mới theo tháng, lệnh mua thêm, số ngày vào lệnh khác nhau, các
   tháng không có lệnh mới.
7. **Riêng tư**: không in NAV/journal thô vào chat trừ khi người dùng yêu cầu; `portfolio.txt` git-ignored.

## 12. Luồng 4 agent theo VeriHarness

### 12.1 Hai phương án

- **A. 1 data-verifier/refresher + 3 analyst.** Ưu: đúng mô tả ban đầu của người dùng. Nhược:
  - Chỉ 3 rollout cùng đề.
  - Agent data-verifier làm **việc khác** nên không so được trong phase giải quyết bất đồng.
  - Việc làm tươi dữ liệu là deterministic: giao cho LLM chỉ thêm rủi ro, không thêm kiểm chứng.
- **B. (Đề xuất) Phase 0 steward deterministic + 4 analyst rollout cùng đề + các phase verifier của
  VeriHarness.** Lý do:
  - Lợi thế của VeriHarness đến từ (1) **nhiều lần chạy cùng một tác vụ** để so và (2) bằng chứng lấy từ môi
    trường, qua hai phiên điều tra **tách ngữ cảnh**. Ablation của bài báo trên Opus 4.8: một lần chạy 49,6 →
    cả hai phiên 56,1; gộp hai phiên vào một ngữ cảnh mất 0,8 (skill `veriharness`, mục "Ý chính").
  - Yêu cầu "verifier chạy trigger để dữ liệu luôn tươi" được đáp ứng hai lớp:
    - Steward chạy `status`/`refresh` trước.
    - Verifier "thách thức đồng thuận" **chạy lại** `status`, tính lại chỉ báo 3 mã từ CSV snapshot, và mở lại
      URL căn cứ.

### 12.2 Quy trình B

| Phase | Ai | Đọc | Ghi | Ghi chú |
|---|---|---|---|---|
| 0 | Steward (script; có thể do session chính gọi) | — | `swing_desk/state/<session>/`, `runs/<session>/{MISSION.md, spec/, workspace/}`; baseline + control đóng băng | Fail closed theo G0 |
| 1 | 4 analyst (cùng model, cùng prompt, song song, ngữ cảnh riêng) | `workspace/` (chỉ đọc) + web | `rollouts/aN/deliverables/{desk_report.md, recommendations.json, sentiment.md}` | Không được khuyến nghị mã BLOCK; WARN phải có lý do; mỗi tin có `S-xx` |
| 2a | Giải quyết bất đồng | CHARTER + INVESTIGATE_ELIM | `ledger_elim.json` | Bất đồng về mã/giá/tin được phân định bằng CSV snapshot và URL, **không bằng số phiếu** |
| 2b | Thách thức đồng thuận | CHARTER + INVESTIGATE_FALS | `ledger_fals.json` | Bắt buộc: hợp đồng đầu ra; độ tươi; tính lại chỉ báo; mở URL; kiểm luật T+2.5/G9 |
| 3 | Phân xử (ngữ cảnh mới) | ADJUDICATE | `finish.json` (base/work/open) | Chỉ dùng hai ledger + spec + workspace |
| 4 | Sửa | REPAIR | `out/deliverables/` + `repair.json` | Giá trị không truy được → để trống |
| 5 | Đóng băng + báo cáo | code `freeze` | `recommendations.jsonl` | Trước 09:00 VN; báo cáo tiếng Việt có các nhánh `open` |

Chi phí: 1 steward + 4 analyst + 2 điều tra + 1 phân xử + 1 sửa. Đường rẻ: N = 2, chỉ chạy phiên thách thức
đồng thuận (skill `veriharness`).

Nguyên tắc:

- "4/4 analyst cùng chọn HPG" **không** là xác suất hay bằng chứng. Skill `veriharness` dẫn bài báo: 34% giá
  trị đồng thuận vẫn sai. Số analyst đồng ý chỉ được ghi để mô tả.
- Analyst phải nêu `p_win_stated` để chấm Brier (§13). Kỳ vọng model bị overconfident [S23].
- Các analyst có thể cùng đọc một bài CafeF nổi bật, nên tương quan nguồn là blind spot chung. Verifier
  FALS phải tìm ít nhất một nguồn độc lập hoặc nguồn gốc (CBTT).

## 13. Giao thức đánh giá tiến cứu (prospective)

Đăng ký `configs/research/swing_desk_prospective_v1.json` **trước** dòng log đầu tiên. Lịch sử đã xem là
development data, không phải holdout.

- **Log**: mỗi khuyến nghị (desk), baseline rule-only và control ngẫu nhiên được đóng băng trước 09:00 VN.
  - Mỗi dòng gồm: sha snapshot, code/params/prompt hash, model id.
  - Log append-only với hash chain; không xoá dòng thua.
  - Các lần `check` của người dùng cũng được log.
- **Nhánh so sánh**:
  - D = desk sau VeriHarness.
  - B = rule-only: S1∪S2 có gate PASS, 3 mã RSI2 thấp nhất, cùng luật entry/stop/target/size.
  - C = 3 mã PASS ngẫu nhiên, seed theo ngày.
  - (tùy chọn) D1 = rollout analyst đơn lẻ trước verify, để đo đóng góp của VeriHarness.
- **Chấm kết quả** (code, quy tắc cố định):
  - Khớp: lệnh giới hạn khớp nếu L ≤ limit, giá = min(O, limit); ATO = O. Không khớp → `no_fill`, đếm riêng.
  - Khóa 2 phiên; thoát sớm nhất ở đóng cửa T+2.
  - Stop: gap → giá O; chạm trong phiên → giá stop. Phiên giảm ≥ 6,5% → coi như không khớp.
  - Target: tương tự. Stop và target cùng nến → tính stop trước.
  - Time exit.
  - Phí 0,4% (chính) và 0,6% (stress).
  - Báo: net %, R, MAE/MFE, hit flags, số phiên giữ.
- **Giả thuyết**:
  - H1 (chính): lợi suất ròng TB mỗi khuyến nghị của D > B, ghép theo phiên.
  - H2: Brier skill score của `p_win_stated` > 0 so với win rate cơ sở. Định nghĩa Brier: B = (1/N)Σ(p−o)²
    (PARTIAL [S35]).
  - H3: D > C.
  - H4 (hành vi): lệnh thật của người dùng gắn PASS có net TB > lệnh WARN/BLOCK/NONE.
- **Ngưỡng mẫu & sức mạnh thống kê** (trung thực):
  - Độ lệch chuẩn lợi suất mỗi lệnh khoảng 4–5% (§1.3).
  - Để phát hiện chênh lệch ghép cặp δ với α = 0,05 và power 0,8 cần `n ≈ (2,8 × σ_d / δ)²`. Với σ_d ≈ 5%:
    - δ = 0,5 điểm % → n ≈ 780.
    - δ = 1 → n ≈ 200.
    - δ = 1,5 → n ≈ 90.
  - Với ~3 khuyến nghị/tuần, trong 6–9 tháng **chỉ phát hiện được khác biệt lớn (≥ 1,5 điểm %/lệnh)**. Không
    được tuyên bố "LLM giúp tăng win rate" trước đó.
  - Sàn tối thiểu cho bất kỳ kết luận nào:
    - ≥ 90 dòng đã resolve mỗi nhánh D và B.
    - ≥ 40 phiên.
    - Có cả hai trạng thái regime (RISK_ON và không RISK_ON).
- **Phân tích**: bootstrap khối theo tuần, khoảng tin cậy 95% cho chênh lệch D − B.
  - Phải dương ở cả hai nửa thời gian và ở phí 0,6%.
  - Báo riêng: tỷ lệ no-fill, số khuyến nghị mới mỗi tháng, các tháng không có khuyến nghị (activity floor do
    người dùng chốt trước, AGENTS.md).
  - Phân biệt "lãi ròng dương" với "chỉ lỗ ít hơn chỉ số".
- **Quyết định**:
  - "Desk hơn rule-only" chỉ khi CI > 0 và qua các stress trên; ngược lại "chưa chứng minh".
  - Mọi thay đổi luật/prompt/model → protocol version mới và reset bộ đếm.
  - Kết quả xanh **không** phê duyệt giao dịch thật.
- **H4**: chỉ mô tả cho tới khi mỗi nhóm ≥ 30 lệnh; báo theo emotion tag.

## 14. Quyết định mở cho người dùng

1. Rủi ro mỗi lệnh: 0,75 / **1,0** / 1,5% NAV. Size theo stop 3% (vị thế lớn, bị dừng nhiều) hay stop −8% đã
   kiểm định (vị thế nhỏ)?
2. B3 (gỡ lỗ) và B6 (hype) là hard hay soft? Có bật cooling-off mặc định không?
3. Có cho phép desk gọi collector Vietstock on-demand (`--with-flows`), hay flows chỉ theo lịch 17:05?
4. Activity floor cho đánh giá, ví dụ ≥ 4 khuyến nghị desk/tháng.
5. Vị trí thư mục (`swing_desk/` gốc repo hay `data/swing_desk/`) và skill (`.agents/` đang bị git-ignore).
6. Có chạy nhánh D1 (rollout đơn lẻ) để đo riêng giá trị của VeriHarness không? Tốn thêm, nhưng là cách
   duy nhất trả lời câu "VeriHarness có giúp không".
7. Xác nhận với CTCK: còn đúng T+2.5 không; có lệnh điều kiện (stop order) không.

## 15. Nguồn

Grade:

- VERIFIED: trang đã mở, nội dung khớp.
- PARTIAL: nguồn thứ cấp, hoặc chỉ một phần.
- PAYWALL: chỉ metadata.
- UNVERIFIED: chưa mở. Không dùng làm căn cứ.

"Mở bởi":

- T10: tôi mở trực tiếp.
- Sx: một trong 5 scout subagent mở. Báo cáo scout là testimony; tôi mở lại kiểm tra một số mục, ghi ✓.

Forum và blog là ý kiến người chơi. Báo cáo CTCK là sell-side opinion.

**Quy tắc thị trường Việt Nam (R)**

- [R1] Tạp chí Thị trường Tài chính Tiền tệ, "Từ ngày 29/8, chính thức rút ngắn thời gian thanh toán… T+2",
  https://thitruongtaichinhtiente.vn/tu-ngay-29-8-chinh-thuc-rut-ngan-thoi-gian-thanh-toan-giao-dich-chung-khoan-xuong-con-t-2-41949.html
  — VERIFIED (T10).
- [R2] LSVN, "Chu kỳ thanh toán chứng khoán chính thức rút ngắn xuống T+2" (30/08/2022),
  https://lsvn.vn/chu-ky-thanh-toan-chung-khoan-chinh-thuc-rut-ngan-xuong-t21661824227-a123203.html
  — VERIFIED (scout).
- [R3] VnEconomy 24/10/2025 (VSDC về bán chứng khoán chờ về/T0, CCP 2027); doanhnhan.baophapluat.vn
  08/02/2021 (TT120/2020) — PARTIAL (scout). TT 68/2024, 18/2025, 08/2026 chỉ áp dụng cho tổ chức nước ngoài
  hoặc nội dung CBTT — PARTIAL.
- [R4] CTS/VBSE, "Quy định giao dịch chứng khoán tại HSX" (KRX, 08/2025),
  vbse.vn/wp-content/uploads/2025/08/KRX_1.Quy-dinh-giao-dich-chung-khoan-tai-HSX.pdf — VERIFIED (scout).
  Biên độ, phiên, lô, tham chiếu GDKHQ, bước giá.
- [R5] Thời báo Tài chính 31/03/2026 (thuế 0,1%/lần chuyển nhượng); VnEconomy 05/09/2025 (rút đề xuất
  20%) — VERIFIED (scout). Ngày hiệu lực Luật 109/2025/QH15: UNVERIFIED.
- [R6] Biểu phí Shinhan Securities hiệu lực 27/03/2026 — VERIFIED (scout). Mức 0% ở CTCK khác: UNVERIFIED.
- [R7] MBS, "Quy định giao dịch HĐTL chỉ số VN30" (05/2025),
  https://mbs.com.vn/files/uploads/2025/05/1.-Quy-dinh-giao-dich-hdtl-csvn30.pdf — VERIFIED (T10 đọc PDF):
  - "Ngày thứ Năm thứ ba trong tháng đáo hạn, nếu trùng ngày nghỉ điều chỉnh lên ngày giao dịch trước đó".
  - Giá thanh toán là trung bình 30 phút cuối, loại 3 cao/3 thấp.
  - Biên độ ±7%.
- [R8] Thời báo Tài chính 17/05/2022: HOSE công bố tự doanh cuối ngày — VERIFIED (scout).
- [R9] TT 96/2020 Điều 33 (sửa bởi 18/2025, 08/2026): giao dịch nội bộ, qua accgroup.vn và hethongphapluat.com
  — VERIFIED (scout, nguồn thứ cấp).
- [R10] Duane Morris Vietnam blog, "Today is the day for Vietnam" (21/09/2026),
  https://blogs.duanemorris.com/vietnam/2026/09/21/today-is-the-day-for-vietnam/ — VERIFIED (T10): hiệu lực
  21/09/2026, phân kỳ 10/20/35/35%. LSEG 07/10/2025 và Vietstock 08/04/2026, 25/06/2026 (MSCI) — VERIFIED
  (scout).
- [R11] Nghị định 155/2020 Điều 139 về room NN, qua luatvietan.vn — PARTIAL (scout).
- [R12] QĐ 87/QĐ-UBCK (2017) về tỷ lệ ký quỹ, qua luatvietan.vn; SHS 2023 — PARTIAL (scout).
- [R13] HOSE factsheet chỉ số (31/03/2025), static2.vietstock.vn; Báo Đầu tư 14/01/2025 (rulebook v4.0);
  Tạp chí Kinh tế Tài chính 15/07/2026 và 03/07/2026 (cơ cấu 7/2026, ETF) — VERIFIED/PARTIAL (scout).

**Giá, đảo chiều, momentum, volume, biến động, rule kỹ thuật (A)** — VERIFIED (scout) trừ khi ghi khác

- [A1] Jegadeesh (1990), JF 45(3):881–898, https://ideas.repec.org/a/bla/jfinan/v45y1990i3p881-98.html —
  VERIFIED (T10 ✓).
- [A2] Lehmann (1990), QJE 105(1):1–28, https://www.nber.org/papers/w2533.
- [A3] Avramov, Chordia & Goyal (2006), JF 61(5):2365–2394,
  https://ideas.repec.org/a/bla/jfinan/v61y2006i5p2365-2394.html.
- [A4] Nagel (2012), RFS 25(7):2005–2039, https://ideas.repec.org/a/oup/rfinst/v25y2012i7p2005-2039.html.
- [A5] Jegadeesh & Titman (1993), JF 48(1):65–91.
- [A6] Daniel & Moskowitz (2016), JFE 122(2):221–247, https://www.nber.org/papers/w20439.
- [A7] Gervais, Kaniel & Mingelgrin (2001), JF 56(3):877–919.
- [A8] Lee & Swaminathan (2000), JF 55(5):2017–2069.
- [A9] Llorente, Michaely, Saar & Wang (2002), RFS 15(4):1005–1047, https://www.nber.org/papers/w8312.
- [A10] Ang, Hodrick, Xing & Zhang (2006), JF 61(1):259–299.
- [A11] Moreira & Muir (2017), JF 72(4):1611–1644, https://www.nber.org/papers/22208.
- [A12] Wilder (1978), *New Concepts in Technical Trading Systems* — metadata qua Wikipedia (PARTIAL).
- [A13] Brock, Lakonishok & LeBaron (1992), JF 47(5):1731–1764.
- [A14] Sullivan, Timmermann & White (1999), JF 54(5):1647–1691; full text LSE FMG DP303.
- [A15] Lo, Mamaysky & Wang (2000), JF 55(4):1705–1765.
- [A16] George & Hwang (2004), JF 59(5):2145–2176 — PAYWALL.
- [A17] Connors & Alvarez, *Short Term Trading Strategies That Work* (TradingMarkets/Connors Group) — metadata
  (trang bán sách). Năm/ISBN bản đầu: UNVERIFIED.
- [A18] StockCharts ChartSchool, "RSI(2)" — practitioner.
- [A19] CXO Advisory, "A Few Notes on Short Term Trading Strategies That Work",
  https://cxoadvisory.com/technical-trading/a-few-notes-on-short-term-trading-strategies-that-work —
  practitioner critique, VERIFIED (T10 ✓): không xử lý data snooping, không out-of-sample, không có ma sát.
- [A20] Alpha Algo Trading Substack — practitioner, bằng chứng yếu.
- [A21] Sweeney (1997), *Maximum Adverse Excursion*, Wiley,
  https://www.wiley-vch.de/en/areas-interest/finance-economics-law/maximum-adverse-excursion-978-0-471-14152-5
  — VERIFIED (T10 ✓, trang nhà xuất bản).
- [A22] Sweeney (1987), TASC 5(4) — metadata.
- [A23] LuxAlgo, "MAE/MFE Distributions" — practitioner.
- [A24] Tharp (2006), *Trade Your Way to Financial Freedom* — metadata.
- [A25] TradeThatSwing, "What R means" — practitioner.
- [A26] Kaminski & Lo (2014), J. Financial Markets 18:234–254.
- [A27] Lo & Remorov (SSRN 2015; luận văn MIT 2016), https://dspace.mit.edu/handle/1721.1/107017.
- [A28] Han, Zhou & Zhu, "Taming Momentum Crashes" — qua tóm tắt CXO.
- [A29] Dummies, "Overnight Gaps and Trend Trading" — practitioner.
- [A30] Nguyễn Thị Yến (2021), Kinh tế & Phát triển số 284 — full text.
- [A31] Duong & Bertrand (2023), Managerial Finance 49(1):13–28.
- [A32] Vo & Truong (2018), JBEF 17:10–15.
- [A33] Truong et al. (2023), IJFS 11(2):58.
- [A34] Metghalchi (2013), J. Prediction Markets 7(2) — cấp chỉ số, không xử lý data snooping.
- [A35] Liberated Stock Trader, Expectancy calculator — practitioner.

**Hành vi (B)**

- [B1] Barber & Odean (2000), "Trading Is Hazardous to Your Wealth", JF 55(2):773–806,
  https://ideas.repec.org/a/bla/jfinan/v55y2000i2p773-806.html — VERIFIED (T10): "those that trade most earn
  an annual return of 11.4 percent, while the market returns 17.9 percent… Overconfidence can explain high
  trading levels".
- [B2] Odean (1998), "Are Investors Reluctant to Realize Their Losses?", JF 53(5):1775–1798,
  https://faculty.haas.berkeley.edu/odean/papers/disposition/disposition.html — VERIFIED (T10): "the tendency
  of investors to hold losing investments too long and sell winning investments too soon".
- [B3] Barber & Odean (2008), "All That Glitters", RFS 21(2):785–818,
  https://ideas.repec.org/a/oup/rfinst/v21y2008i2p785-818.html — VERIFIED (T10).
- [B4] Barber, Huang, Odean & Schwarz (2022), "Attention-Induced Trading and Returns: Evidence from Robinhood
  Users", JF 77(6):3141–3190,
  https://profiles.wustl.edu/en/publications/attention-induced-trading-and-returns-evidence-from-robinhood-use/
  — VERIFIED (T10): "Average 20-day abnormal returns are −4.7% for the top stocks purchased each day."
- [B5] Coval & Shumway (2005), "Do Behavioral Biases Affect Prices?", JF 60(1):1–34,
  https://ideas.repec.org/a/bla/jfinan/v60y2005i1p1-34.html — VERIFIED (T10).
- [B7] Seasholes & Wu (2007), "Predictable behavior, profits, and attention", J. Empirical Finance 14:590–610,
  https://www.bauer.uh.edu/wu/Papers/JEF2007_Attention.pdf — VERIFIED (T10 đọc trang abstract PDF).
- [B6] Shefrin & Statman (1985), JF 40(3):777–790,
  https://ideas.repec.org/a/bla/jfinan/v40y1985i3p777-90.html — META (scout; không có abstract).
- [B8] Odean (1999), "Do Investors Trade Too Much?", AER 89(5):1279–1298 — VERIFIED (scout, full text).
- [B9] Barber & Odean (2001), "Boys Will Be Boys", QJE 116(1):261–292 — VERIFIED (scout): "Men trade 45
  percent more than women".
- [B10] Barber, Lee, Liu & Odean (2014), J. Financial Markets 18:1–24 — VERIFIED (scout). Cùng nhóm tác giả
  (2009), RFS 22(2):609–632: NĐT cá nhân Đài Loan mất 3,8 điểm %/năm. Chague, De-Losso & Giovannetti
  (2020), FGV WP 525: 97% day trader bền bỉ ở Brazil lỗ — VERIFIED (scout, working paper).
- [B11] Thaler & Johnson (1990), Management Science 36(6):643–660 — VERIFIED (scout).
- [B12] Kahneman & Tversky (1979), Econometrica 47(2):263–291 — VERIFIED (scout).
- [B13] Ariely & Wertenbroch (2002), Psychological Science 13(3):219–224 — VERIFIED (scout).
- [B14] Luca, Malhotra & Poliquin (2017), PNAS 114(46):12162–12165 — VERIFIED (scout).
- [B15] Haynes et al. (2009), NEJM 360(5):491–499 — PARTIAL (qua Harvard Gazette).
- [B16] Steenbarger, TraderFeed blog — practitioner.
- [B17] Vo & Phan (2017), JBEF 13:33–41 — VERIFIED (scout).
- [B18] Phan et al. (2023), Cogent Economics & Finance 11(2) — VERIFIED (scout).
- [B19] Nguyen Duc Hien et al. (2012), J. Economics and Development 14(2):52–71 — VERIFIED (scout).
- [B20] Trần Thị Hải Lý & Huỳnh Ngọc Thương (2013), J. Economic Development — VERIFIED (scout).
- [B21] Pham, Nguyen & Tô (2007), *Asia-Pacific Financial Markets* ch.10, Elsevier/Emerald — VERIFIED (scout,
  tóm tắt).
- [B22] Farber, Nguyen & Vuong (2006), ULB WP; Lê Đình Nghi (2012), J. Economic Development 214 — VERIFIED
  (scout, abstract). Kim & Rhee (1997), JF 52(2):885–899, về giới hạn giá Tokyo — VERIFIED (scout).
- UNVERIFIED (không dùng): Chapkovski et al. (gamification), Strahilevitz, Odean & Barber (2011).

**Sentiment / LLM (S)** — VERIFIED (scout) trừ khi ghi khác

- [S1] Tetlock (2007), JF 62:1139–1168, https://www.columbia.edu/~pt2238/papers/Tetlock_Media_Sentiment_JF.pdf.
- [S2] Tetlock, Saar-Tsechansky & Macskassy (2008), JF 63(3):1437–1467.
- [S3] Loughran & McDonald (2011), JF 66(1):35–65: gần 3/4 từ "negative" của Harvard Dictionary không tiêu
  cực trong ngữ cảnh tài chính.
- [S4] Chan (2003), JFE 70(2):223–260 — PAYWALL. Citation VERIFIED (T10, RePEc không có abstract); phát hiện
  chỉ từ snippet.
- [S5] Baker & Wurgler (2006), NBER w10449.
- [S6] Da, Engelberg & Gao (2015), RFS 28(1):1–32.
- [S7] Lopez-Lira & Tang, arXiv:2304.07619v6 — VERIFIED (T10 ✓).
- [S8] Araci (2019), FinBERT, arXiv:1908.10063.
- [S9] Huang, Wang & Yang (2023), CAR 40(2).
- [S10] Kirtac & Germano (2024) — UNVERIFIED, không dùng.
- [S11] Chen, Kelly & Xiu — PARTIAL.
- [S12] Glasserman & Lin (2023), arXiv:2309.17322 — VERIFIED (T10 ✓).
- [S13] Sarkar & Vafa (2024/2025), ICML workshop.
- [S14] Lopez-Lira, Tang & Zhu (2025), arXiv:2504.14765.
- [S15] He et al. (2025), arXiv:2502.21206.
- [S16] DatedGPT (2026), arXiv:2603.11838.
- [S17] FinanceBench, arXiv:2311.11944.
- [S18] TradingAgents, arXiv:2412.20138: thử 3 tháng, 3 mã.
- [S19] FinMem, arXiv:2311.13743: chọn cấu hình theo giai đoạn test.
- [S20] StockBench, arXiv:2510.02209 — VERIFIED (T10 ✓).
- [S21] AlphaForgeBench, arXiv:2602.18481.
- [S22] Harris (2026), arXiv:2607.20645 — VERIFIED (T10 ✓): "matches all expert labels in only 52.4%".
- [S23] Chen, Green, Gulen & Zhou, arXiv:2409.11540.
- [S24] Bernard & Thomas (1989) — PARTIAL (qua Wikipedia).
- [S25] Cohen, Malloy & Pomorski (2012), JF 67(3).
- [S26] Lakonishok & Lee (2001), RFS 14(1).
- [S27] Loughran & Ritter (1995), JF 50(1).
- [S28] Hartzmark & Solomon (2013), JFE 109(3).
- [S29] Tetlock (2011), RFS 24(5).
- [S30] Heston & Sinha (2017), FAJ 73(3).
- [S31] PhoBERT, arXiv:2003.00744.
- [S32] Vu, Pham, Kieu & Pham (2023), IJFS 11(3):101,
  https://ideas.repec.org/a/gam/jijfss/v11y2023i3p101-d1212361.html — VERIFIED (scout; T10 xác nhận nội dung
  qua tìm kiếm).
- [S33] Tran et al. (2025), IEEE DEFI — abstract.
- [S34] Phan Việt et al. (2026), JCC 42(2).
- [S35] Brier (1950), qua NOAA McNulty (1988) — PARTIAL.
- [S36] Mellers et al. (2014) — qua APS Observer.
- [S37] Gneiting & Raftery (2007) — PAYWALL.

**Vĩ mô, ngành, dòng tiền (M)** — scout; grade theo từng mục

- [M1a] vnba.org.vn 17/06/2023, giảm lãi suất điều hành lần 4 — VERIFIED.
- [M1b] Tạp chí Thị trường Tài chính Tiền tệ 25/05/2026, "NHNN quyết tâm hạ nhiệt lãi suất" — VERIFIED.
- [M1c] Viet Nam News 13/01/2026, mục tiêu tín dụng 15% — VERIFIED.
- [M1d] VietnamFinance 28/07/2026, tỷ giá trung tâm kỷ lục, biên ±5% — VERIFIED.
- [M2a] Nguyen, Mai & Tran (2020), AEFR (ARDL) — VERIFIED.
- [M2b] Le, Dang & Ho (2018), NARDL — VERIFIED.
- [M2c] Nguyễn An Phú (2026), Tạp chí Kinh tế Tài chính (VAR) — VERIFIED.
- [M3] smoney.com.vn, danh mục E1VFVN30 tháng 8/2026 — PARTIAL (aggregator; tổng tỷ trọng do scout tự cộng).
- [M4d] trungtamwto.vn 03/04/2026, QĐ 612/QĐ-BCT — VERIFIED.
- [M4e] VnEconomy 01/07/2026 (dầu khí); baodauthau (NĐ 80/2023) — VERIFIED.
- [M4f] doanhnghiephoinhap 17/06/2025, giảm VAT 2% đến 31/12/2026 — VERIFIED.
- [M4i] lsvn.vn 14/09/2026, giải ngân đầu tư công — PARTIAL.
- Các mục ngành khác (ngân hàng NIM, BĐS, VJC, SSI) xem báo cáo scout; tất cả là PARTIAL/VERIFIED tin báo.
- [M5] Nguyen Thi Nhung et al. (2023), VNU JEB, https://jeb.ueb.edu.vn/index.php/jeb/article/view/156 —
  VERIFIED.
- [M5c] VnEconomy 05/01/2026, NN bán ròng kỷ lục 2025 — VERIFIED.
- [M6] Trần Văn Trung (2025), Tạp chí Tài chính — VERIFIED.
- [M7a] VietnamFinance 21/07/2026 — VERIFIED.
- [M7b] VnEconomy 21/07/2026 — VERIFIED.
- [M7c] bbw.vn — PARTIAL.
- [M8] StockCharts ChartSchool: McClellan; Percent Above MA — định nghĩa, practitioner.
- [M9] Brown & Cliff (2004), JEF 11(1) — PARTIAL (qua tóm tắt CXO).
- [M10] Lịch sự kiện Vietstock https://finance.vietstock.vn/lich-su-kien.htm và CafeF
  https://cafef.vn/du-lieu/lich-su-kien.chn — VERIFIED (tồn tại, có tab cổ tức/ĐHĐCĐ). hsx.vn render bằng JS
  (PARTIAL). ssc.gov.vn và vsd.vn không mở được (UNVERIFIED). F247 tồn tại — forum.
- [M11] Trần Văn Khởi & Phạm Tiến Đạt (2025), Tạp chí Tài chính: cá nhân ~80–85% giá trị GD — VERIFIED;
  Đặng Công Hoàn (2022) — VERIFIED.

**Repo (bằng chứng nội bộ, development data)**

- memory:
  - `strategy-edge-findings`
  - `swing-rsi2-edge`
  - `crash-response-by-shape`
  - `win-probability-metalabeling`
  - `portfolio-power-verdict`
  - `momentum-excess-gate`
  - `foreign-flow-data-sources`
- audits:
  - `docs/audits/2026-10-04-r5-weekly-results.md`
  - `docs/audits/2026-10-06-data-update-flow.md`
- code:
  - `stock_agent/features/{indicators,swing_scan,mr_scan,market_regime,position_manager}.py`
  - `stock_agent/pipeline/paper_runner.py`
  - `stock_agent/data/{foreign_flows,exchange_calendar}.py`

## Giới hạn

- Thống kê §1.2–1.3 dùng rổ VN30 hiện tại áp ngược (survivorship) và dữ liệu đã điều chỉnh của provider.
  Đây là mô tả cấu trúc rủi ro, không phải kiểm định tín hiệu, và không qua research gate.
- Không mở được trang gốc hsx.vn / ssc.gov.vn / vsd.vn. Các quy tắc thị trường lấy từ tài liệu CTCK, văn bản
  pháp luật qua trang thứ cấp và báo chí.
- Báo cáo scout là testimony. Tôi mở lại một phần (✓); phần còn lại dựa vào kỷ luật "chỉ báo cáo nguồn đã
  mở" của brief scout.
- Ngưỡng của gate là **tham số thiết kế có lý do**, không phải tham số đã tối ưu. Mọi tuyên bố rằng gate
  "cải thiện lợi nhuận" phải qua research gate và §13.
- Tài liệu này không phải chấp thuận giao dịch hay triển khai.
