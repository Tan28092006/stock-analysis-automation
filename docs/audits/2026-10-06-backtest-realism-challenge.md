# Backtest trong thực tế: challenge trực tiếp, T+2, thời điểm thoát lệnh, thiết kế portfolio backtest

> Bản cuối VeriHarness 06/10/2026 (Firstmate run `2026-10-06-ml-roadmap`): 3 rollout độc lập → giải bất đồng + đánh đổ đồng thuận (ngữ cảnh tách) → phân xử (base r01, 8 work items) → repair. Coordinator kiểm lại: patch áp vào repo thật trên branch `firstmate/mr-exit-single-path` (commit `8b244f2`), `pytest` 8 file liên quan **86 passed**. Ledger/finish/repair JSON nằm trong scratchpad của phiên, không commit.

Rollout r01 · 06/10/2026 · source revision `4403fc7` (bản copy read-only `workspace/repo`).
Phạm vi: chỉ đọc code + audit + memory, script **synthetic** trong `rollouts/r01/trajectory/`.
**Không chạy backtest trên giá thật, không có con số hiệu suất mới.** Mọi số hiệu suất trích dẫn
dưới đây là số đã có trong `workspace/memory` hoặc `docs/audits` (development data, đã bị xem,
không phải holdout). Mã claim `[Cxx]` tra trong `claims.json`.

## 0. Kết luận ngắn (đọc phần này nếu chỉ có 2 phút)

1. Repo có **năm** bộ mô phỏng thoát lệnh với năm hệ kế toán khác nhau, và chỉ **một** trong
   số đó (`scripts/period_backtest.py` Broker, dùng bởi `research_gate.py`) gần với thực tế giao
   dịch HOSE. Các đường legacy (`build_labeled_dataset_fast`, `run_backtest`,
   `run_portfolio_backtest`) chứa các giả định lạc quan có hệ thống: khớp stop đúng mức stop khi
   giá gap qua, clamp giá vào biên độ thay vì coi là không khớp được, tiền bán dùng ngay (không
   settlement), không có ràng buộc thanh khoản, universe cố định hiện tại (survivorship).
2. `run_portfolio_backtest` có **lỗi logic** chứ không chỉ là giả định: stop/target chỉ được kiểm
   tra vào đúng ngày exit dự kiến (`backtest.py:622-629`), stop khớp ở mức cao hơn cả high của
   ngày (synthetic: 18.950 so với high 15.150), vị thế thiếu bar ngày cuối bị **xóa khỏi equity**
   (synthetic: −24,97% "lỗ ma"), tiền bán lúc đóng cửa được dùng mua lúc mở cửa cùng ngày, và
   Sharpe theo trade luôn nhân `sqrt(252/2)` bất kể horizon (phóng đại ×3,16 ở horizon 20 phiên).
   Đây là đường cần viết lại nếu user định đổi portfolio backtest.
3. "T+2" trong repo là **hai thứ khác nhau bị trùng tên**: (a) settlement lock 2 bar; (b) target
   `net_t2_win`. Nhưng cấu hình mặc định `configs/rules_t2.json:3` đặt `time_horizon_days: 20`,
   nên nhãn `net_t2_*` hiện tại thực chất là lệnh bracket **20 phiên**. Không có bằng chứng nào
   trong memory/audit cho thấy giữ 2 phiên có edge sau phí; có nhiều bằng chứng ngược lại (IC
   T+2 ≈ 0, chi phí 0,6%/vòng lớn hơn biến động kỳ vọng 2 ngày). Kết luận: **T+2 nên là ràng buộc
   settlement, không phải horizon mục tiêu**.
4. Về thời điểm thoát lệnh tối ưu: tài liệu học thuật (US/developed) cho câu trả lời có điều
   kiện: horizon tối ưu phụ thuộc tốc độ suy giảm của tính dự báo (half-life mean reversion) so
   với chi phí; reversal ngắn hạn phần lớn bị chi phí ăn; momentum là 3–12 tháng. Không có nghiên
   cứu nào tôi mở được đưa ra "số phiên tối ưu" cho HOSE ngắn hạn; bằng chứng VN tìm được là theo
   tuần/tháng.
5. Mâu thuẫn `mr_exit.py:7` được giải quyết bằng refactor **giữ nguyên hành vi** (không cần
   preregistration) + docstring đúng; các thay đổi ngữ nghĩa (gap fill, T+3/T+2-chiều, unfillable,
   horizon theo bar) phải đăng ký trong research gate trước khi xem kết quả. Xem
   `mr_exit_resolution.md` và `.patch`.

## 1. Q1 — Repo đang backtest như thế nào

| Đường | File:line | Entry | Exit | Settlement | Cost/slippage | Universe | Kết thúc mẫu |
|---|---|---|---|---|---|---|---|
| Fast label builder (training `net_t2_*`) | `stock_agent/agents/parallel_engine.py:189-365` | open bar kế tiếp, clamp ±6,9% (`:290`) | vòng lặp chép riêng `:298-312` (stop trước, target sau, lock 2 bar), clamp exit `:316` | lock 2 bar, tiền không mô hình | trừ cộng dồn 0,6% (`:224,318`) | danh sách truyền vào, `fixed_requested_universe_not_point_in_time` (`:364`) | row bị bỏ nếu thiếu bar exit theo lịch (`:266-269`) |
| `run_backtest` (1 mã) | `stock_agent/features/backtest.py:251-403` | open kế tiếp, clamp + 1 tick (`:321,345-346`) | `simulate_mr_exit` (`:334-335`), clamp + 1 tick (`:342,348-349`) | lock 2 bar | 1 tick + phí 0,4% (`:352`); summary lại báo 0,6% có slippage % (`:367,379`) | 1 mã | dừng nếu chưa resolved (`:336-337`) |
| `run_portfolio_backtest` | `backtest.py:406-506, 605-782` | open + 1 tick rồi clamp (`:734-735`) | **chỉ** kiểm tra stop/target vào ngày `planned_exit_date` (`:622-629`), không clamp exit | **không** — tiền bán cộng ngay (`:437,689`) | tick + phí; `cash_room` dùng `slippage_pct` (`:741`) | file CSV hiện có (`:525-537`) | `FORCED_CLOSE` tại close cuối (`:465-470,635-651`) |
| Research Broker (`replay_mr`, `replay_momentum`) | `scripts/period_backtest.py:98-171, 220-335` | open × (1+s), bỏ nếu open ngoài (stop,target) hoặc gap ≥ biên (`:244-247,206-211`) | stop `min(stop, open)` (`:265`), target, time; không bán nếu open chạm sàn (`:260-262`) | **T+3** cho cổ phiếu và tiền (`:133,152`) | nhân (1±s), phí trên notional (`:105-107,125-147`) | PIT membership qua `research_gate` (`research_gate.py:137,227`) | mark-to-close, không ép bán (`:369`) |
| `research_ml.labels` (R3) | `stock_agent/research_ml/labels.py:36-80` | open `e` | close `e+horizon`, không bracket | không | trừ cộng dồn 0,6% (`:72-73`) | do caller | `pending`/`exit_unfillable` tường minh (`:59-71`) |
| Forward label resolver (live) | `stock_agent/pipeline/label_resolver.py:100-137` | open clamp | close tại horizon, **không bracket** | không | trừ cộng dồn | — | pending |

Hệ quả trực tiếp: model được train trên nhãn bracket của fast builder, nhãn live được resolve
bằng close-to-close không bracket (`label_resolver.py:131-137`), còn đánh giá danh mục dùng
Broker T+3. Ba "sự thật" khác nhau cho cùng một quyết định [C01, C02]. Audit R3
(`docs/audits/2026-10-06-r3-labels-cost-parity.md:34-44`) đã liệt kê phần lớn skew fast↔Broker;
tôi bổ sung label_resolver, run_portfolio_backtest và các bằng chứng synthetic.

**Ai dùng `run_portfolio_backtest`:** lỗi của hàm này (mục 2.2, 2.8, 2.9) lan sang
`optimizer.py:331-345,535-544` (split train/validation/test), `robustness.py:35,142,173`, và qua
`run_portfolio_backtest_from_csvs` (`backtest.py:509-539`) sang `cli.py:84` và `app.py:216` [C27].
Mọi tham số tối ưu và kết quả robustness rút ra từ các đường này kế thừa cùng các giả định.

## 2. Q1 — Challenge từng giả định (mức độ, chiều lệch)

Thang mức độ: **Cao** = có thể đổi dấu kết luận; **TB** = đổi độ lớn đáng kể; **Thấp** = sai số nhỏ
hoặc đã được giảm thiểu. Chiều lệch: Lạc quan (+) / Bi quan (−) / Không xác định (±).

### 2.1 Settlement và khả năng bán

- **Quy tắc thật (đã verify):** từ 29/08/2022 nhà đầu tư được bán chứng khoán mua ngày T ngay
  **phiên chiều T+2** (VSD Quyết định 109/QĐ-VSD ngày 19/08/2022; tiền và chứng khoán phân bổ
  trước 13:00 T+2); trước đó phải chờ đến **T+3**. Nguồn báo chí sơ cấp: Thị trường Tài chính
  Tiền tệ (Quyết định 109/QĐ-VSD ký 19/08/2022, hiệu lực 29/08/2022; ngân hàng thanh toán và VSD
  hoàn tất thanh toán 11h00–11h30 ngày T+2; nhà đầu tư "thực hiện mua, bán chứng khoán trong
  phiên giao dịch buổi chiều cùng ngày") [C30]. Chu kỳ thanh toán T+2 có từ 01/01/2016 (UNVERIFIED,
  [C31]).
- **Quy tắc này vẫn có hiệu lực năm 2026 (đã verify):** The Investor 27/02/2026: cổ phiếu, chứng
  chỉ quỹ, chứng quyền mua ngày 25/02/2026 (và trái phiếu mua 26/02) lẽ ra được phân bổ **trong
  phiên chiều 27/02** (T+2) nhưng bị chậm vì hệ thống KRX đang bảo trì [C35]. T+0 (mua bán trong
  ngày) và bán chứng khoán chờ về: năm 2025 sẵn sàng về pháp lý/hệ thống nhưng chờ phê duyệt
  (16/08/2025 [C32]); theo tin báo chí 06/2026 (Chủ tịch VSDC, 11/06/2026) **vẫn chưa triển khai**,
  sớm nhất H2-2026 [C36].
- **Repo:** `settle_lock=2` bar (`mr_exit.py:45`, `parallel_engine.py:298-301`,
  `backtest.py:335`) cho phép khớp stop/target trên **toàn bộ high/low của bar T+2**, tức là cả
  phiên sáng khi cổ phiếu chưa về. Với dữ liệu trước 29/08/2022 (phần lớn lịch sử train
  2018–2022) lock 2 bar còn **ngắn hơn luật một phiên**.
  - Mức độ: **TB-Cao** với horizon ngắn (đúng đường `net_t2`), Thấp với hold 15–20 phiên.
  - Chiều: (+) với target (high buổi sáng không bán được); (±) với stop. Broker dùng T+3 bảo thủ
    (`period_backtest.py:3-4,133`) — đúng hướng cho dữ liệu trước 2022, hơi bi quan sau 2022.
- **Tiền bán:** `run_portfolio_backtest` cộng tiền bán vào `cash` ngay trong ngày rồi dùng mua ở
  giá **mở cửa cùng ngày** (`backtest.py:437-450`). Synthetic `[D]`: AAA bán lúc close 06/03, BBB
  mua 900 cp (9 triệu) lúc open 06/03 với vốn ban đầu 10 triệu đã kẹt 8 triệu trong AAA
  (`trajectory/portfolio_mechanics.out`) [C10]. Mức độ: **Cao** khi turnover cao; chiều (+).

### 2.2 Giá khớp entry/exit, gap-through

- **Stop khớp đúng mức stop khi giá gap qua:** `simulate_mr_exit` trả `float(stop)`
  (`mr_exit.py:46-47`), fast builder cũng vậy (`parallel_engine.py:303-305`). Synthetic: open 80,
  stop 95 → ghi nhận khớp 95 (`exit_equivalence.out` [2]) [C11]. Thực tế lệnh stop ở VN là lệnh
  điều kiện của công ty chứng khoán, kích hoạt trong phiên liên tục, không áp dụng cho phiên
  định kỳ (audit `2026-09-30-execution-receipts.md:18-20`), và phải khớp với hàng chờ — giá mở
  cửa gap là mức tốt nhất có thể kỳ vọng. Broker và paper scoring đã dùng `min(stop, open)`
  (`period_backtest.py:265`, `paper_scoring.py:83-84`). Mức độ: **TB** (MR mua vào đúng lúc hoảng
  loạn → gap-down thường gặp nhất); chiều (+).
- **Portfolio backtest khớp stop ở mức không tồn tại trong ngày:** vì chỉ xét ngày planned exit
  và lấy `position["stop_loss"]` khi `low <= stop` (`backtest.py:624-626`), nếu cả ngày nằm dưới
  stop thì giá khớp còn **cao hơn high**: synthetic `[B]` khớp 18.950 trong khi high 15.150
  (+26%) [C12]. Mức độ: **Cao**; chiều (+).
- **Stop bị bỏ qua giữa kỳ:** cùng hàm chỉ chạy khi `current_date >= planned_exit_date`
  (`backtest.py:622-623`). Synthetic `[A]`: low 15.000 thủng stop 19.000 ngày 06/03 nhưng vị thế
  thoát 11/03 lý do `T2_CLOSE` [C13]. Tái lập độc lập (synthetic `fals/scratch/pb.py`): giá giảm
  25% (20.000 → 15.000, low 14.800 thủng stop 19.000) trong lúc giữ rồi hồi phục trước ngày planned
  exit ⇒ trade ghi `T2_CLOSE` −0,90% (−0,8975%) thay vì `STOP_LOSS` [C13]. Với `rules_t2` (horizon 20) stop gần như vô hiệu. Chiều (±):
  bỏ stop có thể cải thiện trung bình (memory: "no-stop survives", `strategy-edge-findings.md`)
  nhưng che rủi ro đuôi; quan trọng hơn là **khác hẳn** run_backtest/label nên số portfolio
  không kiểm chứng được nhãn.
- **Target gap-up:** khớp ở target thấp hơn low của ngày (synthetic: target 110, low 128)
  → (−) nhỏ.
- **Precedence cùng bar stop-trước-target** (`mr_exit.py:46-49`): bảo thủ hợp lý với dữ liệu ngày;
  giữ nguyên.
- **Entry next-open:** hợp lý, nhưng giá mở cửa là kết quả phiên ATO; một lệnh mua ở ATO không
  đảm bảo khớp toàn bộ ở giá đó. Daily OHLC không chứng minh được (audit execution-receipts
  `:5-13`). Thấp-TB; (+).

### 2.3 Biên độ giá (±7% HOSE) và ngày khóa trần/sàn

- **Luật (đã verify):** biên độ ±7% so với giá tham chiếu; 20% cho ngày giao dịch đầu tiên, ngày
  GDKHQ trả cổ tức bằng cổ phiếu quỹ, và ngày giao dịch lại sau tạm ngừng > 25 ngày; lô chẵn 100
  (Quyết định 352/QĐ-SGDHCM, 2021) [C33].
- **Repo:** `price_limit_pct: 6.9` (`rules_t2.json:68`) và `enforce_price_limit` **clamp** giá
  vào biên (`backtest.py:162-168`), dùng close điều chỉnh của bar trước làm tham chiếu
  (`parallel_engine.py:289-290,314-316`; `label_resolver.py:127-134`). Clamp tạo ra **giá khớp
  bịa** thay vì trạng thái "không khớp được". Không đường legacy nào kiểm tra **ngày khóa sàn**
  (high == low == floor, không có người mua): stop vẫn được giả định khớp. Broker chỉ chặn ở giá
  mở cửa (`period_backtest.py:206-211`), không bắt trường hợp mở cửa trên sàn rồi khóa sàn cả
  ngày; paper scoring có kiểm tra `high == low` (`paper_scoring.py:85`).
- Mức độ: **TB-Cao trong khủng hoảng** (2022, 04/2025, 03/2026 — đúng lúc MR giao dịch); chiều
  (+): thực tế không bán được ở sàn, lỗ kéo sang ngày sau. Với ngày biên 20% hoặc giá điều chỉnh
  (tham chiếu chính thức ≠ close điều chỉnh, audit `strategy-reassessment.md:175-177`) clamp 6,9%
  còn bóp méo giá thật (±).

### 2.4 Lô, thanh khoản, phí, thuế, slippage

- **Lô 100:** portfolio (`backtest.py:743`) và Broker (`period_backtest.py:114-115`) đúng; fast
  builder và `run_backtest` tính % nên không cần. Thấp.
- **Thanh khoản/participation:** không đường legacy nào có trần % ADV; chỉ có lọc
  `min_avg_volume_20: 100000` cổ phiếu (`rules_t2.json:14`). Bộ momentum events của gate có
  "prior-bar volume caps" (audit execution-receipts `:161-162`), Broker MR thì không. Với vốn 1 tỷ
  trên VN30: Thấp; với VN100 mid/small cap hoặc vốn lớn hơn: TB-Cao; chiều (+).
- **Phí và thuế:** 0,15%/chiều + thuế bán 0,1% (`backtest.py:20-21`, `rules_t2.json:65-66`) khớp
  thông lệ VN (thuế TNCN 0,1% giá trị bán — chưa verify bằng nguồn sơ cấp trong rollout này, đánh
  dấu UNVERIFIED). `run_backtest` trừ thuế như % notional mua (`backtest.py:352`) — sai số bậc hai,
  Thấp.
- **Slippage — bốn dạng khác nhau** [C14]: cộng dồn 0,2% (fast builder `parallel_engine.py:318`,
  `research_ml/labels.py:72-73`), 1 tick (`backtest.py:345-349`, `:663,734`), nhân 0,1%
  (`period_backtest.py:125,140`). Một tick HOSE = **0,08%–0,50%** giá tùy vùng giá
  (`trajectory/tick_cost.out`: 10.050 VND → 0,498%; 25.000 → 0,200%; 120.000 → 0,083%;
  bảng tick 10/50/100 VND lấy từ code `backtest.py:140-148`, nguồn web chưa mở được → UNVERIFIED).
  Với cổ phiếu 10–30 nghìn (nhiều mã ngân hàng/thép) 0,1%/chiều **thấp hơn nửa spread tối thiểu**
  một tick. Mức độ: TB; chiều (+) cho mã giá thấp. Ngoài ra `run_backtest` báo
  `round_trip_cost_pct` 0,6% (`:367,379`) nhưng thực trừ tick + 0,4% (`:352`) — báo cáo sai lệch
  với chính nó.

### 2.5 Survivorship / PIT universe

- Fast builder gắn nhãn chính nó là không-PIT (`parallel_engine.py:364`); `run_portfolio_backtest`
  lấy các CSV đang tồn tại (`backtest.py:529-537`). Research gate dùng PIT VN30
  (`research_gate.py:137,227-229`).
- Bằng chứng định lượng đã có (memory `momentum-excess-gate.md`): cùng engine momentum, rổ VN30
  hiện tại áp ngược **+190% / Sharpe 0,88**, 25 rổ ngẫu nhiên median **+121%**, rổ PIT động
  **+33% / Sharpe 0,31** [C20]. Mức độ: **Cao**; chiều (+). Quy tắc: mọi kết quả trên rổ cố định
  hiện tại phải coi là cận trên.

### 2.6 Corporate actions / giá điều chỉnh

- Giá là bản điều chỉnh của nhà cung cấp tại thời điểm tải, không có lineage
  (`strategy-reassessment.md:179-180`, `2026-09-26-h1-2026-backtest.md:103-105`); không có sổ cổ
  tức tiền mặt (`period_backtest.py:377`). Fast builder chỉ loại cửa sổ có bước nhảy > 15%
  (`parallel_engine.py:279-281`, `validation.py:82-83`) — lỗi điều chỉnh < 15% lọt qua.
- Hệ quả thêm (suy luận): giá điều chỉnh lùi cổ tức tiền mặt làm return chiến lược gần
  total-return, trong khi benchmark VNINDEX là **chỉ số giá** (`period_backtest.py:351-353`,
  `strategy-reassessment.md:36-38`) → so sánh excess lệch (+) cho chiến lược khoảng tỷ suất cổ tức
  × exposure. Giá điều chỉnh cũng không nằm trên lưới tick, nên slippage tick tính trên giá
  không có thật. Mức độ: TB; chiều (+) cho excess.

### 2.7 Look-ahead

- **Sizing bằng close cùng ngày:** vòng lặp portfolio mark-to-market ở **close** của
  `current_date` (`backtest.py:438`) rồi dùng equity đó để chia vốn cho lệnh khớp ở **open** cùng
  ngày (`:439-450, 738-742`). Broker làm đúng: dùng NAV phiên trước (`period_backtest.py:231,
  249-253`). Mức độ: TB; chiều (+).
- **Lịch giao dịch trước 2021 thiếu ngày nghỉ:** `HOSE_HOLIDAYS` bắt đầu từ 2021
  (`exchange_calendar.py:20-24`). `is_trading_day(2020-01-24)` trả True (`exit_equivalence.out`
  [3a]). Fast builder tính exit theo lịch (`parallel_engine.py:266-269`): trên dữ liệu synthetic
  quanh Tết 2020 (giả định sàn nghỉ 23–29/01/2020 — kiến thức chung, chưa verify nguồn sơ cấp)
  5/30 row bị **bỏ im lặng**, 10/30 row có horizon thực < 15 bar [C15]. Đây không phải look-ahead
  nhưng là bias chọn mẫu theo lịch; chiều (±).
- **ML override trong `run_backtest`** đã được chặn bởi `model_available_at` (`ml_models.py:231-
  233`), nên không look-ahead, nhưng kết quả phụ thuộc artifact đang nằm trên đĩa → không tái lập.
  Thấp.
- Feature causal: đã có test (`tests/test_backtest_leakage.py`, remediation audit
  `2026-09-22-training-remediation.md:40`). Thấp (đã giảm thiểu).

### 2.8 Kết thúc mẫu, pending, forced close

- `run_portfolio_backtest` ép bán tất cả ở close ngày cuối và ghi thành trade `FORCED_CLOSE`
  (`backtest.py:465-470,635-651`), trộn vào win rate/expectancy. **Lỗi:** vị thế không có bar ở
  ngày cuối (tạm ngừng) không được bán nhưng `equity_curve[-1].equity = cash` (`:468-470`) xóa
  giá trị của nó: synthetic `[C]` NAV giảm **−24,97%** trong một ngày không có biến động giá
  [C16]. Tái lập độc lập với vị thế lớn (synthetic `fals/scratch/pb.py`, vốn 1.000 triệu, mã CCC
  hết dữ liệu 5 phiên trước ngày cuối): equity từ 998.331.920 xuống 98.490.032,5 VND trong ngày
  cuối, tức **~900 triệu trên 1.000 triệu biến mất**, CCC không có trade nào được ghi [C16]. Mức độ:
  **Cao** khi xảy ra; chiều (−) (lỗi), nhưng `FORCED_CLOSE` nói chung (±).
- Fast builder bỏ row thiếu exit im lặng (`parallel_engine.py:267-269`) và gắn
  `label_resolved=True` cho mọi row (`:330`) → giai đoạn cuối mẫu (regime gần nhất) bị cắt.
  `research_ml.labels` xử lý đúng (`labels.py:59-71`). Broker không ép bán (`period_backtest.py:
  369`). TB; (±).

### 2.9 Kế toán tiền/vị thế và chỉ số

- Portfolio: khi thiếu bar, exposure mark ở `entry_value` (`backtest.py:778-780`) → che drawdown
  trong thời gian tạm ngừng (+).
- Chỉ số trade-level: `_sharpe_ratio` luôn nhân `sqrt(252/2)` (`backtest.py:209,222`). Với hold 20
  phiên (`rules_t2.json:3`), Sharpe bị phóng đại **×3,16** (synthetic `[E]`: 1,34 so với 0,42)
  [C17]. `_max_drawdown` nhân dồn return từng trade như thể toàn bộ vốn vào mỗi trade
  (`:171-180`) — không phải drawdown danh mục. `recall` mặc định bằng `precision` vì
  `opportunity_count` mặc định = `selected` (`:235-236`). Mức độ: TB-Cao cho báo cáo; chiều (+).
- Broker đã được đối soát độc lập NAV/tiền/lô với sai số < 1e-6 VND
  (`strategy-reassessment.md:11-13`) — đây là chuẩn nên giữ.
- **Broker crash khi đang giữ một mã có khoảng ngừng giao dịch đã đăng ký:** `replay_mr` chỉ dựng
  `bars` cho mã có bar trong ngày (`scripts/period_backtest.py:235`), rồi với **mọi** vị thế đang
  giữ đọc `bar = bars[symbol]` (`:259`), và `mark()` đọc `bars[s]['close']` (`:167`) ⇒ `KeyError`
  khi mã đang giữ không có bar. `validate_frames` (`:45-49`) lại chấp nhận đúng các khoảng trống
  trong `SYMBOL_NONTRADING_INTERVALS` (`exchange_calendar.py:126-132`: BSR 07–16/01/2025, MCH
  18–24/12/2025), và cả hai nằm trong cửa sổ gate 2022-09-01..2026-09-29 (`research_gate.py:233`).
  Synthetic (`adj/scratch/broker.py`: AAA thiếu 3 phiên trong lúc giữ, VNINDEX đầy đủ) ⇒
  `EXC KeyError 'AAA'` [C25]. Mức độ: **TB**, kiểu **fail-closed** (crash, không phải bias im
  lặng); chiều: không lệch kết quả, chặn bằng chứng. Việc run gate đã đăng ký có thực sự giữ BSR
  hoặc MCH qua khoảng đó hay không là **UNVERIFIED** (cần pinned snapshot, không có ở đây). Nếu có,
  mọi lần chạy gate 09/2022–09/2026 đều crash và không tạo được bằng chứng (khi đó mức độ là Cao vì
  chặn toàn bộ gate); nếu không, lỗi ở trạng thái tiềm ẩn và chỉ nổ khi universe/tín hiệu thay đổi.

### 2.10 Multiple testing / overfitting

- Optimizer có không gian tìm kiếm gồm cả `time_horizon_days` 2–30 và hàng chục ngưỡng
  (`optimizer.py:67-75`); `rules_t2.json` mang dấu vân tay của tối ưu hóa: `volume_surge_min
  1.647285`, `target_rr 2.546033`, `max_position_weight 0.296294` (`rules_t2.json:22,36-37`).
  Memory ghi "3 năm optimizer trials không đạt target" (`strategy-edge-findings.md`) và nhiều
  biến thể đã thử trên cùng mẫu (`portfolio-power-verdict.md`: "~10 strategy variants tried on the
  same 2020-26 data"). Số trial lịch sử không biết → không thể tính DSR toàn cục
  (`strategy-reassessment.md:111-112`). Mức độ: **Cao**; chiều (+).
- Research gate đã làm đúng phần có thể: registry trước, Bonferroni 22 so sánh, bootstrap khối
  20/40 (`research_gate.py:51-68,246`; `strategy-reassessment.md:104-110`). Thiếu: control kiểu
  "shuffled label"/exposure-matched cho MR (R5 weekly cho thấy control xáo nhãn còn "thắng" model:
  `2026-10-04-r5-weekly-results.md:23,30`) [C21].
- **Fast builder gắn nhãn mọi bar, cửa sổ chồng lấn:** `build_labeled_dataset_fast`
  (`parallel_engine.py:257-341`) gắn nhãn mọi ngày tín hiệu có bar entry và bar exit, bất kể
  `signal.decision` (không có bộ lọc BUY_SETUP trước khi chấm điểm và gắn nhãn; `decision` chỉ được
  lưu thành cột ở `:332`), với horizon 15–20 phiên chồng lấn nhau. Vì vậy các nhãn này **không phải
  kết quả trade của chiến lược**, và chúng phụ thuộc chuỗi (serially dependent). Đánh giá cần purge/embargo theo `label_available_date`, như audit R3 và
  remediation mô tả (`2026-09-22-training-remediation.md:13,30`) [C26]. Mức độ: TB; chiều (+):
  thổi phồng cỡ mẫu hiệu dụng và mức ý nghĩa nếu tính như quan sát độc lập.

### 2.11 Bảng tổng hợp

| # | Giả định | Đường bị ảnh hưởng | Mức độ | Chiều |
|---|---|---|---|---|
| 1 | Bán được cả ngày T+2 (trước 2022: T+3) | mr_exit, fast builder, run_backtest | TB-Cao (ngắn hạn) | + |
| 2 | Tiền bán dùng ngay, mua cùng sáng | run_portfolio_backtest | Cao | + |
| 3 | Stop khớp ở mức stop dù gap | mr_exit + mọi caller trừ paper_scoring, Broker | TB | + |
| 4 | Stop chỉ xét ngày planned exit; khớp trên high | run_portfolio_backtest | Cao | + (±) |
| 5 | Clamp ±6,9% thay vì unfillable; không xét khóa sàn | fast builder, run_backtest, label_resolver, portfolio | TB-Cao (khủng hoảng) | + |
| 6 | Không trần participation | tất cả | Thấp (VN30, 1 tỷ) → Cao (VN100/vốn lớn) | + |
| 7 | Slippage 0,1% < 1 tick ở giá thấp; 4 dạng khác nhau | tất cả | TB | + |
| 8 | Universe hiện tại áp ngược | fast builder, portfolio, scripts cũ | Cao | + |
| 9 | Giá điều chỉnh vs chỉ số giá | tất cả so với VNINDEX | TB | + (excess) |
| 10 | Sizing bằng close cùng ngày | run_portfolio_backtest | TB | + |
| 11 | Lịch thiếu ngày nghỉ trước 2021 | fast builder, label_resolver | TB | ± |
| 12 | Forced close / xóa vị thế thiếu bar | run_portfolio_backtest | Cao khi xảy ra | − / ± |
| 13 | Sharpe sqrt(126) mọi horizon; DD trade-level | run_backtest, portfolio | TB-Cao (báo cáo) | + |
| 14 | Multiple testing không đếm được | optimizer, scratch research | Cao | + |
| 15 | Train/live/eval khác target | fast builder vs label_resolver vs Broker | Cao (ML) | ± |
| 16 | Vị thế giữ qua khoảng ngừng giao dịch đã đăng ký ⇒ `KeyError` | Broker (`replay_mr`, `mark`) | TB (fail-closed; reachability UNVERIFIED) | crash, không lệch im lặng |
| 17 | Gắn nhãn mọi bar (không lọc BUY_SETUP), cửa sổ 15–20 phiên chồng lấn | fast builder | TB | + (cỡ mẫu/ý nghĩa) |

## 3. Q2 — T+2 có hiệu quả dài hạn và trong thực tế?

**Trả lời thẳng: Không có bằng chứng; bằng chứng hiện có nghiêng về "không".**

1. **Tên gọi đang đánh lừa.** Cấu hình mặc định là `rules_t2.json` (`config.py:21`) với
   `time_horizon_days: 20` (`rules_t2.json:3`), và `BacktestConfig.holding_days` lấy từ khóa này
   (`backtest.py:41`). Giá trị mặc định 2 (`backtest.py:24`) chỉ dùng khi rules thiếu khóa. Vậy
   `net_t2_return_pct` của fast builder là lợi nhuận bracket **20 phiên** với lock 2 bar
   (`parallel_engine.py:266,338-340`). Optimizer được phép chọn 2–30 (`optimizer.py:68`) và cấu
   hình đang dùng là 20: chính quá trình tối ưu đã rời xa 2 phiên (suy luận, không phải phép thử)
   [C03, C04]. Plan R-roadmap cũng đã ghi `net_t2_*` "không tự coi là target phù hợp"
   (`docs/audits/2026-10-04-data-ml-research-plan.md:25`).
2. **Thực tế vi cấu trúc VN.** Mua phiên T, sớm nhất bán phiên chiều T+2 (từ 29/08/2022; trước đó
   T+3) [C30]; quy tắc này vẫn áp dụng năm 2026 (phân bổ phiên chiều T+2, The Investor
   27/02/2026 [C35]). Một "lệnh T+2" thật sự là ≥ 2,5 phiên với **hai đêm không thể thoát**, không
   có T+0 (chưa triển khai tính đến 06/2026, sớm nhất H2-2026 [C32, C36]), retail không short được. Biên ±7% chặn biến
   động một ngày nhưng cũng tạo ngày khóa sàn không bán được — đúng lúc stop cần nhất.
3. **Tín hiệu ở horizon 2 ngày ≈ nhiễu.** Memory `strategy-edge-findings.md`: Rank IC tại T+2 gần
   như mọi feature ≈ 0; các feature scorecard momentum (breakout, RSI cao) **âm** ở T+2 (rsi2
   −0,032, ret_5d −0,026, breakout −0,021); dấu IC đổi theo horizon (T+10: adx14 +0,031, rsi14
   +0,027) [C05]. Scorecard 18 luật vì thế "đặt cược sai chiều ở chính horizon của nó".
4. **Chi phí lớn hơn edge.** Round-trip 0,6% (`round_trip_cost_pct`, `tick_cost.out`) là ngưỡng
   hòa vốn **mỗi lệnh**, bất kể giữ 2 hay 15 phiên. Memory: MR lỏng RSI2<5 thắng > 53% nhưng **lỗ
   ròng** ở 0,6% (`strategy-edge-findings.md`); swing RSI2 (giữ ~3–8 phiên) chỉ +0,52%/lệnh ở
   0,4% và +0,3% ở 0,6% (`swing-rsi2-edge.md`) [C06]. R5 weekly: lô 5 phiên trả ~0,6% so với biến
   động universe ~0,3%/tuần, ở chi phí ×2 mọi excess âm hoặc ≈ 0
   (`2026-10-04-r5-weekly-results.md:35`) [C07]. Số học đơn giản: nếu quay vòng mỗi ~2,5 phiên,
   ~100 vòng/năm × 0,6% ≈ 60% chi phí/năm trên vốn quay vòng — không edge ngắn hạn nào trong
   memory gần với mức đó.
5. **Bằng chứng danh mục dài hạn (development, không phải holdout).** Research gate liên tục
   09/2022–09/2026: MR brackets +7,904%, MR fixed15 +3,087% so với VNINDEX giá +38,857%; fixed15
   chỉ 27 lô đóng < sàn 30 (`strategy-reassessment.md:74-76,92-93`) [C08]. Không có biến thể
   "giữ 2 phiên" nào trong registry (`research_gate_v1.json` "mr": fixed5/10/15/20) — nghĩa là
   chưa từng có phép thử T+2 đạt chuẩn gate. Giai đoạn 2020–2026 là đại bull nên mọi chiến lược
   ít phơi nhiễm đều thua B&H (`portfolio-power-verdict.md`).
6. **Ý nghĩa đúng của T+2:** một **ràng buộc settlement** (không thoát được trước chiều T+2) phải
   có trong mọi simulator; không phải horizon mục tiêu. Nếu muốn kiểm tra horizon ngắn, đăng ký
   một giả thuyết mới (ví dụ fixed3 tương ứng "bán chiều T+2/sáng T+3") trong research gate trước
   khi xem kết quả, với sàn số lệnh và activity floor đã chốt.

## 4. Q3 — Nghiên cứu về thời điểm thoát lệnh / holding period / time stop

Quy ước: **[V]** = tôi đã mở và xác nhận metadata (Crossref API hoặc trang RePEc/NBER/nhà xuất
bản); nội dung tóm tắt lấy từ abstract đã mở. **[V-meta]** = chỉ xác nhận được metadata, nội dung
nêu ở mức tiêu đề → coi phần nội dung là UNVERIFIED. Danh sách kiểm tra:
`trajectory/crossref_check.txt`, `trajectory/abstracts.txt`.

### 4.1 Thị trường phát triển (chủ yếu US)

| Trích dẫn | Phát hiện liên quan tới horizon/exit |
|---|---|
| Lehmann, B. N. (1990). Fads, Martingales, and Market Efficiency. *QJE* 105(1), 1–28. doi:10.2307/2937816 [V] | Reversal **theo tuần**: winners/losers tuần này đảo chiều tuần sau; tác giả cho rằng lợi nhuận còn sau hiệu chỉnh bid-ask và chi phí "hợp lý" (abstract NBER w2533). |
| Jegadeesh, N. (1990). Evidence of Predictable Behavior of Security Returns. *J. Finance* 45(3), 881–898. doi:10.1111/j.1540-6261.1990.tb05110.x [V] | Tự tương quan âm bậc 1 của return **tháng** rất có ý nghĩa; tự tương quan dương ở lag dài (12 tháng). |
| Lo, A. W., & MacKinlay, A. C. (1990). When Are Contrarian Profits Due to Stock Market Overreaction? *RFS* 3(2), 175–205. doi:10.1093/rfs/3.2.175 [V] | Một phần lớn lợi nhuận contrarian tuần đến từ **lead-lag** (mã lớn dẫn mã nhỏ), không chỉ overreaction. |
| Avramov, D., Chordia, T., & Goyal, A. (2006). Liquidity and Autocorrelations in Individual Stock Returns. *J. Finance* 61(5), 2365–2394. doi:10.1111/j.1540-6261.2006.01060.x [V] | Reversal ngắn hạn lớn nhất ở mã **kém thanh khoản**; lợi nhuận contrarian **nhỏ hơn chi phí giao dịch khả dĩ**. |
| Nagel, S. (2012). Evaporating Liquidity. *RFS* 25(7), 2005–2039. doi:10.1093/rfs/hhs066 [V] | Return reversal ngắn hạn = phần thưởng cung cấp thanh khoản; dự báo được bằng VIX, tăng vọt khi thị trường căng thẳng. |
| Novy-Marx, R., & Velikov, M. (2016). A Taxonomy of Anomalies and Their Trading Costs. *RFS* 29(1), 104–147. doi:10.1093/rfs/hhv063 [V] | Chiến lược turnover một chiều < 50%/tháng thường còn ý nghĩa sau chi phí; **ít** chiến lược turnover cao còn ý nghĩa (abstract NBER w20721). |
| Jegadeesh, N., & Titman, S. (1993). Returns to Buying Winners and Selling Losers. *J. Finance* 48(1), 65–91. doi:10.1111/j.1540-6261.1993.tb04702.x [V] | Momentum có lợi nhuận ở holding **3–12 tháng**; một phần tan biến trong 2 năm sau. |
| Avellaneda, M., & Lee, J.-H. (2010). Statistical arbitrage in the US equities market. *Quant. Finance* 10(7), 761–782. doi:10.1080/14697680903124632 [V] | Mô hình phần dư như quá trình mean-reverting (OU); hiệu quả phụ thuộc tốc độ và độ bền của mean reversion theo chu kỳ thị trường. |
| Bertram, W. K. (2010). Analytic solutions for optimal statistical arbitrage trading. *Physica A* 389(11), 2234–2243. doi:10.1016/j.physa.2010.01.045 [V] | Với giá OU: công thức đóng cho **thời lượng lệnh kỳ vọng** (first-passage time) và ngưỡng vào/ra tối ưu theo expected return hoặc Sharpe. |
| Leung, T., & Li, X. (2015). Optimal Mean Reversion Trading with Transaction Costs and Stop-Loss Exit. *IJTAF* 18(3), 1550020. doi:10.1142/S021902491550020X [V] | Bài toán optimal double stopping (thời điểm vào và thoát) dưới chi phí; stop-loss cao hơn ⇒ take-profit tối ưu thấp hơn. |
| Kaminski, K. M., & Lo, A. W. (2014). When do stop-loss rules stop losses? *J. Financial Markets* 18, 234–254. doi:10.1016/j.finmar.2013.07.001 [V] | Khung phân tích: giá trị của stop-loss phụ thuộc vào động học return (tính dự báo/serial correlation) và tần suất; với một số tần suất dài hơn, stop-loss có thể tăng expected return và giảm volatility (trang RePEc). Abstract bản working paper MIT: dưới Random Walk Hypothesis, stop-loss 0/1 đơn giản **luôn làm giảm** expected return; khi có momentum, stop-loss có thể tăng giá trị [V qua abstract working paper; full-text chưa đọc]. |
| López de Prado, M. (2018). *Advances in Financial Machine Learning*. Wiley. ISBN 978-1-119-48208-6 [V-meta: mục lục có Chương 3 "Labeling"] | Triple-barrier labeling (stop/target/time) và meta-labeling: nhãn phải mô phỏng đúng đường đi giá và horizon. Tên phương pháp xác nhận qua nguồn thứ cấp, không qua mục lục → UNVERIFIED ở mức nội dung. |
| Bailey, D. H., & López de Prado, M. (2014). The Deflated Sharpe Ratio. *JPM* 40(5), 94–107. doi:10.3905/jpm.2014.40.5.094 [V-meta] | Hiệu chỉnh Sharpe cho selection bias/số trial (theo tiêu đề). |
| Bailey, D. H., Borwein, J., López de Prado, M., & Zhu, Q. J. (2016). The probability of backtest overfitting. *J. Computational Finance*. doi:10.21314/JCF.2016.322 [V-meta] | PBO (theo tiêu đề). |
| Harvey, C. R., Liu, Y., & Zhu, H. (2016). …and the Cross-Section of Expected Returns. *RFS* 29(1), 5–68. doi:10.1093/rfs/hhv059 [V-meta] | Ngưỡng t-stat cao hơn khi có nhiều phép thử (theo tiêu đề). |

### 4.2 Emerging / frontier, Việt Nam

| Trích dẫn | Phát hiện |
|---|---|
| Rouwenhorst, K. G. (1999). Local Return Factors and Turnover in Emerging Stock Markets. *J. Finance* 54(4), 1439–1464. doi:10.1111/0022-1082.00151 [V] | Emerging markets có momentum, size, value tương tự developed; liên hệ mạnh với turnover. (Mẫu trước khi HOSE mở cửa năm 2000 → không có VN.) |
| Griffin, J. M., Ji, X., & Martin, J. S. (2003). Momentum Investing and Business Cycle Risk: Evidence from Pole to Pole. *J. Finance* 58(6), 2515–2547. doi:10.1046/j.1540-6261.2003.00614.x [V] | Momentum quốc tế lớn trong cả trạng thái kinh tế tốt và xấu; **đảo chiều ở horizon 1–5 năm**. |
| Chui, A. C. W., Titman, S., & Wei, K. C. J. (2010). Individualism and Momentum around the World. *J. Finance* 65(1), 361–392. doi:10.1111/j.1540-6261.2009.01532.x [V] | Lợi nhuận momentum tương quan dương với chỉ số individualism (Hofstede); thấp hơn ở các nước ít individualism. |
| Chan, S. H., Kim, K. A., & Rhee, S. G. (2005). Price limit performance: evidence from transactions data and the limit order book. *J. Empirical Finance* 12(2), 269–290. doi:10.1016/j.jempfin.2004.01.001 [V-meta] | Ảnh hưởng của biên độ giá tới price discovery/order imbalance (abstract không mở được → nội dung UNVERIFIED). |
| Truong, L. D., Cao, G. N., Friday, H. S., & Doan, N. T. (2023). Overreaction in a Frontier Market: Evidence from the Ho Chi Minh Stock Exchange. *IJFS* 11(2), 58. doi:10.3390/ijfs11020058 [V] | HOSE 392 mã, dữ liệu **tháng** 01/2004–06/2021: danh mục loser vượt winner 1,80% và 2,17% ở tháng thứ 2 và 3 → contrarian theo tháng. |
| Vo, X. V., & Truong, Q. B. (2018). Does momentum work? Evidence from Vietnam stock market. *J. Behavioral and Experimental Finance* 17, 10–15. doi:10.1016/j.jbef.2017.12.002 [V] | HOSE 06/2007–10/2015: chiến lược chọn danh mục theo 6 tháng trước và **giữ 9 tháng** có lợi nhuận đáng kể → momentum có ở VN, nhưng ở horizon **tháng**. |
| Le Quy Duong, & Bertrand, P. (2023). Overreaction and momentum in the Vietnamese stock market. *Managerial Finance* 49(1), 13–28. doi:10.1108/MF-01-2022-0013 [V] | Dữ liệu **tuần** 2009–2019, > 300 cổ phiếu phi tài chính: có momentum; lợi nhuận momentum mất ý nghĩa sau khi điều chỉnh overreaction → momentum ở VN đến từ overreaction của nhà đầu tư. |
| Vu, L. T., Nguyen, M. P., & Hoang, Q. A. (2026). An integrated momentum strategy based on entropy and behavioral overreaction: Evidence from Vietnam. *Investment Management and Financial Innovations* 23(1), 154–171. doi:10.21511/imfi.23(1).2026.12 [V] | VNINDEX 2015–2025: momentum **ngắn hạn** tồn tại; horizon trung/dài cho overreaction âm, tức **mean reversion**. Sharpe 3,96 (so với 0,64 của thị trường) đến từ backtest 2023–2025: cửa sổ ngắn, rủi ro overfit cao, **không** dùng làm benchmark. |

Tôi **không tìm được** nghiên cứu peer-reviewed nào (mở được) ước lượng horizon thoát tối ưu
hoặc time-stop cho mean reversion ngày/tuần trên HOSE có tính tới T+2/T+3 và biên ±7% (kết luận
dựa trên một lượt tìm kiếm, không phải tìm kiếm hệ thống). Bằng chứng VN đã verify ở trên đều ở
horizon tuần/tháng/nhiều năm, không nói về exit theo ngày.

### 4.3 Cái gì chuyển giao được, cái gì không

**Chuyển giao được (cơ chế):**
- Horizon tối ưu là **nội sinh**: được xác định bởi tốc độ suy giảm tính dự báo (half-life OU,
  Bertram 2010; Leung & Li 2015) so với chi phí cố định mỗi vòng. Đây chính là thứ repo nên ước
  lượng (IC theo horizon, như memory đã làm) thay vì cố định "T+2".
- Reversal ngắn hạn tập trung ở mã kém thanh khoản và bị chi phí ăn (Avramov et al. 2006;
  Novy-Marx & Velikov 2016) → khớp với kết quả nội bộ "RSI2 lỏng thắng > 53% nhưng lỗ ròng".
- Reversal là phần thưởng cung cấp thanh khoản, tăng trong căng thẳng (Nagel 2012) → khớp với
  memory: edge MR tập trung trong khủng hoảng 2022/2023 (`win-probability-metalabeling.md`).
- Stop-loss không phải "quản trị rủi ro miễn phí": giá trị phụ thuộc động học return (Kaminski &
  Lo 2014) → khớp memory "the REAL risk manager is the time-stop + entry gate, not the ATR stop".
- Nhãn phải mô phỏng đúng barrier + time (triple-barrier) và phải đếm số trial (DSR/PBO/HLZ).

**Không chuyển giao (hoặc phải kiểm lại):**
- Độ lớn lợi nhuận và Sharpe của US (Avellaneda & Lee: Sharpe 1,1–1,5) dựa trên long-short,
  thanh khoản cao, T+0/T+1 và short selling — retail VN không short được, không bán trong T+2.
- Reversal tuần của US dựa trên bán được ngay; ở VN lock T+2 cắt mất 2 trong 5 phiên đầu.
- Biên độ ±7% và khóa sàn biến stop thành lệnh "có thể không khớp" — mô hình OU/stop liên tục
  không mô tả được.
- Momentum quốc tế (3–12 tháng) là bằng chứng gián tiếp; nghiên cứu VN cho kết quả **lẫn lộn
  theo horizon**: Truong et al. (2023) contrarian ở tháng 2–3; Vo & Truong (2018) momentum 6/9
  tháng; Le Quy Duong & Bertrand (2023) momentum tuần gắn với overreaction; Vu et al. (2026)
  momentum ngắn hạn và mean reversion ở horizon dài hơn. Dữ liệu tuần/tháng không nói gì về exit
  theo ngày.

## 5. Q5 — Khuyến nghị thiết kế portfolio backtest (ưu tiên)

Nguyên tắc: **một engine thực thi duy nhất** cho đánh giá danh mục, nhãn train và replay
paper; mọi khác biệt là tham số có tên và có test. Không khẳng định lợi nhuận nào.

**P0 — làm trước, không đổi kinh tế của gate:**
1. **Đừng sửa `run_portfolio_backtest` tại chỗ; hãy hạ cấp nó.** Đánh dấu deprecated, dẫn sang
   `scripts/period_backtest.Broker` (đã reconcile, T+3, gap fill, opening gate, không ép bán).
   Nếu cần giữ API, bọc Broker. Bốn lỗi ở mục 2.2/2.8/2.9 (stop chỉ ngày cuối, khớp trên high,
   xóa vị thế thiếu bar, Sharpe sqrt(126)) là lý do không vá từng chỗ [C12, C13, C16, C17].
   - **Sửa Broker trước khi dựa vào nó:** vị thế của mã không có bar trong ngày (khoảng ngừng giao
     dịch đã đăng ký) phải được giữ nguyên số lượng và mark ở close cuối cùng; exit và kiểm tra
     fillability hoãn sang bar kế tiếp của mã đó, thay vì `KeyError` ở `period_backtest.py:259`
     và `:167`. Thêm test synthetic giữ một mã xuyên qua một khoảng trong
     `SYMBOL_NONTRADING_INTERVALS` [C25]. Đây là thay đổi hành vi của Broker nên đi cùng
     preregistration như mục P1.
   - **Phạm vi thay đổi:** `optimizer.py`, `robustness.py`, `cli.py`, `app.py` đều gọi
     `run_portfolio_backtest` (trực tiếp hoặc qua `run_portfolio_backtest_from_csvs`) [C27]. Đổi
     logic portfolio nghĩa là phải **re-baseline** các output của optimizer/robustness (tham số
     đã tối ưu, split train/validation/test, kịch bản robustness) và viết test hành vi (behaviour-
     change tests) ghi rõ output nào được phép đổi, như một phần của cùng thay đổi.
2. **Tách "exit rule" khỏi "fill model" khỏi "cost model"** thành ba hàm thuần có tên:
   `exit_rule` (= `simulate_mr_exit`, đã thống nhất bởi patch), `fill(bar, side, level, policy)`
   (open-gate, `min(stop, open)`, locked-bar ⇒ unfillable/deferred), `cost(notional, side, cfg)`
   (một công thức dùng chung cho Broker, nhãn, paper). Audit R3 đề xuất `costs.py` tương tự
   (`2026-10-06-r3-labels-cost-parity.md:105`).
3. **Invariant test (RED/GREEN)**: (a) cash không âm, tổng lô bội số 100; (b) NAV = cash +
   receivables + Σ qty×close, đối soát độc lập mỗi ngày; (c) không lệnh bán nào trước ngày
   available; (d) giá khớp ∈ [low, high] của bar và ∈ [sàn, trần] tính từ tham chiếu; (e) không
   dùng dữ liệu của ngày D (close/high/low/NAV) cho quyết định khớp ở open ngày D; (f) vị thế
   tạm ngừng giữ nguyên số lượng và mark ở giá cuối, không biến mất; (g) prefix-invariance: cắt dữ
   liệu sau ngày D không đổi mọi fill ≤ D; (h) parity: cùng một trade synthetic qua Broker và qua
   label builder cho cùng exit_idx/fill (đúng RED test audit R3 đề xuất).

**P1 — thay đổi ngữ nghĩa (mỗi thay đổi = một hypothesis data-contract trong
`configs/research/research_gate_v1.json`, version protocol, đăng ký trước khi xem kết quả):**
4. **Settlement theo ngày hiệu lực:** T+3 (sáng) cho giao dịch trước 29/08/2022, "chiều T+2" sau
   đó. Với dữ liệu ngày, mô hình bảo thủ: ở bar T+2 chỉ cho phép thoát theo **close** (ATC/phiên
   chiều), không dùng high/low của cả ngày; stop/target nội phiên từ T+3. Tiền bán về theo cùng
   lịch; không dùng tiền chưa về.
5. **Unfillable thay vì clamp:** mua ở open ≥ trần hoặc bán ở open ≤ sàn ⇒ không khớp; bar khóa
   (high == low ở trần/sàn) ⇒ không khớp phía bị khóa, lệnh chuyển sang phiên sau và ghi trạng
   thái `exit_deferred`. Biên 20% cho ngày đầu niêm yết/sau tạm ngừng > 25 ngày/ngày GDKHQ cổ
   tức bằng cổ phiếu quỹ cần giá tham chiếu chính thức — khi chưa có, đánh dấu `reference_unknown`
   và loại khỏi thống kê thay vì đoán.
6. **Slippage theo tick:** tối thiểu nửa spread = 1 tick ở giá **không điều chỉnh**, cộng hệ số
   participation (ví dụ `k × sqrt(qty/ADV20)`), trần participation (ví dụ ≤ 5–10% ADV20 bar trước)
   — các con số là tham số cần đăng ký, không phải khuyến nghị đã kiểm chứng.
7. **Horizon theo bar giao dịch thực của mã** (phiên có giao dịch), không theo lịch tĩnh; sửa
   `HOSE_HOLIDAYS` trước 2021 hoặc từ chối dữ liệu trước ngày đầu của bảng lịch.
8. **Sizing bằng NAV phiên trước** (Broker đã làm), exposure theo giá phiên trước.

**P2 — đo lường và báo cáo:**
9. Sharpe/Sortino/drawdown chỉ từ **chuỗi NAV ngày**; bỏ Sharpe theo trade với hệ số cố định.
10. Benchmark song song: VNINDEX giá (như hiện tại) **và** proxy total-return/equal-weight PIT
    có chi phí; control exposure-matched và (học từ R5) control xáo nhãn / ngẫu nhiên cùng
    exposure cho mọi chiến lược chọn mã.
11. Báo cáo activity đúng AGENTS.md: entries flat-to-long, add-on, số ngày vào lệnh theo tháng,
    tháng không có lệnh; tách cash-restart và inherited-portfolio.
12. Universe: chỉ PIT (`vn30_membership_2022_2026.json`); mọi kết quả rổ cố định ghi rõ là cận
    trên survivorship.

**Giữ nguyên:** next-open entry; stop-trước-target cùng bar; không ép bán cuối kỳ (mark-to-close
+ báo cáo phí thoát giả định riêng); FIFO lots; đối soát NAV độc lập; registry trước kết quả;
Bonferroni + block bootstrap; `live_eligible=false` cho đến khi có bằng chứng prospective.

**Thứ tự triển khai đề xuất:** (1) merge patch refactor (không đổi kinh tế) → (2) viết RED test
invariant (a)–(h) trên engine mới → (3) đăng ký hypothesis data-contract cho P1 (expectation:
"replay economics của baseline sẽ thay đổi theo hướng bảo thủ; không promotion") → (4) chạy đầy
đủ `research_gate` + `research_timeslices` trên manifest đã verify, báo cáo cả block BLOCKED.

## 6. Bản đồ claim

- Q1: C01, C02, C10–C27, C30, C33, C34, C35 (sizing bằng close cùng ngày = C18; tick vs slippage = C19;
  clamp/khóa sàn = C22; giá điều chỉnh vs chỉ số giá = C23; universe không-PIT = C24; Broker KeyError = C25; nhãn mọi bar = C26;
  consumer của run_portfolio_backtest = C27).
- Q2: C03–C08, C30–C32, C35, C36. Q3: L01–L21. Q5: dựa trên C10–C18, C21, C22, C25, C27, C49.

## 7. Giới hạn của rollout này

- Không chạy test toàn repo (bản copy thiếu `tests/__init__.py` nên `tests/test_backtest.py` lỗi
  import ở cả base và patched — lỗi môi trường); đã chạy 45 test liên quan, xem
  `mr_exit_resolution.md`.
- Chạy script với `PYTHONPATH=workspace/repo` lúc đầu khiến CPython ghi `__pycache__` vào
  `workspace/repo` (chỉ bytecode, không đổi source); sau đó dùng `python -B`.
- Settlement 2026 đã xác minh qua The Investor 27/02/2026 [C35]; tin 06/2026 về T+0 chỉ thấy qua
  kết quả tìm kiếm, trang chưa mở [C36]; bảng tick hiện hành chưa xác minh bằng nguồn sơ cấp.
