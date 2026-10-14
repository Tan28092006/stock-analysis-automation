# Giải quyết mâu thuẫn `mr_exit.py:7` ↔ vòng exit chép riêng trong fast builder

> Bản cuối VeriHarness 06/10/2026 (Firstmate run `2026-10-06-ml-roadmap`): 3 rollout độc lập → giải bất đồng + đánh đổ đồng thuận (ngữ cảnh tách) → phân xử (base r01, 8 work items) → repair. Coordinator kiểm lại: patch áp vào repo thật trên branch `firstmate/mr-exit-single-path` (commit `8b244f2`), `pytest` 8 file liên quan **86 passed**. Ledger/finish/repair JSON nằm trong scratchpad của phiên, không commit.

Rollout r01 · revision `4403fc7` · bằng chứng chỉ dùng dữ liệu **synthetic**.

## 1. Mâu thuẫn chính xác là gì

`stock_agent/features/mr_exit.py:4-8` nói vòng exit trước đây bị chép ở **bốn** nơi
(win_probability, position_manager, forward_test, features.backtest) và "All four now call
`simulate_mr_exit`". Câu này đúng cho bốn nơi đó, nhưng docstring sai theo ba cách:

1. **Bỏ sót một bản sao còn sống:** `build_labeled_dataset_fast` có vòng lặp riêng
   (`stock_agent/agents/parallel_engine.py:294-312`). Đây chính là builder tạo nhãn train
   `net_t2_*` cho ensemble (`pipeline/daily_runner.py:160-169`, `features/calibration.py:89-96`).
2. **Bỏ sót một caller:** `pipeline/paper_scoring.py:80` (`_outcome`) cũng gọi
   `simulate_mr_exit` (và thêm gap fill `min(stop, open)` ở `:84`, kiểm tra bar khóa ở `:85`).
3. **Ngầm hứa quá mức:** "exit logic identical" không có nghĩa nhãn/backtest/alert nhất quán, vì
   có các đường **không** gọi hàm này với ngữ nghĩa khác hẳn: `run_portfolio_backtest`
   (`backtest.py:605-632`, chỉ xét stop/target ngày planned exit), `period_backtest.replay_mr`
   (`scripts/period_backtest.py:257-271`, T+3 + gap fill + opening gate), `label_resolver`
   (`pipeline/label_resolver.py:100-137`, close-to-close không bracket), `research_ml/labels.py`
   (`:58-73`, fixed horizon không bracket).

## 2. Bảng khác biệt ngữ nghĩa (fast builder cũ vs `simulate_mr_exit`)

| Khía cạnh | Fast builder cũ (`parallel_engine.py`) | `simulate_mr_exit` (`mr_exit.py`) | Khác biệt thực? |
|---|---|---|---|
| Horizon | **Theo lịch**: `exit_date = add_trading_days(entry_date, holding_days)` rồi tra **chính xác** bar (`:266-269`); không có bar ⇒ bỏ row im lặng | **Theo bar**: `entry_idx + max_hold` (`:43-44`) | Có, tùy cách caller truyền `max_hold`. `run_backtest` quy đổi lịch→bar (`backtest.py:322-335`, lấy bar đầu tiên ≥ ngày đích); win_probability/forward_test/paper_scoring đếm bar trực tiếp. Synthetic: tạm ngừng 3 phiên ⇒ horizon lịch = 12 bar so với 15 bar (`exit_equivalence.out` [3b]); Tết 2020 thiếu trong `HOSE_HOLIDAYS` ⇒ 5/30 row bị bỏ, 10/30 row horizon < 15 bar ([3a]) |
| Settle lock | Hardcode `j < entry_idx + 2` ⇒ continue (`:298-301`) | Tham số `settle_lock`, vòng từ `entry_idx + settle_lock` (`:45`) | Không (ở 2) |
| Precedence cùng bar | stop trước, target sau (`:303-312`) | stop trước, target sau (`:46-49`) | Không |
| Guard giá trị falsy | Không guard: `low <= stop_loss`, `high >= take_profit_1` | `if stop and …`, `if target and …` — 0/None vô hiệu chân đó | Chỉ khi target == 0: cũ chốt lời ở **giá 0** tại bar khóa đầu tiên; mới bỏ qua (synthetic [2]). stop = 0, NaN: giống nhau. Không reachable: risk plan luôn có `take_profit_1 ≥ close×1.005 > 0` (`signal_engine.py:148,463`) |
| Time exit khi horizon < lock | `holding_days` 0/1 ⇒ thoát close tại bar trong lock | giống hệt (`:50-51`), không chặn `max_hold < settle_lock` | Không khác nhau, nhưng **cả hai** cho phép bán trong lock (synthetic [2], `hold=1`). Optimizer cho phép tối thiểu 2 (`optimizer.py:68`) nên chưa xảy ra với cấu hình hiện tại |
| Resolved/pending | Luôn `label_resolved=True` (`:330`); horizon vượt dữ liệu ⇒ row bị bỏ ở `:267-269` trước vòng lặp | Trả `resolved=False` + mark ở bar cuối khi horizon vượt dữ liệu (`:50-52`) | Không với cách gọi mới (exit_idx luôn là bar có thật ⇒ luôn resolved) |
| Giá exit | stop/target thô hoặc close tại `exit_idx` | như nhau | Không |
| Clamp biên độ | Ngoài vòng lặp: clamp exit theo close bar trước (`:314-316`) | Không clamp ("callers layer slippage/price-limits/costs", `:27`) | Không — clamp giữ nguyên sau refactor |
| Gap-through | khớp đúng mức stop | khớp đúng mức stop | Không — **cả hai lạc quan**; Broker/paper_scoring dùng `min(stop, open)` |
| Callers | chỉ chính nó | run_backtest, win_probability, position_manager, forward_test, paper_scoring (+ fast builder sau patch) | — |

**Kết luận phân tích:** ở mọi input reachable (stop/target dương, `exit_idx` là bar có thật),
vòng lặp cũ **tương đương chính xác** với
`simulate_mr_exit(data, entry_idx, stop, target, max_hold=exit_idx-entry_idx, settle_lock=2)`.
Khác biệt thật duy nhất giữa "các họ caller" là **cách chọn horizon** (lịch→bar vs đếm bar), và
nó nằm **ngoài** hàm exit.

## 3. Bằng chứng synthetic

`rollouts/r01/trajectory/exit_equivalence.py` (output `.out`):
- [1] 20.000 trade ngẫu nhiên (độ dài 5–60 bar, horizon 0..n, stop/target dương ngẫu nhiên):
  **0 khác biệt** giữa vòng lặp cũ (chép nguyên văn) và lời gọi mới; phân bố lý do T2_CLOSE 9.940,
  TAKE_PROFIT_1 5.115, STOP_LOSS 4.945 (cả ba nhánh được phủ).
- [2] Edge case: target=0 là khác biệt duy nhất; stop=0/NaN, target=NaN giống nhau; hold=1 cả hai
  thoát trong lock; gap-down open 80 < stop 95 cả hai khớp 95.
- [3] Lịch vs bar: Tết 2020 và tạm ngừng như bảng trên.

Test trong patch (`tests/test_fast_builder_exit_parity.py`):
- Trên source **chưa patch**: 2 test cấu trúc FAIL (`calls_canonical_exit_and_has_no_inline_loop`,
  `docstring_lists_every_caller`), 4 test characterization PASS — tức là test end-to-end
  `build_labeled_dataset_fast` trên CSV synthetic (có tạm ngừng 3 phiên) cho **cùng** exit
  date/reason/price/net trước và sau refactor.
- Test `test_zero_target_is_the_one_documented_divergence` ghim khác biệt duy nhất: với
  `take_profit_1 == 0`, vòng lặp cũ "chốt lời" ở giá 0 tại bar đầu tiên sau lock
  (`(3, 0.0, TAKE_PROFIT_1)`), còn lời gọi canonical bỏ qua chân target (falsy guard) và thoát
  theo thời gian (`(4, 104.0, T2_CLOSE)`). Trường hợp này không reachable từ `signal_engine`
  (`take_profit_1 ≥ close×1.005 > 0`, `signal_engine.py:148,463`); test giữ cho khác biệt này có
  chủ ý và nhìn thấy được.
- Trên source **đã patch**: 6/6 PASS.
- Lệnh (từ bản copy sạch đã `git apply`):
  `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. python -B -m pytest -q -p no:randomly -p no:cacheprovider tests/test_fast_builder_exit_parity.py tests/test_mr_exit.py tests/test_ml_features_labels.py tests/test_period_backtest.py tests/test_period_backtest_boundaries.py tests/test_period_backtest_integration.py tests/test_backtest_leakage.py`
  → **45 passed**. `git apply --check` sạch trên bản copy `workspace/repo` (đã bỏ `__pycache__`).
  `tests/test_backtest.py` lỗi `ModuleNotFoundError: tests.test_backtest` ở **cả** base và patched
  (bản copy thiếu `tests/__init__.py`) — lỗi môi trường, không do patch.

## 4. Khuyến nghị

**(a) Làm ngay — refactor giữ nguyên hành vi (patch kèm theo):**
- `parallel_engine.py`: thay vòng lặp `:294-312` bằng lời gọi `simulate_mr_exit(..., max_hold=
  exit_idx - entry_idx, settle_lock=2)`; giữ horizon theo lịch, clamp, cost, schema row y nguyên.
  Nhánh `if not resolved: continue` là phòng thủ, không reachable.
- `mr_exit.py`: viết lại docstring: liệt kê **đủ sáu** caller với cách chọn horizon của từng caller,
  liệt kê bốn đường **không** gọi hàm và vì sao, nói rõ module chỉ sở hữu *exit rule* (không sở
  hữu fill/clamp/settlement tiền/cost), ghi chú daily bar không thể hiện "bán từ chiều T+2", và
  rằng hàm không chặn `max_hold < settle_lock`.
- Test mới: characterization so với oracle vòng lặp cũ (ngẫu nhiên + end-to-end), test cấu trúc
  chống tái xuất hiện vòng lặp chép, test "docstring liệt kê mọi caller" để docstring không trôi
  lại.
- Không đổi target==0: không reachable; ghi trong tài liệu này và ghim bằng test
  `test_zero_target_is_the_one_documented_divergence` để reviewer biết.

**(b) Không làm trong patch này — thay đổi ngữ nghĩa, cần đăng ký research gate:**
1. Gap fill `min(stop, open)` (và `max(target, open)` cho target gap-up) bên trong
   `simulate_mr_exit` hoặc một lớp fill chung → đổi nhãn win_probability, forward_test, run_backtest,
   fast builder, alert position_manager.
2. Settlement theo ngày hiệu lực (T+3 trước 29/08/2022; chiều T+2 sau đó, chỉ cho thoát ở close
   tại bar T+2) và chặn `max_hold < settle_lock`.
3. Unfillable/deferred thay cho clamp; nhận diện bar khóa sàn.
4. Thống nhất horizon (bar giao dịch thực của mã) cho mọi caller + sửa lịch trước 2021.
5. Truyền `cost_config` qua `build_labeled_t2_dataset` (`calibration.py:81-96`), bỏ `COST = 0.60`
   hardcode (`win_probability.py:37`, `forward_test.py:39`).

## 5. Trạng thái research gate (AGENTS.md)

- **(a) refactor**: là thay đổi engineering không đổi kinh tế. AGENTS.md yêu cầu đăng ký giả
  thuyết khi thay đổi "signals, selection, exits, sizing, costs, calendar, or data-contract" và cấm
  xem kết quả mới trước khi đăng ký. Patch này không đổi exit (đã chứng minh tương đương trên mọi
  input reachable), không đổi Broker/`research_gate` (không import `parallel_engine`), nên **không
  cần preregistration**. Vẫn phải: chạy toàn bộ test suite trên repo thật, và — vì AGENTS.md coi
  calendar/exit là vùng nhạy cảm — tốt nhất chạy lại regression 2.618 replay để xác nhận payload
  kinh tế không đổi (kỳ vọng: không đổi, vì các replay đó không đi qua fast builder). Đây là
  khuyến nghị, rollout này không chạy được (không có snapshot).
- **(b) thay đổi ngữ nghĩa**: **bắt buộc** đăng ký một `data_contract_hypotheses` mới trong
  `configs/research/research_gate_v1.json` (hoặc version protocol mới) **trước** khi xem bất kỳ
  kết quả nào, với expectation ghi rõ: nhãn train/P(win)/alert thay đổi theo hướng bảo thủ;
  replay Broker (đã dùng `min(stop, open)` và T+3) không đổi trừ khi chính Broker bị đổi; không
  promotion; dữ liệu đã xem là development. Audit R3 đã phân loại tương tự
  (`docs/audits/2026-10-06-r3-labels-cost-parity.md:147-164`).
- `run_portfolio_backtest` (lỗi ở `backtest_realism_report.md` §2) chạm đường backtest/trading:
  tách task riêng, cần user chấp thuận rõ; khuyến nghị thay bằng Broker thay vì vá.

## 6. Rủi ro còn lại của patch

- `simulate_mr_exit` gọi `.to_numpy()` trên ba cột mỗi lần; với cột float64 đây thường là view
  (không copy) nên chi phí thêm mỗi row nhỏ, nhưng chưa được đo trên dữ liệu kích thước thật.
- Test end-to-end monkeypatch `signal_engine.score_symbol` và `precompute_signal_frames`; nó chứng
  minh vòng exit/clamp/cost, không chứng minh scorer.

## 7. Bản đồ claim

C40 (docstring sai), C41 (sáu caller sau patch, năm trước patch), C42 (đường train), C43 (tương đương
20.000 trade), C44 (target==0), C45 (lịch vs bar), C46 (time exit trong lock), C47 (patch apply +
45 test), C48 (đường không gọi), C49 (quy tắc preregistration), C11 (gap-through), C15 (lịch trước 2021).
