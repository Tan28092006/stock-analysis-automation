# Kiểm tra indicator nâng cao + nguồn dữ liệu của swing desk (T18, scout/verification)

- Ngày: 2026-10-08 (giờ VN ~10:30–11:10; `now_utc` desk = 2026-10-08T03:34Z).
- Revision: `bc8e09e` (branch `codex/data-integrity-audit`). Dirty tree chỉ gồm các file có sẵn từ đầu phiên (docs/audits 2026-10-06/07, `data/pipeline/*.log|cache`, plan doc). Không sửa code, không ghi vào `data/` hay `swing_desk/`.
- Scratch desk: `C:/Users/acer/AppData/Local/Temp/claude/d--Chungkhoan/e65ae674-4bcd-4501-996a-7c1a23ac2c88/scratchpad/adv-desk` (đã copy `swing_desk/portfolio.txt`). Lệnh: `python -m stock_agent.swing_desk --repo D:/Chungkhoan --data-root D:/Chungkhoan --desk <scratch> refresh` → `snapshot_action reused`, state `2026-10-07`.
- Snapshot dùng: `data/paper/snapshots/20261007T104346304078Z` (as_of 2026-10-07, manifest sha `58b02f10…022bc`, nguồn VCI `gap-chart`). History store mới nhất không-aborted là `20261006T161307Z_6ea2ce75` (as_of 2026-10-06) → **không khớp as_of 2026-10-07**, nên mọi trường long-window lấy từ snapshot 650 ngày (`history_source: snapshot_650d`, 440 phiên). Đây là đúng thiết kế (WARN trong `history_notes`), nhưng có nghĩa SMA200/high_250/excursion hôm nay được tính trên 440 phiên, không phải 10 năm.
- Network: chỉ WebFetch các trang công khai (CafeF `PriceHistory.ashx` cho VRE/VHM; PDF quy chế HSX của VBSE). Không chạy collector, không `refresh --history/--with-flows`.
- Script kiểm độc lập (thuần Python, không dùng pandas để tính): `scratchpad/recompute_adv.py`; so sánh: `scratchpad/compare_adv.py`; output đầy đủ: `scratchpad/compare_out.txt` (272 phép so sánh).
- Mã ngẫu nhiên: `random.Random(20261008).choice(VN30 \ {VRE,SSB,TCB,MBB,VPB,HDB})` = **MCH**. Thêm VHM vì liên quan câu hỏi chia cổ tức.

## 1. Kết luận

**Verdict tổng: PASS về tính toán (0 sai lệch > 1e-6 trên 272 phép so sánh, lớn nhất 5,3e-13); 1 FAIL logic mức MEDIUM (định nghĩa episode/panic_low lệch tài liệu), 2 FAIL dữ liệu mức LOW–MEDIUM (volume không điều chỉnh theo chia tách; flows gồm thỏa thuận nhưng ADV chỉ khớp lệnh), vài quan sát LOW.**

| # | Hạng mục | Kết quả | Bằng chứng chính |
|---|---|---|---|
| 1 | Inventory trường | PASS | §2: 60 trường swing + 45 trường bottomfish, công thức file:dòng, dùng bởi rule hay chú thích |
| 2a | Recompute swing (VRE, SSB, MCH, VHM) | PASS | Tất cả trường không-core khớp; max rel 5,3e-13 (`pct_b`, do thứ tự cộng) |
| 2b | Recompute bottomfish | PASS (tính toán) | Tất cả khớp kể cả `beta_60` (1,4e-16), `fall_type_auto` 4/4 |
| 2c | Tick/biên độ HOSE | PASS | Bảng bước giá 10/50/100 và ±7% khớp PDF [R4] trang 4–5; quy tắc làm tròn sàn LÊN/trần XUỐNG khớp dữ liệu: 63 low = sàn làm tròn lên, 0 = làm tròn xuống; 117 high = trần làm tròn xuống |
| 2d | Logic so với tài liệu | **FAIL F1 (MEDIUM)** | `episode_start`/`panic_low` neo vào đợt giảm cũ (VRE: từ 2026-05-11, 105 phiên) thay vì "từ đỉnh 60 phiên" như BF-A12 |
| 3 | Look-ahead | PASS | §4: mọi trường dùng bar ≤ S; excursion chỉ dùng entry e có e+1 (hoặc e+4) ≤ S; panic_low/episode chỉ dùng low/close ≤ S; beta cắt tại S−10; flows lọc `available_at ≤ fetched_at` |
| 4a | Giá điều chỉnh vs giá thô | PASS (có giải thích) | VRE/VHM 3 phiên gần nhất khớp CafeF tuyệt đối; giá trước GDKHQ = giá điều chỉnh lùi (VRE ×0,965800, VHM ×0,5) |
| 4b | panic_low VRE 20.900 | PASS dữ liệu / FAIL logic (F1) | 20.900 là **giá thô thật** (low 28/07/2026 theo CafeF, sau GDKHQ 30/06) — không phải artefact điều chỉnh. 21.300 là **giá đóng cửa** cùng ngày |
| 4c | Đơn vị volume | PASS + **FAIL F2 (LOW–MEDIUM)** | Volume = cổ phiếu khớp lệnh (không gồm thỏa thuận); **không** điều chỉnh theo chia 1:1 của VHM trong khi giá đã chia đôi |
| 4d | Snapshot vs store 10 năm | PASS | 0 ô OHLCV khác trên 439 phiên chung (VRE, VHM, SSB), 434 (MCH); MCH thiếu 5 phiên 18–24/12/2025 ở **cả hai** nguồn |
| 5a | `flow_status UNVERIFIED` | PASS (giải thích) | Health store flows = `partial` (proprietary thiếu 2026-10-07 cho 30/30 mã) → `service.py:238-239` hạ OK → UNVERIFIED cho cả trường khối ngoại vốn đầy đủ |
| 5b | Trường flows | PASS tính toán / **FAIL F3 (MEDIUM, chú thích)** | f_net_1d/5d/20d/pct_adv khớp tính lại từ `records.jsonl`; HDB −184,7% (10-07) / −176% (10-06) do **thỏa thuận** khối ngoại nằm trong tử số |

## 2. Inventory trường

Ký hiệu: **R** = dùng bởi rule (gate/behaviour/bottomfish/bf_arms/bf_service/review), **A** = chỉ chú thích/hiển thị/log. Dòng code là của `bc8e09e`. `base.*` = `stock_agent/features/indicators.py`.

### 2.1 State ticker JSON (`indicators`, `stock_agent/swing_desk/indicators.py:compute_ticker` dòng 79–151)

| Trường | Công thức (file:dòng) | Cửa sổ | Dùng |
|---|---|---|---|
| date, rows, close, open, high, low | bar cuối ≤ as_of (`indicators.py:104-111`) | 1 | R (close: G3, sizing) |
| prev_close | `cv[-2]` (`:107`) | 1 | A |
| ret_1d / ret_20d | core (đã kiểm T17) | 1/20 | R (G4, B1, B6) |
| ret_3d | `_ret(cv,3)` (`:113`, `_ret` `:28`) | 3 | R (G4 soft, `gate.py:153`) |
| ret_5d | `_ret(cv,5)` (`:114`) | 5 | A |
| gap_open_1d | `O_t/C_{t-1} − 1` (`:116`) | 1 | A |
| sma5 | `base.sma` rolling mean (`:117`, `base:11`) | 5 | R (review HOLD/SELL, `review.py:138`) |
| sma20 | (`:118`) | 20 | R (G8 target khi có setup, `gate.py:46`) |
| sma50, sma200 | core | 50/200 | R |
| ema21, ema50 | `ewm(span, adjust=False, min_periods=span//2)` (`:120-121`, `base:7`) | 21/50 | A (ema21 chỉ xuất hiện trong text `exit_signal` của P1, không có code đọc) |
| rsi2 | core | 2 | R (S1) |
| rsi14 | `base.rsi`: ewm α=1/14 seed delta đầu, `fillna(50)` (`:123`, `base:15-23`) | 14 | A |
| atr14, atr_pct | core | 14 | R (G6) |
| bb_lower, bb_upper | SMA20 ± 2·std **ddof=1** (`:125`, `base:33-39`) | 20 | A |
| pct_b | `(C−L)/(U−L)` (`:130`) | 20 | A |
| adx14, plus_di14, minus_di14 | Wilder qua ewm α=1/14; DM bar 0 = 0; `fillna(0)` (`:126`, `base:78-102`) | 14 | A |
| vol_ratio_20 | `V_t / mean(V_{t-19..t})` — **gồm cả hôm nay** (`:134`) | 20 | R (B6 hype, `behaviour.py:65`) |
| avg_value_20_bn | `mean(C·V)` 20 phiên / 1e9 (`:135`) | 20 | R (G7, BF liquidity, sizing) |
| high_20, low_20 | max High / min Low 20 phiên (`:101,137`) | 20 | high_20: R (G8 target không setup); low_20: A |
| high_250 | core | 250 | R |
| dist_high_20_pct | `high_20/C − 1` (`:139`) | 20 | A |
| limit_up_t, limit_down_t | `ret_1d ≥ +6,5%` / `≤ −6,5%` (proxy, `:140-141`) | 1 | limit_up_t: R (B1, `behaviour.py:45`); limit_down_t: A |
| limit_up_t1 | ret_1d của t−1 ≥ 6,5% (`:142`) | 1 | R (B1) |
| up_streak | số phiên tăng liên tiếp tính từ t (`:96-102`) | — | A |
| vsa_stopping_volume, vsa_no_supply | `base.add_vsa_features` (`base:150-192`): spread/vol ratio so với mean 20 (min 5) **gồm hôm nay** | 20 | A (research A2) |
| locked_mae_p50/p80/p90 | `excursion.py:27-36`: `min(L_e,L_{e+1})/O_e − 1`, entry e ∈ [n−251, n−2]; p80 = quantile 0,2 | 250 entry | p80: R (G6 soft, `gate.py:182`); p50/p90: A |
| locked_breach_3pct_rate | % entry có locked ≤ −3% (`excursion.py:37`) | 250 | A |
| day2_ret_given_breach | median `C_{e+2}/O_e − 1` trên entry breach có e+2 ≤ t (`:29-30`) | 250 | A |
| n_locked, n_breach, n_window5 | đếm (`:39-44`) | — | A |
| mfe5_p50, mae5_p50 | median `max H_{e..e+4}/O_e −1`, `min L…` với e ≤ n−5 (`:32-34`) | 250 | A |
| stop_atr_multiple | `3,0 / atr_pct` (`:150`) | — | A |
| flows.* (f_net_1d/5d/20d_bn, f_net_5d_pct_adv, td_net_5d_bn, flow_history_sessions, flow_status) | `indicators.py:flow_fields` 154-185; status hạ cấp ở `service.py:238-239` | 1/5/20 | A (chú thích A4, không đổi verdict) |

### 2.2 Bottomfish (`stock_agent/swing_desk/bf_indicators.py:compute_bottomfish_fields` dòng 126–227)

| Trường | Công thức (file:dòng) | Cửa sổ | Dùng |
|---|---|---|---|
| ret_1d, floor_t | `ret1 ≤ −6,5%` (`:143-145`, `:191`) | 1 | R (BF-G3, BF-B1) |
| locked_floor_t / _t1 | floor & O=H=L=C (`:146-147`) | 1 | R (BF-G3, BF-B1) |
| floor_streak, locked_floor_streak | `_streak` (`:195-196`) | — | floor_streak: A (bảng scan); locked_floor_streak: A |
| floors_in_last_5/10 | (`:197-198`) | 5/10 | A |
| last_floor_date, sessions_since_floor | sàn cuối trong 60 phiên (`:148-151`) | 60 | R (in_fall_episode, stab_state) |
| floor_price_today / _next | `ceil(ref×0,93 / tick)×tick` (`floor_price` `:39-48`, tick `:25-36`) | 1 | _next: R (BF-G3/B1 với `--price`); _today: R gián tiếp (capitulation) |
| higher_low_t | `L_t > L_{t−1}` (`:168`) | 2 | R (qua stab_state) |
| close_pos_in_range | `(C−L)/(H−L)` (`:206`) | 1 | A |
| high_60, drawdown_60_pct | max **High** 60 phiên; `C/high_60 −1` (`:164-165`) | 60 | R (in_fall_episode, stab NONE, BF-B2) |
| high_250, drawdown_250_pct | (`:166`, `:210`) | 250 | R (BF-B2/B3) |
| episode_start_date, episode_sessions, episode_depth_pct | start = 1 + phiên cuối có `C ≥ 0,95 × rolling-max(High,60)` (`:153-162`) | không giới hạn | start/sessions: R (in_fall_episode, episode_id → BF-G11); depth: A |
| panic_low | `min(Low[start:])` (`:157`) | episode | R (BF-G4 qua new_low_2, invalidation P1/P2 `bottomfish.py:154-156`, `bf_arms.py:52,101`) |
| new_episode_low_2 | `min(L_{t−1},L_t) ≤ panic_low` (`:159`) | 2 | R (stab_state) |
| stab_state | LOCKED/FALLING/NONE/STAB1/STAB2/WEAK (`:170-181`) | — | R (BF-G4, bf_arms) |
| ret_10d, sector_rel_ret_10d | (`:183`, `:217`) | 10 | ret_10d: R (bf_arms random arm); sector_rel: A |
| vol_ratio_20_prior | `V_t / mean(V_{t−20..t−1})` — **không** gồm hôm nay (`:184-185`) | 20 | A (qua capitulation) |
| capitulation_t | vr ≥ 3 và (ret1 ≤ −4% hoặc mở ≤ sàn×1,001 và H > O) (`:187`) | 20 | A (bảng scan) |
| dist_sma50_atr, avg_value_20_bn | (`:220-221`) | 50/14, 20 | avg_value: R (BF liquidity); dist: A |
| idx_ret_5d, idx_ret_10d | VNINDEX (`:98-101`) | 5/10 | idx_ret_5d: R (phân loại a, fall_type_auto) |
| beta_60, beta_obs | cov/var (ddof 0) của pct_change, inner-join theo ngày, cắt tại ngày t−10, 60 obs, tối thiểu 40 (`:102-113`) | 60, gap 10 | R (classify_fall) |
| idio_ret_10d, idio_share | `ret_10d − β·idx_ret_10d` (`:114-116`) | 10 | idio: R (BF-G2, fall_type_auto); share: A |
| index_atr_ratio | ATR%14 VNINDEX / median 250 (`:117-121`) | 250 | R (giảm size khi biến động cao, `bottomfish.py:151`) |
| f_sell_streak, f_net_turn_positive, f_net_1d_pct_vol, flow_status | `_flows` (`:67-84`) | — | A — **luôn UNVERIFIED** vì hai call site (`bf_service.py:102, 212`) không truyền `flows` (O3) |
| fall_type_auto | `bf_service.py:179-192` MARKET/IDIO/NONE | — | R (bf_arms nhánh `MARKET`) |

## 3. Recompute độc lập

Phương pháp: đọc CSV snapshot bằng `csv`, tính bằng vòng lặp Python (EMA/Wilder viết tay, quantile tuyến tính tự viết, beta bằng tổng thủ công trên dict theo ngày). So sánh với `state/2026-10-07/tickers/<S>.json` (swing) và `bf_service.scan_rows` (bottomfish; hàm này read-only). Ngưỡng: 1e-6 relative.

### 3.1 Swing (giá trị desk; "khớp" = rel diff ≤ 5,3e-13)

| Trường | VRE | SSB | MCH | VHM | Kết quả |
|---|---|---|---|---|---|
| ret_3d / ret_5d / gap_open_1d | −1,4831 / −4,5175 / −0,4202 | khớp | khớp | khớp | 0 diff |
| sma5 / sma20 | 23.720 / 24.760 | khớp | khớp | khớp | 0 |
| ema21 / ema50 | 24.526,304 / 25.140,378 | khớp | khớp | khớp | 0 |
| rsi14 | 33,6396 | khớp | khớp | khớp | 0 (Wilder seed-SMA khác < 1e-12) |
| bb_lower / bb_upper / pct_b | 23.001,53 / 26.518,47 / 0,07065 | khớp | khớp | khớp | ≤ 5,3e-13 |
| adx14 / +DI / −DI | 22,258 / 11,153 / 26,590 | khớp | khớp | khớp | 0 (bản textbook khác < 2e-13) |
| vol_ratio_20 / avg_value_20_bn | 0,6791 / 99,133 | khớp | khớp | khớp | 0 |
| high_20 / low_20 / dist_high_20_pct | 26.500 / 23.050 / 13,978 | khớp | khớp | khớp | 0 |
| limit_up_t / limit_down_t / limit_up_t1 | F/F/F | khớp | khớp | khớp | eq |
| up_streak | 0 | khớp | khớp | khớp | eq |
| vsa_stopping_volume / vsa_no_supply | 0 / 1 | khớp | khớp | khớp | eq |
| locked_mae p50/p80/p90 | −3,002 / −5,454 / −7,441 | khớp | khớp | khớp | 0 |
| locked_breach_3pct_rate / n_breach | 50,4 / 126 | khớp | khớp | khớp | 0 |
| day2_ret_given_breach | −2,715 | khớp | khớp | khớp | 0 |
| mfe5_p50 / mae5_p50 | 3,554 / −4,882 | khớp | khớp | khớp | 0 |
| stop_atr_multiple | 0,9644 | khớp | khớp | khớp | 0 |

Ghi chú quy ước (không phải lỗi tính): Bollinger dùng std mẫu (ddof=1); với std tổng thể (chuẩn Bollinger) VRE `pct_b` = 0,0595 thay vì 0,0706 (band rộng hơn √(20/19) ≈ 2,6%). Chỉ là chú thích.

### 3.2 Bottomfish

| Trường | VRE | SSB | MCH | VHM | Kết quả |
|---|---|---|---|---|---|
| floor_t / locked / streaks / floors_in_last_5,10 | F / F / 0 / 0,0 | khớp | khớp | khớp | eq |
| last_floor_date / sessions_since_floor | None / None | khớp | khớp | khớp | eq |
| floor_price_today / next | 22.150 / 21.650 | khớp | khớp | khớp | 0 |
| high_60 / drawdown_60_pct | 27.950 / −16,816 | khớp | khớp | khớp | 0 |
| episode_start / sessions / depth | 2026-05-11 / 105 / −32,94 | khớp | khớp (không có episode) | khớp | eq |
| panic_low (ngày) | 20.900 (2026-07-28) | khớp (2026-10-05) | None | khớp (2026-09-25) | 0 |
| new_episode_low_2 / stab_state | F / STAB2 | khớp | khớp | khớp | eq |
| ret_10d / vol_ratio_20_prior / capitulation | −6,627 / 0,668 / F | khớp | khớp | khớp | 0 |
| idx_ret_5d / idx_ret_10d | −0,861 / −2,679 | khớp | khớp | khớp | 0 |
| beta_60 / beta_obs | 1,55909 / 60 | khớp | khớp | khớp | ≤ 1,4e-16 |
| idio_ret_10d / idio_share | −2,450 / 0,370 | khớp | khớp | khớp | 0 |
| fall_type_auto | MARKET | IDIO | NONE | MARKET | 4/4 eq |

Market dùng cho fall_type_auto: `PANIC_BEAR`, breadth 26,7% (state `market.json` = scan).

### 3.3 Tick / biên độ / sàn-trần

- Nguồn [R4] (VBSE, "Quy định giao dịch chứng khoán tại HSX", 08/2025, đã mở PDF): trang 4 "Đơn vị yết giá — cổ phiếu: <10.000 đồng: 10 đồng; 10.000–49.950: 50 đồng; >=50.000: 100 đồng"; trang 5 "Biên độ … ± 7% (so với giá tham chiếu)", "Giá sàn = Giá tham chiếu × (100% − Biên độ)". `tick_size` (`bf_indicators.py:25-36`) **khớp**. PDF không nêu chiều làm tròn.
- Kiểm thực nghiệm chiều làm tròn trên 3.366 cặp (prev close, bar) có giá nguyên của 30 mã: 63 phiên có Low = sàn làm tròn **lên**, 0 phiên = sàn làm tròn xuống; 117 phiên High = trần làm tròn **xuống**. 7 ngoại lệ (VHM 22/07, 29/07/2026; VIC 2025) đều nằm trong vùng giá đã điều chỉnh ×0,5 (giá nguyên nhưng không phải giá bảng) → artefact điều chỉnh, không phải sai quy tắc. → `floor_price` đúng.
- Trần thật hôm nay: VRE 24.850 (+6,88%), SSB 18.400 (+6,98%), MCH 152.600 (+6,94%), VHM 72.700 (+6,91%). Proxy `limit_up_t` ≥ 6,5% có thể gắn cờ một phiên +6,5…6,9% chưa chạm trần — hướng bảo thủ (chặn thêm), chấp nhận được (O4).
- Không cài đặt: luật "trần/sàn điều chỉnh = tham chiếu ± 1 bước khi bằng tham chiếu" (chỉ xảy ra với giá rất thấp, không liên quan VN30); `behaviour.B1` dùng `close×1,07` chưa làm tròn tick (sai lệch ≤ 1 tick, LOW).

## 4. Look-ahead

**Verdict: PASS — không trường nào cho phiên S dùng dữ liệu sau S.**

- Cắt dữ liệu: `_sessions` (`indicators.py:33-40`) lọc `date ≤ as_of`; `long_series` (`:49-61`) yêu cầu long series kết thúc đúng phiên cuối của snapshot, raise nếu không.
- Excursion (`excursion.py`): entry e ∈ [n−251, n−2] dùng `L_e, L_{e+1}` với e+1 ≤ n−1 = S; breach chỉ khi `e+2 < n`; MFE/MAE 5 phiên chỉ e ≤ n−5 (e+4 ≤ S). Kiểm thủ công: tôi tính lại đúng các tập chỉ số này và khớp `n_locked=250`, `n_window5=250`, `n_breach`. Các thống kê này là phân phối của các entry **quá khứ** đã hoàn tất, không phải dự báo của phiên S.
- Episode/panic_low: `rolling(60, min_periods=1).max()` của High gồm bar hiện tại (không nhìn trước); `panic_low = min(Low[start..S])`; start xác định bằng close ≤ S. Causal.
- Beta: returns inner-join, `index ≤ date[S−10]`, 60 obs → causal (gap 10 phiên tránh chồng với cửa sổ idio).
- Floors: `floor_lookback_sessions=60` chỉ nhìn lùi.
- Flows: `load_flows(as_of=fetched_at snapshot)` lọc `available_at ≤ 2026-10-07T10:43:46Z`; run flows mới nhất `available_at 10:43:10Z` → hợp lệ. `flow_fields` lọc `date ≤ as_of`.
- Lưu ý (không phải look-ahead): giá điều chỉnh lùi của VCI tại thời điểm fetch 10-07 phản ánh các sự kiện doanh nghiệp đã xảy ra **trước** S, nên không rò rỉ tương lai cho phiên S hiện tại; nhưng khi tái lập phiên quá khứ từ snapshot mới hơn, giá quá khứ đã bị điều chỉnh theo sự kiện xảy ra sau phiên đó (chuẩn của mọi chuỗi adjusted; chỉ ảnh hưởng mức giá tuyệt đối như floor_price/panic_low lịch sử, không ảnh hưởng tỉ suất).

## 5. Nguồn dữ liệu

### 5.1 Giá điều chỉnh vs giá thô (CafeF `PriceHistory.ashx`, cột GiaDongCua = thô, GiaDieuChinh = điều chỉnh)

| Mã | Ngày | Snapshot (C / H / L / V) | CafeF thô | CafeF điều chỉnh | Nhận xét |
|---|---|---|---|---|---|
| VRE | 2026-10-07 | 23.250 / 23.700 / 23.250 / 2.708.700 | 23,25 / 23,7 / 23,25 / KL khớp 2.708.700, TT 0 | 23,25 | khớp tuyệt đối |
| VRE | 2026-10-06 | 23.800 / 24.000 / 23.050 / 3.747.500 | 23,8 / 24 / 23,05 / 3.747.500 | 23,8 | khớp |
| VRE | 2026-10-05 | 23.800 / 23.850 / 23.450 / 3.082.000 | 23,8 / 23,85 / 23,45 / 3.082.000 | 23,8 | khớp |
| VRE | 2026-09-07 | H 27.950, C 27.000, V 11.633.600 | H 27,95, C 27, V 11.633.600 | 27 | khớp (đỉnh 60 phiên) |
| VRE | 2026-07-28 | L 20.900, C 21.300, V 3.761.100 | L 20,9, C 21,3, V 3.761.100 | 21,3 | khớp — **giá thô** |
| VRE | 2026-03-06 | C 25.352,25, V 3.672.725 | C 26,25, V 3.666.200 | 25,35 | ×0,965800 (cổ tức tiền) |
| VRE | 2026-03-04 / 03-05 / 02-02 | 27.042,40 / 26.076,60 / 27.138,98 | 28,0 / 27,0 / 28,1 | 27,04 / 26,08 / 27,14 | ×0,965800 không đổi |
| VHM | 2026-10-07 / 10-06 / 10-05 / 10-02 | 68.000 / 69.000 / 68.300 / 67.900; V 4.666.500 / 7.022.900 / 5.443.600 / 6.216.200 | giống hệt; TT 2.322.900 / 0 / 1.632.200 / 966.000 | = thô | khớp; volume **không** gồm thỏa thuận |
| VHM | 2026-07-28 | C 65.950, V 2.683.826 | C 131,9, V 2.674.400 | 65,95 | giá ×0,5, volume **không** ×2 |
| VHM | 2026-08-05 / 08-06 | 76.500 / 77.100 | 153,0 / 77,1 | 76,5 / 77,1 | GDKHQ 1:1 = **2026-08-06** theo CafeF |

Kết luận 5.1: snapshot VCI là chuỗi **điều chỉnh lùi** (backward-adjusted) — sau GDKHQ là giá thô (VRE từ 2026-06-30, VHM từ 2026-08-06), trước đó nhân hệ số (VRE 0,965800 ≈ 1 − 1.000/29.2xx; VHM 0,5). Phù hợp với cổ tức tiền 1.000 đ của VRE (GDKHQ trong snapshot là 30/06/2026, chi trả tháng 7) và cổ tức cổ phiếu 1:1 của VHM (niêm yết bổ sung 09/09/2026). Ghi chú: `data/raw/prices_hist` (legacy, không dùng bởi desk) có bước nhảy −49,9% của VHM ở 2026-07-31, lệch với GDKHQ 08-06 của CafeF — thêm một lý do để không dùng nguồn legacy; không kiểm sâu hơn.

### 5.2 panic_low VRE 20.900 và drawdown −16,8%

- 20.900 là **Low thô** của phiên 2026-07-28 (CafeF: GiaThapNhat 20,9; GiaDieuChinh = GiaDongCua → không có điều chỉnh nào sau ngày đó). Không phải artefact giá điều chỉnh. Con số "≈21.300" của researcher là **giá đóng cửa** 28/07.
- `drawdown_60_pct` −16,82% = 23.250 / 27.950 − 1, với 27.950 = **High** 07/09/2026 (CafeF xác nhận). Nếu dùng max **Close** 60 phiên (27.000) → −13,9%; nếu dùng close 10-06 (23.800) → −14,85%. Con số −14,7% của researcher không tái lập chính xác được từ dữ liệu; nhiều khả năng do dùng mốc đỉnh/đóng cửa khác. Desk nhất quán với docstring/BF doc ("drawdown từ đỉnh 60 phiên" đo bằng High).
- Vấn đề thật là **logic episode** (F1): xem §6.

### 5.3 Đơn vị volume

- Snapshot `volume` = **số cổ phiếu khớp lệnh** (lô chẵn), không gồm thỏa thuận: VRE/VHM 10-02…10-07 khớp chính xác `KhoiLuongKhopLenh` của CafeF, kể cả ngày có thỏa thuận lớn (VHM 10-07 TT 2.322.900 không nằm trong 4.666.500).
- Một số phiên cũ snapshot lớn hơn CafeF 0,15–0,5% (VRE 03/2026: +0,15…0,18%; VHM 07-27: +0,43%; 08-07: +0,51%) với số lẻ không tròn 100 → có thể là lô lẻ được VCI cộng vào. **UNVERIFIED** (không ảnh hưởng đáng kể).
- `avg_value_20_bn` = mean(C×V) là giá trị **khớp lệnh** ước lượng, dùng giá đóng cửa (không phải VWAP).
- **F2**: volume **không** được điều chỉnh theo sự kiện chia tách (VHM 07-28: giá ×0,5 nhưng volume giữ đơn vị cổ phiếu cũ). Chuỗi C×V trước GDKHQ bị thấp 50%; `vol_ratio_20`, VSA, `vol_ratio_20_prior` trong 20 phiên sau GDKHQ (06/08 → ~03/09/2026) bị thổi phồng tới ~2×. Hôm nay (cửa sổ 20 phiên đã qua GDKHQ) không ảnh hưởng VHM; nhưng ảnh hưởng mọi tính lại lịch sử và các mã sẽ chia tách trong tương lai.

### 5.4 Snapshot vs store 10 năm

| Mã | Phiên chung | Ô OHLCV khác | Store rows / bắt đầu |
|---|---|---|---|
| VRE | 439 (2024-12-26 → 2026-10-06) | 0 | 2.225 / 2017-11-06 |
| VHM | 439 | 0 | 2.288 / 2016-10-06 |
| SSB | 439 | 0 | 1.381 / 2021-03-24 |
| MCH | 434 | 0 | 2.412 / 2017-01-05 |

MCH thiếu 2025-12-18…24 (5 phiên) ở cả snapshot và store (snapshot 435 rows) — trùng giai đoạn chuyển sàn; `ret_k` của MCH tính theo hàng (không theo lịch) qua khoảng này, beta dùng inner-join theo ngày nên không lệch ngày. Store as_of 2026-10-06 ≠ snapshot 2026-10-07 nên hôm nay desk không dùng store (đúng thiết kế).

## 6. Flows

- **Vì sao UNVERIFIED**: `state/2026-10-07/manifest.json → flows_health.status = "partial"`, `missing_latest_count = {foreign: 0, proprietary: 30}` (run `20261007T103956104644Z_a3f7eef4`: gap `proprietary:* ['2026-10-07']`). `service.py:238-239` hạ mọi `flow_status OK` thành `UNVERIFIED` khi health ≠ `ready`. Dữ liệu khối ngoại thực ra đủ 20/20 phiên tới 10-07 cho cả 30 mã; chỉ tự doanh thiếu (nên `td_net_5d_bn = null` đúng quy tắc "thiếu = NaN"). Đây là hành vi đúng thiết kế nhưng thô (O1).
- **Tính lại** từ `records.jsonl` (investor=foreign, `net_value_vnd/1e9`): VRE 1d −16,20492 / 5d −18,232075 / 20d −130,906645 / pct_adv −3,6783; HDB −88,979507 / −2.965,42139 / −2.770,225965 / −184,653; SSB 5,987375 / 5,41004 / 392,200905 / 1,4058; MCH 30,35696 / 20,67542 / 600,18075 — **khớp desk tuyệt đối**.
- **HDB "−176% của 5×ADV" (10-06; hôm nay −184,7%)**: `trade_scope = provider_chart_aggregate` (Vietstock chart) **gồm giao dịch thỏa thuận**. Bằng chứng từ raw `foreign_HDB.json`: 2026-10-02 khối ngoại bán 39.683.039 cp (1.042,3 tỷ) trong khi tổng khớp lệnh cả thị trường chỉ 15.620.900 cp; giá bán bình quân ngầm định 26.265 < Low phiên 27.650. 2026-10-06: bán 76.603.452 cp (2.045,5 tỷ) vs khớp 13.844.700 cp; giá ngầm định 26.703 < Low 27.350. Không thể là khớp lệnh → là thỏa thuận (bước giá 1 đ, theo [R4] trang 4). Trong khi mẫu số `avg_value_20_bn` chỉ là giá trị **khớp lệnh** → tỉ lệ > 100% là so sánh khác phạm vi (F3).
- Ghi chú phụ: `flow_fields.tail_sum` đếm theo **hàng**, không theo phiên; nếu một phiên flows bị thiếu, "5d" sẽ trải dài > 5 phiên mà không báo (O5).

## 7. Lỗi và quan sát (đề xuất, chưa áp dụng)

| ID | Mức | Mô tả | Đề xuất sửa |
|---|---|---|---|
| F1 | MEDIUM (logic dùng bởi rule) | `episode_start` = sau phiên cuối có `C ≥ 0,95 × max(High,60)` (`bf_indicators.py:153-157`). BF doc BF-A12 định nghĩa `episode_sessions` "từ đỉnh 60 phiên tới hôm nay". Với VRE, episode bắt đầu 2026-05-11 (105 phiên) và `panic_low` = 20.900 (28/07), dù giá đã hồi +34% lên 27.950 (07/09) rồi mới giảm lại; tỉ lệ C/rolling-high cao nhất từ 05-11 chỉ 0,945 nên không bao giờ reset. Theo định nghĩa của tài liệu, đáy của cú giảm hiện tại là 23.050 (06/10) và episode ≈ 22 phiên. Hệ quả: invalidation P1 (`close < panic_low`) xa hơn ~10% so với giá, BF-G4 (`new_episode_low_2`) gần như không bao giờ kích hoạt, `episode_id`/BF-G11 gộp hai cú giảm thành một | Định nghĩa lại start = ngày của High 60 phiên (khớp BF-A12) hoặc reset khi hồi ≥ X% từ panic_low; nếu giữ quy tắc 95%, sửa docstring + BF doc và hiển thị cả `panic_low` và "đáy từ đỉnh 60 phiên". Thay đổi ngưỡng/định nghĩa → bump `bottomfish_v1` (bf_params_hash) theo notes của config |
| F2 | LOW hiện tại / MEDIUM cho backtest | Volume không điều chỉnh theo chia tách/cổ tức cổ phiếu trong khi giá đã điều chỉnh (VHM 1:1) | Khi fetch: phát hiện hệ số điều chỉnh (so `raw`/legacy hoặc bước nhảy) và nhân volume với 1/hệ số; hoặc tính các tỉ lệ volume trên giá trị (C×V) dùng giá thô; tối thiểu ghi cờ "corporate action trong cửa sổ 20 phiên" vào `history_data_issues` |
| F3 | MEDIUM (chú thích, có thể gây hiểu sai) | `f_net_5d_pct_adv` chia flows **gồm thỏa thuận** cho ADV **chỉ khớp lệnh** (HDB −184,7%) | Ghi rõ phạm vi trong annotation ("gồm thỏa thuận; ADV = khớp lệnh"); hoặc lấy giá trị khớp lệnh của khối ngoại từ nguồn có tách match/put-through nếu có; hoặc cờ `put_through_suspected` khi khối lượng ngoại > khối lượng khớp hay giá ngầm định ngoài [Low, High] |
| O1 | LOW | `flow_status` của khối ngoại bị hạ UNVERIFIED chỉ vì tự doanh thiếu | Tách status theo investor (`f_status`, `td_status`) |
| O2 | LOW | `vol_ratio_20` gồm hôm nay trong mẫu số (bf dùng `_prior` không gồm) → B6 ngưỡng 3 tương đương ≈3,5× trung bình 20 phiên trước. Bollinger dùng std mẫu (ddof=1) | Thống nhất định nghĩa và ghi vào docstring/research doc A1 |
| O3 | LOW | Trường flows bottomfish (`f_sell_streak`, `f_net_turn_positive`, `f_net_1d_pct_vol`) là code chết: `bf_service.py:102, 212` không truyền `flows` → luôn UNVERIFIED; BF-A1 trong doc nói "snapshot + flows v2" | Truyền flows hoặc bỏ trường khỏi output |
| O4 | LOW | `limit_up_t` dùng proxy +6,5% thay vì trần thật (+6,88…6,98%); B1 dùng `close×1,07` chưa làm tròn tick | Dùng hàm `ceiling_price` đối xứng với `floor_price` (làm tròn xuống) cho B1/G4 |
| O5 | LOW | `flow_fields.tail_sum` đếm hàng, không kiểm liên tục phiên | Reindex theo lịch giao dịch trước khi lấy tail |
| O6 | INFO | Hôm nay long-window dùng snapshot 440 phiên vì store as_of 10-06; SMA200/high_250/excursion vẫn đủ cửa sổ nhưng là chuỗi ngắn | Chạy `refresh --history` khi người dùng cho phép (không chạy trong kiểm tra này) |

## 8. Tái lập

```
python -m stock_agent.swing_desk --repo D:/Chungkhoan --data-root D:/Chungkhoan --desk <scratch>/adv-desk refresh
python -I <scratch>/recompute_adv.py <scratch>/adv-desk <scratch>/mine.json      # random ticker: MCH
python <scratch>/compare_adv.py <scratch>/adv-desk <scratch>/mine.json             # 272 so sánh, max rel 5.3e-13
```
`<scratch>` = `C:/Users/acer/AppData/Local/Temp/claude/d--Chungkhoan/e65ae674-4bcd-4501-996a-7c1a23ac2c88/scratchpad`. Script và output nằm ngoài repo (không commit). Bước tiếp theo có giới hạn: quyết định định nghĩa episode (F1) với người dùng trước khi sửa, vì thay đổi sẽ tạo phiên bản giao thức bottomfish mới. Kết quả kiểm này không phê duyệt giao dịch hay triển khai.
