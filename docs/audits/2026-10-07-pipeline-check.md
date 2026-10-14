# Kiểm tra end-to-end data pipeline + swing-desk triggers (T17, scout/verification)

- Ngày: 2026-10-07 (chạy lúc ~00:08–00:40 giờ VN; `now_utc` của desk = 2026-10-06T17:09Z)
- Revision: `72b93de` (branch `codex/data-integrity-audit`), working tree CRLF (`core.autocrlf=true`); dirty tree chỉ gồm các file đã có từ đầu phiên (docs/audits 2026-10-06-*, `data/pipeline/*.log|cache`, plan doc). Không sửa code.
- Scope: health read-only của pipeline; chuỗi trigger desk trong scratch desk `C:/Users/acer/AppData/Local/Temp/claude/d--Chungkhoan/e65ae674-4bcd-4501-996a-7c1a23ac2c88/scratchpad/pipecheck-desk` (portfolio.txt copy, sha256 `a257f444…26c` khớp file thật); recompute độc lập; gate sanity; bottom-fish PNJ; hash chain; G0.
- Network đã dùng: đúng 1 lần `refresh --held` (VCI: PNJ, TPB, VNINDEX → scratch desk). Không chạy `refresh --history`, collector, `paper_runner --refresh`, `--quarantine-orphans --apply`.
- Python: `C:/Users/acer/anaconda3/python.exe`. Prefix desk: `python -m stock_agent.swing_desk --repo D:/Chungkhoan --data-root D:/Chungkhoan --desk <scratch>`.
- Người kiểm không tham gia xây dựng code; mọi kết luận dựa trên output lệnh và tính lại độc lập.

## 1. Kết luận

**Verdict tổng: PASS có điều kiện (1 FAIL mức MEDIUM, 1 gap provenance MEDIUM, 3 quan sát LOW).**
Các trigger `status`/`refresh`/`scan`/`check` đọc đúng snapshot đã verify (sha khớp, `as_of` = completed session), dùng đúng history store không-aborted `20261006T161307Z_6ea2ce75` cho cả 30 mã VN30, NAV từ journal; indicator khớp tính lại độc lập tới ≤ 6e-15 relative. FAIL duy nhất: `review` không dùng dữ liệu `refresh --held` cho PNJ/TPB (lãi/lỗ n/a, không có nhãn HOLD/SELL).

## 2. Bảng kết quả

| # | Hạng mục | Kết quả | Bằng chứng chính |
|---|---|---|---|
| 1a | `paper_runner --status` | PASS | `latest.session=2026-10-06 recorded`; `missed=["2026-10-01"]`; `calendar_warning=null`; exit 0. `eod_update.log`: block 10-01 20:03 kết thúc bằng `^C` (bị ngắt) → giải thích session mất |
| 1b | `foreign_refresh --status` | PASS | `"status":"ready","expected_session":"2026-10-06","rows":22010,"lock":null,"orphan_runs":[],"calendar_warning":null,"collector_plausibly_running":false` |
| 1c | `history_store status` | PASS (quan sát O1) | `completed_session 2026-10-06, sessions_behind 0, stale false`; nhưng headline `store` = `20261006T163532Z_76f5be2b` (aborted, ok 45/partial 5/failed 57) |
| 1d | `history_store verify 20261006T161307Z_6ea2ce75` | PASS | `"verified": true, "symbols_checked": 103, counts ok 90/partial 13/failed 4`, exit 0 (4,8 s) |
| 1e | `retention_report --days 30` | PASS | snapshots 10, flow runs 9, `older_count 0`, "report only; nothing deleted" |
| 1f | `data/pipeline/eod_status/` | UNVERIFIED (đúng kỳ vọng) | Thư mục chưa tồn tại. `eod_status.py` được commit ở `fd915af` 2026-10-06 23:20 +07, sau run EOD 17:05 → file đầu tiên chỉ có từ run 10-07. Log: `EOD PAPER SUCCESS Tue 10/06/2026 17:10:43` |
| 2a | `status` | PASS | `snapshot.ok true, as_of 2026-10-06, manifest_sha256 1e4058a1…aa8176`; `sha256sum data/paper/snapshots/20261006T100836844041Z/manifest.json` = `1e4058a1069f41f993a962459091d85eb9b134d1a83695e2b2c1f83e76aa8176` |
| 2b | `refresh` reuse | PASS | `"snapshot_action": "reused"`, `"history_fetch": null`; listing `data/paper/snapshots` trước/sau giống hệt (mới nhất vẫn `20261006T100836844041Z`, mtime 17:10:20) |
| 2c | State history provenance | PASS | `state/2026-10-06/manifest.json`: `history.stores = {6ea2ce75: 14ceed52…e653}`, `skipped_stores [] mismatches [] fallbacks []`; 30/30 `tickers/*.json` = (`history_store`, `20261006T161307Z_6ea2ce75`); store aborted không xuất hiện trong state/ticker nào (chỉ ở dòng `newest_store` của status). `sha256sum` manifest store = `14ceed52…e653` khớp |
| 2d | NAV | PASS | state `nav_source: journal`; `review --show-nav`: `NAV: 11.186.207 VND`; `check` JSON `plan.nav 11186207.0` |
| 2e | `refresh --held` | PASS | `action fetched, status verified, held_outside_vn30 [PNJ,TPB], failures []` → `snapshots/extra/20261006T171018771274Z` (manifest sha `b2acc786…e342`, 374 rows/mã, as_of 2026-10-06). Extra CSV PNJ/TPB = history store trên 374 phiên (0 khác biệt) |
| 2f | Nhãn HOLD/SELL-only cho PNJ/TPB | **FAIL một phần** (F1) | `check --mode bottomfish`: dùng `snapshot extra sha b2acc7868c5e`, `BF-G1 … held position: check/review only (HOLD/SELL/AVOID), never BUY` ✔. `check` swing mode: `G1 OUT_OF_SCOPE` BLOCK, không nhãn HOLD/SELL. `review`: `PNJ … lãi/lỗ n/a … gate hôm nay n/a … không có giá trong snapshot (ngoài VN30?)`, tương tự TPB ✘ |
| 2g | `scan` | PASS | header `snapshot sha 1e4058a1069f`, `Regime: PANIC_BEAR`, 30 dòng, tất cả BLOCK (G2), dòng `history_source history_store (store 20261006T161307Z_6ea2ce75); snapshot_650d fallback: none` |
| 2h | `scan --mode bottomfish` | PASS | 30 dòng; structural PASS: LPB, MWG, VHM, VIC; còn lại BLOCK (BF-G4, SAB thêm BF-G7); cùng dòng history |
| 2i | `review` | PASS trừ F1 | 8 vị thế, activity theo tháng tách entries/add-on/ngày vào lệnh, first session 2026-03-16 (10 lệnh), zero-entry months 04–09 |
| 3 | Recompute độc lập | PASS | 4 mã × 9 trường, sai lệch lớn nhất 6,1e-15 relative (bảng §3) |
| 4 | Gate sanity (regime, G3, G9) | PASS | VNINDEX 1759.08 < EMA50 1787.037 → PANIC_BEAR; breadth 7/30 = 23.3%; G3/G9 khớp số học (§4) |
| 5 | PNJ bottom-fish | PASS | `check PNJ --mode bottomfish --profile P1 --price 20150 --amount 100000 --no-log` → `BLOCK`, exit 2; floor/locked detection khớp CSV (§5) |
| 6a | `verify-log` (desk thật) | PASS | `recommendations ok, rows 19, errors []`; `checks ok, rows 0`; exit 0; sha256 của các file log trước/sau giống nhau |
| 6b | code_hash CRLF | PASS | working tree 24/24 file CRLF: hash raw bytes `835a644701e7a3b8`, hash normalized `7c7052a8d4a92b25` = hash tính từ git blob HEAD (LF) `7c7052a8d4a92b25` = `desk_code_hash` trong state manifest |
| 6c | Provenance của frozen rows 2026-10-07 | **GAP** (F2) | 19 rows (11 desk, 8 single) có `desk_code_hash ae6cbc0494604104`; không tái tạo được từ bất kỳ commit nào trong 25 commit chạm `stock_agent/swing_desk` (thử cả LF và CRLF) |
| 7 | G0 freshness | PASS | code path + mô phỏng read-only + test (§7) |

## 3. Recompute độc lập (snapshot `20261006T100836844041Z` và store `20261006T161307Z_6ea2ce75`)

Công thức desk (`stock_agent/swing_desk/indicators.py`, `features/indicators.py`, `features/swing_scan._rsi`):
- SMA: `rolling(n, min_periods=n).mean()`; SMA200 và `high_250` lấy từ long series (store) khi khớp snapshot.
- RSI2: `ewm(alpha=1/n, adjust=False)` của gain/loss, **seed bằng delta đầu tiên** (không phải Wilder seed SMA của n delta đầu).
- ATR14: TR = max(H−L, |H−Cprev|, |L−Cprev|), `ewm(alpha=1/14, adjust=False, min_periods=14)`, seed bằng TR của bar đầu (= H−L).
- `ret_k = C[-1]/C[-1-k] − 1` (%); `drawdown_250 = C/high_250 − 1` (%) với `high_250` = max High 250 phiên.

Tính lại bằng vòng lặp Python thuần (không pandas) theo cả hai biến thể seed (Wilder chuẩn seed SMA, và seed bar đầu). Mã ngẫu nhiên: `random.Random(20261007).choice(VN30 \ {TCB,MBB,VPB})` = **HDB**. Snapshot = store trên toàn bộ 440 phiên của snapshot cho cả 4 mã (0 ngày lệch OHLCV).

| Mã | Trường | Tính lại (snapshot) | Tính lại (store) | Desk JSON | Rel. diff max |
|---|---|---|---|---|---|
| TCB | close / SMA50 | 32450 / 31510 | — | 32450 / 31510.0 | 0 |
| TCB | SMA200 | 32264.55325 | 32264.55325 | 32264.553249999997 | 1,1e-16 |
| TCB | RSI2 | 42.06442643 | 42.06442643 | 42.0644264294967 | 0 |
| TCB | ATR14 / ATR% | 756.760059 / 2.332080305 | 756.760059 | 756.7600590431526 / 2.332080305217728 | 6,0e-15 |
| TCB | high_250 / dd_250 | — | 41134.8 / −21.11302 | 41134.8 / −21.11302352266208 | 0 |
| TCB | ret_1d / ret_20d | −0.153846 / 1.564945 | — | −0.15384615 / 1.56494523 | 0 |
| MBB | close / SMA50 | 19200 / 19919.148 | — | 19200 / 19919.148 | 0 |
| MBB | SMA200 | 20568.1212 | 20568.1212 | 20568.121199999998 | 1,8e-16 |
| MBB | RSI2 | 37.03183427 | 37.03183427 | 37.03183426598017 | 0 |
| MBB | ATR14 / ATR% | 342.8479805 / 1.785666565 | 342.8479805 | 342.84798051196117 / 1.7856665651664643 | 6,1e-15 |
| MBB | high_250 / dd_250 | — | 23627.42 / −18.73848 | 23627.42 / −18.73848266124697 | 0 |
| MBB | ret_1d / ret_20d | 0.261097 / −4.714640 | — | 0.26109661 / −4.71464020 | 0 |
| VPB | close / SMA50 | 23200 / 21209.968 | — | 23200 / 21209.968000000004 | 3,4e-16 |
| VPB | SMA200 | 21330.10425 | 21330.10425 | 21330.104250000004 | 3,4e-16 |
| VPB | RSI2 | 71.46846115 | 71.46846115 | 71.46846115161344 | 2,0e-16 |
| VPB | ATR14 / ATR% | 575.5780603 / 2.480939915 | 575.5780603 | 575.5780603005621 / 2.48093991508863 | 2,4e-15 |
| VPB | high_250 / dd_250 | — | 27010.21 / −14.10655 | 27010.21 / −14.106554521419856 | 0 |
| VPB | ret_1d / ret_20d | 0.869565 / 6.719855 | — | 0.86956522 / 6.71985486 | 0 |
| HDB | close / SMA50 | 28050 / 27042 | — | 28050 / 27042.0 | 0 |
| HDB | SMA200 | 26721.1242 | 26721.1242 | 26721.124200000006 | 2,7e-16 |
| HDB | RSI2 | 46.71003198 | 46.71003198 | 46.71003198237579 | 0 |
| HDB | ATR14 / ATR% | 598.5519106 / 2.133874904 | 598.5519106 | 598.5519106442853 / 2.1338749042577017 | 3,8e-15 |
| HDB | high_250 / dd_250 | — | 30000 / −6.5 | 30000.0 / −6.499999999999995 | 0 |
| HDB | ret_1d / ret_20d | 0 / 2.559415 | — | 0.0 / 2.55941499 | 0 |

`dd_250` của desk lấy từ JSON `check <T> --mode bottomfish --json --no-log` (`drawdown_250_pct`). Không có sai lệch > 1e-6. Khác biệt seed (RSI2 seed delta đầu vs Wilder seed SMA; ATR seed H−L bar đầu vs SMA 14 TR) suy giảm theo (1−α)^~430 nên < 1e-14 ở cuối chuỗi 440 phiên; với chuỗi ngắn (< ~100 phiên) hai biến thể sẽ khác nhau đáng kể, cần nhớ khi so với công cụ bên ngoài.

## 4. Gate sanity

- Regime (`mr_scan._market_state`): PANIC_BEAR ⟺ VNINDEX close < EMA50; RISK_ON cần close > EMA50 và breadth ≥ 40. Tính lại: VNINDEX 2026-10-06 close 1759.08; EMA50 (α = 2/51) = 1787.0371 (seed bar đầu) / 1787.0371 (seed SMA50) → PANIC_BEAR ✔ (desk: 1759.08 vs 1787.04).
- Breadth = % VN30 có close > SMA20: 7/30 = 23.3% (BSR, GVR, HDB, MSN, MWG, VIB, VPB) ✔. Limit up/down (±6.5%): 0/0 ✔.
- G3 (close > SMA50): HDB 28,050 > 27,042 → PASS (JSON `G3 PASS`) ✔; MBB 19,200 ≤ 19,919.148 → FAIL (scan liệt kê G3) ✔.
- G9 trên `check HDB --price 28050 --amount 5000000 --no-log` (NAV journal 11,186,207; lot_size 1):
  - Budget = 1% × NAV = 111,862.07 ✔; effective loss = 3 + 1 + 0.6 = 4.6% ✔.
  - Requested: floor(5,000,000 / 28,050) = 178 cp → 4,992,900 VND; lỗ hiệu dụng 229,673.4 ✔ > budget; weight 44.63% > 25% ✔; open risk 4.60% + 2.05% = 6.65% ✔; exposure 100.0% + 44.6% = 144.6% ✔; positions 8+1 = 9 > 4 ✔; bank 6 > 2 (MBB, SHB, TCB, TPB, VPB + HDB) ✔.
  - Size gợi ý: risk-only = floor(111,862.07 / (28,050 × 4.6%)) = 86 cp; bị cap về 0 vì `max_exposure` (room = 60% NAV − exposure hiện tại < 0) ✔ (`capped_by: max_exposure`).

## 5. Bottom-fish PNJ

`check PNJ --mode bottomfish --profile P1 --price 20150 --amount 100000 --no-log` → **BLOCK** (exit 2) với BF-G1 (ngoài VN30, held → HOLD/SELL/AVOID), BF-G2/G5/G6/G8 (fail-closed, chưa có bf-news/events/margin), BF-G3 (close at floor proxy −6.93%), BF-G4 (`stab_state FALLING`), BF-G10 (4.74% > 4%), BF-G12, BF-B1, BF-B4 (BLOCK), BF-B3/B6 (WARN). Dữ liệu: snapshot extra sha `b2acc7868c5e`, long window từ store `6ea2ce75` (2497 phiên từ 2016-10-06).

Kiểm floor trên CSV (`floor_proxy_pct −6.5`, band 7%, sàn = ceil(ref × 0.93 / tick) × tick, tick 50 cho 10,000–49,950):

| Ngày | O/H/L/C | ret_1d | Sàn tính lại từ C trước | Floor? | Locked (O=H=L=C)? |
|---|---|---|---|---|---|
| 09-28 | 30700 ×4 | −6.97% | 33000×0.93=30690 → 30700 | ✔ | ✔ |
| 09-29 | 28600 ×4 | −6.84% | 28551 → 28600 | ✔ | ✔ |
| 09-30 | 26600 ×4 | −6.99% | 26598 → 26600 | ✔ | ✔ |
| 10-01 | 24750 ×4 | −6.95% | 24738 → 24750 | ✔ | ✔ |
| 10-02 | 23050/25450/23050/23050 | −6.87% | 23017.5 → 23050 | ✔ | ✘ (H 25450) |
| 10-05 | 23050/23300/21450/21650 | −6.07% | 21436.5 → 21450 (L chạm sàn) | ✘ (proxy −6.5%) | ✘ |
| 10-06 | 21600/22350/20150/20150 | −6.93% | 20134.5 → 20150 | ✔ | ✘ |

Desk: `floor_t true, locked_floor_t false, locked_floor_streak 0, floor_streak 1, floors_in_last_5 4, floors_in_last_10 6, last_floor_date 2026-10-06, floor_price_today 20150, floor_price_next 18750 (20150×0.93=18739.5 → 18750), panic_low 20150, high_250 84670.9, drawdown_250 −76.2%` — tất cả khớp bảng trên. 4 phiên sàn trắng bên 28/09–01/10 được nhận đúng là floor; streak locked hiện tại = 0 là đúng vì 10-06 không đứng giá. Ghi chú: 10-05 chạm sàn intraday nhưng đóng −6.07% nên proxy không tính là floor (thiết kế, không phải lỗi). Store không có ngày nào |ret| > 7.05% từ 2025-10 (không có gap điều chỉnh bất thường trong cửa sổ high_250).

## 6. Hashing và provenance

- `verify-log` (desk thật, read-only theo code `reclog.verify_chain`): recommendations ok 19 rows, checks ok 0 rows; sha256 3 file trong `swing_desk/recommendations` trước = sau.
- 19 frozen rows: `target_session 2026-10-07`, `snapshot_manifest_sha256` bắt đầu `1e4058a1069f` (= snapshot đang dùng), `created_at 2026-10-06T14:46:54Z` (21:46 VN, trước 09:00 ngày 10-07). Rows không có trường `history_*` (đông cứng trước khi tính năng history có) → chúng dùng snapshot-only, nhất quán với thời điểm.
- code_hash: xem 6b ở bảng; normalization CRLF hoạt động đúng.

## 7. Freshness G0

- Code: `swing_desk/data.py:53-87 freshness()` — re-verify snapshot (`verify_snapshot`), từ chối `fetched_at/completed_at > now`, universe phải = VN30 + VNINDEX, `as_of` phải = `completed_session_date(now)` (cutoff 16:00 VN), lỗi → `reason "stale: …"`. `gate.py:125`: `freshness.ok` false → trả về BLOCK chỉ với rule G0 (không đánh giá rule khác).
- Mô phỏng read-only (gọi `dl.freshness` với `now` giả, không ghi file): `now=2026-10-07T08:00Z` → expected 2026-10-06, ok True; `now=2026-10-07T09:30Z` (16:30 VN) → `ok False | stale: snapshot as_of 2026-10-06 != completed session 2026-10-07; run refresh`.
- Tests: `tests/test_swing_desk_gate.py::test_g0_stale_snapshot_blocks_everything_and_evaluates_nothing_else` và `tests/test_swing_desk_check.py::test_stale_snapshot_blocks_with_g0_only` pass; BF-G0 có `tests/test_swing_desk_bottomfish_check.py:58`, `…_refresh.py:77`. Toàn bộ `pytest -q tests/ -k "swing_desk or history_store"`: **360 passed, 866 deselected (67 s)**; `git status --porcelain` trước/sau giống hệt.

## 8. Failures và quan sát (đề xuất sửa — CHƯA áp dụng)

| ID | Severity | Mô tả | Đề xuất |
|---|---|---|---|
| F1 | MEDIUM | `review` không dùng snapshot `refresh --held`: PNJ/TPB (tổng ~8.8% NAV theo giá vốn; TPB đang −18.9% theo check) hiện "lãi/lỗ n/a, gate n/a, không có giá trong snapshot (ngoài VN30?)" và không có action/nhãn HOLD/SELL. Nguyên nhân: `build_review` (`review.py:95-135`) chỉ load frames từ snapshot chính; `extra_freshness` chỉ được gọi trong `bf_review` cho vị thế `reason=bottomfish` (`review.py:217-221`), còn journal ghi PNJ/TPB `reason=other`. Không nhất quán với `check --mode bottomfish` (BF-B6 thấy TPB −18.9%). | Trong `build_review`, với mã đang giữ không có trong frames, đọc `dl.extra_freshness(paths, now, t)` nếu ok; tính close/unrealized/stop-breach, ghi `source: extra` + nhãn "HOLD/SELL/AVOID only, never BUY"; thêm test cho vị thế non-VN30 `reason=other`. Có thể in thêm nhãn này ở `check` swing mode khi G1 OUT_OF_SCOPE và mã đang giữ. |
| F2 | MEDIUM (provenance) | Frozen rows 2026-10-07 mang `desk_code_hash ae6cbc0494604104` không tái tạo được từ commit nào chạm `stock_agent/swing_desk` (25 commit, LF và CRLF) → code rule sinh ra rows không gắn được với một git revision (nhiều khả năng là working tree chưa commit và/hoặc hash trước khi normalize CRLF ở `6f7e324`). Chain hash vẫn hợp lệ. | Ghi thêm `git_rev` + `dirty` (và danh sách file khác HEAD) vào frozen row/protocol; `freeze` từ chối hoặc gắn cờ khi tree dirty trong `stock_agent/swing_desk`; ghi chú trong `configs/swing_desk/protocol_amendments.md` rằng batch 10-07 có code hash không bind được. |
| O1 | LOW | `history_store status` và dòng `newest_store` của desk status chọn store aborted `76f5be2b` làm headline (counts ok 45/failed 57), dễ gây hiểu nhầm cho người vận hành; desk vẫn chọn đúng store không-aborted khi tính. | Headline ưu tiên store không-aborted mới nhất, hoặc in cờ `headline_store_aborted: true` + store được desk dùng. |
| O2 | LOW | Doc drift: `swing_desk/README.md:34` "suggested quantity (lot 100)" và SKILL description "lot-100 size" trong khi `configs/swing_desk/desk_v1.json money.lot_size = 1` (odd lot, user decision) và output thực tế làm tròn 1 cp. | Sửa README/SKILL thành "lot_size từ config (hiện 1 cp, lô lẻ)". |
| O3 | LOW / UNVERIFIED | Flows HDB: `f_net_1d −1,957.6 tỷ`, `f_net_5d −2,839.9 tỷ` = −176% của 5×ADV20 (ADV từ khớp lệnh 322.6 tỷ). Nhiều khả năng flows gồm giao dịch thỏa thuận còn ADV chỉ khớp lệnh; chỉ là chú thích A4, không đổi verdict. Chưa đối chiếu nguồn ngoài. | Ghi rõ đơn vị/phạm vi (khớp lệnh vs thỏa thuận) trong flow annotation, hoặc tách `pt_net` nếu nguồn có. |

## 9. Giới hạn

- Time-box ~35 phút; recompute chỉ 4 mã (TCB, MBB, VPB, HDB) và các trường đã nêu; không recompute EMA21/ADX/BB/excursion stats/flows.
- Chỉ verify store `6ea2ce75`; store aborted `76f5be2b` không được verify (desk không dùng). 4 symbol failed của `6ea2ce75` không thuộc VN30 dùng trong desk (30/30 VN30 có `history_store`).
- Giá là provider-adjusted (VCI, `adjustment_basis provider_adjusted_as_retrieved_unverified`); không đối chiếu giá bảng điện/nguồn thứ hai. Kiểm "đúng dữ liệu" ở đây nghĩa là đúng file đã verify và đúng công thức, không phải đúng so với sàn.
- Scratch desk không có `events`/`bf-news`, nên G10/BF-G2/G5/G6/G8 là UNVERIFIED/fail-closed theo thiết kế; không chạy `freeze`/`score`.
- `eod_status` chưa thể kiểm (file đầu tiên chỉ có từ run 10-07).
- Kiểm tra code_hash của frozen rows chỉ trên các commit hiện có; không thể khôi phục working tree lúc freeze.
- Kết quả xanh ở đây không phê duyệt giao dịch hay deploy (AGENTS.md); desk chỉ tư vấn.

## 10. Lệnh đã chạy (tóm tắt)

```text
python -m stock_agent.pipeline.paper_runner --status
python -m stock_agent.pipeline.foreign_refresh --status
python -m stock_agent.data.history_store status --root data/history
python -m stock_agent.data.history_store verify data/history/20261006T161307Z_6ea2ce75
python -m stock_agent.pipeline.retention_report --days 30
python -m stock_agent.swing_desk <A> status | refresh | refresh --held | scan | scan --mode bottomfish | review --show-nav | review --json
python -m stock_agent.swing_desk <A> check PNJ --mode bottomfish --profile P1 --price 20150 --amount 100000 --no-log [--json]
python -m stock_agent.swing_desk <A> check {PNJ,TPB} [--mode bottomfish --profile P1] --price … --amount 100000 --no-log
python -m stock_agent.swing_desk <A> check HDB --price 28050 --amount 5000000 --no-log [--json]
python -m stock_agent.swing_desk <A> check {TCB,MBB,VPB,HDB} --mode bottomfish --no-log --json
python -m stock_agent.swing_desk --repo D:/Chungkhoan --data-root D:/Chungkhoan verify-log      # desk thật, read-only
python -m pytest -q tests/ -k "swing_desk or history_store" -p no:cacheprovider                # 360 passed
(<A> = --repo D:/Chungkhoan --data-root D:/Chungkhoan --desk <scratch>/pipecheck-desk)
```

Script recompute và output thô nằm trong scratchpad của phiên (không commit): `recompute.py`, `recompute.json`, `o_*.txt|json`.

Next bounded action: sửa F1 trong `review.py` + test (một PR nhỏ), ghi chú F2 vào `protocol_amendments.md` và thêm `git_rev/dirty` vào frozen rows.
