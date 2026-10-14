# 2026-10-08 — Nguồn lịch sử dòng tiền khối ngoại / tự doanh theo mã (HOSE, 2025-01-01..2026-10-07)

Loại: audit trinh sát (task T20 SCOUT), chỉ đọc. Không đăng nhập, không vượt xác thực,
không giải CAPTCHA. Mọi request là GET công khai, cách nhau ≥3 s (CafeF 5 s).
Không sửa code/dữ liệu trong repo; chỉ thêm file này. Script probe và mẫu nằm ngoài repo:
`C:/Users/acer/AppData/Local/Temp/claude/d--Chungkhoan/e65ae674-4bcd-4501-996a-7c1a23ac2c88/scratchpad/flows-probe/`
(`p.py`, `cmp*.py`, `cafe_kn.py`, `samples/*.json`).

Tham chiếu trước đó: memory note `foreign-flow-data-sources.md` (probe 2026-07-02),
`docs/FOREIGN-FLOWS.md` (store v2 sạch từ 2026-09-30).

## 1. Kết luận ngắn

1. **Có nguồn sơ cấp miễn phí, chưa từng được probe trước đây: API công khai của HOSE**
   (`api.hsx.vn`, backend của trang hsx.vn "Giao dịch NĐTNN"). Mỗi mã một request trả
   lịch sử ngày từ **2009-01-02** (VNM: 4.431 phiên), tách **khớp lệnh / thỏa thuận**,
   giá trị bằng **VND**, khối lượng bằng cổ phiếu, kèm room. Mã đã hủy niêm yết vẫn còn
   (FLC đến 2023-02-17, HNG đến 2024-09-05), nên dùng được cho kiểm tra survivorship.
2. **Nguồn kiểm chứng độc lập: Vietcap IQ `price-history`** (công khai, không cần auth).
   Khớp **tuyệt đối** với HOSE trên 436/436 phiên × 2 mã (khối lượng và giá trị,
   cả khớp lệnh lẫn thỏa thuận).
3. **Tự doanh:** HOSE không công bố theo mã trên site này. Nguồn miễn phí khả dụng là
   **CafeF `GDTuDoanh.ashx`**: một request cho cả giai đoạn (436 phiên), và khớp chính xác
   với Vietstock trên các phiên chồng lấn (47/48). So với FiinPro (bên thứ ba), lệch
   khoảng 0,35–0,4%, nên chất lượng thấp hơn khối ngoại. Kết luận: dùng được cho
   nghiên cứu, gắn nhãn chất lượng B.
4. Phát hiện so với tháng 7: CafeF `GDKhoiNgoai.ashx` **không còn "chết"**. Nó nhận
   khoảng ngày, nhưng cửa sổ bị cắt khoảng 3 tháng, trang tối đa 20 dòng, và xử lý
   thỏa thuận không nhất quán, nên không nên dùng làm nguồn chính.
5. vnstock: bản 4.0.4 vẫn để `foreign_trade`/`prop_trade` là stub. Trang PyPI
   `/simple/vnstock/` hiện báo `project-status: quarantined` và JSON API trả 404
   (2026-10-08), nên không thể dựa vào `pip install vnstock`.

## 2. Bảng nguồn

| Nguồn | URL / endpoint | Bắt đầu | Độ chi tiết | Trường / đơn vị | Khớp lệnh / thỏa thuận | Auth | Ghi chú ToS | Kết luận |
|---|---|---|---|---|---|---|---|---|
| **HOSE (sơ cấp)** | `GET https://api.hsx.vn/mk/api/v1/market/securities/foreign/{SYM}?pageSize=500&pageIndex=N` (headers Origin/Referer `https://www.hsx.vn`) | 2009-01-02 (VNM); theo ngày niêm yết HOSE của từng mã | Ngày, một dòng mỗi phiên, mới nhất trước; `paging.totalCount/totalPages` | `reportDate` (epoch giây, 00:00 UTC = ngày VN), `main/bigLot` × `Buyer/Seller` × `ForeignVolume/ForeignValue`, `totalBuyer/SellerForeignVolume`, `totalRoom`, `currentRoom`, OHLC/avg. **Giá trị VND, KL cổ phiếu**, đã gồm lô lẻ | Có: `main*` = khớp lệnh (gồm lô lẻ), `bigLot*` = thỏa thuận | Không | robots.txt `Disallow:` rỗng (cho phép tất cả); CORS `*`; WAF F5 (cookie TS). Dữ liệu thuộc HOSE, chỉ dùng nội bộ cho nghiên cứu, không phân phối lại | **CHỌN: nguồn chính khối ngoại** |
| HOSE báo cáo toàn thị trường | `.../mk/api/v1/market/foreign-trading-report?fromDate&toDate&securitiesType` | — | — | Trả `data: []` với mọi `securitiesType` đã thử (1, 0, 2, 3, STOCK, ST) | — | Không | — | Chưa giải được tham số, không cần thiết |
| HOSE datafeed tải file | `.../m/api/v1/data-feed/files`, `/data-feed/downloadfiles` | — | File | — | — | **401**, cần Bearer token (tài khoản datafeed) | Dịch vụ đăng ký | Loại (cần tài khoản) |
| **Vietcap IQ (kiểm chứng)** | `GET https://iq.vietcap.com.vn/api/iq-insight-service/v1/company/{SYM}/price-history?timeFrame=ONE_DAY&fromDate=YYYYMMDD&toDate=YYYYMMDD&page=0&size=500` | ≥ 2025-01-02 (đã thử); chưa kiểm sâu hơn | Ngày | `foreign{Buy,Sell}{Volume,Value}{Matched,Deal,Total}`, `foreignNet*`, `foreignTotalRoom/CurrentRoom/OwnedPercentage`, kèm OHLCV. VND / cổ phiếu | Có (Matched / Deal) | Không | Nền tảng nghiên cứu của công ty chứng khoán; không thấy điều khoản API công khai. Chỉ dùng đối chiếu, gọi thưa | **Nguồn đối chiếu số 2** |
| **CafeF tự doanh** | `GET https://cafef.vn/du-lieu/Ajax/PageNew/DataHistory/GDTuDoanh.ashx?Symbol=VNM&StartDate=MM/DD/YYYY&EndDate=MM/DD/YYYY&PageIndex=1&PageSize=1000` | ≥ 2025-01-02 (đã thử) | Ngày, cả giai đoạn trong 1 request | `KLcpMua, KlcpBan` (cổ phiếu), `GtMua, GtBan` (VND), `Date` dd/mm/yyyy | **Không tách**; nhiều khả năng chỉ khớp lệnh (xem §3.4) | Không | Trang tin; dữ liệu không ghi nguồn gốc. Chỉ dùng nghiên cứu nội bộ | **CHỌN (chất lượng B): tự doanh** |
| CafeF khối ngoại | `.../GDKhoiNgoai.ashx?Symbol=&StartDate=MM/DD/YYYY&EndDate=...&PageIndex=&PageSize=20` | Có dữ liệu 2025 | Ngày; cửa sổ ≤ ~3 tháng tính lùi từ EndDate; tối đa 20 dòng/trang | `KLMua/KLBan` (cổ phiếu lô chẵn), `GtMua/GtBan/GTDGRong` (VND), `RoomConLai` (cuối ngày), `DangSoHuu` % | Không tách, không nhất quán (§3.2) | Không | Như trên; server thỉnh thoảng ngắt kết nối (RemoteDisconnected) | Chỉ làm nguồn đối chiếu số 3 |
| Vietstock (đã biết) | `finance.vietstock.vn/data/KQGDGiaoDichNDTNNChartByStock` (+TuDoanh), `...Paging` | Ẩn danh: ~20 phiên; tháng: 12; quý: 12 | Ngày / tháng / quý | BuyVol/Val (tỷ VND), không gồm lô lẻ | Tổng (khớp lệnh + thỏa thuận) | Ẩn danh có token CSRF; lịch sử sâu là tính năng trả phí | Không vượt giới hạn | Giữ cho collector chạy tiếp (store v2); không phải nguồn backfill |
| vnstock (free) | `Trading.foreign_trade/prop_trade` | — | — | Stub (`pass`) trong 4.0.4 | — | Phần cài đặt chỉ có trong `vnstock_data` trả phí | PyPI quarantined | Loại |
| SSI iBoard / fiin-market | `iboard-api.ssi.com.vn/statistics/...foreign-investor`, `fiin-market.ssi.com.vn/MarketInDepth/GetForeignInvestor` | — | — | 404 / **403 "Blocked - SSI"** | — | — | WAF chặn, không thử vượt | Loại |
| FireAnt | `restv2.fireant.vn/symbols/{SYM}/historical-quotes` | — | — | 401 | — | Bearer | — | Loại (cần auth) |
| Wichart | `api.wichart.vn/vietnambiz/vi-mo?key=nn_hose` | — | — | 500 | — | — | — | Loại |
| HNX, VSDC | `www.hnx.vn`, `www.vsd.vn` | — | — | Lỗi xác minh chứng chỉ SSL từ máy này; không đi tiếp (không bỏ kiểm tra TLS) | — | — | HNX không liên quan mã HOSE | Không áp dụng |
| UBCKNN (SSC) | `www.ssc.gov.vn` | — | — | Cổng ADF, không có thống kê dòng tiền theo mã | — | — | — | Không áp dụng |
| KBS / MBS / DNSE | — | — | — | **Chưa probe** (hết thời gian) | — | — | — | Không cần, vì HOSE + Vietcap đã đủ |

## 3. Đối chiếu mẫu

Mẫu: VNM, HPG. HOSE pageSize=500 cho 2024-10-03..2026-10-07 (500 phiên; 436 phiên trong
2025-01-01..2026-10-07; phân bố thứ Hai–Sáu đều, không có ngày cuối tuần).

### 3.1 HOSE so với Vietcap IQ (toàn giai đoạn)

| Mã | Phiên chung | KL khớp + thỏa thuận trùng tuyệt đối | GT trùng (≤1.000 VND) | Ngày chỉ có ở một nguồn |
|---|---|---|---|---|
| VNM | 436 | 436 | 436 | 0 |
| HPG | 436 | 436 | 436 | 0 |

### 3.2 HOSE so với bên thứ ba và CafeF (5 ngày yêu cầu + cửa sổ quanh đó)

| Ngày | Mã | HOSE ròng (tỷ VND, khớp lệnh + thỏa thuận) | Nguồn khác | Ghi chú |
|---|---|---|---|---|
| 2025-04-03 | VNM | **−308,535** | Báo cáo VietinBank Securities 04/04/2025 (nguồn FiinPro): **−308.534,94 triệu**; CafeF −308,348 | Trùng FiinPro đến triệu đồng |
| 2025-03-14 | HPG | −118,711 (thỏa thuận: 1,3 triệu cp) | CafeF −110,307 | CafeF lệch ở ngày có thỏa thuận |
| 2025-08-05 | HPG | +1.083,713 | CafeF +1.083,743 | Khớp |
| 2026-02-04 | HPG | +939,564 | CafeF +942,071 | CafeF bỏ phần thỏa thuận 100.000 cp bên bán |
| 2026-06-10 | VNM | +28,110 | CafeF +28,127 | Khớp (lô lẻ) |
| 2026-09-04 | VNM | +16,100 (thỏa thuận 7,33 triệu cp) | Vietstock v2 +16,11; CafeF +16,105 | Khớp |

CafeF khối ngoại trên 64 phiên mẫu: chỉ 42/64 phiên có giá trị lệch ≤0,1%. Khối lượng của
CafeF là lô chẵn khớp lệnh; ở ngày có thỏa thuận, có lúc gộp, có lúc bỏ (ví dụ VNM
2025-04-10: HOSE −82,37 tỷ, CafeF −60,12 tỷ). Vì vậy CafeF **không** dùng làm nguồn chính.

### 3.3 HOSE so với Vietstock (store v2, 25 phiên 2026-09-03..10-07; và dữ liệu tháng legacy)

- Khối lượng HOSE ≥ Vietstock, lệch vài chục đến vài trăm cổ phiếu và không phải bội số
  100, tức là **lô lẻ**. HOSE gồm lô lẻ, Vietstock không. Giá trị ròng lệch tối đa khoảng
  0,04 tỷ. Có một ngoại lệ lớn hơn: HPG 2026-09-29, bên bán lệch +13.302 cp, ròng
  −51,81 so với −51,54 tỷ.
- Tổng theo tháng (08/2025–06/2026, 2 mã) so với Vietstock monthly: lệch 0,00–0,03%.
  Tháng 07/2026 của Vietstock là tháng dở dang lúc thu thập (2026-07-02), nên bỏ qua.

### 3.4 Tự doanh: CafeF so với Vietstock và FiinPro

- CafeF so với Vietstock chart (store v2, 24 phiên chung): VNM 23/24, HPG 24/24 trùng tuyệt
  đối (KL và GT).
- 2025-04-03, ròng tự doanh: VNM CafeF −52.605,19 triệu so với FiinPro −52.790,98 triệu
  (lệch 0,35%); HPG −35.456,79 so với −35.595,57 triệu (0,39%).
- Tổng tháng CafeF so với Vietstock monthly: trùng ở một số tháng (VNM 08–10/2025), lệch
  lớn ở tháng khác (HPG 12/2025 mua 1.549,6 so với 2.271,2 tỷ). Giả thuyết **chưa kiểm
  chứng**: số tháng của Vietstock gồm thỏa thuận hoặc đã được điều chỉnh, còn số ngày chỉ
  gồm khớp lệnh. Số phiên không thiếu (436 = 436).

## 4. Thời điểm công bố / PIT

- Lúc 2026-10-08 10:35 ICT, HOSE và Vietcap đều trả phiên mới nhất là **2026-10-07**, không
  có dòng trong phiên của ngày 08/10. Không có timestamp công bố trong payload. Giờ xuất
  hiện lần đầu của phiên d **chưa đo được** trong thời gian probe.
- Đề xuất cho replay offline: `assumed_available_at = d 18:00 ICT` (sau cutoff 16:00 và
  lịch 17:05 hiện có), chỉ dùng cho quyết định từ phiên d+1. Ghi rõ đây là **giả định,
  không phải quan sát**. Cần đo song song: poll 15:15 / 16:00 / 17:05 vài ngày để thay
  giả định bằng số đo.
- Backfill là **truy hồi (retrospective)**: không chứng minh được dữ liệu không bị sửa
  sau ngày công bố. Bằng chứng gián tiếp: HOSE trùng tuyệt đối Vietcap, và trùng (trừ lô
  lẻ) với Vietstock được thu gần thời gian thực (available_at 2026-09-30..10-07). Không
  thấy dấu hiệu chỉnh sửa trọng yếu trong cửa sổ đó.
- **Room:** `currentRoom` của HOSE ngày d bằng room **đầu phiên d** (bằng room cuối ngày
  d−1). Room của Vietcap ngày d trùng room HOSE ngày d+1 ở 332/434 (VNM) và 333/434 (HPG)
  phiên; phần còn lại lệch do điều chỉnh thanh toán / sự kiện doanh nghiệp. Không trộn
  hai định nghĩa. Room không được suy ra từ dòng ròng.
- Khóa ngày: `reportDate` = 00:00 UTC của ngày giao dịch VN. Chuyển bằng UTC, **không**
  dùng giờ địa phương. Lỗi lệch một ngày của Vietstock (`/Date(ms)/` ở 00:00 ICT) không
  xuất hiện ở đây.
- Mã phân biệt hoa thường (`vnm` → 0 dòng). Lịch sử chỉ bao gồm thời gian niêm yết trên
  HOSE (SHB: 1.245 phiên); giai đoạn mã còn ở HNX/UPCoM không có.

## 5. Dữ liệu legacy `data/raw/foreign/*.jsonl` (chỉ đọc, vẫn quarantine)

| File | Dòng | Khoảng | Số phiên / kỳ | Số mã | Ghi chú |
|---|---|---|---|---|---|
| `ndtnn.jsonl` (Paging, chi tiết) | 2.500 | 2026-04-02..2026-07-01 | 50 | 51 (A–C, giới hạn 50 dòng ẩn danh) | Ngày chỉ Thứ 2–5, không có Thứ 6, nên có dấu hiệu lệch ngày |
| `ndtnn_chart.jsonl` | 7.400 | 2026-06-04..2026-09-20 | 74 | 100 | Ngày rơi vào CN–Thứ 5, tức **lệch −1 ngày**. Chỉ khớp HOSE khi dịch +1 ngày (khớp tuyệt đối 13/74 VNM, 10/74 HPG; còn lại lệch lô lẻ) |
| `tudoanh_chart.jsonl` | 7.500 | 2026-06-03..2026-09-20 | 75 | 100 | Cùng lỗi lệch ngày |
| `price_board_snapshots.jsonl` | 2.630 | 2026-07-02..2026-09-21 | 25 | 100 | 130 khóa (ngày, mã) trùng; 100 dòng gắn ngày Chủ nhật |
| `ndtnn_monthly` / `tudoanh_monthly` | 1.200 / 1.200 | 08/2025..07/2026 | 12 | 100 | 07/2026 dở dang |
| `ndtnn_quarterly` / `tudoanh_quarterly` | 1.191 / 1.200 | Q4/2023..Q1/2026 | 12 | 100 | — |

Không có `observed_at` hay provenance. Legacy **không** phủ 2025-01..2026-03 ở mức ngày,
và giữ nguyên trạng thái quarantine. Không đề xuất promote.

## 6. Khuyến nghị

**Đường miễn phí tốt nhất:**
- Khối ngoại: HOSE per-symbol, chuẩn A. Mỗi mã 1 request `pageSize=500` phủ khoảng 2 năm,
  ~105 mã × ≥3,1 s ≈ 6 phút.
- Đối chiếu: Vietcap IQ, 1 request/mã. Yêu cầu trùng tuyệt đối; nếu lệch, quarantine
  dòng đó.
- Tự doanh: CafeF `GDTuDoanh`, chuẩn B, 1 request/mã. Đối chiếu 20 phiên chồng lấn với
  Vietstock chart (store v2).
- Toàn bộ ở nhãn `research_only` / `retrospective_backfill`.

**Đường trả phí tốt nhất (không mua, chỉ ghi nhận):** `vnstock_data` (chương trình thành
viên/sponsor của vnstocks.com; trang `/insiders-program` và `/onboard-member` không hiển
thị bảng giá khi fetch tĩnh, cần xem `/onboard-member/chinh-sach-thanh-vien`), Vietstock
Finance (5 hạng FREE/BASIC/STANDARD/PRO/PREMIUM theo hướng dẫn VSTF 2025; giá không lấy
được), FiinPro/FiinTrade (nguồn mà báo cáo CTCK trích dẫn). Chỉ đáng cân nhắc nếu cần
tự doanh chuẩn A, có tách thỏa thuận. Khối ngoại không cần nguồn trả phí.

**Fetcher + store cần có (mô phỏng `data/history`, tách khỏi `data/foreign_flows_v2`):**
- Layout `data/flows_history/<UTC>_<id>/`, mỗi lần fetch là thư mục mới:
  - `raw/hose_foreign_<SYM>.json`, `raw/vci_ph_<SYM>.json`, `raw/cafef_td_<SYM>.json` (byte gốc);
  - `foreign/<SYM>.csv`, `proprietary/<SYM>.csv`, `issues/<SYM>.json`;
  - `manifest.json` ghi sau cùng, nguyên tử, chứa sha256 raw/csv/issues, request,
    `fetched_at`, số lần thử, khoảng ngày, số dòng.
- Schema dòng khối ngoại: `date`, `symbol`, `matched_buy/sell_volume`,
  `matched_buy/sell_value_vnd`, `putthrough_buy/sell_volume`, `putthrough_buy/sell_value_vnd`,
  `total_*`, `net_*`, `room_total`, `room_current_open` (định nghĩa HOSE), `includes_odd_lot=true`,
  `source=hose_api`, `source_grade=A`, `observed_at` (lúc fetch), `assumed_available_at`,
  `availability_basis=assumed|observed`, `vci_reconciled` (bool).
- Schema dòng tự doanh: KL/GT mua bán, `scope=unspecified_probably_matched`,
  `source=cafef`, `source_grade=B`, `vietstock_overlap_ok`.
- Kiểm tra bắt buộc:
  - ngày thuộc lịch phiên VN, không trùng khóa;
  - giá trị không âm, `total = main + bigLot`;
  - khối lượng có thể lẻ, đúng theo thiết kế;
  - giá trị/khối lượng nằm trong biên giá trần/sàn của ngày;
  - HOSE và Vietcap trùng tuyệt đối;
  - mã trả về đúng mã yêu cầu (phân biệt hoa thường);
  - phủ đủ 2025-01-02..as_of;
  - ghi `ends_before_as_of` cho mã đã hủy hoặc mới chuyển sàn.
- Vận hành: pacing ≥3,1 s, retry hữu hạn, dừng sau 5 lỗi liên tiếp, không lưu cookie, có
  `verify` (băm lại và dẫn xuất lại từ raw) và `status`. Không đổi lịch 17:05, không ghi
  vào store v2.
- Theo `AGENTS.md`: đây là thay đổi data contract. Phải đăng ký giả thuyết trong
  `configs/research/research_gate_v1.json` trước khi dùng cho backtest. Dữ liệu 2025–2026
  đã xem là development data, không phải holdout.

**Hành động tiếp theo có giới hạn:** viết fetcher dry-run cho 2–3 mã, đo giờ công bố HOSE
trong 3–5 phiên, rồi mới chạy VN100.
