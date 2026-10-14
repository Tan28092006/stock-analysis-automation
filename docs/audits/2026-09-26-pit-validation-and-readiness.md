# Kiểm chứng VN30 đúng lịch sử và khả năng dùng khuyến nghị

Ngày 26/09/2026. **Kết luận: NO-GO cho việc tin hoàn toàn để giao dịch tiền thật.**
Đã hoàn thành nghiên cứu bổ sung và một đợt sửa lỗi kiểm soát; chưa hoàn thành
tiêu chí “đã chứng minh chạy tiền thật”. Không đặt lệnh, không nâng cấp mô hình,
không bật ML, không thay tham số để chọn ra kết quả thắng.

Người dùng tự đặt lệnh với vốn nhỏ. Không cần kết nối môi giới cho phạm vi này.
Vốn nhỏ không giải quyết dữ liệu sai, tín hiệu dùng thông tin tương lai hay thiếu
lợi thế sau chi phí. Các mô phỏng vẫn dùng 1 tỷ VND và lô 100 cổ phiếu theo giao
thức cũ: tỷ lệ lợi nhuận không tự động áp dụng cho tài khoản nhỏ hơn vì làm tròn
lô, số mã mua được và mức tiền mặt sẽ khác.

## 1. Thí nghiệm đã thực hiện

Giai đoạn trước: 36 lượt của bốn chiến lược × ba kịch bản × ba giai đoạn, dùng
rổ hiện tại cố định. Xem [giao thức](2026-09-26-momentum-hypotheses-protocol.md)
và [kết quả đầy đủ](2026-09-26-momentum-hypotheses-results.md), bao gồm liên kết
bài báo, giả thuyết đối lập và những phần không thể tái lập bằng dữ liệu hiện có.

Giai đoạn bổ sung: **72 lượt** = cùng 36 trường hợp × hai chính sách thành phần:
rổ hiện tại cố định và thành phần VN30 đúng thời điểm. Đây là nghiên cứu độ nhạy
sau khi đã xem dữ liệu, không phải holdout mới. Không tối ưu tham số, không kết
hợp biến thể. [Giao thức bổ sung](2026-09-26-pit-universe-protocol.md) được commit
ở `a42f2b1` trước khi xem lợi nhuận của lượt bổ sung. Mã chạy: `01330f8`.

Hai chính sách dùng cùng một snapshot mới: 35 cổ phiếu từng/đang thuộc VN30 và
VNINDEX; dữ liệu từ đầu 2024 đến 25/09/2026, 36/36 file qua kiểm tra snapshot.
Rổ chỉ thay khi thông báo đã được biết ở phiên quyết định và có hiệu lực ở phiên
khớp. Không chọn mã bằng thông báo tương lai. Giữ tái cân bằng hàng tháng; mã bị
loại ngừng được mua bổ sung ngay khi có hiệu lực, nhưng vị thế cũ chờ lần tái cân
bằng kế tiếp. Đây không phải mô phỏng ETF bám đúng mọi ngày thay rổ.

Các mốc được phục dựng bằng tài liệu nguồn:

| Hiệu lực | Thêm | Loại | Bằng chứng |
|---|---|---|---|
| 03/02/2025 | Rổ đầu kỳ 30 mã | — | [Tài liệu gốc HOSE lưu trên FireAnt, trang 1](https://mobi.fireant.vn/News/NewsAttachedFile/1732373) |
| 04/08/2025 | DGC | BVH | [SSI, kết quả 16/07/2025](https://www.ssi.com.vn/khach-hang-ca-nhan/ban-tin-etf?page=2); ngày hiệu lực theo [lịch Mirae Asset](https://masvn.com/api/attachment/file/1751856428359-202507ForecastHOSEIndexesreviewin3_25_VN.pdf), phù hợp ngày hết hạn rổ cũ trong tài liệu HOSE |
| 02/02/2026 | VPL | BCM | [SSI, kết quả 23/01/2026](https://www.ssi.com.vn/khach-hang-ca-nhan/ban-tin-etf); [danh sách gốc HOSE tháng 1](https://static2.vietstock.vn/vietstock/2026/1/21/21012026_cbtt___danh_sach_thanh_phan_hose_index_thang_1_2026.pdf) |
| 13/05/2026 | BSR | DGC | [SSI, thông báo thay thế bất thường 08/05/2026](https://www.ssi.com.vn/khach-hang-ca-nhan/ban-tin-etf) |
| 03/08/2026 | MCH, TCX | PLX, TPB | [SSI, kết quả 16/07/2026](https://www.ssi.com.vn/khach-hang-ca-nhan/ban-tin-etf) |

Timeline có ngày biết/ngày hiệu lực riêng và không cho truy vấn vượt phạm vi
03/02/2025–25/09/2026. Đây là phục dựng thủ công có nguồn, **không phải** dữ liệu
thị trường point-in-time hoàn chỉnh hoặc bằng chứng toàn hệ thống hết leakage.

## 2. Kết quả H1/2026 sau khi dùng đúng thành phần lịch sử

Đơn vị %, đã trừ phí và trượt giá mô phỏng ở chiến lược. VNINDEX là chỉ số giá
gộp, từ mở cửa phiên đầu đến đóng cửa phiên cuối, không phải ETF có thể mua với
lợi nhuận tổng sau chi phí. Cổ phiếu cuối kỳ được định giá, chưa bán cưỡng bức.

| Phương án | Rổ hiện tại cố định | Đúng rổ lịch sử | So với VNINDEX | Sụt giảm tối đa |
|---|---:|---:|---:|---:|
| Momentum gốc | -2,66 | +1,53 | -2,54 điểm % | -14,35 |
| A: xếp hạng sau điều chỉnh thị trường | -2,60 | +1,56 | -2,51 điểm % | -14,35 |
| B: lọc thời điểm mua bằng giảm tương đối 5 phiên | -1,03 | +2,56 | -1,52 điểm % | -12,94 |
| C: điều chỉnh tỷ trọng theo biến động danh mục | +1,37 | +4,31 | +0,23 điểm % | -11,67 |
| VNINDEX | +4,08 | +4,08 | — | -16,38 |

Rổ cố định trên snapshot mới cho lợi nhuận normal giống hệt báo cáo trước ở
cả ba giai đoạn và cả bốn phương án. Vì vậy, khác biệt ở bảng này không phải do
đổi độ dài lịch sử/snapshot. Với momentum gốc, sáu tháng đều chọn STB trong rổ
lịch sử thay MCH trong rổ cố định. A cũng không được dùng BSR trước hiệu lực vào
VN30. Không thể quy hết chênh lệch lợi nhuận cho một cổ phiếu vì tỷ trọng, tiền
mặt, làm tròn và chi phí cũng thay đổi theo đường đi danh mục.

Không được biến kết quả C thành tuyên bố đã có lợi thế chắc chắn:

| H1/2026, rổ lịch sử | Gốc | A | B | C | VNINDEX |
|---|---:|---:|---:|---:|---:|
| Chi phí thường | +1,53 | +1,56 | +2,56 | +4,31 | +4,08 |
| Gấp đôi chi phí mô phỏng | +1,20 | +0,88 | +2,05 | +4,01 | +4,08 |
| Trễ thêm một phiên, giữ nguyên tín hiệu/khối lượng | +1,40 | +1,15 | +5,72 | +3,78 | +4,08 |

Khoảng bootstrap 98,3333% cho chênh lệch lợi nhuận ngày bình quân quy năm so với
momentum gốc: A **[-9,12; +10,37]**, B **[-3,58; +8,79]**, C **[-9,19; +19,82]**
điểm %/năm. Tất cả chứa 0. Đây không phải khoảng tin cậy của lợi nhuận gộp sáu
tháng; không sửa được toàn bộ lịch sử thử nhiều mô hình. C chỉ vượt chỉ số 0,23
điểm % ở kịch bản thường, mất mức vượt ở hai stress. **Cả ba giả thuyết đều chưa
đạt tất cả tiêu chí đã khóa.**

Các giai đoạn chẩn đoán khác, cùng rổ lịch sử và chi phí thường:

| Giai đoạn | Gốc | A | B | C | VNINDEX |
|---|---:|---:|---:|---:|---:|
| 07–12/2025, 130 phiên | +45,53 | +45,53 | +44,76 | +39,90 | +29,49 |
| 01/07–25/09/2026, 60 phiên | +0,70 | +0,70 | +3,30 | +1,54 | -4,02 |

Mỗi giai đoạn khởi tạo tiền riêng, không nối các lợi nhuận. B và C đều kém gốc
ở nửa cuối 2025. Tín hiệu thuận lợi trong 60 phiên gần đây không thay thế một
kiểm định tương lai chưa từng dùng để nghiên cứu. B là lớp lọc mua momentum,
không phải bằng chứng bổ sung cho hệ bắt đáy RSI/Bollinger khan hiếm giao dịch.

## 3. Những lỗi vận hành đã xác nhận và sửa

Đọc trực tiếp `data/raw/prices_hist` cho thấy: dữ liệu phần lớn đến 21/09 trong
khi phiên hoàn tất là 25/09; thiếu MCH/TCX; VNINDEX/VIB không đạt kiểm tra OHLCV;
thư mục còn nhiều mã ngoài rổ. Đây là nguồn mà dashboard cũ dùng, khác snapshot
nghiên cứu đã kiểm chứng. Không dùng số backtest sạch để chứng nhận nguồn này.

Đã sửa bằng các vòng RED/GREEN thực sự:

- `mr_scan` chỉ tải mô hình khi `ml.enabled is True`; mô hình thử nghiệm hoặc
  thiếu `release.live_approved is True` không phát xác suất trên dashboard.
  Điều kiện thời gian của mô hình vẫn bắt buộc. `override_enabled` phải bật rõ
  ràng mới có PROB_BUY. Artifact lỗi trở về rule-only, không tự nâng cấp mô hình.
- Cache MR phải mang phiên bản hợp đồng mới và đúng `recent_days`; không tái sử
  dụng cache xác suất cũ. Fingerprint đầu vào sẵn có đã gồm hash artifact/universe.
- MR, momentum và swing đều kiểm tra rổ, độ mới, phiên EOD, lịch sử và OHLCV
  **trước khi** lấy cache/quét/đọc vị thế. Dữ liệu lỗi trả `blocked_data` với các
  danh sách khuyến nghị rỗng; không ghi cache hoặc sửa sổ vị thế trên đường này.
- Giao diện phân biệt dữ liệu không hợp lệ với một kết luận “không có kèo”. Lỗi
  tải API xóa các bảng tín hiệu cũ. `UNKNOWN` không còn bị biến thành nhận định
  “chợ hẹp”; vị thế chưa kiểm tra được không bị hiển thị như đã xác nhận bằng 0.

Kiểm tra trực tiếp mã hiện tại: **cả ba scanner trả `blocked_data`** với thư mục
legacy đang có. Đây là hành vi an toàn có chủ đích, không phải đã sửa xong nguồn
dữ liệu. Không tự thay nguyên thư mục legacy, không tự hợp thức hóa cache bằng
snapshot lịch sử, không thay ledger hay vị thế để khiến giao diện xanh.

Các sửa đổi hiện ở checkout/local commits; không push/deploy hoặc khởi động lại
dịch vụ người dùng đang chạy. Tiến trình cũ chưa reload mã không tự có các chặn
mới. Trình duyệt QA dùng một server preview riêng chỉ đọc, chặn POST/DELETE.

## 4. Kiểm chứng kỹ thuật

- 344 test Python đạt, 23 cảnh báo thư viện/kiểu dữ liệu, không test nào bỏ qua.
  Artifact: `data/paper/backtests/pit_validation_20260926/tests-guard-final.xml`.
- Branch-inclusive coverage: runner thành phần lịch sử 95%; MR scanner 80%;
  data guard 100%; tổng ba module đo 87%. Không tuyên bố toàn repo đạt mức này.
- 18 test riêng cho lịch sử thành phần/CLI, bao gồm 72 replay tổng hợp; 12 test
  admission ML; 9 test guard dữ liệu. Bộ test độc lập không đặt lệnh.
- Ledger tính lại độc lập cho cả 72 replay: tiền, khoản chờ về, cổ phiếu, NAV,
  lô 100 và T+3. Sai lệch lớn nhất **0,000000477 VND**.
- Test JavaScript chạy chính các hàm render; Chromium preview ở 1440/768/375px
  không có page error, API scanner không lỗi HTTP, không có nút ghi lệnh trong
  các bảng mua bị chặn. Đã xem ảnh desktop/mobile và sửa lỗi UNKNOWN phát hiện
  qua ảnh. Đây không phải chứng nhận accessibility/toàn bộ giao diện.
- Browser tích hợp lỗi ACL sandbox; fallback dùng Playwright và Chromium đã có
  sẵn, phiên riêng không dùng hồ sơ đăng nhập người dùng. Không tải thêm browser.
- Ba snapshot nguồn của hai giai đoạn được xác minh hash lại và vẫn `verified`.

Lệnh tái lập chính:

```powershell
python -m scripts.historical_universe --manifest data/paper/backtests/pit_validation_20260926/snapshot/manifest.json --output <new-unused-result-path.json>
python -m pytest -q
node tests/dashboard_readiness_ui.cjs
```

Fingerprint artifact 72 replay:

- Result: `e0c5c1f21076fe7b1e9e8d16937b5243c139ee7f969b4a9eba7c6065a3fbba17`
- Snapshot manifest: `bbee26aa3aa83f0573350343c05827230f900356c00003165247771728687e48`
- Timeline: `02cf6916fe086a4b961a6dfb0ba8e2d55cc092f6cef9b94040cf379d15f74a1d`

Artifact được chạy trước các sửa serving/UI; chứa hash từng nguồn và Git SHA
tại lúc chạy. Các sửa sau không thay công thức ranking/replay. Giữ nguyên cả
artifact cũ lẫn mới, không ghi đè để che sự khác nhau của giả định.

## 5. Điều còn thiếu để nói về tiền thật

1. **Nguồn vận hành**: nối dashboard vào snapshot mới đã xác minh, đúng rổ và
   đúng phiên, với nhận dạng nguồn xuyên suốt tới tín hiệu. Guard hiện tại kiểm
   tra chất lượng/độ mới, không chứng nhận lineage điều chỉnh doanh nghiệp.
2. **Một hợp đồng danh mục**: daily top picks của dashboard không tương đương
   danh mục backtest tái cân bằng tháng có buffer cổ phiếu đang giữ. Phải đồng
   nhất lịch quyết định, trạng thái đang giữ, tiền mặt, khối lượng thực tế và
   giới hạn khớp trước khi dùng kết quả backtest cho các nút mua/giữ đó.
3. **Corporate actions**: hash dữ liệu đúng không chứng minh dữ liệu đó đã biết
   trong quá khứ. Thiếu lineage giá chưa điều chỉnh/điều chỉnh, cổ tức và quyền;
   thiếu giá tham chiếu chính thức cho từng ngày có sự kiện.
4. **Lợi thế ngoài mẫu**: mới có một hồ sơ paper ngày 25/09/2026. Forward return
   momentum hiện là thông tin từng mã, không phải P&L danh mục có vốn. Không có
   đủ quan sát độc lập hay khớp lệnh thực tế để kết luận thắng sau chi phí.
5. **Thực thi thủ công**: cần lưu tín hiệu trước phiên, giá/giờ/khối lượng thực
   khớp hoặc bị bỏ qua, chi phí thực và đối chiếu tiền/cổ phiếu. Không cần gửi
   mật khẩu hay thông tin nhận dạng tài khoản; số lượng vốn nhỏ không miễn bước
   đối chiếu này. Lợi nhuận tương lai không thể được chứng minh trong ngày nghỉ.

Không hạ các điều kiện này chỉ vì một biến thể thắng chỉ số một đoạn ngắn.
Mô hình được nghiên cứu kỹ hơn và phần phát tín hiệu an toàn hơn, nhưng không có
cơ sở để khuyên người dùng tin hoàn toàn hoặc gọi toàn bộ yêu cầu là hoàn tất.

## 6. Minh bạch sự cố kiểm thử trước đó trong cùng phiên

Lần pytest toàn thư mục trước khi thêm `pytest.ini` đã thu thập các script
`scratch/test_*.py` có side effect, gây ghi cache giá và append 30 dòng scan thử.
Chi tiết, hash, sao lưu và cách phục hồi nằm trong
[báo cáo giai đoạn trước](2026-09-26-momentum-hypotheses-results.md).
30 dòng được chứng minh là đuôi append đã được tách sau khi sao lưu; 30 cache
giá được phục hồi từ snapshot đã xác minh; bản scan sinh nhầm bị cách ly.
`data/processed/latest_scan.json` cũ đã bị ghi đè trước khi cách ly và không có
bản gốc để phục hồi: hiện file active này vắng mặt, không giả vờ đã khôi phục.
Snapshot nghiên cứu và vị thế thật không bị sửa. Các lần test cuối chỉ thu
thập `tests/`; giữ nguyên toàn bộ chứng cứ cách ly, không xóa để làm sạch báo cáo.
