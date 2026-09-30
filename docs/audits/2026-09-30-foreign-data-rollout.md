# Foreign flow v2 — kiểm chứng triển khai 30/09/2026

## Kết luận và phạm vi

Collector/loader mới đã chạy trên nguồn công khai thực tế; dữ liệu khối ngoại
và tự doanh được tách biệt, chuẩn hóa về VND/cổ phiếu, có phiên từ timestamp
nguồn và thời điểm hệ thống thực sự nhận dữ liệu. Không thay đổi chiến lược,
không retrain, không phát lệnh. Hướng dẫn: [FOREIGN-FLOWS.md](../FOREIGN-FLOWS.md).

Quy trình MLE/TDD ảnh hưởng trực tiếp đến cách triển khai: đóng băng data contract
trước khi sửa, tái hiện lỗi bằng test đỏ, chỉ chấp nhận sau test xanh; dữ liệu cũ
thiếu provenance không được nâng cấp thành lịch sử point-in-time bằng suy đoán.

## Tải thật và kiểm tra dữ liệu

Run đầy đủ:
`data/foreign_flows_v2/runs/20260930T054946415775Z_306168e5/manifest.json`.

- 102 mã: 100 mã thu thập trước đây, cộng MCH và TCX trong universe hiện tại.
  Đây là universe thu thập, không phải thành phần VN30 lịch sử.
- 204 response nguồn; 3.876 dòng hợp lệ = 102 mã × 19 phiên × 2 nhóm nhà đầu tư.
- Phạm vi phiên 03/09–29/09/2026; không thiếu mã/phiên trong phạm vi này.
- 102 dòng khối ngoại ngày 30/09 còn intraday bị cách ly, không dùng làm EOD.
- Batch hoàn tất 12:56:12 giờ Việt Nam ngày 30/09. Tất cả lịch sử backfill chỉ
  khả dụng từ thời điểm batch này, không được giả là đã biết vào ngày giao dịch.
- Loader kiểm lại raw/hash/normalization: 1.938 dòng wide, 102 mã, không trùng
  symbol/date, không thiếu nhóm khối ngoại/tự doanh. Truy vấn `as_of` đầu ngày
  30/09 trả 0 dòng, đúng vì lúc đó hệ thống chưa thu thập vintage này.
- Kiểm tra freshness bằng `foreign_refresh --status`: `ready`, expected session
  29/09, `missing_latest` rỗng và `gaps` rỗng tại thời điểm kiểm tra trước EOD 30/09.

Sau khi chốt code `3e8d122`, chạy smoke riêng MBB/MCH/TCX:
`data/paper/foreign_v2_smoke/runs/20260930T055835742671Z_dbabcd42/manifest.json`.
114 dòng hợp lệ, 3 dòng intraday cách ly, exit 0; không ghi đè status của run 102 mã.

Migration:
`data/foreign_flows_v2/legacy/20260930T054956234151Z_1008a795/manifest.json`.

- 17.400 dòng chuẩn hóa **research-only**, trong đó 2.500 dòng phục hồi được ngày
  từ timestamp nguồn; chart cũ thiếu timestamp giữ `date=null` và `stored_date`.
- 7.421 dòng board/tháng/quý hoặc không đủ hợp đồng daily được cách ly.
- Cả 8 file raw cũ giữ nguyên SHA256. Không đọc hoặc di chuyển file cookie.
- Không dòng legacy nào vào loader mặc định; không có timestamp nhận dữ liệu
  thì không đủ điều kiện đưa vào kiểm định point-in-time.

## Tự động cập nhật

Đã kiểm tra trực tiếp Windows task `StockAgent_EOD_Update`: action
`D:\Chungkhoan\run_eod_update.bat`, 17:05 thứ Hai–thứ Sáu, task enabled,
timezone `SE Asia Standard Time`, `StartWhenAvailable=true`, `WakeToRun=false`.
BAT chạy collector trước, luôn tiếp tục paper runner kể cả collector lỗi;
không báo SUCCESS nếu một bước lỗi. Retry, timeout, circuit breaker, khóa ghi
đồng thời và lấy chồng cửa sổ nguồn nằm trong collector.

Đã kiểm tra BAT bằng subprocess stub trên Windows, không gọi BAT thật để tránh
ghi thêm paper/ledger ngoài phạm vi. Lịch 17:05 sau thay đổi **chưa đến giờ chạy**
tại thời điểm kiểm chứng; đã xác nhận cấu hình và tải thật qua collector riêng,
không khẳng định một lần chạy tương lai đã thành công. Máy/tài khoản/network
phải sẵn sàng. Không tạo lịch trùng hoặc tự bật wake-from-sleep.

Lần lịch cũ 29/09 có LastTaskResult=2: log cho thấy paper ngày 29/09 ghi thành công
nhưng chấm các paper 25/09 và 28/09 bị raw/canonical OHLC mismatch. Đây là lỗi
paper snapshot có trước, chưa sửa trong phạm vi chuẩn hóa khối ngoại. Thành công
của collector không chứng minh cả hệ thống giao dịch đã sẵn sàng.

## Kiểm định

- 36 test tập trung vào flows/CLI/launcher: pass.
- Toàn bộ repo: **455 passed**, 23 warnings có sẵn từ pandas/matplotlib/SHAP và
  thư viện phụ thuộc; không có test thất bại.
- Coverage hai module mới/sửa: 93% từ suite test; 95% khi cộng smoke nguồn thật
  trên code đã chốt. Không dùng coverage thay thế kiểm định chất lượng nguồn.
- CI contract job được bổ sung ba file test flows/CLI/launcher. Chưa push hoặc
  chạy GitHub Actions; job replay từ xa vẫn cần artifact snapshot được cấu hình.
- Full pinned-snapshot regression: **290/290** lần chạy trên 10 block hoàn tất.
  Toàn bộ nội dung `blocks` (kể cả lệnh, NAV, thống kê) bằng chính xác run_v2 trước
  thay đổi; không có source-hash drift. Trạng thái vẫn là
  `research_complete_live_blocked`, không phải chấp thuận giao dịch.
  Artifact: `data/paper/research_gate/20260930/foreign_v2_regression/results.json`.
  SHA256: `4a2a51af57978ee79fdaa31abdc5d0c8ea7108c3b627dbede3f74cf6f9d50b58`.
  Manifest snapshot: `21b2deff6694a2a7218c861a681524b91f311617fc1e84de0598fc8001e155d0`.

## Giới hạn còn giữ nguyên

Nguồn chart cung cấp cửa sổ gần nhất, không đủ khôi phục vô hạn quá khứ khi máy
ngừng chạy lâu. Phạm vi giá trị là `provider_chart_aggregate`, không giả định
đồng nhất với dữ liệu chỉ khớp lệnh hoặc số liệu xác nhận từ Sở. Missing không
được biến thành zero. Snapshot room từ bảng giá không có ngày nguồn đã dừng,
không gán ngày máy cho room hoặc dòng tiền nữa.

`ready` chỉ là sẵn sàng dữ liệu theo hợp đồng hiện tại. Chưa đưa khối ngoại vào
rule bắt đáy/momentum; vẫn cần ablation và bằng chứng prospective riêng.
