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

## Theo dõi phiên tự động và phục hồi dữ liệu công bố muộn

Kiểm tra thực tế sau giờ đóng cửa 30/09: Windows task đã chạy lúc 17:05,
`LastTaskResult=2`, lần kế tiếp 01/10 lúc 17:05. Không còn là một lần chạy
tương lai chưa quan sát. Collector hoàn tất lúc 17:10:52 giờ Việt Nam, không
có lỗi request/schema nhưng trạng thái `partial`: khối ngoại đủ 102 mã,
tự doanh thiếu phiên 30/09 ở 26 mã. Paper scoring cũng vẫn có lỗi riêng;
không suy diễn một nguyên nhân duy nhất cho mã thoát của cả launcher.

Manifest lần tự động:
`data/foreign_flows_v2/runs/20260930T100502494594Z_5d3b63f4/manifest.json`,
SHA256 `088effad804572f40a3cd4decf7e5a96d2741035d5d6f64d1581f44eaddc6543`.
Nó chứa 4.054 dòng; bản gốc được giữ nguyên sau khi phục hồi.

Lần chẩn đoán riêng bắt đầu 18:09:44, hoàn tất 18:11:19, tải lại đúng 26 mã
thiếu: ACB, ANV, BAF, BCM, BID, BMP, BSI, BSR, BVH, BWE, CII, CMG, CTD,
CTG, CTR, CTS, DBC, DCM, DGW, DIG, NVL, POW, REE, TPB, VHM, VPB.
Nguồn lúc này trả đủ phiên 30/09 cho cả hai nhóm: 1.040 dòng hợp lệ,
không quarantine, không gap. Toàn bộ **1.014 dòng chung** với các vintage
đã lưu có cùng giá trị mua/bán và khối lượng; 26 dòng tự doanh ngày 30/09
là dữ liệu mới xuất hiện. Điều này xác nhận độ trễ khả dụng của nguồn qua
các lần nhận response, không xác định chính xác phút công bố giữa hai lần.

Evidence chẩn đoán:
`data/paper/foreign_eod_late_probe/runs/20260930T110944803466Z_924e2998/manifest.json`,
SHA256 `8709d6f85bf1a2e7403bb0fb91230a759345d92793caa45a1d2a419ee17b956c`.
HTTP thành công không bảo đảm dữ liệu phiên mới đã được công bố. Retry lỗi
mạng hiện tại không tự khắc phục trường hợp response hợp lệ nhưng còn cũ.

## Bổ sung ba mã thu thập và kiểm tra availability

Runtime union đã có MCH/TCX qua VN30; so với bảng VN100 tháng 7 đã kiểm tra,
ba mã thực sự chưa được collector mặc định thu thập là TAL/VCK/VPX.
Smoke riêng ba mã nhận 120 dòng hợp lệ, 20 phiên từ 03/09 đến 30/09,
cả khối ngoại và tự doanh. Manifest:
`data/paper/foreign_vn100_july_probe/runs/20260930T110837019429Z_27b7ea13/manifest.json`,
SHA256 `25ebdc6d863f7d562c95bfa85a046dd606d307d85bc9433308e1ea7576e43769`.
Không dùng kết quả này để suy ra ngày hiệu lực thành phần VN100 lịch sử.

Sau khi hai smoke pass raw/hash/normalization, CLI collector hiện có tải
**29 mã** (26 thiếu và ba mã bổ sung) vào kho runtime, không chạy BAT/paper
hay scanner. Lần tải thật mới hoàn tất 18:13:41, chứa 1.160 dòng:
`data/foreign_flows_v2/runs/20260930T111213592550Z_2f9f4016/manifest.json`,
SHA256 `ce0b0b7e2401624edde8b5eaedec5e84f14a625bfbc9798880ddf9c089a87b97`.
Không copy smoke sang runtime và không gán lùi thời điểm nhận nguồn.

Kiểm tra toàn kho cho union 105 mã: `health=ready`, đủ phiên EOD 30/09,
không gap, không thiếu nhóm, không run dở dang. Loader trả **2.100 dòng wide**
(105 mã × 20 phiên), không trùng khóa mã/ngày, không thiếu giá trị ròng
khối ngoại/tự doanh. 9.090 dòng vintage được giữ, không phải 9.090 quan sát
độc lập. Truy vấn `as_of=2026-09-30T10:11:00Z` vẫn thiếu đúng 26 giá trị
tự doanh và chưa có TAL/VCK/VPX: dữ liệu muộn không lọt vào vintage cũ.

Đây là phục hồi dữ liệu hiện tại, **chưa sửa xong tự động hóa**. Config mặc
định vẫn union 102 mã; chưa đăng ký TAL/VCK/VPX cho các lần tự động tiếp theo
và chưa có retry riêng cho nguồn công bố muộn. Mã nguồn/config giữ nguyên
trong lúc hồi quy 2.618 replay đang chạy. Bước sửa tiếp cần đăng ký data
contract, RED/GREEN test cho coverage/retry hữu hạn và giữ timestamp mọi
lần nhận nguồn. Không xóa năm mã rời bảng tháng 7 khỏi lịch sử thu thập;
105 là tập thu thập mở rộng, không phải khẳng định VN100 có 105 thành phần.
