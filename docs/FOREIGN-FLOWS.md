# Khối ngoại & tự doanh — vận hành dữ liệu v2

## Phạm vi

Thu thập dữ liệu, không phát lệnh và không tự bật tín hiệu giao dịch. Mỗi ngày
lấy hai nhóm riêng biệt (khối ngoại / tự doanh), giữ 100 mã đã thu thập trước
đây, bổ sung MCH/TAL/TCX/VCK/VPX và union danh sách giao dịch hiện tại trong
`configs/universe_vn30.json`. Tập hiện tại là 105 mã; không xóa mã cũ và không
coi tập thu thập này là lịch sử thành phần VN30/VN100.

## Cập nhật tự động

Windows task sẵn có `StockAgent_EOD_Update` chạy `run_eod_update.bat` lúc
**17:05, thứ Hai–thứ Sáu, múi giờ Việt Nam**. BAT nay chạy collector v2 trước,
sau đó chạy paper runner độc lập, kể cả khi khối ngoại lỗi. Không stage/commit/
push dữ liệu, không gọi broker, không retrain model.

- Collector retry tối đa 3 lần mỗi request, timeout hữu hạn, giới hạn nhịp gọi.
- BAT dùng `--retry-stale`: nếu response hợp lệ nhưng chưa có phiên EOD mới,
  chỉ tải lại các mã thiếu, tối đa ba batch bổ sung sau các khoảng chờ 60, 300
  và 900 giây. Mỗi batch giữ raw/timestamp riêng; kiểm tra cuối trên **toàn bộ**
  tập mã ban đầu. Hết lượt vẫn thiếu → không báo thành công.
- Không thử lại để che lỗi schema, quarantine bất thường, gap lịch sử, hash
  mismatch hoặc batch dở dang. Đổi phiên EOD hay đồng hồ lùi trong lúc chờ
  sẽ chặn job. Tổng chờ tối đa 21 phút cộng thời gian request hữu hạn; paper
  chạy sau bước này kể cả khi thu thập thất bại. Lịch 17:05 không đổi.
- Dừng gọi hàng loạt khi 5 cặp mã/nguồn lỗi liên tiếp; ghi rõ phần chưa lấy được.
- Lấy chồng cửa sổ gần nhất của nguồn (~20 dòng) để bù ngày bỏ lỡ và giữ revision.
- Máy/người dùng/network phải sẵn sàng; `StartWhenAvailable` đang bật, `WakeToRun`
  không bật. Máy tắt/ngủ quá lâu có thể mất dữ liệu ngoài cửa sổ của nguồn.
- Ngày nghỉ vẫn tính phiên EOD đã hoàn tất theo lịch Việt Nam; không gán dữ liệu
  bảng giá vào ngày cuối tuần hoặc tự giả định thiếu dòng nghĩa là không giao dịch.
- Một bước lỗi → mã thoát khác 0. Paper lỗi vẫn được báo riêng; thành công của
  collector không được trình bày thành thành công toàn bộ hệ thống.

Chạy tay từ thư mục repo, với Python đã cài dependencies:

```text
python -m stock_agent.pipeline.foreign_refresh
python -m stock_agent.pipeline.foreign_refresh --retry-stale
python -m stock_agent.pipeline.foreign_refresh --status
python -m stock_agent.pipeline.foreign_refresh --symbols ACB MBB MCH TCX
python -m stock_agent.pipeline.foreign_refresh --migrate-legacy
```

Lệnh giới hạn mã là để chẩn đoán, không thay đổi danh sách của lịch tự động.
Chạy tay không có `--retry-stale` chỉ thu thập một batch, không chờ dữ liệu
công bố muộn. `--status` chỉ đọc; không kết hợp với chế độ retry/migration.
Kết quả retry có `requested_symbols`, `attempts`, `attempt_manifest_paths`
và `retry_exhausted`. `latest_status.json` vẫn mô tả batch cuối cùng, có thể
chỉ gồm vài mã; dùng `--status` để kiểm tra toàn bộ tập mặc định hiện tại.
Không chạy lại toàn bộ BAT chỉ để sửa dữ liệu khối ngoại vì BAT còn ghi paper
recommendations; dùng CLI collector riêng ở trên.

## Hợp đồng lưu trữ và đọc dữ liệu

Thư mục `data/foreign_flows_v2/` là runtime state, không commit lên Git:

```text
runs/<UTC time + id>/
  raw/foreign_<symbol>.json
  raw/proprietary_<symbol>.json
  records.jsonl
  quarantine.json
  manifest.json
latest_status.json
legacy/<id>/research_only.jsonl + quarantine.json + manifest.json
```

`manifest.json` được ghi sau cùng. Mỗi response nguyên bản và bảng chuẩn hóa có
SHA256. Loader kiểm tra hash, đọc lại raw và đối chiếu phép chuẩn hóa trước khi
trả dữ liệu. Không lưu cookie, tài khoản hoặc CSRF token; lỗi log chỉ ghi loại lỗi.
Collector có khóa chống chạy đồng thời, status được thay thế nguyên tử.

- Canonical: `buy_value_vnd`, `sell_value_vnd`, `net_value_vnd` là **VND**;
  `buy_volume/sell_volume/net_volume` là cổ phiếu.
- `date` là ngày Việt Nam từ `TradingDate` gốc. `observed_at` là thời điểm nhận
  response UTC; `available_at` không trước lúc hoàn tất và công bố batch.
- `investor=foreign|proprietary`; `trade_scope=provider_chart_aggregate`.
  Không giả vờ chart là chỉ khớp lệnh, không ghép với detailed matched-only.
- Giá trị thiếu, NaN, âm, volume lẻ, sai cấu trúc, phiên tương lai/chưa EOD,
  trùng ngày trong cùng response đều có kiểm tra. Không tự điền zero.
- Chỉ sửa ngày lịch sử khi có timestamp gốc. Dữ liệu cũ không có thời điểm nhận
  **không thể** trở thành bằng chứng point-in-time chỉ nhờ chuẩn hóa.

```python
from stock_agent.data.foreign_flows import load_flows

# Bảng latest để xem dữ liệu. f_* = khối ngoại, td_* = tự doanh.
# Các cột *_val là TỶ VND để tương thích interface cũ, không phải canonical VND.
panel = load_flows()

# BẮT BUỘC dùng as_of khi tạo features lịch sử.
panel = load_flows(as_of="2026-10-01T08:00:00+07:00")
```

Lọc availability **trước** khi chọn revision mới nhất theo mã/ngày/nhóm/nguồn.
Tải cùng payload nhiều lần không tạo dòng trùng trong panel; các lần nhận nguồn
vẫn được giữ để audit. Missing nhóm tự doanh không được điền bằng khối ngoại.
Giá trị hoàn toàn zero chỉ hợp lệ khi nguồn trả rõ cả giá trị và khối lượng zero.

## Dữ liệu cũ

Migration ghi vào khu `legacy/`, không thay thế 8 JSONL cũ và không đọc file cookie.
Detailed BuyVal/SellVal: triệu VND → VND. Chart: tỷ VND → VND. Chỉ detailed có
TradingDate gốc mới phục hồi được session; chart cũ còn lưu stored_date và
`date=null`. Board/period aggregates chưa đủ hợp đồng daily được cách ly.
Mọi hàng legacy có `available_at=null`, không có mặt trong `load_flows()` mặc định.

Giữ nguyên nghiên cứu trước: không dùng những ngày vừa backfill để giả rằng hệ
thống đã nhìn thấy chúng trong quá khứ. Lịch sử tự doanh và khối ngoại cũng phải
qua kiểm định tín hiệu riêng trước khi được dùng để ra quyết định.

## Xử lý sự cố

- `partial`: kiểm tra `missing_latest`, `gaps`, `errors`, `quarantine.json`.
  Retry collector; không sửa status thủ công thành ready.
- `blocked`: không có dữ liệu hợp lệ, nguồn đổi schema, artifact bị sửa hoặc
  batch bị gián đoạn. Giữ raw/evidence; điều tra nguyên nhân trước khi dùng.
- `.collect-lock` tồn tại: xác minh collector cũ đã dừng. Không xóa khóa của
  tiến trình đang chạy. Chỉ dọn khóa orphan sau điều tra; thư mục run chưa commit
  phải được giữ để kiểm tra, không giả làm một run thành công.
- Hash mismatch: kiểm tra file nguồn và manifest, không tính lại hash để che lỗi.
- Dữ liệu quá cũ: mất mạng/máy ngủ/nguồn lỗi hoặc lịch chưa chạy. `--status`
  phải so với phiên EOD hiện tại, không chỉ nhìn status từ lần chạy trước.

Định nghĩa sẵn sàng ở đây chỉ là dữ liệu đầy đủ trong phạm vi đã thu thập.
Không bảo đảm bản chart không bị provider điều chỉnh, không có lịch sử vô hạn,
và không thay thế kiểm định lợi thế giao dịch.
