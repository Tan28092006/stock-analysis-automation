# Strategy Lab: kiểm thử bắt đáy và mua theo đà tăng

Backend offline mới: `python -m scripts.strategy_lab`. Chạy từ thư mục gốc repo.
Không khởi động server, không đặt lệnh, không ghi cache dashboard/ledger giao dịch,
không train hoặc bật ML. Chưa thêm HTTP endpoint hay giao diện cho chức năng này.

## Mục tiêu và hợp đồng

- So sánh bắt đáy MR và momentum trên **cùng snapshot, cùng giai đoạn, cùng vốn
  khởi tạo và cùng cách tính chi phí**. Mỗi trường hợp có tài khoản mô phỏng riêng;
  không cộng lợi nhuận để suy ra danh mục phối hợp.
- Dùng thành phần VN30 đúng lịch sử cho cả hai. MR chỉ mua mã thuộc rổ tại phiên
  thực thi, theo thông báo đã biết tại đóng cửa phiên tín hiệu. Khi mã rời rổ,
  vị thế MR cũ vẫn thoát theo stop/target/time của kế hoạch ban đầu.
- Momentum dùng lại runner đã kiểm định: tái cân bằng tháng, buffer cổ phiếu đang
  giữ, số lượng dựa trên đóng cửa trước, mua ở mở cửa, thời gian chờ tiền/cổ phiếu
  T+3 thận trọng và lô 100. Không thay bằng bảng daily top picks của dashboard.
- Snapshot bắt buộc qua kiểm tra hash; thiếu dữ liệu, sai lịch, ngoài phạm vi
  thành phần, cấu hình không hợp lệ hoặc ML bật sẽ dừng. Đọc và kiểm tra lại nguồn
  sau khi chạy để phát hiện file bị sửa trong lúc tính toán.
- Cho phép không giao dịch/giữ tiền mặt; không giảm ngưỡng tự động để ép có lệnh.
  P&L và NAV được tính lại độc lập từ dòng tiền/cổ phiếu mỗi phiên.

## Chạy ví dụ

Ví dụ dùng **100 triệu VND giả lập**, không phải giả định về vốn thật của bạn.
Dữ liệu sẵn có đến 25/09/2026; timeline hiện chỉ được xác nhận đến ngày này.
Không dùng nó để kiểm định 28–29/09 khi chưa bổ sung và xác minh nguồn.

```powershell
python -m scripts.strategy_lab `
  --spec configs/research/strategy_lab_h1_2026.json `
  --manifest data/paper/backtests/pit_validation_20260926/snapshot/manifest.json `
  --output data/paper/experiments/my_h1_trial_001
```

Thư mục output phải **chưa tồn tại**. Mỗi lần đổi giả thuyết/cấu hình, dùng tên
mới; không ghi đè kết quả thắng/thua hoặc lượt thất bại. `--timeline` và `--rules`
có thể trỏ tới bản cấu hình nghiên cứu riêng; không sửa cấu hình vận hành để thử.
Chạy `--help` để xem đường dẫn mặc định.

## Cấu hình thí nghiệm

Sao chép cấu hình ví dụ sang file riêng rồi chỉnh:

```json
{
  "start": "2026-01-01",
  "end": "2026-06-30",
  "capital": 100000000,
  "mr_profiles": ["vn30", "strict"],
  "momentum_variants": ["baseline"],
  "scenarios": ["normal", "double_cost"],
  "mr_overrides": {}
}
```

| Trường | Ý nghĩa |
|---|---|
| `capital` | Vốn mô phỏng VND, 1 đến 1.000 tỷ; vốn nhỏ có thể không đủ một lô hoặc khiến danh mục không đủ mã |
| `mr_profiles` | `vn30`: ngưỡng MR có override VN30; `strict`: ngưỡng MR gốc, tắt override VN30 |
| `momentum_variants` | `baseline`, `market_adjusted`, `reversal_entry`, `own_portfolio_vol` theo định nghĩa thí nghiệm trước |
| `scenarios` | `normal`, `double_cost`; gấp đôi commission/tax/slippage mô phỏng, không phải thay đổi thuế thực |
| `mr_overrides` | Áp dụng sau khi chọn profile, chỉ trong lượt nghiên cứu này |

Các override cho phép: `rsi_max` [0,100], `band_touch_pct` [0,5;2],
`vol_climax_min` [0,20], `stop_atr_multiple` [0,1;20], `min_rr` [0,20],
`max_hold_days` số nguyên [1,120]. Ví dụ `{"min_rr": 1.5}` thử yêu cầu reward/risk
cao hơn. Đây là giới hạn kiểm tra input, **không phải thông số được khuyến nghị**.
Tên sai hoặc trường lạ bị từ chối, không bị bỏ qua âm thầm.

Có thể dùng danh sách MR hoặc momentum rỗng để chỉ chạy một hướng. Không được
để cả hai rỗng; không được lặp tên. Tối đa 12 tổ hợp hiện tại. Các override MR
chung áp dụng cho mọi profile MR được chọn, không áp dụng cho momentum.
Không hỗ trợ `delay_one_session` trong lab v1: runner momentum cũ có, nhưng MR
chưa có cùng hợp đồng trì hoãn, nên backend từ chối thay vì so sánh không cân xứng.

## Đầu ra

Mỗi thư mục chứa:

- `request.json`: yêu cầu và đường dẫn nguồn, ghi trước tính toán.
- `input_hashes.json`: hash đầu vào ban đầu, nếu các file có thể đọc được.
- `result.json`: cấu hình hiệu lực từng lượt, hash nguồn/mã, Git SHA, lịch sử
  thành phần, NAV theo ngày, phí, trượt giá, lợi nhuận tháng, excess vs VNINDEX,
  drawdown, exposure, turnover, lệnh mua/bán, lô đóng, vị thế mở và đối chiếu NAV.
  MR có tín hiệu được chấp nhận và tín hiệu bị loại vì sai thành phần lịch sử.
- `summary.md`: bảng đọc nhanh. `closed_lots` là lô đóng, **không nhất thiết là
  các giao dịch độc lập**. Trường `sample_warning` trong JSON nhắc giới hạn mẫu.
- `failed.json`: lỗi nếu chạy thất bại. Có request nhưng chưa có result/failed có
  thể là tiến trình bị ngắt; không coi đó là lượt thành công hoặc tự xóa nó.

Chỉ `status=completed_research` nghĩa là phép tính thành công.
`live_approved=false` luôn giữ nguyên. Không tự chọn “chiến lược thắng” hoặc
đổi cấu hình vận hành theo kết quả. Vẫn thiếu lineage corporate actions/cổ tức,
bằng chứng khớp lệnh thực tế và holdout tương lai; nhiều lần thử làm tăng rủi ro
chọn may mắn. Hash sạch không đồng nghĩa không còn mọi dạng leakage.

## Bằng chứng chạy mẫu ngày 29/09/2026

Lượt `data/paper/experiments/20260929_h1_100m` chạy bằng commit `114d785`;
bản sửa tiếp theo chỉ bổ sung ghi hồ sơ khi file nguồn không tồn tại, không đổi
tính toán. Giai đoạn H1/2026, vốn riêng 100 triệu VND mỗi phương án:

| Phương án | Phí thường | Gấp đôi phí | Lô đóng (phí thường) |
|---|---:|---:|---:|
| MR VN30 | +1,4199% | +1,2895% | 2 |
| MR strict | +0,6062% | +0,5623% | 1 |
| Momentum baseline | -2,0316% | -2,4598% | 9 |

VNINDEX +4,0780%, là chỉ số giá gộp không trừ phí đầu tư. Không phương án nào
trong ví dụ vượt chỉ số. Không so kết quả vốn 100 triệu với 1 tỷ mà bỏ qua thay
đổi số lô mua được, tiền mặt dư và đường đi danh mục. MR có quá ít lô đóng để
kết luận thắng. Ví dụ này kiểm chứng công cụ, không chọn thông số sinh lời.

## Kiểm tra và phạm vi bàn giao

```powershell
python -m pytest tests/test_strategy_lab.py tests/test_strategy_lab_boundaries.py -q
python -m pytest -q
```

25 test riêng bao gồm: cấu hình sai/NaN/bool, thiếu nguồn, snapshot bị sửa,
không đủ phạm vi ngày, ML bật, từ chối ghi đè, CLI end-to-end, độc lập vốn/cấu hình,
parity momentum và MR có lệnh/lô đóng thật trong dữ liệu giả lập; thay giá sau
giai đoạn không làm đổi các tín hiệu hay giao dịch trước đó. Test dùng thư mục
tạm, không giao dịch hoặc sửa dữ liệu vận hành.

Chạy đo riêng đạt 25/25, coverage cả nhánh của `scripts/strategy_lab.py` đạt
97%. Artifact coverage: `data/paper/experiments/20260929_lab_focused.coverage`.
Snapshot nguồn chạy mẫu đã được xác minh lại sau thí nghiệm và vẫn hợp lệ.
Hồi quy toàn repo: **369 test đạt**, 23 cảnh báo, 200,17 giây;
`data/paper/experiments/20260929_tests.xml`. Không có test lỗi hoặc bị bỏ qua.

Nâng cấp này chỉ bổ sung backend chạy nghiên cứu và cấu hình ví dụ. Các chặn
dashboard đã làm trước được giữ nguyên; không cập nhật giá online, không phục hồi
cache legacy, không deploy, không push. Bước tiếp theo có thể nối backend này vào
một API job thử nghiệm có trạng thái; không cần thay lại engine giao dịch.
