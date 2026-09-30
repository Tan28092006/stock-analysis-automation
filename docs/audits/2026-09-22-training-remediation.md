# Sửa leakage trong code huấn luyện — 22/09/2026

## Trạng thái

Đã sửa các lỗi tái hiện được trong pipeline train/evaluate/serve và bổ sung kiểm thử chống tái phát. **Chưa chứng nhận dữ liệu hoặc artifact hiện hành sạch; chưa retrain model thật, chưa promotion, chưa chạy giao dịch.** Đây là phần tiếp theo của [kiểm toán baseline](2026-09-22-architecture-and-leakage.md), không thay thế hay viết lại bằng chứng trước sửa.

User đã yêu cầu sửa và push. Thay đổi được giữ trên nhánh `codex/data-integrity-audit`; không merge vào `main`, không chạy batch EOD có tác dụng publish/deploy. Quy trình MLE phân biệt kiểm chứng code với chứng nhận artifact/dữ liệu; TDD lưu từng checkpoint RED/GREEN.

## Hợp đồng mới và thay đổi chính

| Đường xử lý | Sau sửa |
|---|---|
| Ensemble | Chia theo ngày, dành 20% ngày cuối làm holdout; purge nhãn có thời điểm sẵn sàng chạm/vượt ranh giới. Walk-forward chỉ trong development. Model lưu là model fit development, **không refit lên holdout**. |
| Daily update | Fit mới, bỏ warm-start và meta fit in-sample; không save nếu training thất bại/thiếu dữ liệu. |
| HPO | Chỉ thấy development, cùng outer split với train; tiền xử lý fit riêng mỗi fold. Không điều chỉnh tham số bằng test cuối. |
| Preprocessing | Không forward-fill xuyên mã/ngày. Dùng giá trị trung lập độc lập từng dòng. Lưu/reuse bounds train; inference thiếu bounds bị từ chối. Batch và single-row dùng cùng chính sách. |
| Model suite legacy | Fit bounds trên train, áp dụng nguyên trạng cho validation/test/serve. Chọn family bằng validation, không bằng test. Lần train lỗi/thiếu dữ liệu không xóa registry đang có. |
| V2 regime | Quantile chỉ từ quá khứ, lag một phiên; bỏ suy luận đơn vị return bằng phân phối toàn chuỗi. Return đầu vào là ratio theo `add_indicators`. |
| MR | Chỉ nhận replay `resolved=True`; stop neo vào close ngày tín hiệu. Split ngày 60/20/20 có purge train/calibration. Classifier fit **một lần**, isotonic fit calibration, cặp cố định được chấm trên test bằng AUC/Brier. Imputation dùng chung ở cả bốn bước. |
| Dataset | `start/end` chọn ngày tín hiệu, tham số `as_of` riêng chặn dữ liệu/nhãn tương lai. Lưu ngày thoát thật, ngày thoát dự kiến, độ chín và chất lượng nhãn. Cờ corporate-action tương lai không được dùng để xóa candidate quá khứ; gắn cờ nhãn, dừng train để xác minh thay vì âm thầm loại mẫu. |
| OHLCV | Kiểm tra cột, date, số hữu hạn/dương, quan hệ high/low/open/close, volume, ngày trùng. Dữ liệu sai làm training fail; không tự đoán/sửa giá. Cắt tại phiên EOD đã hoàn tất. |
| Feature builder | Dataset, scan legacy và portfolio research dùng chung cross-sectional/regime/temporal builder, không âm thầm fallback V1 khi V2 lỗi. |
| Model availability | Ensemble/legacy/MR thiếu protocol mới hoặc model chưa tồn tại tại EOD tín hiệu bị chặn khỏi ML override/probability buy. Kiểm tra cả fit-label, calibration-label và evaluation-label end. Model cũ không tự được “nâng cấp” bằng metadata đoán lại. |
| Backtest đơn mã | Cắt tại `end`, không đọc exit sau `end`, không coi vị thế chưa kết thúc là realized trade. |
| Demo và cache | `daily-run --demo` dừng trước các bước ghi production. Đổi version cache để kết quả có xác suất model cũ không vượt qua gate mới. |

Luồng thời gian của một lần fit mới:

```text
Giá biết đến as_of -> nhãn đã kết thúc -> chia ngày + purge theo label_available_date
  Ensemble: [development / inner folds + HPO] | [test không dùng để fit]
  MR:       [classifier train] | [isotonic calibration] | [test cặp cố định]
```

Metadata lưu `training_protocol=purged-eod-v2`, hash dataset/features, fit-label end và cửa sổ đánh giá; MR lưu thêm calibration snapshot hash. Metadata này phục vụ truy vết, **không phải chứng thư nguồn giá, PIT universe, hay đủ điều kiện tự động promotion**.

## Kiểm chứng

- Kết quả cuối trên code `98503c1`: **193 passed, 23 warnings, 122,21 giây**. Warnings thuộc thư viện/fixture dtype/date parsing; không có test fail/skip. Hai module contract mới đạt **99,36% statement+branch coverage** (111/111 statements, 45/46 branch destinations), qua ngưỡng 80%; không tuyên bố 99% coverage toàn repo.
- Test mới tại `tests/test_training_integrity.py`: fit/predict row disjointness, cùng ngày không cắt đôi, purge nhãn chồng lấn, holdout perturbation, HPO isolation, prefix invariance, maturity, calibration freeze, artifact round-trip, model-vintage gate, malformed input, batch/single imputation, demo/registry safety và backtest as-of.
- Kiểm thử tích hợp có fit LightGBM/XGBoost/CatBoost/Ridge trên **fixture tổng hợp**, artifact thử nghiệm trong thư mục tạm. Đây không phải retrain dữ liệu thị trường hoặc kiểm chứng lợi nhuận.
- [JUnit toàn suite](2026-09-22-remediation-tests.xml) ghi kết quả chạy cuối; [coverage](2026-09-22-remediation-coverage.json) đo hai module contract mới `temporal_validation` và `training_quality`, không đại diện toàn repository.
- [Bằng chứng audit chạy lại](2026-09-22-remediation-evidence.json): spy có **0/16** dòng holdout đã đi qua fit; số regime quá khứ thay đổi khi append tương lai là **0**. Baseline tương ứng trước sửa ghi 20/20 overlap và 65 regime đổi. Số dòng test khác nhau vì cách chia đã thay đổi.
- Hash kho `prices`, `prices_hist`, ledger legacy và artifact MR khớp baseline. Git diff không có thay đổi ở giá, trọng số model hay ledger gốc.
- Phần `audit_mr` trong script audit được đặt tên rõ là **tái dựng cơ chế legacy** để kiểm tra artifact cũ; không gắn các lỗi legacy đó cho trainer đã sửa. Phần audit này không huấn luyện model.

## Những gate vẫn chưa đạt

1. **76 dòng OHLCV sai trong 12 file history**, gồm VNINDEX/SCS. Chưa có nguồn đã xác minh để đối soát; không chữa bằng nội suy hay xóa hàng âm thầm. Training MR đọc VNINDEX theo strict contract và dừng khi gặp lỗi.
2. Thiếu snapshot nguồn/adjustment basis và corporate-action lineage; chưa đủ dữ liệu thành phần universe point-in-time. Universe cố định/available-files được gắn nhãn không-PIT, không chứng minh loại bỏ survivorship bias.
3. Hai artifact cũ giữ nguyên bytes, thiếu manifest hợp lệ nên không được dùng qua gate ML mới. AUC/win-rate cũ không được diễn giải lại thành kết quả sạch. Chỉ báo rule-based vẫn hoạt động; `prob_buys` MR sẽ không dùng artifact legacy.
4. Ledger cũ vẫn quarantine theo báo cáo trước: không bịa timestamp/model version để “hợp thức hóa” lịch sử. Chưa có mẫu forward sạch đủ để xác nhận hiệu quả ngoài thị trường.
5. Dùng chung builder mới chỉ đảm bảo cùng phép tính khi đầu vào/history giống nhau. Độ dài history, membership và nguồn context giữa các nhánh vẫn cần snapshot thống nhất để chứng nhận train/serve parity toàn hệ thống.
6. Giá fill, thanh khoản, gap/slippage, portfolio forced-close và các script nghiên cứu cũ chưa được xác nhận là mô phỏng giao dịch thực. Không suy từ unit test thành hiệu quả tài chính.

Không chạy `daily-run` production hay `train_and_save` để tạo artifact mới trước khi đóng các gate dữ liệu trên và chốt protocol đánh giá/promotion. Các entry point này vẫn có khả năng ghi model khi được gọi thành công; thay đổi code lần này không phải một sandbox retraining hoặc một quy trình phê duyệt deployment mới.

## Chạy lại

```powershell
python -m pytest tests -q --junitxml=docs/audits/2026-09-22-remediation-tests.xml
python scripts/audit_data_integrity.py --output docs/audits/new-integrity-evidence.json
```

Lưu audit mới vào tên mới để bảo toàn baseline. Không chạy job cập nhật giá/model chỉ để kiểm tra code.
