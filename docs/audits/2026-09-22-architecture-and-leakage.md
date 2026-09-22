# Kiểm toán kiến trúc và tính nhân quả dữ liệu — 22/09/2026

## Kết luận trước khi thay đổi pipeline

**FAIL: chưa thể xác nhận hệ thống/mô hình sạch về dữ liệu hoặc đánh giá ngoài mẫu.** Có leakage tái hiện được trong code ensemble và feature V2; nhánh xác suất MR có vấn đề về độ chín nhãn, ranh giới calibration và tính tái lập. Ba blocker vận hành (universe, EOD, ledger) không đủ để sửa các lỗi huấn luyện này. Chưa được dùng các chỉ số hiện có làm căn cứ retrain/promotion hoặc giao dịch tiền thật.

Phạm vi: mã production, hai kho giá, hai artifact ML, hai ledger, code nghiên cứu có liên quan. Không giao dịch, không chạy job tự động, không huấn luyện/thay artifact, không gọi lệnh Git publish. Bằng chứng baseline gắn với HEAD `086b328d707cd8e770c29a7c6a2a1e1bf9434c1a`, trước sửa pipeline; không có snapshot gốc để tái lập chính xác hai lần train cũ.

Bằng chứng máy đọc: [baseline-evidence.json](2026-09-22-baseline-evidence.json). Phép thử: [audit_data_integrity.py](../../scripts/audit_data_integrity.py). Dữ liệu tổng hợp chỉ dùng để kiểm tra nhân quả và overlap; không dùng làm kết quả đầu tư. Estimator spy không fit mô hình thật.

## 1. Kiến trúc thực tế

```text
CLI / API scan (legacy)
  rules_t2 + universe_vn30
    -> parallel_fetch: local prices -> Yahoo -> VCI fallback
    -> validation -> indicators/rules -> ensemble override -> risk sizing
    -> latest_scan / feature snapshots / báo cáo

daily_runner (legacy, có quyền ghi model)
  resolve pending -> build labels từ prices -> train/update/save ensemble
    -> scan (lúc này mới fetch) -> pending_predictions -> performance report

EOD/dashboard (nhánh đang có dữ liệu mới)
  VCI -> prices_hist (100 cổ phiếu + VNINDEX)
    -> MR rules + win_prob_mr.pkl -> MR cache
    -> momentum 12-1 + inverse-vol -> momentum cache
    -> RSI2 swing rules -> swing cache
    -> position checks + forward_test ledger -> replay kết quả
```

| Thành phần | Vai trò thực tế / ranh giới |
|---|---|
| `ensemble_model.py` | Ensemble legacy; registry ngày 08/06/2026, 2.218 dòng, 49 feature (19 rule + 30 V1). Không được nhầm với MR model. |
| `win_probability.py` | ML xác suất MR riêng; artifact ngày 03/07/2026, 20.420 candidate, `calib_auc≈0,5445`. Dashboard vẫn load model này dù `rules_mr.ml.enabled=false` (cờ đó thuộc đường legacy). |
| `momentum_scan.py` | Rule-based, không phải ML; rank trên kho khoảng 100 mã, VN30 là nhãn thành viên. Luôn active, exposure theo volatility; không có gate chỉ chạy RISK_ON như một số mô tả cũ. |
| `swing_scan.py` | Rule-based RSI2, không phải model được train. |
| Hai kho giá | `prices` phục vụ legacy; `prices_hist` phục vụ dashboard/MR. Không có contract chứng minh nguồn, adjustment basis và snapshot thống nhất. |
| Hai ledger | `pending_predictions` thuộc legacy; `forward_test` thuộc MR/momentum. Không cộng chung để chấm một “mô hình”. `load_resolved_labels` dùng theo dõi performance; dataset của daily runner được dựng lại từ giá, không trực tiếp merge ledger này. |
| `run_eod_update.bat` | Ngoài refresh còn Git commit/pull/push; không phải lệnh chẩn đoán chỉ đọc. |

Lưu ý: `run_scan(persist=False)` chỉ ngăn một số output scan; provider mạng vẫn có thể ghi price cache. `daily-run --demo` hiện chưa cách ly đầy đủ đường train/save/log khỏi production. Không dùng hai lựa chọn này như một sandbox kiểm chứng.

## 2. Phát hiện, theo mức độ ảnh hưởng

### P0 — Ensemble báo “test” trên chính dữ liệu đã fit

`EnsembleTrainer.train` chạy CV rồi fit các base estimator trên **toàn bộ X/y**, sau đó gọi prediction trên last fold để ghi `test_metrics`. `daily_update` cũng cập nhật/fitting bằng toàn bộ dataset rồi chấm đoạn cuối. Phép thử spy xác nhận **20/20 dòng được chấm test đã đi qua fit**. Registry ghi 100% thắng / 50 lệnh được chọn không phải kết quả độc lập. Fold OOF cuối trong registry chỉ chọn 6 lệnh, thắng 50%; không được thay thế một cách tùy tiện bằng chỉ số refit.

Khắc phục cần thiết: tách evaluation khỏi full-data refit, freeze test theo ngày, lưu prediction OOF từ đúng model từng fold. Chỉ sau khi chốt đánh giá mới fit artifact phục vụ tương lai; không chấm artifact refit trên training history rồi gọi test.

### P0 — Tiền xử lý và split không bảo toàn thời gian

`ensemble_model._prepare_features(fit=True)` học quantile winsor trên toàn bộ dataset trước split. Trong fixture, giới hạn trên đổi từ `197,01` thành `1.000.197,01` chỉ vì sửa nửa dữ liệu tương lai; giới hạn fit train-only là `98,01`. `calibration.preprocess_features_robust` còn forward-fill trên panel, không group theo mã.

`TimeSeriesSplit` chia **dòng**, không chia toàn bộ ngày giao dịch, không purge theo thời điểm nhãn kết thúc. Fixture panel 40 ngày × 7 mã: cả 5 fold đều có cùng signal-date ở train và test; số train-label chưa biết tại đầu test lần lượt 50/96/142/146/143. Đây là chứng minh lỗi cơ chế, không phải số lượng leakage đã đo trong snapshot train cũ.

`ml_models._train_one_family` fit preprocessing riêng trên train/val/test, artifact không lưu bounds để serve giống train. `_select_model` dùng test để lựa chọn model: test không còn là holdout cuối cùng.

### P0 — Feature V2 dùng phân phối tương lai

`feature_engineering_v2.add_regime_features` dùng quantile volatility toàn chuỗi để phân loại regime quá khứ; append đoạn tương lai volatility cao làm đổi **65 nhãn regime quá khứ** trong fixture. Suy luận đơn vị theo median toàn chuỗi cũng cần thay bằng schema đơn vị tường minh.

Không kết luận artifact ensemble cũ chắc chắn đã chứa tất cả feature V2: danh sách feature lưu trong registry là V1/rules. Nhưng đường dựng dataset hiện tại gọi V2 và một số rule có thể tiêu thụ các cột bổ sung. Lỗi phải được sửa trước một lần retrain mới.

### P1 — Nhãn MR chưa chín và split calibration chồng lấn

`win_probability._label_trade` bỏ qua cờ `resolved` của `simulate_mr_exit`, nhận mark-to-market cuối đoạn ngắn làm nhãn cuối cùng. Dựng lại candidate trên **dữ liệu hiện tại** cho 30.155 candidate, có **314 nhãn chưa kết thúc** vẫn được code chấp nhận. Split row 85/15 tại 27/11/2025 có cùng signal-date hai phía; **184 train-label** kết thúc từ ngày mở calibration trở đi. Các số này không mô tả chính xác lần train tháng 7 vì thiếu snapshot gốc.

Sau fit classifier train / isotonic calibration, code refit classifier trên cả calibration nhưng giữ isotonic cũ; cặp classifier/calibrator đã đổi và không có test độc lập kiểm tra lại. Việc fit toàn bộ trước khi dự báo thật trong tương lai tự nó không phải leakage tương lai; sai ở tuyên bố chất lượng/calibration của cặp cuối mà không có đánh giá độc lập.

Code nghiên cứu `scratch/win_probability.py` fit isotonic và đánh giá reliability trên cùng tập OOF. Raw OOF AUC và reliability sau isotonic có ý nghĩa khác nhau; không coi reliability cùng-tập là bằng chứng calibration ngoài mẫu. Script nghiên cứu dùng horizon 8 so với production 15.

### P1 — Train / backtest / serve chưa cùng hợp đồng giao dịch

MR label train đặt stop theo **next-open entry − 3 ATR**, nhưng risk plan serve đặt stop theo **signal-close − 3 ATR**. Khoảng gap làm thay đổi mục tiêu mà xác suất dự báo. Legacy resolver dùng holding-days **hiện tại**, không phải horizon/risk plan lúc dự báo, rồi clip entry vào trần/sàn quanh `entry_reference` có thể sai. Không được xem giá clip giả định này là khớp lệnh thật.

`backtest.py` có đường load artifact hiện tại để override quyết định ở các ngày lịch sử, không kiểm tra trained-at trước signal. Mini rule-backtest trong orchestrator có disable ML nên không mắc đúng lỗi đó. `parallel_engine` chỉ lọc signal-date theo `end`, chưa bảo đảm exit/label đã biết ở cutoff; `exit_date` có thể là ngày hết hạn kế hoạch dù đã thoát sớm. Các vấn đề fill khi gap, cổ tức/chia tách và selection do loại corp-action cũng cần kiểm tra trước đánh giá hiệu suất.

### P1 — Giá đầu vào chưa đạt contract

Baseline có 32 file `prices` (30 file tới 16/09, 2 file tới 03/07); `prices_hist` có 101 file tới 21/09. `prices_hist` có **76 dòng OHLC/volume không hợp lệ ở 12 file**, bao gồm SCS close=0 ngày 23/07/2018 và VNINDEX có OHLC lệch. Danh sách/ví dụ nằm trong JSON. Không có duplicate-date hoặc weekend trong snapshot này. Giá nhảy >15% là cờ điều tra corporate action/adjustment, không tự động kết luận giá sai.

Provider comparison có thể so local cache của Yahoo với chính Yahoo, nên “hai nguồn khớp” chưa chứng minh hai nguồn độc lập. Validation last-close không bảo đảm chất lượng toàn lịch sử. Chưa có dữ liệu adjustment/corporate-action và source lineage để chứng nhận toàn kho sạch.

### P1 — Universe, intraday và stale cache

Universe file 26/05 còn PLX/TPB, thiếu MCH/TCX của rổ đang công bố. Membership hiện tại được dùng ngược về lịch sử cho MR gate: survivorship/point-in-time bias, không được khắc phục chỉ bằng cập nhật rổ hôm nay.

Legacy scan lấy `date.today()`; các loader dashboard đọc cuối file mà không chặn bar chưa hoàn tất. EOD job có chặn giờ nhưng dùng timezone máy và không áp dụng đồng bộ toàn đường đọc. Cache keyed chủ yếu bằng ngày VNINDEX/rules, không có universe/model/data content hash; đổi dữ liệu cùng ngày hoặc đổi universe có thể giữ kết quả cũ.

Đối chiếu lịch sàn phát hiện bảng calendar thiếu 02/01, 31/08 và 01/09/2026. Ngày đi làm bù thứ Bảy không phải ngày giao dịch. Đây là lỗi vận hành đã được sửa trong bước EOD sau audit; chưa kiểm toán đầy đủ lịch các năm ngoài 2026.

### P1 — Ledger không đủ chứng minh dự báo ex-ante

Legacy: **143 dòng nhưng 65 cặp mã/ngày**; **93 dòng** có reference lệch >5% so với close lịch sử cùng signal-date. 50 dòng còn lại chỉ là “tương thích close”, **không phải 50 dự báo đã chứng nhận sạch**. Không có mode/model/rules/horizon/source/timestamp ghi thực. Phép so generator demo mặc định không tìm được exact match: **không đủ chứng cứ để khẳng định 93 dòng là demo**, có thể còn sai basis hoặc nguồn khác.

Forward: **737 dòng**, từ 03/07 tới 21/09; đều thiếu model-version. `logged_date` lấy từ data-date, không phải timestamp lúc append nên không chứng minh đã ghi trước giá tương lai. Momentum/watch được chấm close-to-close proxy, không phải PnL portfolio. Scorer còn cho horizon chưa đủ 21 phiên thành “resolved” chỉ sau 3 phiên.

**Đính chính diễn giải kết quả trước:** tỷ lệ thắng 30% / lợi nhuận trung bình −4,56% từ ledger legacy không đủ cơ sở để kết luận hiệu quả mô hình. Gọi đó là T+2 cũng không chính xác khi phép chấm dùng holding-days=20 hiện tại. Phải loại khỏi bằng chứng hiệu quả đã kiểm chứng, không kết luận mô hình thua hoặc thắng dựa trên con số này.

## 3. Những kiểm tra đã đạt và giới hạn

- Prefix-invariance trên fixture: 69/69 cột numeric của base indicator frame (bao gồm OHLCV, không phải 69 chỉ báo riêng) không đổi khi append tương lai.
- Vector 20 feature MR trên fixture không đổi khi append tương lai. Rolling/EWM/positive-shift của base indicators phù hợp nhân quả trên đường đã kiểm tra.
- Không thấy đưa trực tiếp target/return tương lai vào danh sách FEATURES của MR.
- MR historical list chủ động bỏ `win_prob`; điều này tốt, nhưng latest bar của một mã stale vẫn cần kiểm tra so với trained-at.
- Các điểm đạt này **không chứng minh toàn hệ thống không leakage**: split, preprocessing, label, selection, PIT universe và timing log vẫn thất bại/chưa biết.

## 4. Data contract cần có trước retrain

1. Khóa chính `(engine, symbol, signal_session, model_version, rules_hash)`; dedup/conflict phải explicit.
2. Mọi feature chỉ dùng phiên HOSE đã hoàn tất tại thời điểm dự báo (VN UTC+7, cutoff bảo thủ 16:00); index/breadth/cổ phiếu cùng as-of. Source, đơn vị, adjustment-basis, snapshot hash phải có.
3. Nhãn có `entry_date`, `exit_date` thực, `label_available_at`, `resolved`; chỉ fit nhãn đã chín trước cutoff. Cùng stop/target/fees/settlement/fill-policy giữa train và replay/serve.
4. Chia theo **ngày**, purge mọi train label chưa biết tại đầu validation/test. Preprocessor fit train-only theo fold, lưu cùng artifact. Calibration và chọn threshold không được dùng final test.
5. Historical universe point-in-time hoặc công khai cố định-universe/survivorship bias; không gọi là backtest thị trường không thiên lệch.
6. Artifact lưu code SHA, config+feature+data hashes, train/label cutoffs, preprocessing, dependencies và đầy đủ định nghĩa metric. Prediction log append-time UTC + source/mode + model hash; không backfill như live.

## 5. Trình tự tiếp theo và gate

Audit đã hoàn tất ở mức phát hiện lỗi có thể tái hiện, **không đạt gate sạch**. Có thể sửa ba blocker vận hành đã được yêu cầu trong một thay đổi riêng, nhưng không retrain để “rửa” lịch sử hoặc sửa số liệu bằng cách bỏ các lệnh thua. Giữ nguyên ledger/model/raw snapshot gốc; mọi ledger chuẩn hóa/quarantine là output dẫn xuất có lý do và truy về dòng gốc.

Trước retrain còn bắt buộc: sửa split/preprocessing/OOF; sửa label maturity và parity MR; sửa V2; xử lý 76 dòng giá từ nguồn đáng tin; quyết định adjustment và PIT universe. Sau đó mới freeze unseen time window, chạy walk-forward có purge, so baseline rules-only, báo cả độ phủ/chi phí/drawdown/CI. Forward shadow phải append trước phiên vào lệnh và không chỉnh rule theo chính tập kiểm định đó.

## 6. Nguồn đối chiếu

- [scikit-learn: common pitfalls](https://scikit-learn.org/1.8/common_pitfalls.html): split trước preprocessing, fit chỉ train, không dùng test để chọn model.
- [scikit-learn: calibration](https://scikit-learn.org/stable/modules/generated/sklearn.calibration.CalibratedClassifierCV.html): dữ liệu fit classifier và calibrator cần tách biệt theo giao thức phù hợp.
- [SSIAM VN30 ETF](https://ssiam.com.vn/quy-etf-ssiam-vn30): bảng rổ 30 mã hiện tại kiểm tra ngày 22/09/2026.
- [SSI ETF bulletin](https://www.ssi.com.vn/khach-hang-ca-nhan/ban-tin-etf): công bố review quý 3/2026, MCH/TCX thay PLX/TPB, hiệu lực 03/08/2026.
- [HNX lịch nghỉ 2026](https://www.gov.hnx.vn/vi-vn/chi-tiet-lich-nghi-gd-60021971.html?_page=1) và [HNX điều chỉnh Tết Dương lịch](https://hnx.vn/vi-vn/ket-qua-tim-kiem/chi-tiet-tin-591376-0.html): nghỉ 31/08–02/09 và bổ sung 02/01; đối chiếu thêm thông báo HOSE 2410/TB-SGDHCM ngày 25/12/2025.

Phương pháp theo skill `mle-workflow`: tách contract dữ liệu, bằng chứng evaluation và điều kiện promotion; không suy luận artifact cũ sạch từ việc unit tests pass.

## 7. Ba blocker vận hành — thay đổi SAU audit

| Blocker | Đã làm | Giới hạn còn lại |
|---|---|---|
| Universe | Cập nhật 30 mã hiện tại, thêm MCH/TCX, bỏ PLX/TPB; giữ snapshot cũ và nguồn/effective-date. Content hash làm mất hiệu lực cache khi universe đổi. | Chưa có chuỗi membership PIT để huấn luyện/backtest lịch sử không survivorship. Reserve list cũ được đánh dấu chưa xác minh. |
| Intraday/EOD | Dùng cutoff bảo thủ 16:00 UTC+7 + calendar; provider clip trước khi cache; legacy scan/index có cutoff; MR/momentum/swing/position alerts/forward replay chỉ đọc phiên hoàn tất. Dashboard bỏ mã stale so với ngày VNINDEX. Refresh lấy lại tail đã tồn tại để thay bar từng lưu dở. Cache có fingerprint giá, universe, artifact và cutoff. | Không tự sửa 76 dòng OHLC lỗi. Không có giao dịch mạng/full refresh thực hiện trong audit; chứng minh bằng fixture/mocked-provider. Giờ 16:00 là buffer dữ liệu của ứng dụng, không phải giờ đóng cửa sàn. |
| Ledger | Xuất clean/quarantine dẫn xuất, không xóa/gán lại ngày/ghi đè nguồn. Resolver legacy chặn bản ghi thiếu provenance/horizon. Forward scorer loại bản ghi chưa đủ provenance, duplicate/conflict, reference mismatch và bar không hợp lệ; horizon chưa đủ vẫn pending. Log MR/momentum mới có UTC append-time, snapshot/rules/model identity và mode paper. | Ledger cũ không thể biến thành ex-ante chỉ bằng bổ sung metadata hôm nay. Legacy logger chưa được migration đầy đủ; bản ghi của đường đó tiếp tục bị gate chặn. Không gọi daily-run để retrain. |

Model-loader MR chỉ được thay đổi để ghi hash đúng **bytes đã load** và làm mới cache khi artifact đổi; **không thay estimator, trọng số, feature hay cách train**. Các lỗi P0/P1 về train/evaluation nêu ở trên vẫn mở. Quy tắc gate hiện không phải cơ chế vô hiệu hóa toàn bộ CLI training: lệnh train/daily-run cũ vẫn tồn tại, không được vận hành trước khi sửa pipeline học.

### Kết quả chấm ledger sau phân loại

Xem [summary.json](2026-09-22-ledger/summary.json), các file `*-quarantine.jsonl` truy về dòng/hash nguồn, và `*-clean.jsonl` (rỗng).

| Ledger | Tổng dòng | Đủ provenance | Cách ly khỏi metric đã kiểm chứng | Reference lệch >5% |
|---|---:|---:|---:|---:|
| Legacy | 143 | 0 | 143 | 93 |
| Forward MR/momentum | 737 | 0 | 737 | 11 |

Legacy có 114 dòng thuộc nhóm key trùng nhưng nội dung xung đột, 2 dòng duplicate exact dư (nhóm lý do có thể chồng lấn). Không coi 143−65 là 78 bản ghi có thể xóa một cách an toàn. **Không có win-rate được chứng nhận để báo cáo**; `n=0` nghĩa là thiếu bằng chứng, không phải win-rate 0%. Bản gốc được giữ nguyên để có thể đối chiếu thêm append-time/model/source từ chứng cứ độc lập nếu tìm được.

**Bằng chứng bổ sung từ Git:** kiểm tra 23 commit thay đổi forward ledger, khớp nội dung chính xác 737 bản ghi. 649 bản ghi xuất hiện lần đầu trong commit nằm giữa EOD signal và trước 09:00 phiên vào lệnh kế tiếp; 88 xuất hiện ngoài cửa sổ đó. Cả 737 đều ở commit có cùng hash MR artifact như hiện tại. Đây là cơ sở để nghiên cứu một tập **paper replay khôi phục từ Git**, không nên vứt bỏ lịch sử. Tuy nhiên committer-time cục bộ có thể sửa và artifact được track không chứng minh model thực sự đã load; không tự điền chúng vào bản ghi gốc rồi gọi là schema-v2 đã kiểm chứng. Lỗi calibration/label của model vẫn tồn tại bất kể timestamp. Các con số này cũng có trong summary JSON và script audit tái lập.

### Tái lập và kiểm thử

Chạy từ root repository với Python có dependencies dự án:

```powershell
python scripts/audit_data_integrity.py --output docs/audits/recheck-integrity.json
python scripts/audit_prediction_ledgers.py --output-dir docs/audits/recheck-ledger
python -m pytest tests -q --disable-warnings
```

Lệnh audit thứ nhất không train estimator thật (chỉ spy), nhưng tái dựng candidate từ dữ liệu và code **đang có**, nên không ghi đè evidence baseline. Lệnh thứ hai chỉ viết output dẫn xuất vào thư mục audit. Không chạy batch EOD hay daily-run để làm kiểm thử.

Baseline trước sửa: **116 test đạt**. Sau sửa: **153 test đạt, 22 warnings**, không skip/fail, trong 98,65 giây khi bật đo coverage. Kiểm thử bổ sung bao gồm boundary UTC/VN, cuối tuần/ngày lễ, provider trả dư, refresh bar cuối, fingerprint cache, artifact identity, timestamp ex-ante, duplicate/conflict, malformed JSON/date, OHLC replay, horizon đầy đủ và luồng dashboard → cache → ledger → pending cách ly. Coverage hai module guard mới **96% (có tính nhánh)**: EOD 100%, ledger-integrity 96%. Chi tiết trong [test XML](2026-09-22-tests.xml) và [guard coverage JSON](2026-09-22-guard-coverage.json); coverage này **không phải coverage toàn dự án** và test pass không đảo ngược kết luận FAIL leakage.

Snapshot SHA của cả `prices` và `prices_hist`, MR artifact và legacy ledger so với baseline đều không đổi; forward ledger hash không đổi qua hai lần phân loại. Ensemble file/registry vẫn có last-write 08/06/2026. Không phát sinh lệnh giao dịch, retrain, model promotion hoặc Git push.

TDD dùng checkpoint cục bộ trên branch `codex/data-integrity-audit`; các RED là lỗi thực được chạy trước sửa. Có một message checkpoint `08e6260` ghi nhầm GREEN/count; checkpoint `54e65ed` đã đính chính rõ (lần chạy thực có 1 fail/56 pass do fixture ngày `d0`). Fixture đã chuyển sang phiên thật, lỗi input toàn ngày không hợp lệ được tái hiện và sửa; không dùng checkpoint message đó làm bằng chứng kết quả cuối.
