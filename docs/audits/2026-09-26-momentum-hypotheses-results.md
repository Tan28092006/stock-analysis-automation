# Kiểm chứng giả thuyết nghiên cứu bằng dữ liệu hiện có — 26/09/2026

## Kết luận

**Chưa biến thể nào đạt tiêu chí đã khóa để gọi là ứng viên vượt VNINDEX.**
Hai thay đổi về thời điểm mua và rủi ro danh mục cải thiện bản gốc trong H1/2026
và giai đoạn 01/07–25/09/2026, nhưng làm giảm lợi nhuận trong H2/2025. Khoảng
bất định của cả ba phép thử chính H1/2026 vẫn chứa 0. Không triển khai, không
retrain, không bật ML, không ghép các biến thể để tìm một kết quả đẹp hơn.

Đây là kiểm chứng **các cách áp dụng cụ thể**, không phải xác nhận/bác bỏ toàn bộ
bài báo. Dữ liệu dùng rổ cổ phiếu hiện tại và giá điều chỉnh lấy về sau thời điểm
giao dịch: chưa chứng minh sạch point-in-time/survivorship/corporate-action bias.

## Cách thử đã khóa trước khi xem kết quả

Quy trình MLE/TDD được dùng để khóa giả thuyết, kiểm thử tính nhân quả và giữ
chứng cứ RED/GREEN. [Protocol](D:/Chungkhoan/docs/audits/2026-09-26-momentum-hypotheses-protocol.md)
được commit ở `8dd29e5`, trước khi thực thi biến thể trên dữ liệu thị trường.

- 4 chiến lược × 3 kịch bản × 3 giai đoạn = **36 replay**; không dò tham số.
- A: xếp hạng momentum sau điều chỉnh beta VNINDEX; giữ cách phân bổ/khớp lệnh.
- B: chỉ mua tăng thêm khi lợi suất 5 quan sát gần nhất thấp hơn hoặc bằng chỉ
  số; giữ mục tiêu tháng và cửa sổ khớp 4 phiên. Không phải mô hình RSI bắt đáy.
- C: cùng danh sách/cách chia tỷ trọng tương đối; điều chỉnh tổng vốn theo biến
  động 126 phiên của chính danh mục, có tính tương quan, trần vốn 100%.
- Đối chứng: momentum 12-1 hiện tại, 10 cổ phiếu, buffer top 20, inverse-vol.
- Mỗi giai đoạn bắt đầu riêng với 1 tỷ VND; không nối thành một đường NAV.
- Phí mua 0,15%, bán 0,25%, trượt giá 0,10% mỗi chiều; thêm kịch bản gấp đôi
  tất cả chi phí mô hình và kịch bản khớp chậm thêm 1 phiên.
- Lệnh dựa trên dữ liệu trước phiên; giả định bảo thủ T+3 cho tiền/cổ phiếu,
  lô 100, không vay/margin, không tự thanh lý cuối kỳ.
- VNINDEX là chỉ số **giá gross**, tính từ open phiên đầu đến close phiên cuối;
  chiến lược là **net** chi phí mô hình. Không phải ETF/total-return benchmark.

## Kết quả chính: 01/01–30/06/2026, 119 phiên

Phiên đầu có giao dịch là 05/01. Đơn vị chênh lệch là điểm phần trăm (đpt).

| Chiến lược | Lợi nhuận net | Hơn/kém VNINDEX | Hơn/kém bản gốc | Sụt giảm NAV tối đa | Vốn cổ phiếu bình quân |
|---|---:|---:|---:|---:|---:|
| Bản gốc | -2,66% | -6,74 đpt | — | -17,68% | 87,50% |
| A — Điều chỉnh theo thị trường | -2,60% | -6,68 đpt | +0,06 đpt | -17,68% | 86,53% |
| B — Lọc điểm mua hồi quy ngắn hạn | -1,03% | -5,11 đpt | +1,63 đpt | -14,88% | 83,74% |
| C — Rủi ro chính danh mục | +1,37% | -2,71 đpt | +4,03 đpt | -14,18% | 77,29% |
| VNINDEX, gross | +4,08% | — | — | -16,38% | 100% |

### Chi phí, rủi ro tương đối và giao dịch

Turnover dưới đây = tổng giá trị mua + bán / NAV bình quân, **không chia hai**.
IR là trung bình lợi suất ngày vượt chỉ số / độ lệch chuẩn, annualize sqrt(252).
Sụt giảm tương đối đo trên tỷ số NAV / giá trị danh mục chỉ số giả định.

| Đo lường H1/2026 | Gốc | A | B | C |
|---|---:|---:|---:|---:|
| Beta ngày với VNINDEX | 0,825 | 0,797 | 0,767 | 0,708 |
| Tracking error/năm | 10,83% | 11,18% | 10,51% | 10,33% |
| Information ratio | -1,328 | -1,281 | -1,061 | -0,616 |
| Sụt giảm tương đối tối đa | -10,49% | -10,10% | -8,35% | -7,52% |
| Turnover hai chiều | 2,052× | 2,252× | 1,899× | 1,314× |
| Phí đã trả, triệu VND | 3,408 | 3,812 | 3,191 | 2,148 |
| Trượt giá mô hình, triệu VND | 1,947 | 2,143 | 1,826 | 1,282 |
| Số lần mua / bán | 38 / 20 | 40 / 22 | 33 / 19 | 32 / 15 |

B có 41 lần đề nghị mua bị chặn trong H1. Đây là **lượt thử khớp**, có thể lặp
cùng mã/mục tiêu qua nhiều ngày; không được coi là 41 giao dịch độc lập.
Tỷ lệ thắng theo mảnh FIFO trong JSON cũng không phải tỷ lệ thắng độc lập của
chiến lược, nên không dùng để kết luận có alpha.

### Kiểm tra độ nhạy H1/2026

| Kịch bản | Gốc | A | B | C |
|---|---:|---:|---:|---:|
| Bình thường | -2,66% | -2,60% | -1,03% | +1,37% |
| Chi phí mô hình ×2 | -3,41% | -3,18% | -1,52% | +0,91% |
| Khớp chậm thêm một phiên | -2,60% | -2,62% | +1,55% | +0,59% |

So với đối chứng **cùng kịch bản**, lợi ích của A đổi dấu khi khớp chậm
(-0,019 đpt). B và C vẫn cải thiện bản gốc trong cả hai stress, nhưng tất cả
vẫn thấp hơn mức +4,08% của chỉ số. Stress ×2 chỉ là độ nhạy chi phí, không phải
thay đổi thuế luật định hay mô hình tác động thị trường đã hiệu chuẩn.

## Độ bền theo thời kỳ

| Giai đoạn | Số phiên | Gốc | A | B | C | VNINDEX gross |
|---|---:|---:|---:|---:|---:|---:|
| 01/07–31/12/2025 | 130 | +45,53% | +45,53% | +44,76% | +39,90% | +29,49% |
| 01/01–30/06/2026 | 119 | -2,66% | -2,60% | -1,03% | +1,37% | +4,08% |
| 01/07–25/09/2026 | 60 | +1,55% | +1,55% | +3,16% | +2,47% | -4,02% |

- **A:** gần như không thay đổi kết quả ở 2/3 giai đoạn. Trong H1, danh mục mới
  bắt đầu khác từ tháng 4 (BSR thay TCB), rồi tháng 5 có GAS thay VRE. Thêm chi
  phí và xáo trộn chỉ thu được +0,06 đpt; chưa có bằng chứng nên thêm độ phức tạp.
- **B:** cải thiện +1,63 và +1,61 đpt trong hai giai đoạn 2026, nhưng mất
  0,77 đpt trong H2/2025. Gần đây tỷ lệ vốn cổ phiếu chỉ 73,27%, so với 92,75%
  của bản gốc. Chưa tách được đầy đủ lợi thế điểm mua khỏi tác dụng giữ tiền mặt.
- **C:** cải thiện +4,03 và +0,92 đpt trong hai giai đoạn 2026, nhưng mất
  5,63 đpt ở H2/2025. Đầu tháng 4/2026, bản gốc chỉ đặt mục tiêu 56,33% vốn vì
  biến động VNINDEX 20 ngày, còn C đặt 77,81%; lợi suất tháng 4 là +6,34% so với
  +7,93%. Đây là khác biệt đường phân bổ quan sát được, không chứng minh một cơ
  chế nhân quả độc lập hoặc đảm bảo sẽ lặp lại.

Kết quả H2/2025 rất tốt của cả nhóm vẫn chịu rủi ro chọn rổ cổ phiếu hiện tại.
Không dùng nó để tuyên bố hiệu suất thực tế có thể đạt được trong quá khứ.
Giai đoạn 2026 gần đây chỉ có 60 phiên; tháng 9 chưa kết thúc.

## Độ bất định: không bỏ qua kết quả âm

Bootstrap cặp trên **chênh lệch lợi suất ngày** so với bản gốc, block tròn 10
quan sát, 4.000 mẫu, seed 20260926. Khoảng dưới đây được annualize từ trung bình
ngày; **không phải** khoảng tin cậy cho lợi nhuận kép 6 tháng.

| Biến thể, H1/2026 | Trung bình chênh lệch annualized | Khoảng 95% | Khoảng 98,3333%, điều chỉnh 3 phép thử |
|---|---:|---:|---:|
| A | +0,05%/năm | [-11,48; +13,93]% | [-13,54; +17,91]% |
| B | +3,23%/năm | [-2,50; +9,73]% | [-3,55; +11,44]% |
| C | +8,01%/năm | [-3,18; +21,20]% | [-5,74; +23,83]% |

C có khoảng dương trong block gần đây khi khớp bình thường, nhưng đây là kiểm
tra phụ đã xem dữ liệu, chỉ 60 phiên; khoảng của stress khớp chậm lại chứa 0.
Không lấy riêng ô tốt này để thay kết luận của phép thử chính đã khóa.
Bootstrap chỉ là xấp xỉ; chưa điều chỉnh số thử nghiệm lịch sử không được ghi
đầy đủ, thay đổi chế độ thị trường, hoặc thiên lệch dữ liệu. Không công bố một
Deflated Sharpe Ratio giả tạo khi thiếu số lần thử trước đây.

## Từng bài báo được kiểm chứng đến đâu?

| Nghiên cứu | Số đo tại hệ thống | Kết luận có giới hạn |
|---|---|---|
| [Blitz, Huij & Martens — Residual Momentum](https://repub.eur.nl/pub/22252/ResidualMomentum-2011.pdf) | A: +0,06 đpt so gốc ở H1; stress chậm -0,019 đpt; 2 block còn lại gần như giống gốc | Chưa hỗ trợ bản áp dụng một nhân tố VNINDEX. Không phải tái lập FF3/tháng/36 tháng/long-short của bài. |
| [Vo & Truong — momentum Việt Nam](https://www.sciencedirect.com/science/article/pii/S2214635017300965) | Bản gốc vượt chỉ số +16,05 đpt H2/2025, thua -6,74 đpt H1/2026, vượt +5,58 đpt gần đây | Momentum hiện tại phụ thuộc giai đoạn; không suy ra chiến lược formation 6 tháng/holding 9 tháng của bài đã được kiểm chứng. |
| [Nguyen và cộng sự — giao dịch kỹ thuật ngắn hạn Việt Nam](https://yoksis.bilkent.edu.tr/pdf/files/14964.pdf) | B mất 0,77 đpt ở H2/2025, nhưng thêm 1,63 và 1,61 đpt ở hai block 2026 | Có bằng chứng thực tế rằng lọc nhịp giảm có thể bỏ lỡ xu hướng; chưa phân biệt chắc chắn continuation/reversal, chưa tái lập tất cả quy tắc của bài. |
| [Da, Liu & Schaumburg — short-term reversal](https://academicweb.nd.edu/~zda/Reversal.pdf) | Chỉ đo được B, một proxy giá tương đối; không có chuỗi dự báo lợi nhuận/điều chỉnh kỳ vọng PIT | **Chưa kiểm chứng được cơ chế loại tin cơ bản.** Không gọi giá giảm là cú sốc thanh khoản. |
| [Dai và cộng sự — liquidity provision](https://rpc.cfainstitute.org/research/financial-analysts-journal/2024/reversals-and-the-returns-to-liquidity-provision) | B: turnover H1 1,899× so 2,052×, phí 3,191 so 3,408 triệu; vẫn thua VNINDEX 5,11 đpt | Bộ lọc khi tái cân bằng tự nhiên có ích cục bộ, chưa đủ alpha. Không có shares-outstanding PIT/order book nên chưa xác nhận cơ chế turnover/liquidity. |
| [Moreira & Muir — Volatility-Managed Portfolios](https://onlinelibrary.wiley.com/doi/10.1111/jofi.12513); Barroso & Santa-Clara — Momentum Has Its Moments | C: H1 +4,03 đpt so gốc, drawdown -14,18% so -17,68%; H2/2025 -5,63 đpt so gốc | Quản lý rủi ro tốt hơn ở một số block, nhưng không đạt mục tiêu vượt chỉ số của H1. Không kiểm chứng leveraged WML/inverse-variance hay “loại bỏ crash”. |
| [Barroso & Detzel — giới hạn arbitrage](https://www.sciencedirect.com/science/article/pii/S0304405X21000775); [de Groot và cộng sự — chi phí reversal](https://repub.eur.nl/pub/25718/AnotherLook_2011.pdf) | Chi phí ×2: gốc -3,41%, B -1,52%, C +0,91%; lợi ích B/C còn nhưng vẫn thua chỉ số | Lợi ích quan sát không hoàn toàn mất ở stress tuyến tính này; chưa chứng minh chịu được spread/impact/queue thật. |
| [Bailey & Lopez de Prado — Deflated Sharpe Ratio](https://www.davidhbailey.com/dhbpapers/deflated-sharpe.pdf) | Đăng ký 36 replay, 3 so sánh chính; tất cả khoảng H1 điều chỉnh chứa 0 | Giữ kết luận chưa đủ bằng chứng; chưa thể báo DSR tổng thể vì thiếu registry lịch sử. |

## Kiểm tra tính đúng đắn và định danh kết quả

- Snapshot được xác minh lại raw/canonical/hash trước chạy, kiểm tra lịch và
  OHLCV, cắt dữ liệu ở cuối giai đoạn trước mọi tính toán.
- Kiểm thử thay toàn bộ tương lai và giá high/low/close trong ngày: lệnh open
  đầu tiên của cả bốn biến thể không đổi. Khớp chậm không cập nhật lại tín hiệu.
- So sánh bản gốc H1 với kết quả trước: **toàn bộ replay bằng nhau chính xác**,
  gồm NAV, lệnh, FIFO, tỷ trọng và mục tiêu; không chỉ khớp tỷ suất cuối kỳ.
- Đối chiếu độc lập tiền, khoản chờ về, lượng nắm giữ và NAV ở cả 36 replay;
  sai số lớn nhất **0,000000477 VND**. Không âm tiền, đúng lô và T+3.
- Bộ kiểm thử mặc định sau sửa collection: **305 passed**, 23 cảnh báo thư viện/cũ; 40 kiểm thử tập
  trung cho hai runner đạt; branch-inclusive coverage runner nghiên cứu **97%**,
  runner backtest **96%**. Kiểm thử an toàn collection bổ sung cũng đạt.

Một sửa lỗi báo cáo sau lần chạy đầu: A ở H2/2025 có chênh lệch
`2,13e-14` đpt do thứ tự cộng số thực. Lần đầu tính nhầm đó là một block cải
thiện. Đã thêm test RED/GREEN và tolerance zero `1e-9` đpt (= 0,01 VND trên
1 tỷ NAV). **Không sửa tham số hoặc bất kỳ kết quả kinh tế nào**: đối chiếu
toàn bộ 36 replay giữa hai artifact bằng nhau chính xác. Giữ nguyên bản đầu
`result.json`; chỉ dùng `result_validated.json` để đọc kết luận.

Artifact chính:

- [Toàn bộ NAV, giao dịch, số đo và kết luận](D:/Chungkhoan/data/paper/backtests/hypotheses_20260926/result_validated.json).
- SHA256 artifact: `94e73b5d90ed445fba3f30a8fbd8159521208dccc396a0fc1bc46361f20956fb`.
- Code chạy: `b988324587f370aedfd82de69520a0a760a9ec5c`.
- Protocol SHA256: `0ee1a75f31dcbe71773481354a45dae112c5974168bda998c04b2bef327edef9`.
- Snapshot H1/H2-2025 SHA256: `5062296d7607b4782c001a72d0726d69ce16405c9e6308a03d6ad13c9cad7f34`.
- Snapshot gần đây SHA256: `bf4a310704db4a2124f9b3e47ab59c3698347170f9f77d1853582018933d5ffd`.
- [Coverage](D:/Chungkhoan/data/paper/backtests/hypotheses_20260926/coverage-final.json),
  [kết quả mặc định cuối](D:/Chungkhoan/data/paper/backtests/hypotheses_20260926/tests-default-final.xml).

Tái chạy vào **đường dẫn output mới**, không ghi đè:

```powershell
C:/Users/acer/anaconda3/python.exe -m scripts.momentum_hypotheses `
  --historical-manifest data/paper/backtests/h1_2026_20260926/snapshot/manifest.json `
  --recent-manifest data/paper/snapshots/20260926T022757171239Z/manifest.json `
  --output data/paper/backtests/hypotheses_20260926/reproduction.json
```

## Sự cố kiểm thử phụ và khôi phục dữ liệu

Lệnh pytest toàn thư mục ban đầu thu thập chương trình thủ công trong `scratch`.
`scratch/test_failures.py` monkey-patch provider tại import mà không phục hồi;
`scratch/test_persist.py` gọi scan với `persist=True`. Lỗi Yahoo của lượt đó
không tái hiện khi chạy độc lập hoặc chạy đúng `tests/`. Một test roundoff được
thêm khi tiến trình cũ đang chạy cũng dùng module trước bản sửa; kết quả cuối
trên tiến trình mới được ghi riêng, không che giấu lượt thất bại ban đầu.

Tác động đã xác định: scan `scan-20260926T041844Z-4813849f` ghi đè cache scan
và thêm 30 sự kiện vào training_events; các scan scratch cũng refresh 30 file
cache giá legacy trong `data/raw/prices`. Đã:

1. Sao lưu toàn bộ ledger trước phục hồi.
2. Xác minh đúng 30 sự kiện thuộc scan trên và chúng nằm liền ở cuối file;
   chỉ cắt phần append này, bảo toàn từng byte prefix dữ liệu cũ.
3. Chuyển cache scan và feature snapshot phát sinh sang khu cách ly, không xóa.
4. Thêm `pytest.ini` và test an toàn để mặc định chỉ thu thập `tests/`.
5. Sao lưu 30 cache giá phát sinh; thay bằng canonical CSV từ snapshot 25/09 đã
   xác minh lại raw/hash. Mỗi file sau phục hồi khớp hash nguồn. Đây là phục hồi
   từ nguồn đã xác minh, không phải khôi phục byte của cache trước sự cố (không
   có bản sao đó). Không sửa snapshot bất biến hay giá dùng cho 36 replay.

Ledger sau phục hồi: 9.991.783 byte,
SHA256 `6502830b7b0d5e39fc25b3edfc97e07218bcfd91d9a2a1f71b23fb7ffbaab7e8`.
Bản sao trước phục hồi:
`b0750cf532462e6d163dfb51995bac598b44388f569d9af5a21fcd101dc3eed0`.
Các file cách ly có thể khôi phục tại
[scratch_quarantine](D:/Chungkhoan/data/paper/backtests/hypotheses_20260926/scratch_quarantine).
**Không tìm thấy bản sao cache scan cũ đã bị ghi đè; không giả lập lại lịch sử.**
Cache phát sinh đã được đưa ra khỏi đường sử dụng, nên dashboard legacy cần
lượt quét hợp lệ mới để có `latest_scan.json`. Không sửa model/config giao dịch,
không đặt lệnh; dữ liệu snapshot/backtest chính không phụ thuộc cache này.

## Quyết định sau kiểm định

**A: chưa có lợi ích thực dụng để ưu tiên. B/C: giữ là giả thuyết nghiên cứu,
không nâng cấp production.** Mục tiêu là vượt VNINDEX nên cải thiện Sharpe hay
drawdown do giữ tiền mặt không tự động được tính là thành công.

Điều kiện để tăng mức tin cậy tiếp theo: dựng thành phần rổ và corporate actions
point-in-time; kiểm tra đóng góp chọn cổ phiếu so với thay đổi mức đầu tư bằng
các đối chứng đã khóa mới; sau đó quan sát forward chưa từng xem. Mọi phép thử
tiếp theo cần mã thử và tiêu chí riêng, không đổi tên các giai đoạn đã xem thành
holdout, không chỉnh ngưỡng đến khi H1 có lời. Chưa có cơ sở hứa thắng thị trường.
