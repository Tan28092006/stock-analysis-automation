# Vì sao ít lệnh và nghiên cứu momentum từ sách

## Phạm vi đã khóa

Người dùng muốn nhiều cơ hội hơn và momentum có tín hiệu tốt hơn. Hai mục tiêu
không đồng nghĩa: nhiều lần mua bổ sung không phải nhiều ý tưởng độc lập; tăng
vòng quay có thể chỉ tăng phí. Chưa thay rule production, không đặt lệnh hoặc
retrain. Bộ thử nghiệm nằm trong `scripts/momentum_books.py` và protocol
`configs/research/momentum_books_v1.json`, đăng ký trước khi đọc kết quả mới.

Trong lúc chạy, người dùng xác nhận ưu tiên **cả vị thế mới lẫn mua/bán bổ sung,
nhưng phải hiệu quả sau phí**. Dùng các kịch bản phí/delay đã khóa để đánh giá;
không thay tham số hoặc thêm trial sau khi xem kết quả vì câu trả lời này.

Sử dụng MLE/TDD: định nghĩa điều kiện kiểm chứng trước, test đỏ rồi xanh, giữ
baseline và mọi thất bại, không tự ghép các thành phần thắng thành một bộ tối ưu.

## Chẩn đoán ít lệnh

Snapshot giá đã xác minh đến 29/09/2026, cửa sổ 01/04–29/09. Với universe VN30
theo lịch sử, bộ lọc bắt đáy hiện tại đi qua các bước:

| Bước AND theo thứ tự | Lượt mã–phiên còn lại |
|---|---:|
| Đủ lịch sử và thành phần PIT, có phiên vào lệnh trong cửa sổ | 3.720 |
| RSI14 < 35 | 276 |
| Close <= 1,02 × Bollinger lower | 199 |
| Nến tăng và volume climax, hoặc VSA stopping volume | 6 |
| Khoảng về Kijun đủ bù 0,5 lần khoảng stop | 4 |
| Đường về mục tiêu không bị cloud chặn | 4 |

Đây là số đếm có điều kiện theo thứ tự, không phải đóng góp nhân quả của mỗi
tín hiệu. Có 160 lượt chỉ trượt confirmation trong khi các gate còn lại đạt,
nhưng chưa có nghĩa đó là 160 giao dịch tốt hoặc đều khớp được.

4 tín hiệu thực tế: MSN ngày 08/07; BID, HPG, VRE ngày 23/07. Cả 4 được mua
ở phiên kế tiếp trong replay 1 tỷ VND. Không có tín hiệu baseline bị mất vì
thiếu tiền hoặc giá vào lệnh trong cửa sổ này. Do đó tăng vốn/max_positions
không giải quyết được nguyên nhân tín hiệu hiếm của giai đoạn đã đo.

Momentum khác: paper 29/09 có 10 picks, TCX chưa đủ 254 bar cho 12-1. Xếp hạng
bỏ 21 phiên gần nhất, tái cân bằng theo tháng, giữ mã cũ trong top20. Nó không
được thiết kế như máy phát điểm mua mới hằng ngày. Replay 6 tháng có 6 đợt
tái cân bằng, 67 lượt mua trên 16 ngày; cần tách mua mới khỏi bổ sung cùng mã.
Danh sách 10 picks hằng ngày cũng không đồng nghĩa 10 lệnh mua mới.

Không đánh đồng scorer momentum dạng scorecard trong `signal_engine` với
`momentum_scan` đang dùng xếp hạng 12-1. Thêm indicator vào scorer không dùng
sẽ không cải thiện danh mục momentum thực tế.

## Nguồn sách và điều chuyển thành giả thuyết

Đã tra tài liệu công khai chính thức của tác giả/nhà xuất bản; **không tuyên bố
đã đọc toàn văn những sách trả phí**. Kênh Exa trong skill nghiên cứu không có
trong phiên này, nên dùng công cụ web sẵn có. Không lấy bản PDF sách lậu hoặc
dùng kết quả quảng cáo làm bằng chứng hiệu quả.

| Sách/hướng nghiên cứu | Ý tưởng có thể kiểm chứng | Cách triển khai và giới hạn |
|---|---|---|
| Andreas Clenow — *Stocks on the Move* | Xếp hạng cả tốc độ tăng lẫn độ đều của xu hướng | OLS log-price 90 phiên, annualized slope × R². Giữ nguyên history gate, sizing, buffer; chỉ thay ranking. Không sao chép toàn bộ strategy trong sách. |
| Wesley Gray & Jack Vogel — *Quantitative Momentum* | Cùng mức tăng nhưng đường đi đều có thể khác tăng giật cục | Top20 theo 12-1, rồi xếp theo negative information discreteness từ dấu return ngày; top10 và giữ buffer. Đây là adaptation xuống VN30, không phải danh mục Mỹ 50 mã của tác giả. |
| William O’Neil — *How to Make Money in Stocks* | Xác nhận cung/cầu ở điểm vượt giá bằng volume | So sánh breakout với breakout+volume. Kênh high50 là proxy tự định nghĩa, không nhận dạng cup-with-handle/flat-base và không phải CAN SLIM đầy đủ. |

Nguồn công thức Clenow: [bài mô hình Python của chính tác giả](https://www.followingthetrend.com/2017/01/getting-started-with-python-modeling-making-an-equity-momentum-model/).
Ông dùng hệ số annualization 250; thử nghiệm ở đây khóa 252 theo hệ thống hiện
tại và không nhân 100 vì chỉ dùng thứ hạng. [Tài liệu portfolio](https://www.followingthetrend.com/premium/equity-momentum-model-documentation/)
nhấn mạnh thành phần index lịch sử và corporate actions; dữ liệu giá hiện có
vẫn chưa chứng nhận đầy đủ corporate-action ledger.

Nguồn chất lượng momentum: [Alpha Architect — phương pháp Quantitative Momentum](https://alphaarchitect.com/quantitative-momentum-investing-philosophy/),
[nghiên cứu gốc về information discreteness](https://business.uq.edu.au/sites/default/files/events/files/mitch-warachka-paper.pdf).
Formation khớp baseline 252→21 phiên: dấu cumulative return nhân chênh lệch tỷ lệ
ngày giảm/ngày tăng; ID thấp hơn được ưu tiên. Chỉ số đo đường đi giá, không trực
tiếp đo mức chú ý hoặc tâm lý nhà đầu tư.

Nguồn O’Neil: [nhà xuất bản McGraw Hill](https://www.mheducation.com/highered/mhp/product/how-make-money-stocks-and-become-successful-investor-tablet-ebook.html),
[tài liệu IBD về volume breakout](https://shop.investors.com/images/promotional/flat-b-b_112408.pdf).
Rule thử nghiệm yêu cầu close vượt high của 50 phiên trước; bản volume thêm
volume >=1,5 × trung bình 50 phiên trước. Cả hai baseline đều loại phiên tín hiệu
khỏi cửa sổ so sánh, dùng dữ liệu EOD rồi mới vào lệnh ở phiên sau.

*Trade Like a Stock Market Wizard* của Mark Minervini đáng dùng cho nghiên cứu
tiếp về điểm mua và quản trị rủi ro, nhưng [mục lục chính thức](https://www.mheducation.com/highered/mhp/product/trade-like-stock-market-wizard-how-achieve-super-performance-stocks-any-market.html)
còn bao gồm fundamentals, earnings quality, nhóm ngành/catalyst. Không gọi một
bộ MA/volume là SEPA đầy đủ. Earnings revision, ngày công bố, nhóm ngành lịch sử
và catalyst availability chưa đủ dữ liệu PIT nên chưa thêm vào kiểm định này.

## Thiết kế kiểm định

6 nhánh: monthly baseline; weekly control; slope90; quality; breakout50;
breakout50_volume. Trừ monthly baseline, tất cả dùng phiên đầu tuần giao dịch
Việt Nam. Breakout/volume kiểm tra từng lần thử mua, kể cả mua bổ sung; các rule
thoát và target quantities vẫn từ rebalance. Đây là kiểm định một thành phần,
không đại diện mọi cách giao dịch breakout.

- 10 cửa sổ cũ giữ nguyên; mỗi nhánh chạy normal, double_cost và delay_one_session:
  tổng 180 replay mới. Đồng thời chạy lại đầy đủ 290 replay baseline/ablation cũ.
- Giữ PIT universe, next-open, lô 100, T+3 bảo thủ, không margin, vốn 1 tỷ VND.
  Số lệnh không suy rộng tuyến tính về tài khoản nhỏ vì giới hạn lô.
- Báo lợi nhuận ròng, vượt VNINDEX price gross, drawdown, exposure, turnover,
  phí/slippage và số lần từ không nắm giữ sang nắm giữ riêng với mua bổ sung.
- Chỉ suy luận chính trên continuous 2022–2026; các cửa sổ chồng lấp mô tả regime,
  không đếm như các mẫu độc lập. So sánh từng biến với control đã đăng ký.
- Bootstrap 4.000 lần, block20/40, Bonferroni family27 = 22 so sánh cũ + 5 mới.
  Không có hiệu chỉnh toàn cục cho mọi thử nghiệm lịch sử không ghi lại; tất cả
  giai đoạn đã nhìn là development, không phải holdout.
- Cửa sổ 6 tháng thuận lợi không đủ nâng cấp production. CI được thêm contract
  tests và replay180 nhưng chưa chạy remote/push.

## Kết quả

### Sáu tháng 01/04–29/09/2026

| Nhánh | Lượt mua | Mở vị thế từ 0 | Mua bổ sung | Lãi/lỗ ròng | Max DD | Chi phí gấp đôi | Chậm 1 phiên |
|---|---:|---:|---:|---:|---:|---:|---:|
| Monthly baseline | 67 | 13 | 54 | −0,105% | −12,353% | −0,547% | +0,767% |
| Weekly control | 109 | 13 | 96 | +0,644% | −12,300% | −0,235% | +0,100% |
| Weekly slope90 | 149 | 24 | 125 | −4,110% | −14,125% | −5,628% | −4,192% |
| Weekly quality | 136 | 14 | 122 | +1,803% | −12,591% | +0,985% | +0,288% |
| Weekly breakout50 | 10 | 7 | 3 | −1,090% | −2,679% | −1,308% | −0,406% |
| Weekly breakout50 + volume | 5 | 5 | 0 | −2,111% | −2,176% | −2,249% | −0,908% |

VNINDEX gross price cùng kỳ +3,723%. Tất cả nhánh đều chưa vượt chỉ số trong cửa
sổ này. Lãi/lỗ gồm mark-to-market cuối kỳ, không cưỡng bức bán vị thế chưa thoát;
chưa trừ phí thoát giả định của những vị thế còn mở.

Weekly tăng lượt mua 62,7% nhưng không tăng số lần mở vị thế mới. Slope90 tăng số
lần mở mới nhưng kết quả xấu hơn. Quality đáng nghiên cứu theo giả thuyết, nhưng
chưa thể chọn chỉ vì 6 tháng tốt hơn; kết quả nhạy thời điểm thực thi. Breakout
và volume làm giảm số lệnh; exposure trung bình chỉ 22,65% và 7,88%, nên DD thấp
không tự động là chứng cứ stock-selection alpha.

### Toàn kỳ và kiểm định thống kê

**180/180 replay hoàn tất**, không thay rule production. Continuous là
01/09/2022–29/09/2026, 1.014 phiên, lợi nhuận cộng dồn sau phí/slippage, không
phải lợi nhuận mỗi năm. VNINDEX gross price +38,857%.

| Nhánh | Lượt mua | Mở từ 0 | Lợi nhuận | Max DD | Exposure TB | Phí gấp đôi | Chậm 1 phiên |
|---|---:|---:|---:|---:|---:|---:|---:|
| Monthly baseline | 403 | 37 | +43,436% | −27,438% | 90,67% | +36,665% | +48,455% |
| Weekly control | 1.095 | 39 | +38,123% | −26,612% | 89,70% | +30,049% | +36,999% |
| Weekly slope90 | 1.229 | 78 | +23,542% | −21,974% | 88,64% | +14,459% | +21,063% |
| Weekly quality | 1.172 | 63 | +23,339% | −26,917% | 89,49% | +15,399% | +18,732% |
| Weekly breakout50 | 135 | 30 | +38,990% | −17,736% | 61,97% | +35,971% | +55,322% |
| Weekly breakout50 + volume | 93 | 25 | +43,113% | −15,998% | 54,91% | +40,065% | +43,220% |

Weekly tạo 1.056 lượt mua bổ sung, chỉ 39 lần mở từ 0. Phí thực trả trong mô
phỏng tăng từ 28,03 triệu lên 40,12 triệu VND; slippage 14,25 → 20,28 triệu VND.
Hai chiều turnover 13,87 → 19,76 lần NAV bình quân. Số lệnh tăng không đáp ứng
mục tiêu cải thiện sau phí trong mẫu dài này.

Breakout+volume có drawdown nhỏ hơn và giữ được +40,065% ở double-cost, nhưng
không phải giải pháp tăng tần suất. Exposure thấp hơn là một phần khác biệt;
control VNINDEX theo exposure hôm trước cho +28,55%, vẫn không thay thế kiểm
định tín hiệu. Nhánh breakout không-volume nhạy đáng kể với delay (+38,99% so
với +55,32%); không được chọn độ trễ tốt nhất sau khi xem số liệu.

Tất cả 5 so sánh mới đều **inconclusive_or_no_edge** ở cả block20 và block40,
Bonferroni p=1, các family CI đều chứa 0. Ví dụ đóng góp riêng volume so với
breakout không-volume có family CI của chênh lệch mean-return annualized
**[−8,00; +9,06] điểm %/năm** (block20), **[−9,59; +10,68]** (block40). Đây
không phải khoảng CAGR, không phải chứng minh volume vô dụng; chưa đủ bằng
chứng nó cải thiện ổn định theo protocol này.

**Quyết định:** không thay monthly bằng weekly chỉ để có nhiều lượt mua; không
promote slope90/quality; giữ breakout+volume như một giả thuyết cần kiểm định
prospective riêng, không phải winner đã được chọn. Các thử nghiệm chỉ đánh giá
thành phần cụ thể và cách ghép hiện tại, không bác bỏ toàn bộ sách/phương pháp.

Riêng kiểm tra Jan–Sep2026 đã chỉ ra lựa chọn cửa sổ rất quan trọng: baseline
−1,593%, weekly −4,005%, slope90 −9,607%, quality −10,217%, breakout −4,971%,
breakout+volume −5,821%. Quality bắt đầu tháng 4 đẹp hơn không xóa được kết quả
đầu năm xấu; các block bắt đầu bằng cash và buffer tạo danh mục khác nhau. Đây
là lý do cần continuous replay, không nối lợi nhuận của các block cash-start.

### Kiểm chứng phần mềm và tái lập

- 17 contract tests mới: ranking thực sự đổi danh sách mã, công thức đối chứng,
  cutoff breakout/volume, ngày nghỉ, weekly/monthly mặc định, không nhìn tương lai,
  mua bổ sung khác mở vị thế mới, fees/delay/T+3 và không mutate rules.
- Full repo: **472 passed**, 23 warnings; không thất bại hoặc skip.
- Coverage module mới **96%** khi cộng unit/integration với replay180 thực tế;
  suite unit/integration riêng trước 4 test bổ sung là 72%, không nhầm là 96% unit-only.
- 290 replay cũ giữ nguyên toàn bộ `blocks` so với run_v2, không source-hash drift.
  Artifact `data/paper/research_gate/20260930/books_baseline_regression/results.json`,
  SHA256 `79cf889527b427f263217e4532f86578cdcabd5c2d0c0820c4a37f58d9801886`.
- Snapshot dùng chung: `data/paper/research_gate/20260930/snapshot_20210824/manifest.json`,
  SHA256 `21b2deff6694a2a7218c861a681524b91f311617fc1e84de0598fc8001e155d0`.
- Replay mới: `data/paper/research_gate/20260930/books_v1/results.json`,
  SHA256 `18afa32989ff5013e3557cc756ed755edaf7eddeb2f8337f5e3e06746df65289`.
  Không source-hash drift; source checkpoint `5dfc960`. Sai số đối soát NAV/cash/
  receivables tối đa 0,00000144 VND trên 180 replay. Baseline monthly trong từng
  cửa sổ bằng chính xác replay cũ, không đổi đối chứng giữa chừng.

```text
python -m pytest -q --tb=short
python -m scripts.research_gate --manifest data/paper/research_gate/20260930/snapshot_20210824/manifest.json --output data/paper/research_gate/20260930/new_baseline_run
python -m scripts.momentum_books --manifest data/paper/research_gate/20260930/snapshot_20210824/manifest.json --output data/paper/research_gate/20260930/new_books_run
```

Output phải là thư mục mới; không ghi đè trial đã xem. Đã sửa README không còn
gọi bảng legacy là kiểm định ngoài mẫu, không nói momentum tự tắt trong panic,
không gọi inverse-vol là risk parity đầy đủ hoặc artifact ML cũ là ex-ante sạch.

## Những lỗi tư duy cần tránh

Nhu cầu có hành động mỗi ngày dễ gây overtrading; giữ lệnh thua vì muốn hòa vốn
khác với giữ theo rule có kiểm định. Chạy theo mã vừa tăng mạnh có thể là phản
ứng với tin cũ; ngược lại bỏ tháng gần nhất cũng có thể chậm nhận ra regime mới.
Các khả năng này phải được đo, không gán tâm lý là nguyên nhân chắc chắn.

Không thêm nhiều chỉ báo tương quan rồi gọi đó là nhiều bằng chứng. Không bỏ
confirmation bắt đáy chỉ để tăng số lệnh: ablation cũ no_confirmation lỗ 21,05%
ở cuối 2022 và 30,61% liên tục. Không áp today's VN100 vào quá khứ để có nhiều mã:
cần lịch sử membership/listing/liquidity trước khi mở rộng universe.

Một giới hạn cấu trúc là top10 trong khoảng 30 mã, cộng buffer top20: mức chọn lọc
rộng hơn nhiều quy trình momentum trên universe lớn. Điều này giải thích vì sao
thay lịch giao dịch có thể chủ yếu chỉnh tỷ trọng chứ không tìm được mã mới.
Không thể khắc phục thiếu breadth bằng cách thêm vô hạn indicator trên cùng
30 mã; cũng không được bỏ PIT để mở rộng rổ cho số liệu trông đẹp hơn.

Những hướng kế tiếp cần protocol riêng, **chưa được test hoặc bật** trong lần này:

1. Đồng nhất planner danh mục thực tế và replay: tách HOLD / ADD / NEW / REDUCE /
   EXIT, tính cash/settlement và phí rồi mới phát hành động. UI top10 không phải
   portfolio-aware order planner, nên tăng số dòng picks không giải quyết nhu cầu.
2. Mở rộng universe thanh khoản có lịch sử thành phần, niêm yết và khả năng giao
   dịch. Đánh giá có thu được thêm cơ hội độc lập, không chỉ thêm rủi ro penny.
3. Tách setup, trigger và exit của breakout/trend theo sách thành các giả thuyết
   riêng. Proxy high50 + exit theo rank không phủ hết một phương pháp như SEPA.
4. Bổ sung earnings surprise/revisions, sức mạnh nhóm ngành và catalyst chỉ sau
   khi có ngày công bố/lịch sử nhãn. Không lấy báo cáo hiện tại gán ngược quá khứ.

Khối ngoại v2 đã chuẩn hóa/cập nhật thật, nhưng mới có vintage nhận dữ liệu
từ 30/09. Không dùng backfill này giả lập quyết định trước thời điểm nhận nguồn.
