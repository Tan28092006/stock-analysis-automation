# Bắt đáy (bottom-fishing) cho VN30 + mã đang giữ — bằng chứng, phân loại cú giảm, mục tiêu, luật và đánh giá tiến cứu — 06/10/2026

## Phạm vi / revision

- Ngày: 06/10/2026. Reviewer: Firstmate run `2026-10-06-ml-roadmap` / T12 (scout + design).
- Source revision đã đọc: `08dc727` (branch `codex/data-integrity-audit`). Working tree đã có sẵn các file
  untracked (log/cache trong `data/pipeline/`, các audit ngày 06/10) và `docs/plans/...` đang sửa; T12 không
  đụng tới. T12 chỉ ghi hai file: tài liệu này và builder spec
  `.firstmate-native/runs/2026-10-06-ml-roadmap/reports/T12-bottom-fish-design.md`.
- Loại tài liệu: **nghiên cứu + thiết kế**. Không viết code, không đổi lịch, không đổi luật swing/MR/momentum,
  không đặt lệnh, không commit.
- Xây trên: `docs/audits/2026-10-06-swing-desk-signal-research.md` (gọi tắt **T10-doc**), builder spec T10,
  `.claude/skills/swing-desk/SKILL.md`, `configs/swing_desk/desk_v1.json`, `stock_agent/swing_desk/`.
  Mọi luật G0–G11, B1–B9 của desk giữ nguyên; T12 chỉ thêm một **mode riêng**.
- Phương pháp:
  - Đọc code/data local (chỉ đọc).
  - Ba script mô tả trong scratchpad (`bf_events.py`, `bf_clusters.py`, `bf_vn30.py`, không commit) trên
    `data/raw/prices_hist` (VN100, 2017→21/09/2026). Đây là **thống kê mô tả trên development data**, không
    phải backtest chiến lược, không qua research gate.
  - Một lần tải giá PNJ (VCI, `stock_agent.data.reconciliation.fetch_vci_history`, 06/10/2026 14:42 UTC) vào
    scratchpad cho case study, vì PNJ không thuộc snapshot VN30 của desk. Không ghi vào `data/`.
  - Đọc foreign/tự doanh v2 local của PNJ (run `20261006T100504578791Z_1febad33`).
  - Khảo sát tài liệu theo kỷ luật trích dẫn của skill `deep-research`. Bốn scout subagent chạy song song (A:
    đảo chiều/tin tức/pha loãng/nội bộ; B: biên độ giá/thanh khoản/margin; C: hành vi; D: quy tắc HOSE + sự
    kiện PNJ). Mỗi nguồn phải được **mở**. Tôi mở lại kiểm tra các nguồn gánh kết luận (ghi ✓).
  - Mục nào chỉ có snippet ghi PARTIAL/UNVERIFIED và **không** dùng làm căn cứ chính.
- Bối cảnh người dùng (coordinator chuyển lại):
  - NAV ≈ 11,2 triệu VND, giao dịch lô lẻ, tự nhận hay mua theo cảm xúc.
  - Hôm nay (06/10) đã mua PNJ trong phiên giảm sàn và gọi đó là bắt đáy.
  - Người dùng chưa biết đặt mục tiêu gì và luật gì cho bắt đáy.

## Kết luận ngắn

1. **"Bắt đáy" không phải một chiến lược mà là ba tình huống khác nhau**, và văn liệu cho kết quả ngược dấu
   giữa chúng (§4, §5):
   - (a) **Hoảng loạn toàn thị trường** ở một mã VN30 chất lượng. Có bằng chứng EV dương **trước phí, theo
     trung bình, có điều kiện**: lợi nhuận cung cấp thanh khoản cao nhất khi thị trường hỗn loạn [P13, P14,
     P10, P11]. Nhưng rủi ro thời điểm và đuôi trái rất lớn.
   - (b) **Cú sốc cơ bản riêng của doanh nghiệp** (bê bối, trích lập, lỗ, pha loãng, gian lận) — "dao rơi".
     Bằng chứng nghiêng về **trôi giảm tiếp** sau tin xấu [L10✓, L13, L5✓, L14–L18, L25, L26]. **Không có bằng
     chứng EV dương.**
   - (c) **Xả kỹ thuật/thanh khoản không có tin**. Có bằng chứng đảo chiều ở thị trường phát triển [L10✓, L13,
     P16, P19], nhưng nhỏ, nhạy phí [L6, P22], và "không tìm thấy tin" ≠ "không có thông tin".
2. **Dữ liệu repo nói cùng chiều, kèm cảnh báo nặng** (§4.2; VN100 hiện tại áp ngược 2018–09/2026,
   survivorship **có lợi** cho bắt đáy):
   - Mua sau cú giảm ≥15%/5 phiên trong bối cảnh **toàn thị trường**: lãi TB 10 phiên +4,6% ròng, win 62%.
   - Cũng mua như vậy nhưng cú giảm **riêng lẻ**: +1,5%, win 47%, trung vị âm.
   - Mua sau chuỗi sàn riêng lẻ khi xuất hiện phiên "ổn định" đầu tiên: win 10 phiên 46%, trung vị −1,9%.
   - Ở mọi nhóm, p10 MAE trong 20 phiên ≈ −23% đến −32%.
   - Năm 2026 là năm xấu: các nhóm (a) cũng chỉ thắng 27–47%.
   - Memory repo xác nhận: không có bộ lọc thời điểm đáy nào đứng vững (đã bác bỏ hai lần), và scale-in vẫn lỗ
     trong đợt sập chưa hồi.
3. **Biên độ ±7% và T+2.5 làm bắt đáy tệ hơn so với thị trường không biên độ**:
   - Sau phiên đóng cửa sàn, giá thường giảm tiếp 1–3 ngày rồi mới hồi: Trung Quốc ±10% [P4, P5]; hồi sau
     sàn ở Đài Loan/Thượng Hải [P6, P7].
   - Không tìm thấy nghiên cứu nào về lợi suất sau **N phiên sàn khóa liên tiếp**, kể cả ở HOSE (khoảng trống
     bằng chứng).
   - Người mua trong phiên sàn có thể không bán được ở chiều T+2 nếu phiên đó lại sàn trắng bên mua.
4. **Mục tiêu**: đề xuất 4 hồ sơ (§6).
   - P0 "không bắt đáy" là mặc định trung thực.
   - P1 "hồi ngắn sau hoảng loạn thị trường" và P2 "tích lũy theo tầng" chỉ cho loại (a)/(c): rủi ro 0,5–1%
     NAV = **56–112 nghìn VND/lệnh** với NAV 11,2 tr; vị thế khoảng 2,6–5% NAV vì lỗ hiệu dụng ~20–33%.
   - P3 "tình huống đặc biệt" cho loại (b) **không có bằng chứng edge**. Chỉ cho phép sau khi sự kiện đã ngã
     ngũ, size ≈ 1% NAV.
5. **BASIC GATE BF-G0..BF-G12** (§7), deterministic. Fail-closed khi thiếu dữ liệu, thiếu phân loại tin hoặc
   thiếu lịch sự kiện. Các luật cứng:
   - không mua khi đóng cửa/giá ở sàn hoặc đang chuỗi sàn khóa;
   - phải có ổn định (≥2 phiên không sàn + đáy cao hơn);
   - không mua trước sự kiện pha loãng/ĐHĐCĐ;
   - veto tin FACT tiêu cực;
   - không mua mã bị cắt margin hoặc bị cảnh báo/kiểm soát;
   - có ngân sách rủi ro và trần số lần thử;
   - phải viết THESIS **trước** lệnh.
6. **PNJ 06/10** (§10, ví dụ minh họa, **không** dùng để chỉnh luật): gate cho kết quả **BLOCK** với ≥7 luật
   cứng độc lập. Mã ngoài VN30, cú sốc cơ bản loại (b) có văn bản gốc, đóng cửa sàn, 6/7 phiên gần nhất đóng
   sàn, ĐHĐCĐ 21/10 có tờ trình phát hành riêng lẻ 550 triệu cp, người nội bộ/gia đình đang bán.
   Ngay trong năm 2026, PNJ đã có một "phiên đầu hàng" 25,6 triệu cp (08/07). Ai mua ở giá mở cửa phiên kế tiếp
   thì lỗ −37% sau 10 phiên.
7. Đánh giá tiến cứu (§11): giao thức riêng `bottomfish_prospective_v1`, đơn vị là **episode**. Không được
   tuyên bố gì trước ≥30 episode đã resolve mỗi nhánh và ≥3 đợt hoảng loạn thị trường khác nhau, tức nhiều
   khả năng là nhiều năm. Desk chỉ tư vấn, edge chưa được chứng minh.

## 1. Câu hỏi nghiên cứu (đóng băng trước khi tìm)

- RQ1: Có bằng chứng nào cho EV dương sau phí khi mua sau cú giảm mạnh, và nó phụ thuộc vào cái gì (tin có/không,
  toàn thị trường/riêng lẻ, biên độ giá, bán bắt buộc, khối lượng)? Tách thị trường phát triển và EM/Việt Nam.
- RQ2: Phân loại cú giảm nào cho phép luật khác nhau, và loại nào có EV dương sau phí + khóa T+2.5?
- RQ3: Mục tiêu (horizon, win rate, payoff, stop, size, số lần thử) nào là trung thực cho tài khoản 11,2 triệu?
- RQ4: Luật cơ bản và tín hiệu nâng cao nào có căn cứ, đo bằng trường dữ liệu nào?
- RQ5: Cờ hành vi và checklist nào nhắm đúng lỗi của người bắt đáy theo cảm xúc?
- RQ6: Ghi log và chấm điểm thế nào để sau này trả lời được "bắt đáy có lời không" mà không tự lừa?

Góc nhìn (angle): **thông tin hay áp lực thanh khoản là yếu tố quyết định**. Nhiều dòng văn liệu độc lập (tin tức,
bán bắt buộc, khối lượng, biên độ giá) hội tụ ở cùng một kết luận: giá giảm vì **thông tin** thì trôi tiếp, giá
giảm vì **áp lực không mang thông tin** thì hồi. Bài toán thực tế của người dùng là phân biệt hai loại này
**trước** khi mua, trong điều kiện ±7% và T+2.5.

## 2. Sự thật thị trường liên quan (đã kiểm nguồn)

| Quy tắc | Nội dung | Nguồn (grade) |
|---|---|---|
| Biên độ HOSE ±7% | "biên độ dao động giá … 7%"; ngày đầu niêm yết 20%. Vẫn áp dụng 10/2026: PNJ giảm "hết biên độ 6,93%" ngày 06/10 | [V1] PARTIAL (2021); [V2] VERIFIED ✓ (gián tiếp); T10-doc [R4] VERIFIED |
| Lô lẻ (1–99 cp) | Sổ lệnh tách biệt hoàn toàn với lô chẵn: lô lẻ "chỉ được khớp với nhau"; chỉ lệnh LO, không ATO/ATC/MP. Trên KRX: khớp định kỳ + liên tục + thỏa thuận, vẫn "Chỉ áp dụng giá LO" | [V3] PARTIAL (2022); [V4] PARTIAL (tài liệu CTCK "dự kiến"); [V5] PARTIAL |
| Lô lẻ ngày sàn trắng bên mua | **Không tìm thấy nguồn.** Suy luận từ [V3]: người bán lô lẻ cần người mua lô lẻ, cùng biên độ → ngày sàn khóa không có đảm bảo thoát hàng | UNVERIFIED — hỏi CTCK |
| T+2, bán chiều T+2 | Áp dụng từ 29/8/2022 (T10-doc [R1], [R2]). Còn hiệu lực 10/2026: cổ phiếu PNJ "bắt đáy" phiên 2/10 "bắt đầu về tài khoản theo chu kỳ T+2" ngày 6/10 | [V6] PARTIAL; [V7] VERIFIED |
| Giao dịch trong ngày / bán chờ về | Không thấy bằng chứng đã triển khai 10/2026 | UNVERIFIED (vắng bằng chứng) |
| Ký quỹ | QĐ 87/QĐ-UBCK (2017): ký quỹ ban đầu "không được thấp hơn 50%", duy trì "không được thấp hơn 30%". Loại khỏi margin: niêm yết < 6 tháng; "bị cảnh báo, bị kiểm soát, bị tạm ngừng giao dịch"; "có lỗ tại kỳ xem xét và/hoặc lỗ lũy kế"; công bố trong 2 ngày làm việc | [V8] PARTIAL (trang luật tổng hợp); TT 121/2020: UNVERIFIED (chưa mở được) |
| HOSE áp dụng tiêu chí lỗ | Theo BCTC kiểm toán năm/soát xét bán niên. 09/2026 HOSE bổ sung 5 mã vào danh sách cắt margin vì "lợi nhuận sau thuế … soát xét bán niên năm 2026 là số âm"; 69 mã HOSE không đủ điều kiện tính đến 3/9/2026 (PNJ không có trong danh sách) | [V9] VERIFIED; [V10] VERIFIED |
| CTCK cắt margin trước sàn | CTCK có quyền tự cắt sớm hơn. Tháng 7/2026, sau vụ P-Lab, PHS 50%→0, FPTS 50%→30%, TCBS loại, SSI 40%→0, KIS loại PNJ | [V16] VERIFIED |
| Call/force-sell | Ví dụ một CTCK: ngưỡng duy trì 35%, xử lý 30%, "thường là 3 ngày làm việc" để bổ sung; dưới ngưỡng xử lý bán phiên kế tiếp không cần đồng ý | [V11] PARTIAL (một CTCK, khác nhau theo CTCK) |
| VN30 review | Bán niên (01, 07), kỳ 7/2026: MCH, TCX vào, PLX, TPB ra, hiệu lực 3/8. Ngoài kỳ: DGC bị loại khi chuyển diện kiểm soát (6/5/2026), BSR thay từ 13/5. Quy tắc chỉ số mới của HOSE (QĐ 736–749, 29/9/2026): công bố thứ Hai tuần 1, hiệu lực thứ Hai tuần 3 của tháng 2/5/8/11 | [V12], [V13], [V14] VERIFIED |
| Tác động giá khi vào/ra VN30 | Không tìm thấy nghiên cứu lợi suất bất thường nào cho VN30. Chỉ có tin về cơ chế dòng ETF | UNVERIFIED (khoảng trống) |
| Giá tham chiếu GDKHQ quyền mua/phát hành | Cần văn bản gốc HOSE | UNVERIFIED |

Hệ quả cho thiết kế:

- Với lô lẻ, desk không được giả định thoát được ở chiều T+2 khi phiên đó sàn. Lỗ hiệu dụng phải cộng thêm ít
  nhất **một biên độ sàn (7%)**, không phải 1 điểm % như mode swing.
- Tình trạng cảnh báo/kiểm soát/cắt margin là **thông tin thị trường** (§7 BF-G8). Nó vừa báo hiệu cú sốc cơ bản,
  vừa tạo bán bắt buộc.
- Các kênh này là sự kiện riêng lẻ, nên không biến được cú giảm thành loại (a).

## 3. Bằng chứng văn liệu

Ký hiệu: DEV = thị trường phát triển; EM = mới nổi/cận biên; VN = Việt Nam. Grade theo §14.

### 3.1 Đảo chiều ngắn hạn sau cú giảm lớn

| Nguồn | TT | Phát hiện | Lưu ý |
|---|---|---|---|
| Lehmann 1990 [L2] | DEV | "'winners' and 'losers' one week experience sizeable return reversals the next week … which persist after corrections for bid-ask spreads and plausible transactions costs" | Danh mục tuần, không phải sự kiện sập riêng lẻ |
| Jegadeesh 1990 [L3] (T10 [A1] ✓) | DEV | Đảo chiều 1 tháng; chênh lệch decile 2,49%/tháng trước phí | Trước phí |
| Bremer & Sweeney 1991 [L4] | DEV | Giảm cực lớn được theo sau bởi lợi suất dương, "lasts approximately two days"; "perhaps not representing abnormal profit opportunities" | Định nghĩa sự kiện (1 ngày hay 10 ngày) cần kiểm lại bản gốc |
| Atkins & Dyl 1990 [L6] | DEV | Quá phản ứng "especially in the case of price declines", nhưng "small compared to the bid-ask spreads" | **Không qua được spread** |
| Cox & Peterson 1994 [L5] ✓ | DEV | Hồi ngắn do bid-ask bounce và thanh khoản; "do not find evidence consistent with the overreaction hypothesis"; "securities with large one-day price declines perform poorly over an extended time horizon" | **Dài hạn trôi giảm**, cốt lõi của rủi ro dead-cat bounce |
| Mazouz, Joseph & Joulmer 2009 [L7] | DEV (UK) | Sốc ≤ −5% được theo sau bởi **tiếp diễn** (CAR 1 ngày −0,43%) | Ngược với "luôn hồi" |
| Avramov, Chordia & Goyal 2006 [P22] (T10 [A3]) | DEV | Đảo chiều lớn nhất ở mã kém thanh khoản, turnover cao; lợi nhuận không qua phí | — |
| Pham, Nguyen & Tô 2007 [L28] | VN 2001–05 | "stock prices tend to reverse over the short-term period after large price changes"; lợi nhuận không bù được so với đầu tư thụ động | Mẫu HOSE rất sớm, mỏng |
| Truong et al. 2023 [L29] | VN 2004–21 | Danh mục thua thắng danh mục thắng 1,80–2,17% (tháng 2–3) | Danh mục tháng; tóm tắt không nói phí/survivorship |
| Nguyễn Thị Yến 2024/25 [L30] | VN (chỉ số) | Sau các ngày VN-Index biến động cực đoan: "phản ứng trễ và phản ứng thái quá … cường độ phản ứng với sự kiện tiêu cực mạnh hơn"; có momentum | Cấp chỉ số, không phải mã |
| Lin 2005 [L31]; Schaub, Lee & Chun 2008 [L32]; Mazouz, Joseph & Palliere 2009 [L33]; Parthasarathy & Sendilvelu 2022 [L35] | EM (TW, KR/HK/JP, Asia, IN) | Đài Loan quá phản ứng với tin xấu; chỉ số hồi 35–45% sau ngày giảm quá mức; kết quả HK phụ thuộc mô hình (OLS tiếp diễn, GARCH không); Ấn Độ đảo chiều đến 6 tháng | Phần lớn cấp chỉ số |

Đọc chung: đảo chiều ngắn hạn **tồn tại nhưng nhỏ**, phần lớn là phần thưởng cho việc cung cấp thanh khoản, và
không qua được phí ở nhiều mẫu. Riêng mã giảm mạnh một ngày thì **dài hạn kém** [L5]. Ở VN có dấu hiệu đảo
chiều [L28, L29], nhưng không có nghiên cứu nào kiểm cú sập mã đơn lẻ có phân loại tin, với T+2.5, ±7% và phí.

### 3.2 Có tin vs không tin (cốt lõi cho PNJ)

| Nguồn | TT | Phát hiện |
|---|---|---|
| Savor 2012 [L10] ✓ | DEV | "price events accompanied by information are followed by drift, while no-information ones result in reversals"; "drift exists only when the direction of the price move and of the change in analyst recommendations have the same sign" |
| Boudoukh, Feldman, Kogan & Richardson 2013 [L13] | DEV | "Conditional on extreme moves, stock price reversals occur on no news days, while identified news days show … a strong degree of cont[inuation]" |
| Chan 2003 [L8] | DEV | PARTIAL. Chỉ có metadata; nội dung qua snippet: tin xấu → trôi giảm, cú lớn không tin → đảo chiều, chủ yếu mã nhỏ kém thanh khoản. Đã ghi PARTIAL ở T10-doc [S4]; không dùng làm căn cứ độc lập |
| Pritamani & Singal 2001 [L9] | DEV | PAYWALL, không dùng phát hiện |
| Larson & Madura 2003 [L11] | DEV | Với mã tăng: quá phản ứng khi không có thông tin, không quá phản ứng khi có. Tóm tắt không nói rõ về mã giảm |
| Tetlock 2011 [L12] | DEV | Lợi suất ngày có **tin cũ** dự báo đảo chiều tuần sau, mạnh hơn ở mã nhiều NĐT cá nhân |

Kết luận: phân loại "có tin cơ bản mới hay không" là biến **quan trọng nhất** để quyết định có được bắt đáy hay
không. Đây là bằng chứng DEV, đo tin bằng báo/khuyến nghị analyst. Việt Nam không có nghiên cứu tương đương mở
được. Riêng PNJ: tin ở đây là văn bản gốc của chính doanh nghiệp [V20, V21] (FACT_PRIMARY), không phải tin đồn.

### 3.3 Tin xấu cơ bản: PEAD, gian lận, xóa sổ, kiệt quệ

| Nguồn | TT | Phát hiện |
|---|---|---|
| Bernard & Thomas 1989 [L14] | DEV | PARTIAL (qua Wikipedia): mã có tin lợi nhuận xấu trôi giảm khoảng 60 ngày |
| Hirschey & Richardson 2003 [L16] | DEV | Xóa sổ goodwill: khoảng −3% lúc công bố rồi "roughly −11.02 percent … by the end of a one-year post-announcement period". 80 công ty |
| Karpoff, Lee & Martin 2008 [L15] | DEV | Khi sai phạm bị lộ, công ty mất thêm $3,08 cho mỗi $1 thổi phồng, trong đó $2,71 là danh tiếng. **Không** kiểm trôi giá sau tin, nên không dùng làm bằng chứng "tiếp tục giảm" |
| Coelho & Taffler 2008 [L25] | DEV (WP) | Phá sản không chiến lược: "post-event drift of around −29% over the subsequent 12-months" |
| Campbell, Hilscher & Szilagyi 2008 [L26] | DEV | Mã kiệt quệ tài chính có lợi suất thấp bất thường, độ lệch chuẩn cao hơn nhiều |
| Dawkins et al. 2007 [L36] | DEV | PARTIAL: hồi khoảng 1/3 sau ngày nộp phá sản ở 1993–99, không hồi ở 2000–03 (dead-cat bounce phụ thuộc giai đoạn) |

Kết luận: tin xấu **cơ bản** (lỗ, trích lập, xóa sổ, kiệt quệ) đi kèm trôi giảm từ vài tháng đến một năm ở DEV.
Không tìm thấy nghiên cứu nào về "trích lập dự phòng" kiểu PNJ.

### 3.4 Phát hành thêm, phát hành riêng lẻ, pha loãng

| Nguồn | TT | Phát hiện |
|---|---|---|
| Loughran & Ritter 1995 [L17] | DEV | 5 năm sau SEO: "only 7 percent per year" |
| Hertzel, Lemmon, Linck & Rees 2002 [L18] | DEV | Phát hành riêng lẻ: "positive announcements effects, with negative post-announcement stock-price performance"; DN phát hành thường sau giai đoạn kinh doanh kém |
| Dong, Gu & He 2016 [L19] | EM (CN, WP) | Ngược lại: NĐT phổ thông có lợi khi cổ đông kiểm soát tham gia mua |
| Nguyen, Tran & Nguyen 2022 [L20] | VN | Quản trị lợi nhuận quanh phát hành; "significant negative abnormal returns in the third year after the issuance" |
| Asquith & Mullins 1986; Masulis & Korwar 1986; Wruck 1989 | DEV | PAYWALL, không dùng |

Kết luận cho luật: một ĐHĐCĐ sắp biểu quyết **phát hành lượng lớn cổ phiếu ở giá chưa biết** là một sự kiện chưa
ngã ngũ. Hướng giá dài hạn sau phát hành ở DEV và VN là âm. Ở Trung Quốc có kết quả ngược khi cổ đông lớn tham
gia mua [L19]. Luật BF-G5 vì vậy chặn mua **trước** sự kiện, không phải vì biết chắc giá sẽ giảm, mà vì phân
phối kết quả chưa xác định và người mua nhỏ không có lợi thế thông tin.

### 3.5 Biên độ giá, hiệu ứng nam châm, sau phiên sàn

| Nguồn | TT | Phát hiện |
|---|---|---|
| Kim & Rhee 1997 [P1] | DEV (Tokyo) | Biên độ gây "delayed price discovery" và biến động lan sang phiên sau |
| Chen, Gao, He, Jiang & Xiong 2019 [P4] | EM (Thâm Quyến ±10%) | Sau phiên **đóng cửa sàn**: giảm tiếp lúc mở cửa ngày 1 (TB −1,13%), "continues to drop significantly on day 2 and day 3", rồi hồi ngày 6–10 (+0,34%), 11–20 (+1,33%), 21–60 (+1,68%). Không thấy mẫu này ở cú giảm lớn chưa chạm sàn |
| Wan et al. 2015 [P5] | EM (CN 2000–11) | Ngày sau phiên chạm sàn, "price continuation is much more likely … especially for down-limit hits" (75,9%) |
| Wong, Liu & Zeng 2009 [P6] | EM (Thượng Hải) | "price reversal at the floor" (không nêu horizon) |
| Huang 1998 [P7] | EM (Đài Loan 1971–93) | "significant price reversals following … the down-limit cases" |
| Cho et al. 2003 [P2]; Du, Liu & Rhee 2009 [P3] | EM (TW, KR) | Nam châm: hút mạnh lên trần, hút yếu xuống sàn (TW); hoạt động tăng tốc trước khi chạm biên độ (KR) |
| Wang, Chong & Chan 2014 [P8] | EM (CN) | Biên độ làm tăng biến động khi giá đi xuống và trì hoãn khám phá giá |
| Farag 2015 [L34] | EM (Ai Cập) | "price reversal is observed two and three days post lower and upper limit hits" |
| Farber, Nguyen & Vuong 2006 [P9] | VN 2000–06 | Chạm biên độ thành cụm; có chuỗi "upto 127 consecutive trading sessions"; biên độ "prevent the market from self-adjusting to equilibrium" |

Khoảng trống: **không mở được nghiên cứu nào về lợi suất sau N phiên sàn khóa liên tiếp**, cả ở Trung Quốc,
Đài Loan, Hàn Quốc và HOSE. Không có nghiên cứu HOSE nào về nam châm hay "trắng bên mua".

Hệ quả cho luật BF-G3:

- Bằng chứng gần nhất [P4, P5] nói giá **tiếp tục giảm 1–3 ngày** sau một phiên đóng cửa sàn.
- Với T+2.5, người mua ngay phiên sàn chịu đúng các ngày tiếp diễn đó mà không bán được.
- Phần hồi (nếu có) nằm ở ngày 6–60 và chỉ khoảng +3% tích lũy, nhỏ so với rủi ro.

### 3.6 Bán bắt buộc, margin và cung cấp thanh khoản

| Nguồn | TT | Phát hiện |
|---|---|---|
| Coval & Stafford 2007 [P10] | DEV | Người đối ứng quỹ bị rút vốn "earn highly significant returns for providing liquidity when few others are willing or able" trong các tháng sau |
| Bian, He, Shue & Zhou 2018 [P11] ✓ | EM (CN 2015) | Mã do tài khoản sát ngưỡng đòn bẩy nắm giữ nhiều chịu áp lực bán và giảm bất thường, "which subsequently reverse" |
| Nagel 2012 [P13] (T10 [A4]) | DEV | Lợi nhuận cung cấp thanh khoản biến động theo thời gian, tăng mạnh khi VIX cao |
| Hameed, Kang & Viswanathan 2010 [P14] | DEV | "economically significant returns to supplying liquidity following periods of large drops in market valuations" |
| Brunnermeier & Pedersen 2009 [P15] | Lý thuyết | Thanh khoản thị trường phụ thuộc vốn của nhà đầu cơ; vòng xoáy margin |
| Campbell, Grossman & Wang 1993 [P12] | DEV | Tự tương quan ngày giảm khi volume cao, tức giảm do thanh khoản thì hồi |

Kết luận: áp lực bán **không mang thông tin** (quỹ bị rút vốn, margin call) có xu hướng hồi, và người cung cấp
thanh khoản được trả công, đặc biệt trong hoảng loạn. Đây là cơ sở cho loại (a) và một phần loại (c).

Hai cảnh báo:

1. Người được trả công trong các nghiên cứu này là nhà cung cấp thanh khoản **có vốn và đa dạng hóa**, không phải
   một lệnh lô lẻ tập trung vào một mã.
2. Bán bắt buộc **do tin riêng của doanh nghiệp** là hỗn hợp: CTCK cắt margin PNJ tháng 7 [V16] là phản ứng với
   tin. Phần áp lực có thể hồi, nhưng phần thông tin thì không.

### 3.7 Hoảng loạn toàn thị trường vs cú sập riêng lẻ

| Nguồn | TT | Phát hiện |
|---|---|---|
| Da, Liu & Schaumburg 2014 [P16] | DEV | Đảo chiều nằm ở phần lợi suất **không giải thích bằng cơ bản**; lợi nhuận điều chỉnh rủi ro "four times the size of the standard reversal strategy"; "fire sales more likely demand liquidity" |
| Hameed & Mian 2015 [P17] | DEV | Đảo chiều trong ngành mạnh hơn, gắn với "noninformational shocks" |
| Nagel 2012 [P13] | DEV | Ngay cả đảo chiều danh mục ngành cũng sinh lời cao khi VIX cao |
| Daniel & Moskowitz 2016 [L27] (T10 [A6]) | DEV | Ở trạng thái "panic", mã thua quá khứ hồi mạnh cùng lúc với thị trường hồi |

Hệ quả: đo cú giảm **so với thị trường** (idiosyncratic = phần vượt beta × chỉ số) để phân loại (a) với (b)/(c).
Mã giảm cùng thị trường thì khả năng cao là áp lực thanh khoản. Mã giảm 39% trong khi chỉ số −1,5% (PNJ
25/09→06/10) thì gần như chắc chắn là thông tin riêng.

### 3.8 Khối lượng "đầu hàng" (capitulation)

| Nguồn | TT | Phát hiện |
|---|---|---|
| Conrad, Hameed & Niden 1994 [P18] | DEV | Mã giao dịch nhiều đảo chiều, mã giao dịch ít tự tương quan dương (tuần) |
| Llorente et al. 2002 [P19] (T10 [A9]) | DEV | "returns generated by risk-sharing trades tend to reverse … speculative trades tend to continue" |
| Gervais, Kaniel & Mingelgrin 2001 [P20]; Kaniel, Ozoguz & Starks 2012 [P21] | DEV + 41 nước | Volume bất thường cao → lợi suất tháng sau cao hơn (visibility), có ở cả EM |
| La Anh Thao 2012 [P23] | VN | Luận văn: hiệu ứng volume yếu ở VN (câu hỏi khác) |
| Nghiên cứu "selling climax/capitulation" cụ thể | — | [không tìm thấy công trình học thuật; chỉ có blog kỹ thuật] |

Hệ quả: volume cao **không tự nói** cú giảm là áp lực thanh khoản. Nó chỉ nói có nhiều giao dịch. Nếu bên bán là
người có thông tin hoặc thoái vốn vì tin xấu, giá tiếp diễn [P19].

Ví dụ PNJ 02/10/2026 (development data):
- 76,1 triệu cp khớp, khoảng 19× TB 20 phiên, kỷ lục kể từ niêm yết [V23].
- Khối ngoại bán 58,7 triệu, ròng khoảng −1.265 tỷ [V23]; flows v2 local ghi ròng −54,5 triệu cp.
- Giá chạm +2,83% trong phiên rồi đóng cửa sàn. Hai phiên sau giảm tiếp −6,07% và −6,93% [V2].
- Tổng hợp: "volume đầu hàng" là một chú thích (BF-A1), không phải tín hiệu mua.

### 3.9 Giao dịch nội bộ khi giá giảm

| Nguồn | TT | Phát hiện |
|---|---|---|
| Lakonishok & Lee 2001 [L21] | DEV | "Insiders in aggregate are contrarian … informativeness … is coming from purchases, while insider selling appears to have no predictive ability" |
| Cohen, Malloy & Pomorski 2010/2012 [L22] | DEV | Chỉ giao dịch nội bộ "opportunistic" có thông tin: 82 bps/tháng |
| Agrawal & Cooper 2015 [L23] | DEV | Lãnh đạo cấp cao của công ty phải điều chỉnh BCTC "sell substantially more stock during the misstated period" |
| Tran & Hoang 2022 [L24] | VN | Nội bộ mua trước công bố mua cổ phiếu quỹ rồi "sell intensively after the event" |

Hệ quả: **mua** nội bộ/cổ đông lớn là chú thích dương có căn cứ (BF-A6). **Bán** nội bộ nói chung yếu về dự báo,
nhưng bán dồn dập quanh bê bối là cờ đỏ. PNJ: người nội bộ bán 25 triệu cp (11–18/9) và em trai Chủ tịch đăng
ký bán 9 triệu cp trong 01–30/10 [V24].

### 3.10 Dead-cat bounce

Không tìm thấy công trình học thuật nào kiểm định trực tiếp "dead cat bounce" (hai lượt tìm, standard và extended).
Bằng chứng gần nhất:
- Cox & Peterson [L5]: hồi ngắn do vi cấu trúc, dài hạn kém.
- Chen et al. [P4]: giảm tiếp 1–3 ngày rồi hồi nhỏ.
- Dawkins et al. [L36]: hồi 1/3 ở một giai đoạn, không hồi ở giai đoạn khác.
- Coelho & Taffler [L25]: trôi −29%/12 tháng sau phá sản không chiến lược.

Ví dụ PNJ 07/2026 (development data, repo):
- 03–07/07: ba phiên sàn liên tiếp. 08/07 mở cửa sàn, đóng cửa +2,4% với 25,6 triệu cp (khoảng 20× bình thường).
- Một nhịp hồi trông như "đáy đầu hàng", nhưng giá rơi tiếp từ 52.000 xuống 30.750 trong 12 phiên (−41%).
- Hồi lên 37.900 đầu tháng 8, rồi trở lại 33.000 trước đợt sàn thứ hai.

### 3.11 Tóm tắt loại bằng chứng

| Loại bằng chứng | Ví dụ | Độ mạnh cho câu hỏi của người dùng |
|---|---|---|
| Sự kiện DEV có phân loại tin | Savor, Boudoukh | Mạnh về **chiều** (tin → trôi, không tin → hồi); không có T+2.5/±7% |
| Biên độ giá EM | Chen et al., Wan et al. | Trung bình; Trung Quốc ±10%, thiếu nghiên cứu sàn liên tiếp |
| Bán bắt buộc | Coval–Stafford, Bian et al. | Mạnh về cơ chế; người hưởng lợi là nhà cung cấp thanh khoản có vốn |
| VN | Pham et al., Truong et al., Nguyễn Thị Yến | Yếu; không nghiên cứu nào khớp đúng tình huống |
| Repo (mô tả) | §4 | Development data, survivorship, cụm thời gian; chỉ để định hướng |

## 4. Bằng chứng trong repo

### 4.1 Memory (development data, đã dùng nhiều lần, không phải holdout)

- `crash-response-by-shape`:
  - Tốc độ cú sập (không phải độ sâu) quyết định kết quả. Đợt 07/2026: MR bắt đáy thắng 10% (19/183), P(win)
    dự báo 55–79%.
  - 12 bộ lọc × 7 năm: **không bộ lọc trạng thái thị trường nào nâng tỷ lệ bắt đáy đúng một cách đáng tin**.
    Chờ "reclaim EMA20" còn **giảm** win.
  - Guard "chặn khi chỉ số tạo đáy mới 20 phiên" bị bác bỏ trên 7 đợt sập.
  - Đòn bẩy duy nhất còn lại: R:R ≥ 1,5 và size nhỏ. Circuit-breaker nhanh bị bác bỏ vì whipsaw.
- `swing-rsi2-edge`:
  - Scale-in 3 tầng (−7%, −14%) nâng bắt đáy từ 60% win / −1,33% lên 67% win / +0,97%/lệnh.
  - Nhưng dùng 1,7–2× vốn, đợt sập chưa hồi (07/2026) vẫn −6%, 2022 −2,2%, COVID −1,6%.
  - Edge dương thật chỉ đến khi "đổi trò chơi" sang mua dip **trong uptrend**.
- `strategy-edge-findings`: ở T+2 các đặc trưng gần như nhiễu. Stop ATR hẹp chủ yếu trả phí whipsaw; quản lý rủi ro
  thật là time-stop + cổng vào lệnh.
- `win-probability-metalabeling`: AUC gộp 0,496, có chút phân biệt trong khủng hoảng 2022 (0,557), nhưng thất bại
  ngoài mẫu (07/2026).

### 4.2 Thống kê mô tả mới (scratchpad, không commit, không phải backtest)

Thiết lập:

- `data/raw/prices_hist`: 100 mã VN100 **hiện tại** + VNINDEX, giá đã điều chỉnh, 2017-11 → 2026-09-21.
- Vào lệnh ở giá mở cửa phiên sau tín hiệu. Lợi suất tính đến **đóng cửa** e+2 (sớm nhất bán được), e+10,
  e+20, ròng phí 0,4%.
- "Sàn" = đóng cửa ≤ −6,5% (proxy; giá điều chỉnh nên không bắt chính xác giá sàn).
- Bối cảnh: **toàn thị trường** nếu VNINDEX 5 phiên ≤ −5%; **riêng lẻ** nếu > −2%; còn lại "hỗn hợp".
- Episode = tháng chứa sự kiện, để giảm phụ thuộc cụm.

| Sự kiện (VN100) | Bối cảnh | n | Số tháng | Win r10 | TB r10 | Trung vị r10 | TB r20 | p10 r20 | p10 MAE khóa |
|---|---|---|---|---|---|---|---|---|---|
| Giảm ≥15%/5 phiên, mua phiên sau | toàn TT | 738 | 31 | 61,9% | +4,6% | +4,5% | +8,0% | −17,3% | −9,4% |
| idem | riêng lẻ | 90 | 42 | 46,7% | +1,5% | −0,8% | +3,0% | −17,1% | −9,8% |
| Mua sau phiên sàn thứ 2 liên tiếp ("dao rơi") | toàn TT | 381 | 24 | 57,0% | +3,8% | +3,9% | +7,2% | −19,9% | −9,2% |
| idem | riêng lẻ | 62 | 27 | 56,5% | +3,8% | +1,0% | +7,1% | −17,8% | −7,0% |
| Mua sau phiên sàn thứ 3 liên tiếp | riêng lẻ | 8 | 7 | 37,5% | +1,7% | −8,0% | +4,7% | −21,5% | −7,0% |
| Phiên đầu tiên không sàn sau chuỗi ≥2 sàn | toàn TT | 314 | 21 | 63,4% | +3,6% | +4,7% | +6,0% | −18,0% | −10,2% |
| idem | riêng lẻ | 68 | 29 | 45,6% | +1,1% | −1,9% | +3,7% | −16,5% | −9,4% |

Thêm:

- **Rủi ro trong lúc giữ**: tỷ lệ chạm −10% trong 20 phiên là 44–60%; chạm −20% là 21–25% ở mọi nhóm.
- **Theo năm** (giảm ≥15%/5 phiên, mọi bối cảnh): win r10 là 2021 83%, 2025 80%, nhưng **2022 46%**, **2026
  38%** (TB −1,8%).
- **Episode**: trung bình theo tháng, nhóm toàn TT có r20 dương ở 84% số tháng, nhóm riêng lẻ 55%; "phiên ổn định
  đầu tiên" riêng lẻ chỉ 41%.
- **Chỉ VN30 hiện tại**: nhóm riêng lẻ có n = 6–14, quá nhỏ để kết luận. Nhóm toàn TT giảm ≥15%/5 phiên: win r10
  66%, TB r20 +9,0%, p10 MAE20 −23,5%; năm 2026 win r20 27%.
- **Volume ≥3× trên nhóm giảm ≥15%**: n = 38, chỉ là gợi ý (r20 +10,1%, p10 −4,8%). Đừng dùng làm luật.
- **"Đáy cao hơn" (higher low) ở phiên ổn định**: p10 r20 bớt xấu (−11% so với −23%, n = 112). Không đổi lãi TB.
  Gợi ý tác dụng giảm đuôi, không phải edge.
- **Mua sau sàn thứ 4 liên tiếp** (n = 80) trông rất tốt (win r20 81%), nhưng 50/80 sự kiện rơi vào hai ngày
  đáy toàn thị trường (16/11/2022, 10/04/2025) → thực chất chỉ khoảng 8 episode.
- Các mã bị khởi tố/kiệt quệ còn trong mẫu (NVL, PDR 11/2022) mất 50–60% trong 10–20 phiên.

Giới hạn (bắt buộc đọc):

1. **Survivorship nghiêng có lợi cho bắt đáy.** Rổ VN100 hôm nay không chứa các mã đã sập/hủy niêm yết (nhóm
   FLC/ROS 2022…), đúng là các ca loại (b) tệ nhất.
2. **Không phân loại được tin** từ dữ liệu giá. "Riêng lẻ" chứa cả (b) lẫn (c).
3. Sự kiện tụ cụm: vài ngày đáy chung chi phối (03/2020, 05/2022, 10–11/2022, 04/2025).
4. Giá điều chỉnh, proxy sàn, khớp ở giá mở cửa (lạc quan với lô lẻ và ngày sàn).
5. Đây là development data đã xem; không phải holdout.

Đọc chung với văn liệu:
- **Loại (a)** có EV trung bình dương trong mẫu, nhưng phụ thuộc giai đoạn rất mạnh (2022, 2026 xấu) với đuôi
  trái −20% ở 1/4 số lệnh.
- **Loại riêng lẻ** sau chuỗi sàn có trung vị âm và win dưới 50%.
- Không nhóm nào cho "win cao + lỗ nhỏ".

## 5. Phân loại cú giảm (taxonomy) và luật khác nhau

Phân loại là **bước bắt buộc trước mọi lệnh bắt đáy**. Thiếu bằng chứng phân loại thì desk xếp vào UNCLASSIFIED
và xử lý như (b).

| Loại | Định nghĩa vận hành (desk) | Bằng chứng EV sau phí + T+2.5 | Luật |
|---|---|---|---|
| **(a) MARKET_PANIC** | VNINDEX giảm ≥5% trong 5 phiên **hoặc** regime PANIC_BEAR với breadth < 40%; phần giảm riêng `idio_ret_10d = ret_10d − β60 × idx_ret_10d` > −8%; **không** có tin FACT tiêu cực của DN trong 20 phiên; mã ∈ VN30 | **Có, yếu và có điều kiện.** Lợi nhuận cung cấp thanh khoản cao trong hoảng loạn [P13, P14, P10, P11, P16]; repo mô tả dương (§4.2). Nhưng phụ thuộc giai đoạn, cú sập chưa hồi thì thua (memory 07/2026: 10% win), không có bộ lọc thời điểm đáy | P1/P2 được phép sau BF-G; size nhỏ; time-stop |
| **(b) IDIO_FUNDAMENTAL** | Có tin FACT_PRIMARY/FACT_REPORTED tác động ≤ −1 trong 20 phiên: bê bối/khởi tố, trích lập/lỗ/điều chỉnh kế hoạch, pha loãng/phát hành, kiểm toán ngoại trừ, cảnh báo/kiểm soát, cắt margin, vỡ nợ. Hoặc `idio_ret_10d` ≤ −15% kèm sự kiện đã biết | **Không có bằng chứng EV dương.** Bằng chứng nghiêng về trôi giảm [L10✓, L13, L5✓, L16, L17, L18, L25, L26]; repo riêng lẻ trung vị âm | **BLOCK** cho P1/P2. P3 chỉ sau khi sự kiện ngã ngũ, gắn nhãn "không có edge" |
| **(c) TECH_FLUSH_NO_NEWS** | `idio_ret_10d` ≤ −8% **và** agent đã tìm (có trích dẫn nguồn đã mở: CBTT HOSE, website DN, CafeF/Vietstock) mà không thấy tin FACT tiêu cực; không bị cảnh báo/cắt margin | **Có ở DEV, nhỏ và nhạy phí** [L10✓, L13, P16, P19; L6, P22]; VN chỉ có bằng chứng đảo chiều yếu, không qua phí [L28]. "Không tìm thấy tin" ≠ "không có thông tin" (công bố chậm, nội bộ) | P1 (size giảm một nửa), P2 không áp dụng |
| UNCLASSIFIED | Thiếu tin đã kiểm, thiếu dữ liệu chỉ số/beta | — | Xử lý như (b): BLOCK |

Ghi chú:

- Margin call toàn thị trường (ví dụ 22/07/2026, T10-doc [M7c] PARTIAL) là cơ chế của loại (a).
- CTCK cắt margin **một mã vì tin của nó** là loại (b).
- Một mã có thể chuyển loại: PNJ 07/2026 và 09–10/2026 đều là (b).
- Tốc độ cú sập (memory) là chú thích BF-A12, không phải luật. Văn liệu và repo không cho ngưỡng nào đáng tin.

## 6. Đặt mục tiêu: 4 hồ sơ (người dùng chọn một)

Thông số chung (NAV = 11.200.000 VND):

- 1% NAV = 112.000 VND.
- Phí 0,4% (stress 0,6%).
- Lô lẻ: size làm tròn 1 cp.
- **Lỗ hiệu dụng** = mức dừng + **7% (một phiên sàn không thoát được)** + 0,6% phí. Đây là lý do vị thế phải nhỏ.
- Stop 3% **vô nghĩa** khi mã đi được −7%/phiên và không bán được trước chiều T+2 (T10-doc §1.2: 22% lệnh ngẫu
  nhiên đã xuyên −3% trước khi được bán; với mã vừa sập ATR thường 5–9%, PNJ 06/10: 9,2%).

| | **P0 — Không bắt đáy** (mặc định) | **P1 — Hồi ngắn sau hoảng loạn** | **P2 — Tích lũy theo tầng** | **P3 — Tình huống đặc biệt** |
|---|---|---|---|---|
| Loại cú giảm | — | (a); (c) với size × 0,5 | (a) | (b), chỉ sau khi sự kiện ngã ngũ |
| Mục tiêu | Tránh lỗi lớn nhất của người mua theo cảm xúc; chỉ dùng S1 swing trong uptrend | Ăn nhịp hồi thanh khoản 3–10 phiên | Ăn phần hồi 20–40 phiên sau hoảng loạn | Đặt cược nhỏ vào đánh giá cơ bản của chính người dùng |
| Horizon | — | 3–10 phiên; time-stop 10 | 20–40 phiên; time-stop 40 | Tháng; xét lại sau mỗi sự kiện |
| Vào lệnh | — | Sau BF-G4 (≥2 phiên không sàn + đáy cao hơn), lệnh LO ≤ đóng cửa × 1,005 | Tầng 1 sau BF-G4; tầng 2/3 chỉ khi **mỗi tầng** lại qua BF-G3/G4 và nằm trong kế hoạch `plan=scalein` đã ghi trước tầng 1 | Sau ĐHĐCĐ/điều khoản phát hành đã công bố, ≥10 phiên không có FACT tiêu cực mới, qua BF-G3/G4 |
| Win rate thực tế (development, không hứa hẹn) | — | 55–65% ở mẫu toàn TT (§4.2), nhưng **27–47% ở năm xấu (2026)** | 60–67% (memory scale-in) nhưng vẫn lỗ ở đợt chưa hồi | Không biết; mẫu riêng lẻ < 50%, trung vị âm |
| Dạng payoff | — | Thắng nhỏ (+3..+7%), đuôi thua −12..−20% | Thắng +5..+10%; đuôi −20..−25% | Nhị phân, lệch trái; có thể mất 50% vị thế |
| Stop | — | (1) **Vô hiệu luận điểm**: đóng cửa < đáy hoảng loạn (`panic_low`) → bán phiên được phép kế tiếp. (2) **Thảm họa**: đóng cửa ≤ −12% giá vốn (xét từ chiều T+2). (3) **Time-stop** 10 phiên | (1) Vô hiệu: đóng cửa < `panic_low` × 0,95 sau tầng cuối, hoặc mã chuyển sang loại (b). (2) Thảm họa −25% giá vốn bình quân. (3) Time-stop 40 phiên | (1) Vô hiệu: tin FACT tiêu cực mới / điều khoản phát hành bất lợi. (2) Không stop giá: size chính là stop. (3) Xét lại sau mỗi sự kiện, tối đa 60 phiên |
| Lỗ hiệu dụng để tính size | 12 + 7 + 0,6 = 19,6% | — | 25 + 7 + 0,6 = 32,6% | Giả định mất 50% |
| Rủi ro/lệnh | — | **0,5% NAV = 56.000 VND** (mặc định); tối đa 1% = 112.000 | **1% NAV = 112.000 VND tổng các tầng** | **0,5% NAV = 56.000 VND** |
| Vị thế tối đa | — | 56.000 / 19,6% ≈ **285.700 VND (2,6% NAV)**; 1% → ≈ 571.400 (5,1%) | 112.000 / 32,6% ≈ **343.600 VND (3,1% NAV)**, chia ≤3 tầng ≈ 114.500 | 56.000 / 50% ≈ **112.000 VND (1% NAV)** |
| Lỗ tối đa dự kiến (VND) | 0 | 56.000 (112.000 nếu 1%); kịch bản 3 sàn liên tiếp khi đang khóa: −19,6% × vị thế ≈ 56.000 | ≈112.000 | ≈56.000 |
| Số lần thử tối đa/mã | 0 | **1 lần/episode**; sau 1 lần thua → cấm mã đó 20 phiên | 1 episode, ≤3 tầng đã khai báo | 1, không mua thêm |
| Tổng bắt đáy cùng lúc | 0 | ≤2 vị thế BF, tổng BF ≤ 15% NAV, tổng rủi ro mở (swing + BF) ≤ 4% NAV | như P1 | ≤1 vị thế P3 |
| Có bằng chứng edge? | Không áp dụng | **Yếu, có điều kiện, chưa chứng minh tiến cứu** | **Yếu**; scale-in là development data | **Không** |

Diễn giải trung thực:

- Với 11,2 triệu NAV, một lệnh bắt đáy "đúng luật" chỉ khoảng **0,3–0,6 triệu VND**. Nếu người dùng thấy con số
  này "quá nhỏ để đáng làm", đó là thông tin quan trọng: lợi nhuận kỳ vọng của bắt đáy không đủ để biện minh cho
  size lớn.
- Không hồ sơ nào có bằng chứng tiến cứu. P1/P2 dựa trên cơ chế có văn liệu (cung cấp thanh khoản) và mô tả
  development data. P3 dựa trên niềm tin cá nhân của người dùng.
- Phí lô lẻ trên lệnh rất nhỏ: một số CTCK có phí tối thiểu/lệnh (UNVERIFIED, hỏi CTCK). Nếu có, P2 chia 3 tầng
  có thể bị phí ăn mất. Spread sổ lô lẻ rộng hơn lô chẵn ([V3] PARTIAL).

## 7. BASIC GATE cho mode bottomfish (BF-G0..BF-G12)

Quy ước:

- Hard FAIL → BLOCK; soft FAIL → WARN.
- **Fail-closed**: với BF-G0, BF-G2, BF-G5, BF-G6, BF-G8, thiếu dữ liệu = FAIL hard. Ở mode swing, UNVERIFIED chỉ
  = WARN. Lý do: rủi ro chính của bắt đáy nằm đúng ở thông tin còn thiếu.
- Các ngưỡng là **tham số thiết kế có lý do**, không phải tham số đã tối ưu. Mọi tuyên bố "cải thiện lợi nhuận"
  phải qua research gate.

| Id | Luật & công thức | Tham số | Lý do | Nguồn | Data field |
|---|---|---|---|---|---|
| BF-G0 | Snapshot verified của **chính mã** cho phiên hoàn tất; mã ngoài VN30 cần snapshot "extra" verified (`refresh --held`) | — | Không bắt đáy trên dữ liệu cũ/không kiểm. PNJ không có trong snapshot VN30 | T10 G0; `reconciliation.build_market_snapshot` | manifest |
| BF-G1 | Lệnh BF **mới**: mã ∈ VN30 hoặc ∈ `extra_universe` do người dùng khai. Mã đang giữ ngoài phạm vi chỉ được `check`/`review`, không có hồ sơ P1/P2 | VN30 | Thanh khoản, phạm vi; loại (a) cần mã chất lượng | T10 G1 | universe, journal |
| BF-G2 | Phân loại cú giảm (§5) = a hoặc c (P1), a (P2); b → BLOCK (trừ P3 sau BF-G5); UNCLASSIFIED → BLOCK | −5%/5 phiên; −8%/−15% idio; 20 phiên | Tin → trôi, không tin → hồi | [L10✓, L13, P16, P17]; §4.2 | `idx_ret_5d`, `idio_ret_10d`, `beta_60`, `bf_news.class` |
| BF-G3 | Không mua ở sàn: ret_1d > −6,5% **và** không có phiên sàn khóa (O=H=L=C, ret ≤ −6,5%) trong 2 phiên gần nhất **và** `--price` > giá sàn hôm nay × 1,01 | 6,5%, 2 phiên, 1% | Sau đóng cửa sàn giá thường giảm tiếp 1–3 ngày; khóa T+2.5 khiến người mua chịu trọn | [P4, P5]; T10 §1.2 | `ret_1d`, `locked_floor_t`, `floor_streak`, `--price` |
| BF-G4 | Ổn định: `sessions_since_floor` ≥ 2, cả hai phiên không sàn, `low_t > low_{t−1}`, và `low_t ≥ panic_low` (không có đáy mới trong 2 phiên) | 2 phiên | Chủ yếu là luật **khả thi thoát lệnh + hành vi**: tránh mua trong chuỗi tiếp diễn. Higher low giảm đuôi trong mẫu mô tả (n = 112), không tăng lãi TB. Memory: chờ reclaim **giảm** win → BF-G4 không đòi reclaim MA | [P4, P5]; §4.2; memory | `sessions_since_floor`, `higher_low_t`, `panic_low` |
| BF-G5 | Sự kiện: không có ĐHĐCĐ có nội dung phát hành/tăng vốn/tái cấu trúc, phát hành riêng lẻ/quyền mua, GDKHQ, công bố KQKD trong 10 phiên tới; biểu quyết pha loãng còn treo → BLOCK đến khi kết quả + điều khoản được công bố. Lịch chưa kiểm → BLOCK | 10 phiên | Phân phối kết quả chưa xác định; dài hạn sau phát hành âm | [L17, L18, L20]; T10 G10 | `events.json` (agent, có trích dẫn) |
| BF-G6 | Veto tin: có tin FACT (trọng số ≥ 0,7) tác động ≤ −1 trong 20 phiên → BLOCK (= loại b). Đăng ký bán của nội bộ/người liên quan đang hiệu lực → WARN | 20 phiên | Tin xấu cơ bản trôi tiếp; bán nội bộ quanh bê bối là cờ đỏ | [L10✓, L13, L16, L23]; T10 §9.2 | `bf_news.json` (agent) |
| BF-G7 | Thanh khoản: GTGD TB 20 phiên ≥ 20 tỷ; lệnh ≤ 1% GTGD TB | 20 tỷ, 1% | Ngày sàn thanh khoản biến mất; lô lẻ khớp riêng | T10 G7; [V3] | `avg_value_20_bn` |
| BF-G8 | Không thuộc danh sách không đủ điều kiện ký quỹ của HOSE; không bị cảnh báo/kiểm soát/tạm ngừng; không có CTCK lớn vừa cắt margin trong 20 phiên → BLOCK; chưa kiểm → BLOCK | 20 phiên | Đây vừa là tín hiệu thông tin vừa là nguồn bán bắt buộc tiếp diễn | [V8, V9, V10, V16]; [P11, P15] | `bf_news.json` (`margin_cut`, `warning`) |
| BF-G9 | Bối cảnh: loại (c) chỉ khi regime ≠ PANIC_BEAR (nếu PANIC thì là (a)). Nếu VNINDEX ATR% ≥ 2 × trung vị 250 phiên → rủi ro/lệnh × 0,5 | 2× | **Không** dùng bộ lọc timing (đã bác bỏ trong repo); chỉ giảm size khi biến động cao | memory `crash-response-by-shape`; T10 [A11] Moreira & Muir | `market.json` |
| BF-G10 | Ngân sách: rủi ro ở lỗ hiệu dụng ≤ rủi ro hồ sơ; ≤2 vị thế BF; tổng BF ≤ 15% NAV; tổng rủi ro mở (swing + BF) ≤ 4% NAV | §6 | Sống sót qua chuỗi thua là đòn bẩy duy nhất còn lại | memory; T10 G9 | journal, plan |
| BF-G11 | Số lần thử: ≤ trần của hồ sơ cho mỗi episode; cấm 20 phiên sau một lần thua BF trên cùng mã; ≤2 episode BF mới/tháng | 1/episode, 20 phiên, 2/tháng | Chống gỡ lỗ, chống bình quân giá xuống trá hình | [H20, H21, H22]; T10 B3/B4 | journal (`reason=bottomfish`) |
| BF-G12 | Kế hoạch viết trước: dòng `THESIS` có `bf_type`, `profile`, `invalidation`, `time_stop`, `max_attempts`, và câu trả lời checklist §9.2, ghi **trước** lệnh mua đầu → thiếu là BLOCK (không chỉ WARN) | — | If-then plan viết trước có hiệu quả trung bình–lớn với hành vi mục tiêu; consider-the-opposite giảm neo giá | [H31] PARTIAL, [H29] | journal THESIS |

Verdict: dùng `verdict()` của desk. PASS ở mode bottomfish nghĩa là "không thấy lý do cấu trúc/thông tin/hành vi
để chặn **và** có kế hoạch viết trước", **không** có nghĩa là đáy đã hình thành.

Mode bottomfish **không** chạy G2 (regime) và G3 (C > SMA50) của mode swing, vì bắt đáy theo định nghĩa vi phạm
hai luật đó. Thay vào đó là BF-G2/BF-G9. Mọi luật hành vi B1–B7 của swing **vẫn chạy**, trừ khi BF-B thay thế
(§9).

## 8. ADVANCED signals cho bắt đáy (chú thích, không bao giờ nâng WARN/BLOCK thành PASS)

| Id | Tín hiệu & công thức | Tham số | Hướng / lý do | Nguồn | Data |
|---|---|---|---|---|---|
| BF-A1 | Volume đầu hàng: `vol_ratio_20_prior = V_t / mean(V_{t−20..t−1})`; cờ `capitulation_t` nếu ≥3 và (ret_1d ≤ −4% hoặc phiên mở sàn rồi giao dịch) | 3× | Volume cao + giảm do chia sẻ rủi ro thì hồi; do giao dịch có thông tin thì tiếp diễn. Ở (b) thường là thoái vốn có thông tin (PNJ 02/10: 19×, ngoại ròng −54,5 tr cp, rồi thêm 2 phiên giảm) | [P12, P18, P19, P20, P21]; VN yếu [P23] | snapshot + flows v2 |
| BF-A2 | Quá bán: RSI2 < 5, RSI14 < 25, `pct_b` < 0 | 5/25/0 | Chỉ là mô tả. PNJ 06/10: RSI2 = 0, RSI14 = 12,8 nhưng là (b). Quá bán không phân biệt (a) với (b) | T10 S1; [L2, L3] | `rsi2`, `rsi14`, `pct_b` |
| BF-A3 | Khoảng cách: `dist_sma50_atr = (C − SMA50)/ATR14`; `drawdown_250 = C/high_250 − 1` | — | Càng xa càng "rẻ" trong mắt người mua. Gần đỉnh 52 tuần mới là phía mạnh [H2]; xa đỉnh là phía yếu, không đảo chiều. Dùng cho cờ neo giá BF-B2 | [H2, H3] | snapshot |
| BF-A4 | Dòng ngoại đảo chiều: `f_sell_streak` (số phiên ròng bán liên tiếp), `f_net_turn_positive` (ròng mua sau ≥5 phiên ròng bán), `f_net_1d_pct_vol` | 5 phiên | Bên bán bắt buộc/thoái vốn cạn là điều kiện cần cho hồi thanh khoản. Bằng chứng VN về NN ngắn hạn mỏng (T10 [M5]); lịch sử v2 < 20 phiên → INSUFFICIENT_HISTORY | T10 A4; [P10] | flows v2 |
| BF-A5 | Tự doanh ròng 5 phiên | — | Chưa có bằng chứng dự báo (T10 A5). PNJ 02/10: tự doanh ròng mua +2,79 tr cp trong phiên ngoại bán | T10 [R8] | flows v2 |
| BF-A6 | Nội bộ: đăng ký/kết quả **mua** của nội bộ hoặc cổ đông lớn trong cú giảm → dương; **bán** → cờ đỏ (BF-G6 WARN) | 30 ngày | Mua nội bộ có thông tin, bán thì yếu, trừ quanh bê bối | [L21, L22, L23, L24]; T10 [R9] | agent (CBTT) |
| BF-A7 | Áp lực margin: dư nợ margin toàn TT/free-float (báo); số CTCK cắt margin mã; tỷ lệ ký quỹ CTCK cho mã | — | Bán giải chấp hồi khi áp lực hết [P11]; vòng xoáy [P15] | [V9, V10, V11, V16]; T10 [M7a–b] | agent |
| BF-A8 | Breadth: % VN30 > SMA20, số mã VN30 sàn (`limit_down_count`), % mã tăng | — | Để phân loại (a); breadth là đặc trưng quan trọng nhất của mô hình P(win) repo trong khủng hoảng (AUC 0,557 năm 2022, không ổn định) | memory; T10 A9 | `market.json` |
| BF-A9 | Phần giảm riêng: `idio_ret_10d`, `idio_share = idio_ret_10d / ret_10d` | — | Đảo chiều nằm ở phần không giải thích bằng cơ bản/ngành | [P16, P17] | snapshot + VNINDEX |
| BF-A10 | Giảm so với ngành: ret mã − ret TB nhóm ngành trong VN30 (ngân hàng, BĐS…) | 10 phiên | Đảo chiều trong ngành mạnh hơn | [P17] | snapshot + `sectors` |
| BF-A11 | Đồng hồ sàn: `floor_streak`, `locked_floor_streak`, `sessions_since_floor`, `last_floor_date` | — | Sau sàn giảm tiếp 1–3 ngày, hồi ngày 6–60 | [P4, P5, P6, P7] | snapshot |
| BF-A12 | Tốc độ cú sập: `episode_sessions` (từ đỉnh 60 phiên tới hôm nay), `episode_depth` | — | Memory: tốc độ quyết định kết quả. Không có ngưỡng đáng tin, chỉ để ghi log và phân tích tiến cứu | memory | snapshot |

Luật data snooping (T10-doc §6, Sullivan–Timmermann–White): mỗi chú thích chỉ được nâng thành luật sau khi đăng
ký giả thuyết và qua research gate / giao thức tiến cứu.

## 9. Cờ hành vi riêng cho bắt đáy và checklist trước lệnh

### 9.1 Cờ BF-B (chạy thêm vào B1–B7 của desk)

| Id | Cờ | Phát hiện | Mức | Căn cứ |
|---|---|---|---|---|
| BF-B1 | **Bắt dao rơi ngày sàn** | ret_1d ≤ −6,5% hôm nay, hoặc `--price` ≤ giá sàn hôm nay × 1,01, hoặc `locked_floor_t` ở t/t−1 | hard | Ngày sau sàn tiếp diễn 75,9% [P5]; giảm tiếp 1–3 ngày [P4]; NĐT cá nhân mua ròng mã có "extreme one-day returns" [H13] |
| BF-B2 | **Neo giá đỉnh cũ** | `drawdown_250` ≤ −40% và (reason ∈ {cheap, anchor, old_high} hoặc target trong THESIS ≥ `high_60`) | soft + bắt buộc trả lời câu hỏi neo giá | Neo giá [H1]; xa đỉnh 52 tuần làm kỳ vọng lệch lên [H3]; ngay cả chuyên gia M&A cũng neo đỉnh cũ [H4]; consider-the-opposite giảm neo giá [H29] |
| BF-B3 | **"Rẻ rồi"** (ảo giác giá danh nghĩa, xổ số) | reason ∈ {cheap}, hoặc giá giảm ≥50% so với `high_250` mà không có setup | soft | Đánh giá quá cao dư địa tăng của mã giá thấp [H8]; NĐT cá nhân thích mã kiểu xổ số, cầu tăng khi suy thoái, mã đó kém [H7]; MAX effect [H9] |
| BF-B4 | **Bình quân giá xuống** (= B4 của desk) | Mua thêm mã đang lỗ so với giá vốn | hard, trừ `plan=scalein` đã ghi trước tầng 1 **và** tầng này qua BF-G3/G4 | Odean 1998 [H16]; bằng chứng "mua thêm mã đang lỗ" chỉ ở nguồn thứ cấp [H39] PARTIAL; repo: scale-in vào đợt sập chưa hồi vẫn lỗ |
| BF-B5 | **Gỡ lỗ trên cùng mã** | Có lệnh BF lỗ đã đóng trên mã này trong 20 phiên (bổ sung B3 của desk, vốn chỉ xét 1 phiên) | hard | Break-even effect [H20]; trader nhận rủi ro cao hơn để gỡ [H21]; né mua lại mã từng lỗ là hành vi bình thường, nên quay lại ngay là bất thường [H17] |
| BF-B6 | **Lỗ giấy → liều hơn** | Danh mục đang có vị thế lỗ chưa thực hiện ≥10% | soft | Lỗ **chưa thực hiện** làm tăng chấp nhận rủi ro, lỗ đã thực hiện thì không [H22] |
| BF-B7 | **Tự tin quá mức / tự quy công** | Số lần thử > trần hồ sơ; hoặc size BF tăng > 1,25 × lần trước ngay sau một lần BF thắng | hard (trần) / soft (size) | [H23, H24, H25]; T10 B5 |
| BF-B8 | **Chú ý/đám đông** | reason ∈ {forum, group, tip, kol, news} hoặc mã lên tin vì sập | soft | Mã gây chú ý thu hút NĐT cá nhân [H13, H14]; herding trên HOSE kể cả thị trường giảm [H35, H36]; NĐT cá nhân trong bear market mua mã vừa giảm [H15] |
| BF-B9 | **Ngụy biện con bạc** ("giảm N phiên rồi phải hồi") | Phía LLM: THESIS dựa trên số phiên giảm | checklist | [H26] (tên đúng: "Inference by Believers in the Law of Small Numbers"); [H27]. Lưu ý: ở cấp chỉ số NĐT thường **ngoại suy** xu hướng [H28] |
| BF-B10 | **Disposition / giữ lệnh BF thua** (review) | Vị thế BF quá time-stop hoặc dưới mức vô hiệu mà vẫn giữ ≥1 phiên được phép bán | review | [H16, H18]; VN: [H37, H38] |

Phản bằng chứng phải ghi nhận:

- NĐT cá nhân mua sau khi giá giảm, xét tổng thể, có **lợi nhuận dư ngắn hạn dương** (phần thưởng thanh khoản)
  [H10].
- Trên HOSE, danh mục thua có hồi ở tháng 2–3 [L29].

Các cờ trên không nói "mua sau giảm luôn sai". Chúng nhắm vào các **biến thể cảm xúc** đã có bằng chứng là kém:
mua đúng ngày sàn, theo chú ý, bình quân giá xuống, gỡ lỗ, neo đỉnh cũ.

### 9.2 Checklist bắt buộc trước lệnh bắt đáy (trả lời bằng chữ trong THESIS)

1. **Loại cú giảm**: (a) thị trường, (b) tin riêng, hay (c) không tin? Dán link nguồn **đã mở** (CBTT HOSE, website
   DN, báo uy tín).
2. **Tin đã "chạy hết" chưa?** Sự kiện kế tiếp là gì và ngày nào (ĐHĐCĐ, giá phát hành, GDKHQ, KQKD, kết luận điều
   tra)? Tôi có đang mua **trước** sự kiện đó không?
3. **Điều gì chứng minh tôi sai?** Một mức đóng cửa cụ thể, một sự kiện cụ thể và một ngày cụ thể (time-stop).
4. **Nếu mai và 2 phiên nữa lại sàn**, tôi mất bao nhiêu VND? (Desk in sẵn con số.) Tôi chịu được không mà vẫn bán
   theo kế hoạch?
5. **Neo giá**: tôi có mua mã này ở giá hiện tại nếu chưa từng thấy giá cũ không? Viết 3 lý do giá cũ **không còn
   liên quan** [H29].
6. **Ai đang bán cho tôi?** Bán bắt buộc (margin, quỹ bị rút vốn) hay người có thông tin (nội bộ, quỹ thoái vốn vì
   tin)?
7. **Lần thử thứ mấy** với mã này trong đợt giảm này? Đã có kế hoạch tầng trước lệnh đầu chưa?
8. **Trạng thái của tôi**: vừa lỗ lệnh nào? Đang ôm lỗ giấy không? Cảm xúc (fomo, fear, revenge, hope…)?
9. **Premortem**: giả sử lệnh này **đã** lỗ 20%, vì sao? [H32, H33]
10. **Kế hoạch if-then**: NẾU đóng cửa < X **hoặc** đến phiên Y chưa lãi **hoặc** có tin Z, THÌ tôi bán ở phiên được
    phép kế tiếp (sớm nhất chiều T+2) [H31].

## 10. Ví dụ PNJ ngày 06/10/2026

> Đây là ví dụ minh họa áp dụng luật đã thiết kế từ văn liệu và §4.2. Các ngưỡng **không** được chỉnh để khớp PNJ
> theo hồi cứu. PNJ 07/2026 có xuất hiện trong mẫu mô tả §4.2 (1 trong 8 sự kiện "sàn thứ 3 riêng lẻ"), nên
> PNJ là development data, không phải bằng chứng xác nhận luật.

Sự kiện (nguồn §2, §14):

- **03/07/2026**: Công an Thanh Hóa triệt phá đường dây buôn lậu kim cương. Giám đốc Công ty giám định PNJ
  (P-Lab) bị khởi tố. PNJ sàn 58.700, dư bán 12,5 triệu [V15].
- **07/2026**:
  - 3 phiên sàn liên tiếp, "phiên đầu hàng" 08/07 (25,6 tr cp) rồi rơi tiếp tới 30.750 ngày 24/07.
  - CTCK cắt/giảm margin [V16].
  - 24 quỹ thoái hết [V17].
  - NĐT cá nhân trong nước mua ròng 27,1 tr cp trong tháng 7 [V17].
- **24–25/09**: Tờ trình 608/2026 [V20, primary].
  - Trích lập dự phòng 7.071.137.000.000 đồng, giả định khách bán lại/đổi 80% hàng đã bán.
  - Kế hoạch 2026 thành LNST −6.270.554 triệu.
  - Cổ tức 2025 giảm 20% → 10%; 2026 0%.
- **29/09**: Tờ trình 622/2026 [V21 ✓, primary].
  - Phát hành riêng lẻ tối đa 550 triệu cp (> 511 triệu cp đang lưu hành), giá ≥ max(TB 30 phiên, BVPS).
  - Hạn chế chuyển nhượng 1 năm. 60% vốn cho kế hoạch thu đổi.
  - ĐHĐCĐ bất thường 21/10/2026 [V19].
- **Nội bộ**: bán 25 tr cp (11–18/9); em trai Chủ tịch đăng ký bán 9 tr cp trong 01–30/10 [V24].
- **Giá** (VCI scratch fetch, khớp [V2, V22, V23]):

| Ngày | Mở | Cao | Thấp | Đóng | KL | % |
|---|---|---|---|---|---|---|
| 25/09 | 33.200 | 34.400 | 32.800 | 33.000 | 5,3 tr | −0,9 |
| 28/09 | 30.700 | 30.700 | 30.700 | 30.700 | 2,1 tr | −7,0 (sàn khóa) |
| 29/09 | 28.600 | 28.600 | 28.600 | 28.600 | 1,2 tr | −6,8 (sàn khóa) |
| 30/09 | 26.600 | 26.600 | 26.600 | 26.600 | 2,4 tr | −7,0 (sàn khóa) |
| 01/10 | 24.750 | 24.750 | 24.750 | 24.750 | 1,7 tr | −7,0 (sàn khóa) |
| 02/10 | 23.050 | 25.450 | 23.050 | 23.050 | 76,1 tr | −6,9 (sàn, ngoại bán 58,7 tr) |
| 05/10 | 23.050 | 23.300 | 21.450 | 21.650 | 11,0 tr | −6,1 |
| 06/10 | 21.600 | 22.350 | 20.150 | 20.150 | 16,3 tr | −6,9 (sàn, dư bán 14,7 tr) |

- **Bối cảnh**:
  - VNINDEX 25/09 → 06/10: 1.785 → 1.759 (−1,5%); PNJ −38,9%. Từ 02/07: −68%.
  - Regime desk 06/10: PANIC_BEAR (VNINDEX < EMA50, breadth 23%).
  - Chỉ báo PNJ (scratch): RSI2 0, RSI14 12,8, ATR 9,2%, GTGD TB20 228 tỷ.

Gate nói gì nếu người dùng chạy `check PNJ --mode bottomfish --price <giá sàn>` trong phiên 06/10:

| Luật | Kết quả | Lý do |
|---|---|---|
| BF-G0 | **BLOCK** | PNJ không có trong snapshot VN30 verified; không có snapshot extra |
| BF-G1 | **BLOCK** | Ngoài VN30 (rổ hiệu lực 03/08/2026); trước lệnh chưa nắm giữ → không có hồ sơ P1/P2 |
| BF-G2 | **BLOCK** | Riêng lẻ (−38,9% vs chỉ số −1,5%) và có FACT_PRIMARY tiêu cực → loại (b) |
| BF-G3 | **BLOCK** | ret_1d −6,93%; giá đặt ở sàn; 4 phiên sàn khóa 28/09–01/10; 6/7 phiên gần nhất đóng cửa sàn |
| BF-G4 | **BLOCK** | 05/10 không sàn nhưng đáy 21.450 < đáy 02/10 (23.050, không có higher low); 06/10 sàn lại |
| BF-G5 | **BLOCK** | ĐHĐCĐ 21/10 (11 phiên) biểu quyết phát hành 550 tr cp giá chưa xác định |
| BF-G6 | **BLOCK** | Trích lập 7.071 tỷ, kế hoạch lỗ 6.271 tỷ (văn bản gốc, 24/09); WARN thêm do đăng ký bán nội bộ đang hiệu lực |
| BF-G8 | **BLOCK** (chưa kiểm) | CTCK đã cắt margin từ 07/2026 [V16]. Danh sách HOSE 3/9 chưa có PNJ [V10]; trạng thái 10/2026 chưa kiểm → fail-closed. Theo QĐ 87 kế hoạch lỗ chỉ thành tiêu chí khi vào BCTC soát xét/kiểm toán (suy luận) |
| BF-G12 | **BLOCK** | Không có THESIS trước lệnh |
| BF-B1 | **BLOCK** | Mua ở sàn trong phiên sàn |
| BF-B2/B3 | WARN | Drawdown từ đỉnh 250 phiên khoảng −76%; giá danh nghĩa "rẻ" |
| BF-A1 | chú thích | Volume 02/10 19× nhưng bên bán là khối ngoại có thông tin/thoái vốn; 08/07 cũng 20× và giảm tiếp −41% |

**Verdict: BLOCK.** Kiểm lại sớm nhất khi **đồng thời**:

1. ĐHĐCĐ 21/10 đã diễn ra, kết quả và điều khoản phát hành đã công bố.
2. ≥2 phiên không sàn với đáy cao hơn.
3. Có snapshot verified cho PNJ.
4. Người dùng chọn P3 và viết THESIS.

Ngay cả khi đó, desk chỉ cho P3 (≤ 1% NAV ≈ 112.000 VND, nhãn "không có edge"). Desk **không** chuyển PNJ thành
loại (a).

Về vị thế PNJ người dùng **đã mua** hôm nay (đây là review, không phải lệnh):

- Sớm nhất bán được là chiều **08/10/2026** (T+2).
- B4 cấm mua thêm khi giá dưới giá vốn.
- Câu hỏi cần trả lời bằng chữ trước chiều 08/10: loại cú giảm; điều gì chứng minh sai; time-stop; có nắm giữ qua
  ĐHĐCĐ 21/10 không.
- Lô bắt đáy phiên 02/10 về tài khoản ngày 06/10 và tạo áp lực bán [V7]. Lô mua 06/10 sẽ về 08/10, cùng cơ chế.
- Desk không khuyến nghị bán hay giữ. Nó in kịch bản VND và yêu cầu kế hoạch.

## 11. Đánh giá tiến cứu: log và chấm điểm

Đăng ký `configs/swing_desk/bottomfish_prospective_v1.json` **trước** dòng log BF đầu tiên, tách khỏi
`prospective_v1` để không reset bộ đếm swing. Lịch sử đã xem (bao gồm §4.2 và PNJ) là development data.

- **Log**: dùng chung hash chain `recommendations.jsonl` / `checks.jsonl`, thêm `mode=bottomfish`, `bf_type`,
  `profile`, `episode_id` (mã + ngày bắt đầu episode), `attempt_no`, `bf_news_sha`, `events_sha`.
- **Nhánh** (đóng băng trước 09:00 phiên kế, bằng code, trước mọi analyst):
  - `bf_baseline`: mọi mã VN30 có BF structural PASS loại (a), kế hoạch P1 chuẩn (rủi ro 0,5%, stop vô hiệu
    `panic_low`, thảm họa −12%, time 10).
  - `bf_knife` (đối chứng "bắt dao rơi"): cùng mã/episode nhưng giả định mua ở mở cửa phiên sau phiên sàn đầu
    tiên. Đo cái giá phải trả hoặc lợi ích của BF-G3/G4.
  - `bf_random`: 3 mã VN30 ngẫu nhiên có ret_10d ≤ −10%, seed theo ngày, cùng kế hoạch.
  - `bf_desk` (tùy chọn, nếu desk run có mục bắt đáy): khuyến nghị analyst sau VeriHarness.
  - `user_bf`: mọi `check --mode bottomfish` và mọi lệnh journal `reason=bottomfish`.
- **Chấm** (bổ sung scorer):
  - Vào ở O phiên kế (LO ≤ C × 1,005; không khớp → `no_fill`).
  - Khóa 2 phiên.
  - **Phiên đóng cửa ≤ −6,5%: không thoát được, chuyển sang phiên sau**. Ghi `floor_lock_days`.
  - Stop vô hiệu/thảm họa xét theo đóng cửa từ e+2, thoát ở O phiên sau. Time-stop theo hồ sơ.
  - Phí 0,4%/0,6%.
  - Báo: net %, R, MAE/MFE, `floor_lock_days`, `no_fill`.
- **Giả thuyết** (đăng ký trước):
  - BF-H1: `bf_baseline` net TB > 0 ở phí 0,6% (lợi nhuận ròng dương, không chỉ "lỗ ít hơn VNINDEX").
  - BF-H2: `bf_baseline` > `bf_random`, ghép theo phiên.
  - BF-H3: `bf_baseline` − `bf_knife` (đo luật ổn định; được phép âm, tức luật chỉ là phòng thủ).
  - BF-H4 (hành vi): lệnh BF thật của người dùng gắn BLOCK có net TB < PASS. Chỉ mô tả cho đến khi mỗi nhóm
    ≥30 lệnh.
  - BF-H5: tỷ lệ lệnh `user_bf` vi phạm BF-B1/B4/B5 giảm theo thời gian (mục tiêu hành vi chính).
- **Mẫu**:
  - Đơn vị độc lập là **episode** (mã × đợt giảm), cụm theo đợt thị trường.
  - σ lợi suất 20 phiên ≈ 15% (§4.2: p10 ≈ −17%, p90 ≈ +30%). Phát hiện δ = 5 điểm % với α = 0,05, power 0,8 cần
    n ≈ (2,8 × 15/5)² ≈ **71 episode độc lập** mỗi nhánh. Phân cụm theo đợt làm số cần cao hơn.
  - Sàn tối thiểu cho **bất kỳ** câu nào ngoài mô tả: ≥30 episode resolve mỗi nhánh, ≥3 đợt hoảng loạn thị trường
    khác nhau (VNINDEX −5%/5 phiên), ≥12 tháng.
  - Với tần suất lịch sử (khoảng 3–4 đợt hoảng loạn/năm, nhiều năm không có), điều này có nghĩa là **nhiều năm**.
- **Phân tích**:
  - Bootstrap khối theo **đợt**.
  - Báo riêng: năm/đợt xấu, cash restart vs giữ, số episode mới mỗi tháng, các tháng không có episode
    (bình thường với bắt đáy).
  - Activity floor của desk (4 ý tưởng/tháng) **không áp dụng** cho BF. Người dùng quyết định có đặt sàn riêng
    không (AGENTS.md). Không bao giờ ép lệnh để đủ quota.
- **Quyết định**: kết quả xanh **không** phê duyệt giao dịch thật hay tăng size. Mọi thay đổi luật → protocol
  version mới, reset bộ đếm.

## 12. Quyết định người dùng phải đưa ra

1. Chọn hồ sơ mặc định: **P0** (khuyến nghị cho đến khi có kỷ luật ghi THESIS ổn định), P1, P2 hay P3. Có cho phép
   P3 (loại b) không?
2. Rủi ro/lệnh BF: 0,5% (56.000 VND) hay 1% (112.000 VND) NAV.
3. Phạm vi lệnh BF mới: chỉ VN30, hay thêm `extra_universe` (ví dụ mã đang giữ như PNJ)? Nếu thêm, chấp nhận
   `refresh --held` gọi VCI on-demand cho mã đó.
4. BF-B2/B3 (neo giá, "rẻ rồi") là soft hay hard.
5. Có đặt activity floor cho BF không (khuyến nghị: không).
6. Kế hoạch cho vị thế PNJ hiện tại: viết THESIS + điều kiện vô hiệu + time-stop trước chiều 08/10. Desk không
   quyết định thay.
7. Hỏi CTCK: lô lẻ ngày sàn trắng bên mua; phí tối thiểu/lệnh lô lẻ; tỷ lệ margin hiện hành cho PNJ; có lệnh điều
   kiện không.

## 13. Rủi ro hàng đầu và giới hạn

- **Không hồ sơ nào có bằng chứng tiến cứu.** Văn liệu mạnh nhất là DEV, đo tin bằng báo/analyst, không có ±7%,
  T+2.5, lô lẻ.
- **Phân loại tin là khâu yếu nhất**: phụ thuộc agent tìm và đọc tin (T10-doc: agent chỉ khớp nhãn chuyên gia
  52,4% [S22]). Khi nghi ngờ → (b).
- **Survivorship và cụm thời gian** làm thống kê repo đẹp hơn thực tế. Năm 2026 tệ.
- **Đuôi trái**: 1/4 số lệnh chạm −20% trong 20 phiên. Lô lẻ và ngày sàn có thể không thoát được. Size là phòng thủ
  duy nhất.
- **Luật ổn định (BF-G4) có thể bỏ lỡ đáy chữ V** (mẫu: mua dao rơi trong hoảng loạn thị trường không kém hơn).
  Đó là đánh đổi có chủ ý vì khả thi thoát lệnh và hành vi. `bf_knife` đo cái giá này.
- **Nguồn quy tắc HOSE** chủ yếu qua báo/CTCK/trang luật thứ cấp; hsx.vn, ssc.gov.vn, vsd.vn chưa mở được.
- Desk không thể chặn lệnh thật. Tài liệu này không phải chấp thuận giao dịch hay triển khai.

## 14. Nguồn

Grade:

- VERIFIED: trang đã mở, câu trích khớp.
- V(frag): công cụ fetch tóm tắt, chỉ các đoạn trích là nguyên văn.
- PARTIAL: thứ cấp, snippet, hoặc chỉ khớp một phần.
- PAYWALL: chỉ metadata.
- UNVERIFIED: không dùng làm căn cứ.

"Mở bởi": A/B/C/D = scout tương ứng; ✓ = tôi mở lại kiểm tra (T12). Báo cáo scout là testimony. Báo CTCK/analyst
là OPINION. Forum là ý kiến người chơi.

**Quy tắc thị trường và PNJ (V)** — scout D

- [V1] Thời báo Tài chính VN 05/07/2021, "Quy chế giao dịch chứng khoán mới tại HOSE" (QĐ 352/QĐ-SGDHCM),
  thoibaotaichinhvietnam.vn/quy-che-giao-dich-chung-khoan-moi-tai-hose-31056.html — PARTIAL (trước KRX).
- [V2] CafeF 06/10/2026, "Cổ phiếu PNJ lại chất đống sàn, dư bán hơn 14,7 triệu đơn vị",
  https://cafef.vn/co-phieu-pnj-lai-chat-dong-san-du-ban-hon-147-trieu-don-vi-188261006164817222.chn —
  VERIFIED (D, ✓).
- [V3] VnBusiness (09/2022), giao dịch lô lẻ 1–99 cp — PARTIAL.
- [V4] SHS, "KRX — Một số thay đổi dự kiến khi triển khai" (2025, PDF) — PARTIAL.
- [V5] Thị trường Tài chính Tiền tệ 25/04/2025, "Áp dụng hệ thống KRX…" — PARTIAL.
- [V6] SGGP 19/08/2022, T+2 từ 29/8 — PARTIAL; xem thêm T10-doc [R1], [R2] VERIFIED.
- [V7] CafeBiz 06/10/2026, "Buồn của nhà đầu tư bắt đáy PNJ",
  cafebiz.vn/buon-cua-nha-dau-tu-bat-day-pnj-17626100618312138.chn — VERIFIED (D).
- [V8] QĐ 87/QĐ-UBCK (25/01/2017) qua luatvietan.vn và caselaw.vn — PARTIAL (trang tổng hợp luật; QĐ sửa đổi
  1205/2017 chưa kiểm). TT 121/2020/TT-BTC — UNVERIFIED.
- [V9] VnEconomy 05/09/2026, "HOSE bổ sung thêm 5 mã vào danh sách cắt margin",
  vneconomy.vn/hose-bo-sung-them-5-ma-vao-danh-sach-cat-margin.htm — VERIFIED (D).
- [V10] Pháp luật VN (doanhnhan.baophapluat.vn) 06/09/2026, "HOSE nâng lên 69 mã…" — VERIFIED (D).
- [V11] DSC, "Call, force sell, bán giải chấp" (không ngày) — PARTIAL (một CTCK).
- [V12] Tạp chí Kinh tế Tài chính, "HOSE công bố danh mục VN30 kỳ tháng 7/2026" — VERIFIED (D).
- [V13] Thị trường Tài chính Tiền tệ 08/05/2026 (DGC sang diện kiểm soát, BSR thay vào VN30) — VERIFIED (D).
- [V14] Tạp chí Kinh tế Tài chính 05–06/10/2026, quy tắc chỉ số mới (QĐ 736–749) — VERIFIED (D).
- [V15] VnEconomy 03/07/2026, "Cổ phiếu PNJ dư bán sàn 12,5 triệu…" — VERIFIED (D).
- [V16] VnBusiness 23/07/2026, "Quỹ ngoại đua rời đi, margin bị siết…" — VERIFIED (D).
- [V17] Tuổi Trẻ News 21/08/2026, "24 funds exit…" — VERIFIED (D).
- [V18] Tuổi Trẻ News 19/08/2026, Vanguard/Sprucegrove 5,04% — VERIFIED (D).
- [V19] VietnamFinance 03/09/2026, "PNJ chốt lịch họp…" (ĐHĐCĐ 21/10, chốt DS 25/8) — VERIFIED (D).
- [V20] PNJ, Tờ trình 608/2026/TTr-HĐQT-CTY (24/09/2026), "Điều chỉnh kế hoạch kinh doanh 2026", cdn.pnj.io
  (PDF IR) — VERIFIED (D, primary). Ghi chú: LNST 6 tháng trong bảng tờ trình 1.549,45 tỷ ≠ CafeF 1.256 tỷ [V25].
- [V21] PNJ, Tờ trình 622/2026/TTr-HĐQT-CTY (29/09/2026), "Proposal on the Plan for private placement of shares",
  cdn.pnj.io — VERIFIED (D, primary); VnEconomy, "PNJ muốn phát hành thêm 550 triệu cổ phiếu",
  https://vneconomy.vn/pnj-muon-phat-hanh-them-550-trieu-co-phieu.htm — VERIFIED (✓).
- [V22] VietTimes 28/09/2026; CafeF 02/10/2026 "PNJ giải trình…" — VERIFIED (D).
- [V23] CafeF 02/10/2026, "Tiền bắt đáy đột biến, PNJ vẫn nằm sàn"; Người Quan Sát; Vietstock — VERIFIED (D).
  Số liệu ngoại 58,7 tr vs 58,5 tr không thống nhất giữa nguồn; tên người bán UNVERIFIED.
- [V24] CafeF 01/10/2026, "Vốn hóa PNJ giảm hơn 5.000 tỷ…"; Thị trường Tài chính Tiền tệ, "Em trai Chủ tịch PNJ
  đăng ký bán 9 triệu cổ phiếu" — VERIFIED (D).
- [V25] CafeF 31/07/2026 và 06/08/2026 (KQKD bán niên PNJ) — VERIFIED (D).
- [V26] `configs/universe_vn30.json` (local; PNJ không có trong rổ hiệu lực 03/08/2026); Pháp luật VN 15/07/2026 —
  PARTIAL.
- [V27] Tạp chí Kinh tế Tài chính 21/09/2026 (dự báo VNDiamond); Người Quan Sát — VERIFIED (D); dự báo trước
  02/10, là OPINION.
- [V28] VnBusiness 07/07/2026, SSI Research "under review" — PARTIAL, OPINION.
- Local: foreign/tự doanh v2 run `20261006T100504578791Z_1febad33` (PNJ); VCI fetch scratch 06/10/2026 14:42 UTC
  (không phải snapshot verified của desk).

**Đảo chiều, tin tức, tin xấu cơ bản, phát hành, nội bộ, EM (L)** — scout A trừ khi ghi khác

- [L1] De Bondt & Thaler (1985), JF 40(3):793–805 — PARTIAL (IDEAS không có abstract).
- [L2] Lehmann (1990), QJE 105(1):1–28, ideas.repec.org/a/oup/qjecon/v105y1990i1p1-28..html — VERIFIED.
- [L3] Jegadeesh (1990), JF 45(3) — VERIFIED (T10 ✓).
- [L4] Bremer & Sweeney (1991), JF 46(2):747–754 — VERIFIED; định nghĩa sự kiện cần kiểm bản gốc.
- [L5] Cox & Peterson (1994), JF 49(1):255–267, https://ideas.repec.org/a/bla/jfinan/v49y1994i1p255-67.html —
  VERIFIED (A, ✓).
- [L6] Atkins & Dyl (1990), JFQA 25(4):535–547 — VERIFIED.
- [L7] Mazouz, Joseph & Joulmer (2009), JBF 33(8):1481–1493 — V(frag).
- [L8] Chan (2003), JFE 70(2):223–260 — PARTIAL (metadata + snippet; MIT thesis page mở được).
- [L9] Pritamani & Singal (2001), JBF 25(4):631–656 — PAYWALL.
- [L10] Savor (2012), JFE 106(3):635–659, https://ideas.repec.org/a/eee/jfinec/v106y2012i3p635-659.html —
  VERIFIED (A, ✓).
- [L11] Larson & Madura (2003), J. Financial Research 26(1):113–127 — VERIFIED.
- [L12] Tetlock (2011), RFS 24(5):1481–1512 — VERIFIED.
- [L13] Boudoukh, Feldman, Kogan & Richardson (2013), NBER w18725, nber.org/papers/w18725 — V(frag).
- [L14] Bernard & Thomas (1989), JAR 27 — PARTIAL (Wikipedia). Bernard & Thomas (1990), JAE 13(4) — PARTIAL.
- [L15] Karpoff, Lee & Martin (2008), JFQA 43(3):581–611 — VERIFIED.
- [L16] Hirschey & Richardson (2003), FAJ 59(6):75–84 — V(frag).
- [L17] Loughran & Ritter (1995), JF 50(1):23–51 — VERIFIED.
- [L18] Hertzel, Lemmon, Linck & Rees (2002), JF 57(6):2595–2617 — V(frag).
- [L19] Dong, Gu & He (2016), CityU GRU WP 2016-021 — V(frag).
- [L20] Nguyen, Tran & Nguyen (2022), J. International Economics & Management, vjol.info.vn/jieme/article/view/76934
  — V(frag).
- [L21] Lakonishok & Lee (2001), RFS 14(1):79–111 — VERIFIED.
- [L22] Cohen, Malloy & Pomorski, NBER w16454 (2010) — VERIFIED (bản JF 2012 không kiểm).
- [L23] Agrawal & Cooper (2015), J. Corporate Finance 34:169–190 — V(frag).
- [L24] Tran & Hoang (2022), Cogent Economics & Finance 10(1) — V(frag).
- [L25] Coelho & Taffler (2008), working paper v1.2 (NYU Stern seminar PDF) — VERIFIED (WP).
- [L26] Campbell, Hilscher & Szilagyi (2008), JF 63(6):2899–2939 — VERIFIED.
- [L27] Daniel & Moskowitz, NBER w20439 — VERIFIED (T10 [A6]).
- [L28] Pham, Nguyen & Tô (2007), *Asia-Pacific Financial Markets* ch.10 (Emerald) — V(frag).
- [L29] Truong, Cao, Friday & Doan (2023), IJFS 11(2):58 — V(frag) (A, B, C).
- [L30] Nguyễn Thị Yến (2024/2025), Tạp chí Nghiên cứu Kinh tế 556(9), vjol.info.vn/tcvienkinhteVNvathegioi/article/view/112315
  — VERIFIED.
- [L31] Lin (2005), Sun Yat-sen Management Review 13(4S) — VERIFIED.
- [L32] Schaub, Lee & Chun (2008), Research in Finance 24 — V(frag).
- [L33] Mazouz, Joseph & Palliere (2009), Pacific-Basin Finance J. 17(4) — V(frag).
- [L34] Farag (2015), QREF 58:190–199 — V(frag).
- [L35] Parthasarathy & Sendilvelu (2022), Economic Thought — V(frag).
- [L36] Dawkins, Bhattacharya & Bamber (2007), JFQA 42(2) — PARTIAL.
- UNVERIFIED/không dùng: Asquith & Mullins (1986), Masulis & Korwar (1986), Wruck (1989), Seyhun (1986) —
  PAYWALL; Rozeff & Zaman (1998) — không mở được.

**Biên độ giá, thanh khoản, volume (P)** — scout B

- [P1] Kim & Rhee (1997), JF 52(2):885–901 — VERIFIED.
- [P2] Cho, Russell, Tiao & Tsay (2003), J. Empirical Finance 10(1–2) — VERIFIED.
- [P3] Du, Liu & Rhee (2009), Int. Review of Finance 9(1–2):83–110 — VERIFIED.
- [P4] Chen, Gao, He, Jiang & Xiong (2019), J. Econometrics 208(1):249–264; NBER w24014 — VERIFIED (B đọc PDF; số
  liệu sàn từ bảng 3).
- [P5] Wan et al. (2015), PLoS ONE 10(2), arXiv:1503.03548 — VERIFIED.
- [P6] Wong, Liu & Zeng (2009), China Economic Review 20(1):91–102 — VERIFIED.
- [P7] Huang (1998), J. Business Finance & Accounting 25(3–4):469–483 — VERIFIED.
- [P8] Wang, Chong & Chan (2014), MPRA 54146 — VERIFIED (WP).
- [P9] Farber, Nguyen & Vuong (2006), ULB CEB WP 06-005 — VERIFIED (B đọc PDF).
- [P10] Coval & Stafford (2007), JFE 86; NBER w11357 — VERIFIED.
- [P11] Bian, He, Shue & Zhou (2018), NBER w25040, https://www.nber.org/papers/w25040 — VERIFIED (B, ✓ nội dung).
- [P12] Campbell, Grossman & Wang (1993), QJE 108(4) — V(frag).
- [P13] Nagel (2012), RFS 25(7):2005–2039, NBER w17653 — VERIFIED (T10 [A4]).
- [P14] Hameed, Kang & Viswanathan (2010), JF 65(1):257–293 — VERIFIED.
- [P15] Brunnermeier & Pedersen (2009), RFS 22(6):2201–2238 — V(frag).
- [P16] Da, Liu & Schaumburg (2014), Management Science 60(3):658–674 — V(frag).
- [P17] Hameed & Mian (2015), JFQA 50(1–2):89–117 — V(frag).
- [P18] Conrad, Hameed & Niden (1994), JF 49(4) — VERIFIED.
- [P19] Llorente, Michaely, Saar & Wang (2002), RFS 15(4); NBER w8312 — VERIFIED (T10 [A9]).
- [P20] Gervais, Kaniel & Mingelgrin (2001), JF 56(3) — VERIFIED (T10 [A7]).
- [P21] Kaniel, Ozoguz & Starks (2012), JFE 103(2) — VERIFIED.
- [P22] Avramov, Chordia & Goyal (2006), JF 61(5) — V(frag) (T10 [A3]).
- [P23] La Anh Thao (2012), luận văn MBA HCMIU — VERIFIED (paraphrase).
- UNVERIFIED/không dùng: Huang, Fu & Ke (2001) — PAYWALL; Nguyen Thi Thuy Linh (2013) — không mở được.

**Hành vi (H)** — scout C

- [H1] Tversky & Kahneman (1974), Science 185:1124 — VERIFIED (abstract qua OpenAlex).
- [H2] George & Hwang (2004), JF 59(5):2145 — VERIFIED (T10 [A16] chỉ PAYWALL; nay abstract đã mở).
- [H3] Birru (2015), Ohio State WP, ideas.repec.org/p/ecl/ohidic/2014-03.html — VERIFIED.
- [H4] Baker, Pan & Wurgler (2012), JFE 106:49 — VERIFIED (PDF trang 1).
- [H5] Li & Yu (2012), JFE 104:401 — VERIFIED.
- [H6] Heath, Huddart & Lang (1999), QJE 114:601 — V(frag).
- [H7] Kumar (2009), JF 64(4):1889 — VERIFIED (định nghĩa "lottery-type" chưa kiểm).
- [H8] Birru & Wang (2016), JFE 119(3):578 — V(frag).
- [H9] Bali, Cakici & Whitelaw (2011), JFE 99(2):427 — V(frag).
- [H10] Kaniel, Saar & Titman (2008), JF 63(1):273 — VERIFIED (phản bằng chứng).
- [H11] Grinblatt & Keloharju (2000), JFE 55:43 — VERIFIED.
- [H12] Grinblatt & Keloharju (2001), JF 56(2):589 — VERIFIED.
- [H13] Barber & Odean (2008), RFS 21(2):785 — VERIFIED (T10 [B3]).
- [H14] Barber, Huang, Odean & Schwarz (2022), JF 77(6) — VERIFIED (T10 [B4]).
- [H15] Xu, Liu, Hu & Guo (2022), British Accounting Review 54(6) — V(frag).
- [H16] Odean (1998), JF 53(5):1775 — VERIFIED (T10 [B2]).
- [H17] Strahilevitz, Odean & Barber (2011), J. Marketing Research 48 — V(frag).
- [H18] Shefrin & Statman (1985), JF 40(3) — PARTIAL.
- [H19] Barberis & Xiong (2012), JFE 104(2):251 — VERIFIED.
- [H20] Thaler & Johnson (1990), Management Science 36(6):643 — VERIFIED.
- [H21] Coval & Shumway (2005), JF 60(1):1 — VERIFIED (T10 [B5]).
- [H22] Imas (2016), AER 106(8):2086, aeaweb.org/articles?id=10.1257/aer.20140386 — VERIFIED.
- [H23] Barber & Odean (2000), JF 55(2):773 — VERIFIED (T10 [B1]).
- [H24] Barber & Odean (2001), QJE 116(1):261 — VERIFIED.
- [H25] Daniel, Hirshleifer & Subrahmanyam (1998), JF 53(6):1839 — VERIFIED.
- [H26] Rabin (2002), "Inference by Believers in the Law of Small Numbers", QJE 117(3):775 — V(frag). Đề bài ghi
  sai tên ("Inside the law of small numbers").
- [H27] Chen, Moskowitz & Shue (2016), NBER w22026 — V(frag).
- [H28] Greenwood & Shleifer (2014), RFS 27(3):714, NBER w18686 — V(frag).
- [H29] Mussweiler, Strack & Pfeiffer (2000), Personality and Social Psychology Bulletin — VERIFIED.
- [H30] Lord, Lepper & Preston (1984), JPSP 47(6):1231 — PARTIAL.
- [H31] Gollwitzer & Sheeran (2006), Advances in Experimental Social Psychology 38:69 — PARTIAL (d = 0,65 qua tóm
  tắt).
- [H32] Klein (2007), "Performing a Project Premortem", HBR — PARTIAL (paywall teaser).
- [H33] Mitchell, Russo & Pennington (1989), J. Behavioral Decision Making 2(1):25 — V(frag); con số "+30%" hay
  được trích là UNVERIFIED.
- [H34] Haynes et al. (2009), NEJM — V(frag) (T10 [B15]).
- [H35] Vo & Phan (2017), JBEF 13:33 — V(frag) (T10 [B17]).
- [H36] Nguyen & Vo (2024), J. Economic Studies 51(2):357 — VERIFIED.
- [H37] Nguyen Duc Hien et al. (2012), J. Economics and Development 14(2) — V(frag) (T10 [B19]).
- [H38] Trần Thị Hải Lý & Huỳnh Ngọc Thương (2013), J. Economic Development — V(frag) (T10 [B20]).
- [H39] Haas (Berkeley) 14/07/2011, "Professor Terrance Odean: pride and regret…" — PARTIAL (bài báo gốc chưa xác
  định).
- UNVERIFIED/không dùng: Andreassen (1988) (không tìm thấy); Gollwitzer (1999) — PAYWALL; preprint "The
  Averaging-Down Trap" (2026, không bình duyệt).

**Repo (development data)**

- memory: `crash-response-by-shape`, `swing-rsi2-edge`, `strategy-edge-findings`,
  `win-probability-metalabeling`.
- T10-doc và builder spec T10; `stock_agent/swing_desk/{gate,behaviour,sizing,indicators,service,scorer}.py`;
  `configs/swing_desk/{desk_v1,prospective_v1}.json`.
- Scratchpad (không commit): `bf_events.py`, `bf_clusters.py`, `bf_vn30.py`, `bf_events.csv`, `PNJ_vci*.csv`.
