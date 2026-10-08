# Nguồn tin cho agent (analyst, bf-news, sự kiện)

Dùng cùng `SENTIMENT_RUBRIC.md`. Mọi nguồn phải **mở được thật** (WebFetch) mới được trích; nguồn không mở được
là UNVERIFIED, trọng số 0. Không dùng trí nhớ của model thay cho nguồn. Ghi URL, thời điểm đăng (giờ VN), thời
điểm truy cập và trích nguyên văn ≤ 2 câu.

## Bậc 1 — Nguồn chính thức (FACT_PRIMARY, trọng số 1.0)

| Nguồn | Tên miền | Dùng cho |
|---|---|---|
| Ủy ban Chứng khoán Nhà nước (SSC) | `ssc.gov.vn` | văn bản pháp luật, quyết định xử phạt, chấp thuận phát hành/niêm yết, thống kê thị trường |
| Sở GDCK TP.HCM (HOSE) | `hsx.vn`, `hose.vn` | công bố thông tin, danh sách cắt margin, thay đổi rổ chỉ số (VN30/VN100), cảnh báo/kiểm soát |
| Sở GDCK Hà Nội (HNX) | `hnx.vn` | công bố thông tin HNX/UPCoM, trái phiếu doanh nghiệp |
| Tổng công ty Lưu ký (VSDC) | `vsd.vn`, `vsdc.com.vn` | ngày đăng ký cuối cùng, chốt quyền, đăng ký/bổ sung chứng khoán |
| Website IR của doanh nghiệp | theo `configs/swing_desk/company_ir_domains.json` | nghị quyết HĐQT/ĐHĐCĐ, BCTC, tài liệu họp, giải trình biến động giá |
| Ngân hàng Nhà nước, Bộ Tài chính, cơ sở dữ liệu văn bản (vbpl) | `sbv.gov.vn`, `mof.gov.vn`, `vbpl.vn` | lãi suất điều hành, tỷ giá, chính sách tài khóa, văn bản gốc |

## Bậc 2 — Báo tài chính uy tín (FACT_REPORTED, trọng số 0.7)

Chỉ phần **tường thuật sự kiện** là FACT_REPORTED; nhận định, dự báo, khuyến nghị trong cùng bài là OPINION (0.3).

| Nguồn | Tên miền | Mạnh ở | Lưu ý khi lấy dữ liệu |
|---|---|---|---|
| CafeF | `cafef.vn` | tin nhanh doanh nghiệp, vĩ mô, giao dịch nội bộ, dữ liệu lịch sử | trang bài viết đọc được; một số trang dữ liệu (`/du-lieu/...`) tải bằng JavaScript, có thể trống |
| Vietstock | `vietstock.vn`, `finance.vietstock.vn` | dữ liệu 24/7, lịch sự kiện, tin thị trường hằng ngày, báo cáo phân tích | bài trên `vietstock.vn` đọc được; lịch sự kiện / giao dịch nội bộ trên `finance.vietstock.vn` cần đăng nhập/JS → nếu không đọc được thì ghi UNVERIFIED. Báo cáo phân tích = OPINION |
| Tin Nhanh Chứng Khoán (Báo Đầu Tư Chứng Khoán) | `tinnhanhchungkhoan.vn` | nhận định thị trường, thông tin doanh nghiệp, danh sách margin | nhận định = OPINION |
| VnEconomy (Thời báo Kinh tế Việt Nam) | `vneconomy.vn` | chính sách vĩ mô, diễn biến thị trường | |
| Báo Đầu Tư (Kinh doanh – Chứng khoán) | `baodautu.vn` | dự án lớn, khối ngoại, chuyển động thị trường niêm yết | |

Nguồn báo khác đã gặp và dùng được như bậc 2 khi tường thuật sự kiện: VnExpress (`vnexpress.net`), Tuổi Trẻ
(`tuoitre.vn`), Dân Trí (`dantri.com.vn`), Tạp chí Thị trường Tài chính Tiền tệ (`thitruongtaichinhtiente.vn`),
Tạp chí Kinh tế Tài chính (`tapchikinhtetaichinh.vn`), Market Times (`markettimes.vn`), DNSE Senses (`dnse.com.vn`).

## Bậc 3 — Ý kiến và diễn đàn (OPINION 0.3 / RUMOR 0)

Báo cáo phân tích CTCK, bài nhận định, F247, nhóm Zalo/Facebook: chỉ là ý kiến người chơi, **không bao giờ là bằng
chứng**; dùng để phát hiện hype/FOMO (cờ hành vi), không dùng để nâng verdict.

## Cách tìm (thứ tự khuyến nghị)

1. **Sự kiện doanh nghiệp** (ĐHĐCĐ, phát hành, chốt quyền, KQKD, giao dịch nội bộ): website IR doanh nghiệp →
   HOSE/VSDC → CafeF/Vietstock (bài viết) để đối chiếu ngày.
2. **Cú giảm mạnh / tin xấu**: tìm theo mã + "giải trình", "giảm sàn", "nghị quyết", "trích lập", "phát hành"
   trong 20 phiên gần nhất trên CafeF, Vietstock, Tin Nhanh Chứng Khoán, VnEconomy, Báo Đầu Tư; mở văn bản gốc
   nếu bài báo dẫn nguồn.
3. **Margin / cảnh báo**: danh sách không đủ điều kiện ký quỹ của HOSE theo quý (HOSE hoặc bài CafeF/Tin Nhanh
   Chứng Khoán trích lại) — ghi ngày kiểm.
4. **Vĩ mô / chính sách**: SBV, Bộ Tài chính, SSC, vbpl; VnEconomy/Báo Đầu Tư để lấy bối cảnh.
5. **Bẫy năm**: công cụ tìm kiếm hay trộn bài cũ (2023–2025) với bài mới — chỉ dùng ngày trên trang đã mở.
6. `none_found` cần ít nhất 2 nguồn đã mở, trong đó 1 nguồn bậc 1 (sàn/SSC/VSDC hoặc website IR trong
   `company_ir_domains.json`).
