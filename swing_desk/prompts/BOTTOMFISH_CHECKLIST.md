# Checklist bắt buộc trước lệnh bắt đáy (trả lời bằng chữ trong THESIS)

Nguồn: `docs/audits/2026-10-06-bottom-fishing-research.md` §9.2. `check --mode bottomfish` in đường dẫn
file này; lệnh bắt đáy chỉ PASS khi dòng `THESIS` có `bf_type=`, `profile=`, `invalid=`, `time_stop=`,
`max_attempts=` và `checklist=done` (BF-G12).

Chỉ tư vấn; bắt đáy chưa có bằng chứng edge tiến cứu; loại (b) có bằng chứng trôi giảm.

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

Ví dụ dòng THESIS (dữ liệu giả):

```text
2026-11-03|THESIS|HPG|||||||T020|bf_type=a; profile=P1; invalid=close<24100; time_stop=2026-11-17; max_attempts=1; target=26500; checklist=done
```
