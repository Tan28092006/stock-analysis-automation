# 🆓 Deploy miễn phí trên Render (không mất phí, không cần đĩa trả phí)

Ý tưởng: **không lưu dữ liệu trên server**. Server free của Render chỉ *phục vụ* — dữ
liệu giá sống trong **GitHub repo** (đã bake sẵn vào image lúc build). Máy bạn (job EOD
17:05 đang chạy) fetch giá mới rồi `git push` → Render **tự redeploy** với data mới.

```
Máy bạn 17:05 (VN)              GitHub repo              Render FREE (24/7)
  fetch vnstock  ──git push──►   lưu CSV mới  ──auto──►  rebuild image → serve
```

- ✅ $0/tháng, URL công khai, chạy 24/7 kể cả khi tắt máy.
- ✅ Data tươi mỗi phiên (máy bạn ở VN nên vnstock không bị chặn/limit như trên cloud).
- ⚠️ Free tier **ngủ** sau ~15 phút không dùng (mở lại chờ ~30–60s) và **không có ổ bền**.
- ⚠️ Vị thế ghi tay được lưu ở **trình duyệt** (localStorage) nên vẫn còn sau mỗi redeploy;
  giá/scan thì luôn tươi từ GitHub.

---

## 1. Tạo Web Service (một lần)

1. [render.com](https://render.com) → đăng nhập bằng **GitHub**.
2. **New → Web Service** → chọn repo `Tan28092006/stock-analysis-automation`.
3. **Instance Type: Free**.
4. Runtime **Docker** (tự nhận `Dockerfile`) — **không** gắn Disk.
5. **Environment** → thêm biến:
   - `APP_PASSWORD` = *(mật khẩu dài, ngẫu nhiên)* — bắt buộc, để đăng nhập.
   - `APP_USERNAME` = `hao` *(tùy chọn, mặc định `admin`)*.
   - `NVIDIA_API_KEY` = *(tùy chọn, chỉ cho nút chat)*.
6. **Advanced → Health Check Path**: `/api/health`.
7. Bật **Auto-Deploy** (mặc định On) — đây là cái để mỗi lần push data thì Render tự build lại.
8. **Create Web Service** → build lần đầu ~5–10 phút. Xong mở `https://<tên-app>.onrender.com`
   → popup login → vào dashboard. 🎉

> Layer Docker được cache, nên các lần redeploy sau (chỉ đổi data) build lại **nhanh**.

---

## 2. Bật tự cập nhật data hằng ngày

Việc này **đã tự động** nếu Task Scheduler của bạn đang chạy `run_eod_update.bat` lúc 17:05.
File bat giờ có thêm bước `git push` sau khi cập nhật giá → Render thấy commit mới → tự redeploy.

**Chỉ cần kiểm tra 1 lần rằng scheduled task push được lên GitHub:**

```powershell
# Chạy thử thủ công, xem cuối log có dòng push thành công không
D:\Chungkhoan\run_eod_update.bat
Get-Content D:\Chungkhoan\data\pipeline\eod_update.log -Tail 20
```

Nếu `git push` báo lỗi xác thực: mở GitHub Desktop hoặc chạy `git push` một lần trong
terminal để lưu credential (Windows Credential Manager), rồi task sẽ push được.

> Không muốn dùng máy để refresh? Có thể dùng GitHub Actions
> ([.github/workflows/eod-update.yml](../.github/workflows/eod-update.yml)) để POST vào
> `/api/data/update` — nhưng cách đó ghi vào ổ tạm của Render nên **mất khi redeploy**;
> cách push-data-từ-máy ở trên mới giữ được data. Trên free tier hãy dùng cách push từ máy.

---

## 3. Giới hạn của bản free (biết trước cho đỡ bất ngờ)

| Vấn đề | Thực tế | Cách sống chung |
|---|---|---|
| Web **ngủ** sau 15' | Lần mở đầu chờ ~30–60s | Chấp nhận, hoặc ping định kỳ (UptimeRobot free) |
| Không ổ bền | File ghi lúc chạy mất khi redeploy | Data lấy từ GitHub; vị thế ở localStorage → không sao |
| Redeploy hằng ngày | ~5–10' build (cache nên nhanh) | Xảy ra sau 17:05, lúc bạn không dùng |
| Build 750h free/tháng | Đủ cho 1 service chạy liên tục | Đừng tạo nhiều service free |

Muốn **không ngủ + có ổ bền** thì nâng lên Starter (~$7/tháng) + Disk 1GB —
xem [DEPLOY.md](DEPLOY.md). Còn muốn **hoàn toàn free mà không ngủ** thì cân nhắc
Oracle Cloud Always Free VM (tự chạy `docker compose`, xem DEPLOY.md §3).
