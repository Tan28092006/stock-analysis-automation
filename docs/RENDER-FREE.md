# 🆓 Deploy miễn phí trên Render (không mất phí, không cần đĩa trả phí)

Ý tưởng: **không lưu dữ liệu trên server**. Server free của Render chỉ *phục vụ* — dữ
liệu giá sống trong **GitHub repo** (đã bake sẵn vào image lúc build). Render chỉ có data
mới khi có một commit data mới được push lên GitHub.

> [!WARNING]
> **Trạng thái hiện tại (từ 22/09/2026, commit `7d788cb`):** job 17:05
> (`run_eod_update.bat`) **không còn** chạy `eod_update` và **không còn** `git push`.
> BAT chỉ chạy `foreign_refresh --retry-stale` và `paper_runner --refresh --run`, ghi
> local. Vì vậy `data/raw/prices_hist`, các scan cache của dashboard và bản trên Render
> free **đứng yên ở commit data cuối (21/09/2026)**. Muốn data mới phải chạy tay
> `python -m stock_agent.pipeline.eod_update` rồi tự commit/push; bật lại việc này theo
> lịch là quyết định riêng của người dùng, không tự thêm vào BAT.

```
Luồng cũ (đến 21/09/2026, ĐÃ BỎ):
Máy bạn 17:05 (VN)              GitHub repo              Render FREE (24/7)
  fetch vnstock  ──git push──►   lưu CSV mới  ──auto──►  rebuild image → serve

Hiện tại: chỉ khi người dùng tự chạy eod_update + git push thì Render mới rebuild.
```

- ✅ $0/tháng, URL công khai, chạy 24/7 kể cả khi tắt máy.
- ⚠️ Data chỉ tươi khi có người chạy `eod_update` và push bằng tay (xem cảnh báo trên).
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

## 2. Cập nhật data (thủ công)

Job 17:05 **không** cập nhật data cho Render. Từ commit `7d788cb` (22/09/2026) file
`run_eod_update.bat` chỉ chạy flow collector và paper runner, không chạy `eod_update`,
không `git add/commit/push`. Nếu muốn Render free có giá mới, người dùng tự làm:

```powershell
# 1) Cập nhật prices_hist + scan cache (legacy, chạy tay, cần mạng VN)
python -m stock_agent.pipeline.eod_update
# 2) Tự xem lại thay đổi rồi commit/push data nếu muốn (Render auto-deploy)
git status --short data/raw/prices_hist data/pipeline
```

Không chạy lại `run_eod_update.bat` cho việc này: BAT ghi paper record, không push.
Muốn bật lại tự động (`eod_update` theo lịch và/hoặc push) là một quyết định riêng của
người dùng; tài liệu này không bật lại.

> Không muốn dùng máy để refresh? Có thể dùng GitHub Actions
> ([.github/workflows/eod-update.yml](../.github/workflows/eod-update.yml)) để POST vào
> `/api/data/update` — nhưng cách đó ghi vào ổ tạm của Render nên **mất khi redeploy**.
> Trên free tier chỉ có cách push data từ máy mới giữ được data.

---

## 3. Giới hạn của bản free (biết trước cho đỡ bất ngờ)

| Vấn đề | Thực tế | Cách sống chung |
|---|---|---|
| Web **ngủ** sau 15' | Lần mở đầu chờ ~30–60s | Chấp nhận, hoặc ping định kỳ (UptimeRobot free) |
| Không ổ bền | File ghi lúc chạy mất khi redeploy | Data lấy từ GitHub; vị thế ở localStorage → không sao |
| Redeploy khi push data | ~5–10' build (cache nên nhanh) | Chỉ xảy ra khi có người push data bằng tay |
| Build 750h free/tháng | Đủ cho 1 service chạy liên tục | Đừng tạo nhiều service free |

Muốn **không ngủ + có ổ bền** thì nâng lên Starter (~$7/tháng) + Disk 1GB —
xem [DEPLOY.md](DEPLOY.md). Còn muốn **hoàn toàn free mà không ngủ** thì cân nhắc
Oracle Cloud Always Free VM (tự chạy `docker compose`, xem DEPLOY.md §3).
