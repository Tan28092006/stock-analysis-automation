# 🚀 Deploy — VN Swing-Trading Dashboard

The app is a single Python service: the stdlib `http.server` serving the dashboard
(`web/index.html`) + a JSON API (`stock_agent/app.py`). No database, no external
services required. Data lives on disk under `data/`.

> [!IMPORTANT]
> This is a **personal research tool with no authentication**. Anyone who can reach the
> port can see your positions and trigger scans/updates. Do **not** expose it to the open
> internet without putting auth in front of it (reverse-proxy basic-auth, a VPN, Cloudflare
> Access, or Tailscale). See [§5 Security](#5-security).

---

## 1. What runs, and what it needs

| Component | Command | Cadence |
|---|---|---|
| **Dashboard + API** | `python -m stock_agent.app` | long-running |
| **EOD data update** | `python -m stock_agent.pipeline.eod_update` | daily 17:05 T2–T6 |
| **(optional) P(win) model** | `python -c "from stock_agent.features.win_probability import train_and_save; print(train_and_save())"` | once / occasionally |

The dashboard **reads** the price history and caches that the EOD job **writes**. They
share the `data/` directory — keep it on a persistent volume so updates survive restarts.

Runtime deps: `libgomp1` (for LightGBM/XGBoost) — already handled in the `Dockerfile`.

---

## 2. Local run (no Docker)

```bash
pip install -r requirements.txt

# One-time (or daily): pull VN100 EOD prices + build caches. Needs internet (vnstock).
python -m stock_agent.pipeline.eod_update

# Optional: train the win-probability model (dashboard degrades gracefully without it).
python -c "from stock_agent.features.win_probability import train_and_save; print(train_and_save())"

# Serve. Use 0.0.0.0 to accept LAN connections; 127.0.0.1 for local-only.
python -m stock_agent.app --host 127.0.0.1 --port 8000
```

Open http://127.0.0.1:8000. Host/port also read from `HOST` / `PORT` env vars.

---

## 3. Docker

```bash
# Build + run with data persisted to ./data on the host.
docker compose up -d --build          # -> http://localhost:8000
docker compose logs -f dashboard

# Run an EOD update on demand (writes into the shared ./data volume):
docker compose run --rm eod
```

Or plain Docker:

```bash
docker build -t vn-swing-trading .
docker run -d --name vn-swing -p 8000:8000 \
  -e HOST=0.0.0.0 -e PORT=8000 \
  -v "$(pwd)/data:/app/data" \
  vn-swing-trading
```

The image bundles `data/raw/prices_hist/` (~8.5 MB) so it runs out of the box. The
`data/` volume overlays it on the host so EOD updates accumulate there.

---

## 4. PaaS (Render / Railway / Fly.io)

The app honours the platform-injected `$PORT` and binds `0.0.0.0` via `HOST` — no code
changes needed.

- **Render**: New → Web Service → from repo. Runtime **Docker** (uses the `Dockerfile`).
  Add a **Persistent Disk** mounted at `/app/data` (else scans/positions reset on each
  deploy). Health check path: `/api/health`.
- **Railway**: deploy from repo (Dockerfile auto-detected). Add a **Volume** at
  `/app/data`. `PORT` is injected automatically.
- **Fly.io**: `fly launch` (detects Dockerfile) → attach a **fly volume** mounted at
  `/app/data`; set `[[services]] internal_port = 8000` and `[env] HOST = "0.0.0.0"`.

> Free/ephemeral filesystems lose `data/` on restart. Without a persistent disk the app
> still boots (bundled prices), but tracked positions and fresh EOD prices won't persist.

### Scheduling the EOD update on PaaS
The dashboard container does **not** self-schedule. Options:
- A platform **Cron Job** (Render Cron / Railway Cron / `fly machine run … --schedule`)
  running `python -m stock_agent.pipeline.eod_update` **against the same volume**, at
  10:05 UTC (≈17:05 ICT), Mon–Fri.
- Or keep the EOD job on a machine at home (§6) and only host the read-only dashboard.

---

## 5. Security

No auth is built in. Before any public exposure, put a gate in front:

- **Reverse proxy + basic auth** (Caddy example):
  ```
  dash.example.com {
      basicauth { youruser JDJhJDE0... }   # bcrypt hash via `caddy hash-password`
      reverse_proxy 127.0.0.1:8000
  }
  ```
- Or a **private network**: Tailscale / WireGuard / Cloudflare Access — simplest for a
  personal tool.
- The write endpoints (`/api/data/update`, `/api/*/positions`, `/api/model/train`,
  `/api/optimize`, `/api/chat`) mutate state or spend compute — never leave them open.

**Secrets:** the only secret is `NVIDIA_API_KEY` (optional — powers the `/api/chat`
summary only). Set it as an env var / platform secret. Never commit `.env`; copy
`.env.example` → `.env`. The `data/raw/foreign/vietstock_session.json` auth cookie is
gitignored and `.dockerignore`d — keep it that way.

---

## 6. Scheduling the EOD job (self-hosted)

**Windows** (current setup) — `run_eod_update.bat` registered in Task Scheduler as
`StockAgent_EOD_Update`, 17:05 Mon–Fri. Re-create with:
```powershell
schtasks /Create /TN StockAgent_EOD_Update /TR "D:\Chungkhoan\run_eod_update.bat" ^
  /SC WEEKLY /D MON,TUE,WED,THU,FRI /ST 17:05 /F
```

**Linux (cron)** — 17:05 ICT is 10:05 UTC:
```cron
5 10 * * 1-5  cd /app && /usr/local/bin/python -m stock_agent.pipeline.eod_update >> data/pipeline/eod_update.log 2>&1
```

The EOD job is resilient to vnstock's guest rate-limit (20 req/min): it catches the
rate-limit `SystemExit`, retries up to 4×, and backs off 20 s.

---

## 7. Post-deploy checklist

- [ ] `GET /api/health` returns 200.
- [ ] Dashboard loads; regime banner + both engine panels render.
- [ ] `data/` is on a **persistent** volume (positions survive a restart).
- [ ] EOD schedule points at the **same** `data/` the dashboard reads.
- [ ] Port is **not** publicly reachable without auth (§5).
- [ ] `.env` / `vietstock_session.json` are **not** in the image or repo.
- [ ] `python -m pytest tests/ -q` passes (85 tests).
