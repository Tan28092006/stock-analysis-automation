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
| **Scheduled EOD job** (`run_eod_update.bat`, task `StockAgent_EOD_Update`) | `python -m stock_agent.pipeline.foreign_refresh --retry-stale`, then `python -m stock_agent.pipeline.paper_runner --refresh --run` | 17:05 T2–T6 (Windows Task Scheduler) |
| **Legacy dashboard price refresh** | `python -m stock_agent.pipeline.eod_update` | manual only — not scheduled |
| **(optional) P(win) model** | `python -c "from stock_agent.features.win_probability import train_and_save; print(train_and_save())"` | once / occasionally |

What the 17:05 job writes (local only, no Git publishing since commit `7d788cb`, 22/09/2026):

- `data/foreign_flows_v2/` — verified foreign/proprietary flow runs (see [FOREIGN-FLOWS.md](FOREIGN-FLOWS.md));
- `data/paper/snapshots/<UTC>/` — a new source-verified VCI price snapshot (VN30 + VNINDEX);
- `data/paper/runs/<session>/paper.json`, `data/paper/latest.json`, `data/paper/scores/latest.json`;
- `data/pipeline/eod_update.log` — appended by the BAT.

What it does **not** write: `data/raw/prices_hist/*.csv`, the dashboard scan caches
(`data/pipeline/{mr,momentum,swing}_scan_cache.json`) and `data/pipeline/forward_test.jsonl`.
Those are written only by the legacy `eod_update` (manual CLI) or the dashboard's
`POST /api/data/update`. Unless someone runs that by hand, `prices_hist` and the dashboard
stay **stale** (they stopped at 21/09/2026 when the push was removed). Re-enabling a
scheduled `eod_update` is a separate user decision; do not add it to the BAT silently.

The dashboard **reads** `prices_hist` and those caches. They share the `data/` directory —
keep it on a persistent volume so manual updates survive restarts.

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

- **Render** (recommended for beginners): New → **Web Service** → connect the GitHub repo.
  1. Runtime **Docker** (auto-detects the `Dockerfile`).
  2. Add a **Persistent Disk**, mount path `/app/data`, size **1 GB** (data is ~12 MB and
     grows a few MB/year — 1 GB lasts effectively forever). Without it, scans/positions
     reset on every deploy.
  3. Environment: set `APP_PASSWORD` (+ optional `APP_USERNAME`) as secrets; optionally
     `NVIDIA_API_KEY`. `PORT`/`HOST` are handled automatically.
  4. Health check path: `/api/health`.
  5. **Daily EOD refresh (paid disk).** A Render disk can attach to only ONE service, so a
     separate cron job cannot mount `/app/data`. Trigger the update over HTTP — the web
     service (which owns the disk) runs it in a background thread — via a Render **Cron Job**
     (schedule `5 10 * * 1-5`) running
     `curl -fsS -X POST -u "$APP_USERNAME:$APP_PASSWORD" https://<your-app>.onrender.com/api/data/update`.
     > ⚠️ This HTTP-trigger path (including the `.github/workflows/eod-update.yml` workflow)
     > is **paid-disk only**. On the **free** tier it does NOT work: the fetched data lands
     > on ephemeral storage (lost on redeploy) and the fetch runs from Render's IP which
     > vnstock blocks. The local `run_eod_update.bat` **no longer** runs `eod_update` or
     > `git push` (removed in commit `7d788cb`, 22/09/2026), so a free-tier deploy receives
     > no new prices unless someone runs `eod_update` and pushes by hand — see
     > [RENDER-FREE.md](RENDER-FREE.md).
  Note: Render's *free* web service sleeps after inactivity and does not include a
  persistent disk — a paid instance (or the cheapest paid tier) is needed for the disk +
  always-on behaviour.
- **Railway**: deploy from repo (Dockerfile auto-detected). Add a **Volume** at
  `/app/data`. `PORT` is injected automatically.
- **Fly.io**: `fly launch` (detects Dockerfile) → attach a **fly volume** mounted at
  `/app/data`; set `[[services]] internal_port = 8000` and `[env] HOST = "0.0.0.0"`.

> Free/ephemeral filesystems lose `data/` on restart. Without a persistent disk the app
> still boots (bundled prices), but tracked positions and fresh EOD prices won't persist.

### Scheduling the legacy dashboard refresh on PaaS
The dashboard container does **not** self-schedule. This is the legacy dashboard price
refresh, separate from the local paper/flow job in §6. Options:
- A platform **Cron Job** (Render Cron / Railway Cron / `fly machine run … --schedule`)
  running `python -m stock_agent.pipeline.eod_update` **against the same volume**, at
  10:05 UTC (≈17:05 ICT), Mon–Fri. Nothing like this is configured today.
- Or keep the EOD job on a machine at home (§6) and only host the read-only dashboard —
  remembering that the home job no longer refreshes `prices_hist` or the dashboard caches.

---

## 5. Security

**Built-in login (HTTP Basic auth).** Set `APP_PASSWORD` (and optionally `APP_USERNAME`,
default `admin`) as env vars and every request requires that user/password. The browser
shows a native login popup; `/api/health` stays open so platform health checks work.

```bash
APP_USERNAME=hao APP_PASSWORD='a-long-random-password' python -m stock_agent.app --host 0.0.0.0
```

- When `APP_PASSWORD` is **empty the gate is OFF** (fine for `127.0.0.1` local dev). The
  server prints a warning if it's listening on a public interface without a password.
- On Render/Railway/Fly: add `APP_PASSWORD` (and `APP_USERNAME`) as a **secret env var**.
- Basic auth sends credentials base64-encoded, so it's only safe over **HTTPS** — all the
  PaaS options terminate TLS for you. On a raw VPS put it behind an HTTPS reverse proxy
  (Caddy/Traefik auto-TLS) rather than serving `:8000` in the clear.

For a stronger boundary you can still layer a private network (Tailscale / Cloudflare
Access) on top, but the built-in password is enough for a personal deployment.

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

The BAT runs exactly two steps and nothing else — no `eod_update`, no git push, no
broker call, no model retrain:

1. `python -m stock_agent.pipeline.foreign_refresh --retry-stale` — flow collector,
   bounded retries 60/300/900 s for late-published rows;
2. `python -m stock_agent.pipeline.paper_runner --refresh --run` — new VCI snapshot,
   readiness, MR + momentum paper record inside the 16:00 → next session 09:00 VN window,
   then scoring. It runs even if step 1 failed.

Exit code: the paper exit code wins if non-zero, otherwise the flow exit code. Exit 2 from
the paper step can mean "scoring blocked" even though `paper.json` **was** recorded — read
the log line `"status": "recorded"` rather than inferring one cause from the code.

**Linux (cron)** — 17:05 ICT is 10:05 UTC. Untested mirror of the BAT (no git push):
```cron
5 10 * * 1-5  cd /app && (python -m stock_agent.pipeline.foreign_refresh --retry-stale; python -m stock_agent.pipeline.paper_runner --refresh --run) >> data/pipeline/eod_update.log 2>&1
```

The legacy `eod_update` (manual only) is resilient to vnstock's guest rate-limit
(20 req/min): it catches the rate-limit `SystemExit`, retries up to 4×, and backs off 20 s.

### Daily health check (≈2 minutes, read-only)

Run from the repo root after 17:30 on a trading day. None of these commands fetch data,
record paper runs or change the schedule.

1. `Get-Content data\pipeline\eod_update.log -Tail 30` — the last block must end with
   `EOD PAPER SUCCESS`. `FAILED` / `FLOW FAILED` lines name the failing step; a block with
   no closing line means the job was interrupted (e.g. `^C`, sleep, power loss).
2. `python -m stock_agent.pipeline.foreign_refresh --status` — expect `"status": "ready"`
   for today's `expected_session`. A stale `.collect-lock` or a run dir without
   `manifest.json` keeps it `partial` every day until handled — follow the troubleshooting
   steps in [FOREIGN-FLOWS.md](FOREIGN-FLOWS.md).
3. Open `data/paper/latest.json` — `status` should be `recorded` or `already_recorded`
   with `session` = today. Exit 2 after a recorded run usually means scoring was blocked by
   a provider revision; check `data/paper/scores/latest.json`.
4. `python -m stock_agent.pipeline.paper_runner --status` (read-only) — prints
   `calendar_warning` (HOSE holiday table about to run out) and `missed_sessions`: trading
   sessions since the first record with no `runs/<session>/paper.json`. Entries under
   `missed` had their window close (09:00 VN next session) and are **lost**; never backfill
   them. `window_open_unrecorded` can still be recorded today. The same block is written to
   `data/paper/latest.json` on every run.
5. Remember: `prices_hist` and the dashboard are **not** part of this job; their dates
   reflect the last manual `eod_update`.

---

## 7. Post-deploy checklist

- [ ] `GET /api/health` returns 200.
- [ ] Dashboard loads; regime banner + both engine panels render.
- [ ] `data/` is on a **persistent** volume (positions survive a restart).
- [ ] EOD schedule points at the **same** `data/` the dashboard reads.
- [ ] Port is **not** publicly reachable without auth (§5).
- [ ] `.env` / `vietstock_session.json` are **not** in the image or repo.
- [ ] `python -m pytest tests/ -q` passes (97 tests).
