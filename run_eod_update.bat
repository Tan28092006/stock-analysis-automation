@echo off
rem Daily EOD update: prices + foreign flows + room snapshot + MR/momentum scan caches.
rem Registered in Windows Task Scheduler as "StockAgent_EOD_Update" (17:05 Mon-Fri).
rem After updating, it publishes the fresh data to GitHub so a free Render deploy
rem auto-redeploys with today's data (no paid persistent disk needed).
cd /d D:\Chungkhoan

echo ===== EOD run %date% %time% ===== >> data\pipeline\eod_update.log 2>&1
C:\Users\acer\anaconda3\python.exe -m stock_agent.pipeline.eod_update >> data\pipeline\eod_update.log 2>&1

rem --- Publish data to GitHub (only data paths are staged, not unrelated changes) ---
git add data/raw/prices_hist data/raw/foreign data/pipeline/momentum_scan_cache.json data/pipeline/mr_scan_cache.json data/pipeline/forward_test.jsonl data/pipeline/daily_runs.jsonl data/pipeline/pending_predictions.jsonl data/models/win_prob_mr.pkl >> data\pipeline\eod_update.log 2>&1
git commit -m "data: EOD update %date%" >> data\pipeline\eod_update.log 2>&1
git push >> data\pipeline\eod_update.log 2>&1
echo ===== done %date% %time% ===== >> data\pipeline\eod_update.log 2>&1
