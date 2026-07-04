@echo off
rem Daily EOD update: prices + foreign flows + room snapshot + MR scan cache.
rem Registered in Windows Task Scheduler as "StockAgent_EOD_Update" (17:05 Mon-Fri).
cd /d D:\Chungkhoan
C:\Users\acer\anaconda3\python.exe -m stock_agent.pipeline.eod_update >> data\pipeline\eod_update.log 2>&1
