@echo off
setlocal
rem Daily EOD: collect verified foreign/prop flows, then isolated PAPER run.
rem Registered in Windows Task Scheduler as "StockAgent_EOD_Update" (17:05 Mon-Fri).
rem No broker orders, production model replacement, legacy ledger writes, or Git publishing.
rem VN30_PYTHON may override the existing local Anaconda interpreter.
cd /d "%~dp0" || exit /b 2
if not exist "data\pipeline" mkdir "data\pipeline"
if not exist "data\pipeline" exit /b 2
if not defined VN30_PYTHON set "VN30_PYTHON=C:\Users\acer\anaconda3\python.exe"

echo ===== EOD PAPER run %date% %time% ===== >> "data\pipeline\eod_update.log" 2>&1
"%VN30_PYTHON%" -m stock_agent.pipeline.foreign_refresh --retry-stale >> "data\pipeline\eod_update.log" 2>&1
set "VN30_FLOW_EXIT=%errorlevel%"
"%VN30_PYTHON%" -m stock_agent.pipeline.paper_runner --refresh --run >> "data\pipeline\eod_update.log" 2>&1
set "VN30_RUN_EXIT=%errorlevel%"
if not "%VN30_RUN_EXIT%"=="0" goto failed
if not "%VN30_FLOW_EXIT%"=="0" goto flow_failed

echo ===== EOD PAPER SUCCESS %date% %time% - local only, not published ===== >> "data\pipeline\eod_update.log" 2>&1
exit /b 0

:failed
echo ===== EOD PAPER FAILED exit=%VN30_RUN_EXIT% %date% %time% - see runner readiness report ===== >> "data\pipeline\eod_update.log" 2>&1
exit /b %VN30_RUN_EXIT%

:flow_failed
echo ===== EOD FLOW FAILED exit=%VN30_FLOW_EXIT% %date% %time% - paper job completed; inspect data\foreign_flows_v2\latest_status.json ===== >> "data\pipeline\eod_update.log" 2>&1
exit /b %VN30_FLOW_EXIT%
