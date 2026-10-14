@echo off
rem Thin wrapper: forwards arguments only. Advisory desk; never places orders or changes schedules.
cd /d "%~dp0..\.."
python -m stock_agent.swing_desk refresh %*
