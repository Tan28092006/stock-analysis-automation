# Thin wrapper: forwards arguments only. Advisory desk; never places orders or changes schedules.
$repo = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
Push-Location $repo
try { python -m stock_agent.swing_desk refresh @args; exit $LASTEXITCODE } finally { Pop-Location }
