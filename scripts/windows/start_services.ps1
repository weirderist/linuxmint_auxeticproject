# start_services.ps1 - start the project's background services on the WORKSTATION (Windows), hidden, each only if it is not already running (2026-11-27).
# It replaces, on Windows, what systemd (auxetic_webui.service) and the @reboot lines of deploy/crontab_shell.txt do on the server. Safe to run again: every service has its own
# single-instance guard (port in use, runner lock, controller lock, agent lock). Logs: <repo>\logs\win_<service>_<time>.log
#   powershell -NoProfile -ExecutionPolicy Bypass -File scripts\windows\start_services.ps1 [-Campaign shell2] [-NoWeb] [-NoKaggle] [-NoAgent] [-Hub local|server] [-AnalysisCampaigns shell1,shell2]
# What it starts:
#   web service     uvicorn main:app on 127.0.0.1:8080 from webui\ (the same app as the server; view it here, or from the phone through Tailscale, see CLAUDE.md)
#   Kaggle          scripts\kaggle\kaggle_controller.py <Campaign>: pushes sessions only while the Kaggle box on /shell says on
#   workstation     scripts\workstation\ws_agent.py --host <Hub>: runs shells only while the Workstation box on /shell allows it (Hub = local: queue and results on THIS machine)
#   analysis        scripts\shell_learning\shell_analysis_job.py <c> for each campaign in -AnalysisCampaigns (rules and proposer evaluation every 100 runs; nothing if omitted)
param([string]$Campaign = "shell2", [switch]$NoWeb, [switch]$NoKaggle, [switch]$NoAgent, [string]$Hub = "local", [string]$AnalysisCampaigns = "")
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent (Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path))
$py = Join-Path $root "venv\Scripts\python.exe"
if (-not (Test-Path $py)) { throw "no project venv: run  python -m venv venv  and  venv\Scripts\pip install -r requirements.txt  in $root" }
$logs = Join-Path $root "logs"; New-Item -ItemType Directory -Force $logs | Out-Null
$stamp = Get-Date -Format "yyyyMMdd-HHmmss"
function Start-Hidden($name, $argList, $workdir) {
  $out = Join-Path $logs "win_${name}_$stamp.log"; $err = Join-Path $logs "win_${name}_$stamp.err"
  $p = Start-Process -FilePath $py -ArgumentList $argList -WorkingDirectory $workdir -WindowStyle Hidden -RedirectStandardOutput $out -RedirectStandardError $err -PassThru
  "{0,-10} started, pid {1}, log {2}" -f $name, $p.Id, $out
}
if (-not $NoWeb) {
  $busy = Get-NetTCPConnection -LocalPort 8080 -State Listen -ErrorAction SilentlyContinue
  if ($busy) { "web        already listening on port 8080 (pid $($busy[0].OwningProcess))" }
  else { Start-Hidden "web" @("-m", "uvicorn", "main:app", "--host", "127.0.0.1", "--port", "8080") (Join-Path $root "webui") }
}
if (-not $NoKaggle) { Start-Hidden "kaggle" @("-u", (Join-Path $root "scripts\kaggle\kaggle_controller.py"), $Campaign) (Join-Path $root "scripts") }
if (-not $NoAgent) { Start-Hidden "agent" @("-u", (Join-Path $root "scripts\workstation\ws_agent.py"), "--campaign", $Campaign, "--python", $py, "--host", $Hub, "--remote-root", $root) $root }
foreach ($c in ($AnalysisCampaigns -split "," | Where-Object { $_ })) { Start-Hidden "analysis_$c" @("-u", (Join-Path $root "scripts\shell_learning\shell_analysis_job.py"), $c, "--poll-s", "600") (Join-Path $root "scripts") }
