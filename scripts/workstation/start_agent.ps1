# start_agent.ps1 - start the workstation agent hidden, in the background (2026-11-27). Used by the Windows logon task (see campaigns/README.md) so the agent comes back after a power cut.
# A second start is harmless: the agent holds a lock on its work folder and a second copy exits at once. It only runs shell runs while K allows them on the /shell page.
param([string]$Campaign = "shell2")
$py = Join-Path $env:USERPROFILE "auxetic_ws_venv\Scripts\python.exe"
$ws = if ($env:AUXETIC_WS_HOME) { $env:AUXETIC_WS_HOME } else { Join-Path $env:USERPROFILE "auxetic_ws" }          # D:\auxetic_ws since 2026-11-28
$agent = Join-Path (Split-Path -Parent $MyInvocation.MyCommand.Path) "ws_agent.py"
New-Item -ItemType Directory -Force $ws | Out-Null
$stamp = Get-Date -Format "yyyyMMdd-HHmmss"
Start-Process -FilePath $py -ArgumentList @('-u', $agent, '--campaign', $Campaign, '--python', $py, '--work', $ws, '--poll-s', '60') -WindowStyle Hidden `
  -RedirectStandardOutput (Join-Path $ws "agent_$stamp.log") -RedirectStandardError (Join-Path $ws "agent_$stamp.err")
