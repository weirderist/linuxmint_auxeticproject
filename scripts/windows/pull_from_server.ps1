# pull_from_server.ps1 - copy the project's DATA from the server to the workstation, for the switch to the workstation (2026-11-27). Code comes from git; this copies what git does not hold.
# Run it ONLY when K decides to switch and nothing runs on the server (stop the runners, the Kaggle controller and the analysis jobs there first; the web page shows them):
#   powershell -NoProfile -ExecutionPolicy Bypass -File scripts\windows\pull_from_server.ps1 [-WhatIf]
# Copies (scp -r, the ssh host alias 'server'): campaigns\shell_runs (every pure-shell campaign: runs, queue, index, leases, frozen code, Kaggle state), campaigns\shell_runs_legacy,
# webui\results (the simulator's results), webui\presets.json, and the old yarn campaigns (fidelity, screen1, ...). Nothing on the server is changed or removed; an existing folder here is
# never overwritten (the script stops instead). Not copied: .env (K places the Kaggle token himself, see CLAUDE.md), ~/old_runs (reproduction runs, K's decision).
param([switch]$WhatIf)
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent (Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path))
$items = @(
  @{ remote = "auxetic_project/campaigns/shell_runs";        local = "campaigns\shell_runs" },
  @{ remote = "auxetic_project/campaigns/shell_runs_legacy"; local = "campaigns\shell_runs_legacy" },
  @{ remote = "auxetic_project/webui/results";               local = "webui\results" },
  @{ remote = "auxetic_project/webui/presets.json";          local = "webui\presets.json" }
)
foreach ($c in @("fidelity", "fidelity_large", "screen1", "shell_ref")) { $items += @{ remote = "auxetic_project/campaigns/$c/runs"; local = "campaigns\$c\runs" } }
foreach ($it in $items) {
  $dst = Join-Path $root $it.local
  if (Test-Path $dst) { "SKIP  $($it.local): already here (never overwritten; move it aside first if a fresh copy is wanted)"; continue }
  if ($WhatIf) { "WOULD copy server:~/$($it.remote) -> $dst"; continue }
  New-Item -ItemType Directory -Force (Split-Path -Parent $dst) | Out-Null
  & scp -q -r -o BatchMode=yes "server:~/$($it.remote)" $dst
  if ($LASTEXITCODE -eq 0) { "OK    $($it.local)" } else { "FAIL  $($it.local) (scp exit $LASTEXITCODE; may simply not exist on the server)" }
}
"Next: venv\Scripts\python.exe scripts\campaigns\campaign.py shell-status shell2   (checks that the copied campaign reads correctly here)"
