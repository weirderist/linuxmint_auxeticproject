# install_logon_task.ps1 - register ONE Windows logon task that runs start_services.ps1 (2026-11-27), so the services come back after a power cut or a restart of the workstation.
# K runs this once (no administrator rights needed for a task of his own account):
#   powershell -NoProfile -ExecutionPolicy Bypass -File scripts\windows\install_logon_task.ps1 [-Campaign shell2] [-Hub local] [-Extra "-NoKaggle"]
# To remove it later:  schtasks /Delete /TN AuxeticServices /F
param([string]$Campaign = "shell2", [string]$Hub = "local", [string]$Extra = "")
$script = Join-Path (Split-Path -Parent $MyInvocation.MyCommand.Path) "start_services.ps1"
$cmd = "powershell -NoProfile -WindowStyle Hidden -ExecutionPolicy Bypass -File `"$script`" -Campaign $Campaign -Hub $Hub $Extra"
schtasks /Create /SC ONLOGON /TN AuxeticServices /TR $cmd /F /DELAY 0001:00
"registered: AuxeticServices (at logon, one minute after) -> $cmd"
