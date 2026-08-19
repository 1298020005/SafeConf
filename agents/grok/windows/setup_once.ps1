# Run ONCE on Windows (PowerShell). Then keep an SSH session with LocalForward.

$sshConfig = Join-Path $env:USERPROFILE ".ssh\config"
Write-Host "1) Ensure SSH config has LocalForward 18765 127.0.0.1:18765 for your server host."
Write-Host "   Example block:"
Write-Host "   Host your-server"
Write-Host "       HostName <ip>"
Write-Host "       User yyf"
Write-Host "       LocalForward 18765 127.0.0.1:18765"
Write-Host ""
Write-Host "2) Open a NEW SSH/Cursor session so the forward is active."
Write-Host "3) Start this sync loop (keep window open):"
Write-Host ""

$here = Split-Path -Parent $MyInvocation.MyCommand.Path
$sync = Join-Path $here "clip_sync.ps1"

# Allow scripts for current user if needed
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned -Force | Out-Null

# Quick probe
try {
    $r = Invoke-WebRequest -Uri "http://127.0.0.1:18765/health" -UseBasicParsing -TimeoutSec 2
    Write-Host "Bridge health: $($r.Content)"
} catch {
    Write-Host "WARNING: cannot reach http://127.0.0.1:18765 — fix SSH LocalForward first."
}

Write-Host "Starting clip_sync.ps1 ..."
& powershell -NoProfile -ExecutionPolicy Bypass -File $sync
