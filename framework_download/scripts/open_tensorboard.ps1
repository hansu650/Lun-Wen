#Requires -Version 5.1

param(
    [string]$RunName = "early_run1",
    [string]$LogDir = "",
    [int]$Port = 6006
)

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Split-Path -Parent $ScriptDir

if (-not $LogDir) {
    $LogDir = Join-Path $ProjectRoot ("checkpoints\" + $RunName + "\lightning_logs")
}

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Open TensorBoard" -ForegroundColor Cyan
Write-Host "LogDir : $LogDir" -ForegroundColor Cyan
Write-Host "Port   : $Port" -ForegroundColor Cyan
Write-Host "URL    : http://localhost:$Port" -ForegroundColor Cyan
Write-Host "Tip    : If you see version_0 and version_1, check the latest run first." -ForegroundColor Yellow
Write-Host "========================================" -ForegroundColor Cyan

if (-not (Test-Path $LogDir)) {
    throw "LogDir not found: $LogDir"
}

Push-Location $ProjectRoot
try {
    & python -m tensorboard.main --logdir $LogDir --port $Port
}
finally {
    Pop-Location
}
