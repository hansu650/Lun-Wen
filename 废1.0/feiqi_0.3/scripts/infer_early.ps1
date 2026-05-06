#Requires -Version 5.1

param(
    [string]$DataRoot = "",
    [ValidateSet("early", "mid_fusion", "attention", "advanced", "dformer")]
    [string]$Model = "early",
    [string]$RunName = "",
    [string]$Checkpoint = "",
    [int]$NumVis = 10,
    [string]$SaveDir = "",
    [ValidateSet("auto", "true", "false")]
    [string]$EvalTta = "auto"
)

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Split-Path -Parent $ScriptDir

if (-not $DataRoot) {
    $DataRoot = Join-Path (Split-Path -Parent $ProjectRoot) "data\NYUDepthv2"
}

if (-not $RunName) {
    switch ($Model) {
        "mid_fusion" { $RunName = "mid_run1" }
        "attention" { $RunName = "attention_run1" }
        "advanced" { $RunName = "advanced_run1" }
        "dformer" { $RunName = "dformer_run1" }
        default { $RunName = "early_run1" }
    }
}

if (-not $SaveDir) {
    $SaveDir = Join-Path $ProjectRoot ("visualizations\" + $RunName)
}

if (-not $Checkpoint) {
    $CheckpointRoot = Join-Path $ProjectRoot ("checkpoints\" + $RunName)
    $Latest = Get-ChildItem -Path $CheckpointRoot -Recurse -Filter *.ckpt |
        Sort-Object LastWriteTime -Descending |
        Select-Object -First 1
    if (-not $Latest) {
        throw "No .ckpt found under: $CheckpointRoot"
    }
    $Checkpoint = $Latest.FullName
}

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Visualize Predictions" -ForegroundColor Cyan
Write-Host "Model      : $Model" -ForegroundColor Cyan
Write-Host "DataRoot   : $DataRoot" -ForegroundColor Cyan
Write-Host "Checkpoint : $Checkpoint" -ForegroundColor Cyan
Write-Host "NumVis     : $NumVis" -ForegroundColor Cyan
Write-Host "SaveDir    : $SaveDir" -ForegroundColor Cyan
Write-Host "EvalTTA    : $EvalTta" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan

if (-not (Test-Path $DataRoot)) {
    throw "DataRoot not found: $DataRoot"
}
if (-not (Test-Path $Checkpoint)) {
    throw "Checkpoint not found: $Checkpoint"
}

Push-Location $ProjectRoot
try {
    & python infer.py `
        --checkpoint $Checkpoint `
        --model $Model `
        --data_root $DataRoot `
        --num_vis $NumVis `
        --save_dir $SaveDir `
        --eval_tta $EvalTta
}
finally {
    Pop-Location
}
