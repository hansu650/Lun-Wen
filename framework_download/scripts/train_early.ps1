#Requires -Version 5.1

param(
    [string]$DataRoot = "",
    [ValidateSet("early", "mid_fusion", "attention", "advanced", "dformer")]
    [string]$Model = "early",
    [string]$RunName = "",
    [int]$MaxEpochs = 50,
    [int]$BatchSize = 2,
    [double]$LearningRate = 0.0001,
    [int]$NumWorkers = 0,
    [int]$EarlyStopPatience = 15,
    [int]$CheckValEveryNEpoch = 1,
    [double]$WeightDecay = 0.01,
    [int]$WarmupSteps = 1000,
    [double]$MinLrRatio = 0.05,
    [double]$BackboneLrMult = 0.1,
    [ValidateSet("true", "false")]
    [string]$EvalTta = "false",
    [string]$Devices = "1",
    [ValidateSet("gpu", "cpu", "auto")]
    [string]$Accelerator = "gpu"
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

$CheckpointDir = Join-Path $ProjectRoot ("checkpoints\" + $RunName)

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Train Model" -ForegroundColor Cyan
Write-Host "Model         : $Model" -ForegroundColor Cyan
Write-Host "ProjectRoot   : $ProjectRoot" -ForegroundColor Cyan
Write-Host "DataRoot      : $DataRoot" -ForegroundColor Cyan
Write-Host "CheckpointDir : $CheckpointDir" -ForegroundColor Cyan
Write-Host "Epochs        : $MaxEpochs" -ForegroundColor Cyan
Write-Host "BatchSize     : $BatchSize" -ForegroundColor Cyan
Write-Host "LR            : $LearningRate" -ForegroundColor Cyan
Write-Host "NumWorkers    : $NumWorkers" -ForegroundColor Cyan
Write-Host "EarlyStop     : $EarlyStopPatience" -ForegroundColor Cyan
Write-Host "ValEveryNEpoch: $CheckValEveryNEpoch" -ForegroundColor Cyan
Write-Host "WeightDecay   : $WeightDecay" -ForegroundColor Cyan
Write-Host "WarmupSteps   : $WarmupSteps" -ForegroundColor Cyan
Write-Host "MinLrRatio    : $MinLrRatio" -ForegroundColor Cyan
Write-Host "BackboneLRMul : $BackboneLrMult" -ForegroundColor Cyan
Write-Host "EvalTTA       : $EvalTta" -ForegroundColor Cyan
Write-Host "Devices       : $Devices" -ForegroundColor Cyan
Write-Host "Accelerator   : $Accelerator" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan

if (-not (Test-Path $DataRoot)) {
    throw "DataRoot not found: $DataRoot"
}

Push-Location $ProjectRoot
try {
    & python train.py `
        --model $Model `
        --data_root $DataRoot `
        --max_epochs $MaxEpochs `
        --batch_size $BatchSize `
        --lr $LearningRate `
        --num_workers $NumWorkers `
        --early_stop_patience $EarlyStopPatience `
        --check_val_every_n_epoch $CheckValEveryNEpoch `
        --weight_decay $WeightDecay `
        --warmup_steps $WarmupSteps `
        --min_lr_ratio $MinLrRatio `
        --backbone_lr_mult $BackboneLrMult `
        --eval_tta $EvalTta `
        --devices $Devices `
        --accelerator $Accelerator `
        --checkpoint_dir $CheckpointDir
}
finally {
    Pop-Location
}
