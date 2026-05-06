#Requires -Version 5.1
# 批量消融实验脚本 (Windows PowerShell)

param(
    [string]$DataRoot = "data\NYUDepthv2",
    [int]$MaxEpochs = 50,
    [int]$BatchSize = 4
)

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "开始批量消融实验" -ForegroundColor Cyan
Write-Host "数据集: $DataRoot" -ForegroundColor Cyan
Write-Host "Epochs: $MaxEpochs" -ForegroundColor Cyan
Write-Host "Batch Size: $BatchSize" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan

$models = @("early", "mid_fusion")

foreach ($model in $models) {
    Write-Host ""
    Write-Host ">>> 正在训练: $model" -ForegroundColor Green
    & python train.py `
        --model $model `
        --data_root $DataRoot `
        --max_epochs $MaxEpochs `
        --batch_size $BatchSize `
        --num_workers 4 `
        --checkpoint_dir ".\checkpoints_ablation"
}

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "所有实验训练完成！" -ForegroundColor Cyan
Write-Host "检查点目录: .\checkpoints_ablation" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
