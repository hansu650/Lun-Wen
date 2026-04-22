#!/bin/bash
# 批量消融实验脚本 (Linux/Mac)

DATA_ROOT=${1:-"data/NYUDepthv2"}
MAX_EPOCHS=${2:-50}
BATCH_SIZE=${3:-4}

echo "========================================"
echo "开始批量消融实验"
echo "数据集: $DATA_ROOT"
echo "Epochs: $MAX_EPOCHS"
echo "Batch Size: $BATCH_SIZE"
echo "========================================"

MODELS=("early" "mid_fusion")

for MODEL in "${MODELS[@]}"; do
    echo ""
    echo ">>> 正在训练: $MODEL"
    python train.py \
        --model "$MODEL" \
        --data_root "$DATA_ROOT" \
        --max_epochs "$MAX_EPOCHS" \
        --batch_size "$BATCH_SIZE" \
        --num_workers 4 \
        --checkpoint_dir "./checkpoints_ablation"
done

echo ""
echo "========================================"
echo "所有实验训练完成！"
echo "检查点目录: ./checkpoints_ablation"
echo "========================================"
