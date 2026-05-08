# RGB-D Semantic Segmentation Experiments

本工作区用于 RGB-D 语义分割研究，主要数据集为 NYUDepthV2。

## Active Project

当前主要实验代码和结果在：

```
final daima/
```

该目录包含完整的训练框架、模型定义、实验记录和 checkpoint。

## Current Main Line

- Task：RGB-D semantic segmentation
- Dataset：NYUDepthV2（40 类）
- Active baseline：`dformerv2_mid_fusion`
- RGB encoder：DFormerv2_S（Geometry Self-Attention，来自 DFormer 开源项目）
- Depth encoder：ResNet-18（1 通道输入）
- Fusion：GatedFusion（逐 stage 门控融合）
- Decoder：SimpleFPNDecoder（简化 FPN）
- Training framework：PyTorch Lightning

## Current Confirmed Best Baseline

- Model：`dformerv2_mid_fusion`（clean 10-run）
- Mean best val/mIoU：**0.517397**
- Population std：**0.004901**
- Best single run：0.524425

## Active Auxiliary Experiment

- Model：`dformerv2_ms_freqcov`
- Training-only frequency covariance auxiliary loss，推理结构不变
- Best single run：0.520539（lambda_freq=0.01）
- 状态：promising single-run signal，尚未证明 stable improvement

## Important Documents

| 文件 | 作用 |
|---|---|
| `final daima/docs/experiment_log.md` | 实验配置、结果、结论 |
| `final daima/docs/paper_notes.md` | 论文边界和判断 |
| `final daima/docs/model_changes.md` | 代码变更记录 |
| `final daima/miou_list/` | 逐 epoch mIoU 和 summary |
| `final daima/checkpoints/` | checkpoint 和 TensorBoard 日志 |
| `AGENTS.md` | 工作区索引和工作规则 |
| `CLAUDE.md` | Claude Code 指令 |

## Code Entry Points

| 文件 | 作用 |
|---|---|
| `final daima/train.py` | 训练入口、argparse、MODEL_REGISTRY |
| `final daima/src/models/mid_fusion.py` | 核心模型定义 |
| `final daima/src/models/freq_cov_loss.py` | Frequency covariance loss |
| `final daima/src/models/dformerv2_encoder.py` | DFormerv2 backbone |
| `final daima/src/data_module.py` | NYUDepthV2 dataset 和 dataloader |

## Other Folders

| 目录 | 作用 |
|---|---|
| `data/` | NYUDepthV2 数据集 |
| `pretrained/` | 预训练权重（dinov2、swin_tiny） |
| `dformer_work/` | DFormer 相关工作区和预训练权重 |
| `ref_codes/` | 参考论文开源代码（ACNet、CAFuser、CMFormer、DFormer、ESANet 等） |
| `笔记/` | 课程/研究笔记 |
| `versions/` | 版本归档 |
| `废1.0/` | 旧版本归档 |
| `docs/` | 外层文档（部分内容可能过时，以 `final daima/docs/` 为准） |

## How To Train

```bash
cd "final daima"

# DFormerv2 mid-fusion baseline
python train.py \
  --model dformerv2_mid_fusion \
  --data_root "C:\Users\qintian\Desktop\qintian\data\NYUDepthv2" \
  --dformerv2_pretrained "C:\Users\qintian\Desktop\qintian\dformer_work\checkpoints\pretrained\DFormerv2_Small_pretrained.pth" \
  --max_epochs 50 \
  --batch_size 2 \
  --lr 6e-5 \
  --num_workers 4 \
  --checkpoint_dir "./checkpoints/my_run_name"
```

## What Is Not Tracked

- NYUDepthV2 data under `data/`
- Pretrained weights under `pretrained/`
- Training checkpoints under `final daima/checkpoints/`
