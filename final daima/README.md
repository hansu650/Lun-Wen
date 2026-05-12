# RGB-D 多模态融合完整代码框架

本框架是《多模态融合深度学习框架教学》课程的完整可运行代码，整合了课程中讲解的所有模型和技巧，可直接用于 NYU Depth V2 数据集上的 RGB-D 语义分割研究与实验。

## 框架特点

- **模块化设计**：数据、模型、训练、评估完全解耦
- **多种融合策略**：Early Fusion / Mid Fusion (Gated) / Attention Fusion / Advanced (ViT 预训练骨干)
- **PyTorch Lightning 最佳实践**：自动混合精度、Checkpoint、TensorBoard 日志
- **即开即用**：配置好数据路径后即可运行训练和评估

## 目录结构

```
.
├── README.md
├── requirements.txt
├── train.py              # 训练入口
├── eval.py               # 评估入口
├── infer.py              # 推理与可视化入口
├── src/
│   ├── data_module.py    # NYU 数据模块 + 数据增强
│   ├── models/
│   │   ├── encoder.py    # ResNet-18 RGB/Depth 编码器
│   │   ├── decoder.py    # FPN 解码器
│   │   ├── early_fusion.py
│   │   ├── mid_fusion.py
│   │   ├── attention_fusion.py
│   │   └── advanced_fusion.py
│   └── utils/
│       ├── metrics.py    # mIoU / Pixel Accuracy
│       └── visualize.py  # 可视化工具
└── scripts/
    ├── run_all_experiments.sh    # Linux/Mac 批量实验脚本
    └── run_all_experiments.ps1   # Windows 批量实验脚本
```

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 准备数据

确保数据集路径如下：

```
/path/to/NYUDepthv2/
├── RGB/
├── Depth/
├── Label/
├── train.txt
└── test.txt
```

### 3. 训练模型

```bash
# 训练 Mid Fusion (Gated) 模型
python train.py --model mid_fusion --data_root /path/to/NYUDepthv2 --max_epochs 50

# 训练 Attention Fusion 模型
python train.py --model attention --data_root /path/to/NYUDepthv2 --max_epochs 50

# 训练 Early Fusion 模型
python train.py --model early --data_root /path/to/NYUDepthv2 --max_epochs 50

# 训练 Advanced (ViT 预训练骨干) 模型
python train.py --model advanced --data_root /path/to/NYUDepthv2 --max_epochs 50 --batch_size 2
```

### 4. 评估模型

```bash
python eval.py --checkpoint checkpoints/mid_fusion-epoch=xx-xxxx.ckpt --model mid_fusion --data_root /path/to/NYUDepthv2
```

### 5. 推理可视化

```bash
python infer.py --checkpoint checkpoints/mid_fusion-epoch=xx-xxxx.ckpt --model mid_fusion --data_root /path/to/NYUDepthv2 --num_vis 10
```

### 6. 批量消融实验

```bash
# Linux/Mac
bash scripts/run_all_experiments.sh /path/to/NYUDepthv2

# Windows PowerShell
.\scripts\run_all_experiments.ps1 -DataRoot "D:\\Data\\NYUDepthv2"
```

## 支持的模型

| 模型 | 说明 |
|:---|:---|
| `rgb` | RGB-only 基线 |
| `depth` | Depth-only 基线 |
| `early` | Early Fusion (输入层拼接) |
| `mid_fusion` | Mid Fusion (双编码器 + Gated Fusion + FPN) |
| `attention` | Attention Fusion (跨模态通道+空间注意力) |
| `advanced` | Advanced (ViT 预训练 backbone + TTA + Warmup/Cosine) |

## 常用参数

```bash
python train.py \
    --model mid_fusion \
    --data_root /path/to/NYUDepthv2 \
    --num_classes 40 \
    --batch_size 4 \
    --max_epochs 50 \
    --lr 1e-4 \
    --num_workers 4 \
    --checkpoint_dir ./checkpoints
```

## TensorBoard 查看训练日志

```bash
tensorboard --logdir ./lightning_logs
```

## 许可证

本框架仅供教学和研究使用。
# Current Active Project Status

This directory is the active RGB-D semantic segmentation project for NYUDepthV2 40-class experiments.

The old teaching/demo model list below is deprecated. The current default training path is:

- `dformerv2_mid_fusion`: clean main baseline, 10-run mean best mIoU `0.517397`, std `0.004901`, best single `0.524425`.
- `dformerv2_tgga_c34_beta002_aux003_detachsem_simplefpn_v2`: active but unstable. Run01 best `0.522206` at epoch 48, final `0.489865`; run02 best `0.517437` at epoch 49, final `0.486566`; two-run mean best `0.519822`, so this is weak positive but not a stable improvement.
- `dformerv2_geometry_primary_teacher`: active PMAD teacher dependency.
- `dformerv2_primkd_logit_only`: active PMAD logit-only KD branch, marginal positive repeat signal around `0.520795` five-run mean at w0.15/T4.

Archived or default-hidden experiments include DGBF, CGPC, SGBR-Lite, CGCD/ClassContext, context decoder/PPM, FFT freq enhance, FFT HiLo, depth FFT select, CE+Dice, and FreqCov-style auxiliary losses. They can be discussed as negative/unstable ablations, not as active methods.

See `docs/ACTIVE_STATUS.md`, `docs/cleanup_notes.md`, and `feiqi/README.md` for the current active/archive boundary.
