# Qintian Workspace Context

## Workspace Role

`C:\Users\qintian\Desktop\qintian` 是总工作区，包含多个项目、数据集、预训练权重、参考代码和实验记录。

## Main Active Project

当前最重要的 active project 是：

```
C:\Users\qintian\Desktop\qintian\final daima
```

该项目是 RGB-D 语义分割项目，数据集为 NYUDepthV2（40 类），基于 DFormerV2 风格 baseline 研究中间融合和辅助 loss。

> 注意：外层 README.md 是旧版文档，引用的 `framework_download/` 目录已不存在，DINOv2-small + Swin-Tiny 架构已弃用。当前活跃代码和实验全部在 `final daima/` 中。

## Important Folders

| 目录 | 作用 | 备注 |
|---|---|---|
| `final daima/` | 当前主要实验代码、模型、训练脚本、实验记录 | **active project** |
| `data/` | 数据集：NYUDepthV2、NYUDepthv2_matdepth、nyu_depth_v2_labeled.mat | 不要修改 |
| `pretrained/` | 预训练权重：dinov2-main、dinov2_small、swin_tiny | 不要删除 |
| `ref_codes/` | 参考论文开源代码：ACNet、CAFuser、CMFormer、DFormer、ESANet、FAFNet、GeminiFusion、KTB、Mul-VMamba、PGDENet、RGBD_Semantic_Segmentation_PyTorch、TokenFusion 等 | 只读参考 |
| `docs/` | 外层文档：experiment_log.md、model_changes.md、paper_notes.md、next_steps.md | 部分内容可能过时，以 `final daima/docs/` 为准 |
| `dformer_work/` | DFormer 相关工作区：含 DFormer 源码、checkpoints、datasets、notes、papers | 预训练权重路径在此或 `pretrained/` |
| `笔记/` | 课程/研究笔记（第一次课 ~ 第五次课、流程文档） | 个人资料 |
| `skills/` | AI assistant skills / 上下文工具相关 | 工具配置 |
| `versions/` | 版本归档（version_001_stable_baseline_mid_fusion 等） | 历史快照 |
| `废1.0/` | 旧版本归档（decoder_论文、encoder_论文、feiqi_0.3、fusion_论文） | 不要随便动 |
| `查找论文/` | 论文查找资料 | 个人资料 |
| `自己骨架/` | 个人骨架/框架探索 | 尚未完全确认 |
| `Auto-claude-code-research-in-sleep-main/` | 自动研究工具 | 尚未完全确认 |

## Active RGB-D Project Context

- 研究方向：RGB-D semantic segmentation
- 主要数据集：NYUDepthV2（795 train / 654 test，40 类）
- 训练框架：PyTorch Lightning
- Active baseline：`dformerv2_mid_fusion`
- Baseline 结构：
  - RGB encoder：DFormerv2_S（来自 DFormer 开源项目，Geometry Self-Attention）
  - Depth encoder：ResNet-18（1 通道输入，ImageNet 预训练权重均值初始化）
  - Fusion：4 个 GatedFusion 模块（逐 stage 门控融合 `g * rgb + (1-g) * depth`）
  - Decoder：SimpleFPNDecoder（简化 FPN，4 路 lateral + smooth + classifier）
  - 40 类，ignore_index=255
- 训练设置：batch_size=2，max_epochs=50，lr=6e-5，AdamW(weight_decay=0.01)，early_stop_patience=30
- 预训练权重路径：`C:/Users/qintian/Desktop/qintian/dformer_work/checkpoints/pretrained/DFormerv2_Small_pretrained.pth`
- 当前辅助实验：`dformerv2_ms_freqcov`（training-only frequency covariance auxiliary loss，推理结构不变）

## Current Experiment Status

### Clean 10-run Baseline（dformerv2_mid_fusion）
- mean best val/mIoU = **0.517397**
- population std = **0.004901**
- best single run = 0.524425（run05）
- 证据：`final daima/miou_list/dformerv2_mid_fusion_gate_baseline_summary_run01_09_run10_retry.md`

### dformerv2_ms_freqcov Run01
- 设置：lambda_freq=0.01，freq_eta=1.0，freq_proj_dim=64，freq_kernel_size=3，freq_stage_weights=1,1,1,1
- best val/mIoU = **0.520539** at epoch 50
- last val/mIoU = 0.520539
- delta vs baseline mean = **+0.003142**（< 1 sigma）
- freqcov loss：从 epoch1 的 0.001377 降到 epoch50 的 0.000002，乘以 lambda=0.01 后对总 loss 贡献极小
- 状态：**promising single-run signal**，不能说 stable improvement
- 7 个 freqcov 设置的 sweep mean = 0.515697，低于 baseline mean
- 证据：`final daima/miou_list/dformerv2_ms_freqcov_run01.md`

## Important Documents Inside final daima

| 文件 | 作用 |
|---|---|
| `final daima/docs/experiment_log.md` | 实验流水账：配置、结果、delta、结论、下一步 |
| `final daima/docs/paper_notes.md` | 论文层面判断：baseline 数字、candidate 边界、不能引用的结果 |
| `final daima/docs/model_changes.md` | 代码变更记录：新增类、参数量、验证方式 |
| `final daima/miou_list/` | 逐 epoch mIoU 表格 + summary，每次实验一个 .md |
| `final daima/checkpoints/` | checkpoint 和 TensorBoard 日志，只看结构，不要乱删 |

## Code Entry Points Inside final daima

| 文件 | 作用 |
|---|---|
| `final daima/train.py` | 训练入口：argparse、MODEL_REGISTRY、build_model()、trainer.fit() |
| `final daima/src/models/mid_fusion.py` | 核心模型：GatedFusion、DFormerV2MidFusionSegmentor、LitDFormerV2MidFusion、LitDFormerV2MSFreqCov |
| `final daima/src/models/freq_cov_loss.py` | MultiScaleFrequencyCovarianceLoss 定义 |
| `final daima/src/models/dformerv2_encoder.py` | DFormerv2_S/B/L backbone（来自 DFormer 开源） |
| `final daima/src/models/encoder.py` | ResNet-18 RGBEncoder、DepthEncoder、EarlyFusionEncoder |
| `final daima/src/models/decoder.py` | SimpleFPNDecoder |
| `final daima/src/models/base_lit.py` | BaseLitSeg：Lightning 基类，训练/验证逻辑，mIoU 计算 |
| `final daima/src/data_module.py` | NYUDataModule：NYUDepthV2 Dataset + DataLoader，不要随便改 |

## MODEL_REGISTRY（train.py）

```python
MODEL_REGISTRY = {
    "early": LitEarlyFusion,          # 4 通道 early fusion
    "mid_fusion": LitMidFusion,       # ResNet-18 双分支 mid fusion
    "dformerv2_mid_fusion": LitDFormerV2MidFusion,    # DFormerv2 baseline
    "dformerv2_ms_freqcov": LitDFormerV2MSFreqCov,    # freqcov 辅助 loss
}
```

## Working Rules

### 编码规则

1. 不要写兜底代码。
2. 不要自动猜路径、自动兼容旧版本、自动切换到别的实现。
3. 不要为了"尽量跑起来"加 `try/except` 兜底。
4. 不要额外加一层层 helper、wrapper、fallback 分支。
5. 优先保持原版代码风格：结构简单、入口薄、主线直接。
6. 除非用户明确要求，否则不要额外加显式防御性 `raise` 检查。
7. 能直接写清楚的逻辑就直接写，不要过度抽象。
8. 改动优先最小化，尽量只改当前任务真正需要的部分。
9. 不要自动引入大型预训练模型，除非用户明确确认当前环境可用。

### 实验记录规则

10. 任何没有在当前环境真实跑通、没有明确日志/checkpoint/配置支持的结果，都不能写成当前最好结果。
11. 对于旧 README 中出现但用户已确认无效的信息，应标注为 deprecated，不得作为论文实验结果引用。
12. 当前论文项目的活跃代码目录是 `final daima/`（旧 `framework_download/` 已弃用）。
13. 新任务开始前，优先阅读 `final daima/README.md`、`final daima/docs/experiment_log.md`、`final daima/docs/model_changes.md`、`final daima/docs/paper_notes.md`。
14. 每次实验结束后必须更新 `final daima/docs/experiment_log.md`，记录配置、结果、证据路径、结论和下一步。
15. 每次模型结构改动后必须更新 `final daima/docs/model_changes.md`，说明改了什么、为什么改、是否保留。
16. 每次论文叙事变化后必须更新 `final daima/docs/paper_notes.md`，同步动机、方法主线、消融设计和不能引用的结果边界。
17. 不要把 deprecated/invalid 记录重新写成有效 baseline，也不要在论文结果中引用。
18. 当用户要求"总结这次实验结果"或类似实验总结时，必须从真实日志/TensorBoard event 中提取每个 epoch 的 `val/mIoU`，在 `final daima/miou_list/` 下为该实验生成单独的 Markdown 文件，并在 `final daima/docs/experiment_log.md` 中引用该 mIoU 明细文件。
19. GPT Pro / external GPT / reviewer discussion prompts are temporary conversation artifacts. Do not save them as standalone Markdown files under `final daima/docs/` unless the user explicitly asks; provide them directly in chat instead.

### 运行实验原则：决策价值导向实验

20. 实验应以假设和决策价值为导向，而不是为了补齐口径、填满表格、让 ablation 看起来完整。每个实验都应回答一个明确假设，或支持一个后续决策。
21. 不在低边际信息增益的方向上穷举。当可以预见某组实验对性能提升、方向判断或后续改进都没有明显贡献，尤其只是重复确认“不可行 / 无效果”时，就应停止，不做穷举式验证。

### 项目操作规则

22. 修改代码前必须先阅读相关文件，理解现有结构。
23. 不要随便改 backbone（DFormerv2_S）、dataset/dataloader（data_module.py）、decoder（SimpleFPNDecoder）。
24. 不要删除 `final daima/checkpoints/`、`final daima/miou_list/`、`final daima/docs/` 里的已有记录。
25. 新实验必须有独立 run name，checkpoint 目录命名格式：`{model_name}_{run_name}/`。
26. 新模块尽量注册成单独 model name 或参数开关，不要硬改 baseline 代码。
27. 每次修改后说明：改了哪些文件、如何运行、如何验证。
28. 如果只是分析/理解/总结项目，请只读，不要改代码。

<!-- ARIS:BEGIN -->
## ARIS Skill Scope
For ARIS workflows in this project, use only the project-local ARIS skills under `.agents/skills/aris`.
Do not use global skills or non-ARIS project skills unless the user explicitly asks to mix them.
<!-- ARIS:END -->

## Default Experiment Result Workflow

When a training run finishes and the user asks to summarize/discuss/commit/push the result, follow `final daima/docs/experiment_result_workflow.md` by default:

1. Extract true per-epoch `val/mIoU` from TensorBoard/log evidence.
2. Create or update the run file under `final daima/miou_list/`.
3. Update `final daima/docs/experiment_log.md`.
4. Update `final daima/docs/paper_notes.md` when the claim boundary changes.
5. If useful, send the result to a separate GPT/subagent critique, but do not save one-off GPT/Pro prompts or discussion transcripts under `final daima/docs/` unless the user explicitly asks. Summarize any durable conclusion into `experiment_log.md` or `paper_notes.md` instead.
6. Commit and push only relevant code/docs/result files; avoid staging unrelated checkpoint deletions, ignored checkpoint outputs, reference-code folders, or personal workspace files.
