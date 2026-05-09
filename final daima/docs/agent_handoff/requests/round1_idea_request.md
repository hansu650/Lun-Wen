# Claude Idea Request — Round 1 (Revised)

## Round 编号
1

## 当前阶段
Exploration

## 背景：用户否决原因

用户否决了原 3 个候选 idea（alignment / InfoNCE / FFT sweep），理由：
1. alignment 和 FFT sweep 是"试过的方向换皮"，不接受
2. 老师给的 4 个 loss 方向中，freqcov (1) 和 maskrec (2) 已试且均 negative/weak
3. **rec-cov (3)** 和 **InfoNCE (4)** 从未跑过，必须从这 2 个中选

## 3 个候选 idea（全部来自未试过的老师方向）

### Idea 1: 重建式协方差 Loss (rec-cov)

- 设计：先做 maskrec 式跨模态重建（mask depth → 从 RGB 重建），然后对重建特征和原始特征计算 freqcov 式频域协方差对齐
- 即 maskrec + freqcov 的组合：`L_rec-cov = cov_loss(reconstruct(masked_depth), original_depth)`
- 比纯 maskrec 多一层统计约束，比纯 freqcov 多一个重建目标
- 注册为新 model name: `dformerv2_rec_cov`
- 风险：两个已 negative 的机制组合，可能 double negative

### Idea 2: InfoNCE 对比 Loss (contrast)

- 设计：4 个 stage 各加 RGB/Depth projection head（1×1 conv → proj_dim 维 → L2 normalize），同像素位置的 RGB/Depth 投影作为正样本对，不同位置作为负样本对
- `L_nce_i = -log(exp(sim(z_r, z_d)/tau) / sum_k(exp(sim(z_r, z_d_k)/tau)))`
- 推理路径完全不变，仅训练时增加辅助 loss
- 注册为新 model name: `dformerv2_infonce`
- 优势：不做生成/重建，直接在投影空间做对比学习；与 freqcov/maskrec 是完全不同的约束范式
- 风险：temperature tau 敏感，负样本策略需要设计

### Idea 3: InfoNCE + Stage-wise Projection（Idea 2 的简化变体）

- 与 Idea 2 基本相同，但只在 c3/c4 做对比（与 maskrec 的 stage 选择一致）
- 更保守，减少浅层噪声干扰
- 注册为新 model name: `dformerv2_infonce_c34`

## 最终选择

**选择的 idea：Idea 2 — InfoNCE 对比 Loss**

## 选择理由

1. **完全不同的约束范式**：freqcov 比较协方差矩阵，maskrec 做像素级重建，InfoNCE 做对比学习——三种是本质不同的跨模态对齐方式
2. **不做生成**：maskrec 的问题在于重建目标不直接帮助分割；InfoNCE 不需要重建，直接拉近同位置表征、推远不同位置表征
3. **理论基础强**：InfoNCE 在自监督学习（SimCLR, MoCo）中已被证明能有效学习表征对齐
4. **rec-cov 是两个已 negative 机制的组合**：maskrec 和 freqcov 各自都没成功，组合后大概率也不行；InfoNCE 是全新范式
5. **实现简洁**：4 个 1×1 conv head + L2 normalize + cosine similarity + softmax，变量最少

## 与第五次课 4+1 方向的关系

- **+1 辅助 Loss** — InfoNCE 是老师明确列出的第 4 个 loss 方向（contrast）
- **方向 1：映射、对齐、筛选** — projection head 是"映射"，InfoNCE 的正样本拉近是"对齐"
- 不涉及方向 2/3/4（频域增强），不改 encoder/decoder/fusion 结构

## 与已有结果的关系

- **与 negative 结果的关系**：
  - freqcov (sweep mean 0.515697) → 协方差矩阵对齐信号弱；InfoNCE 不用协方差，用对比学习
  - maskrec (0.515327) → 重建目标不帮助分割；InfoNCE 不做重建，直接对比表征
- **与 promising 结果的关系**：
  - FFT freq enhance g01 (0.522688) → 说明增强后的特征更好；InfoNCE 可以帮助增强后的特征更一致（未来可组合）
- **与 rec-cov 的关系**：rec-cov 是 maskrec + freqcov 组合，两个都 negative，组合风险高；InfoNCE 是独立新范式

## 代码修改计划

- 修改哪些文件：
  - `final daima/src/models/mid_fusion.py`（新增 `InfoNCELoss`, `DFormerV2InfoNCESegmentor`, `LitDFormerV2InfoNCE`）
  - `final daima/train.py`（新增 MODEL_REGISTRY 条目 `dformerv2_infonce`，新增 build_model 分支，新增 argparse 参数 `--lambda_infonce`, `--infonce_proj_dim`, `--infonce_tau`, `--infonce_stage_weights`）
- 新增哪些类/函数：
  - `InfoNCELoss`（在 mid_fusion.py 中，projection head + L2 normalize + InfoNCE 计算）
  - `DFormerV2InfoNCESegmentor`（复用 DFormerV2MidFusionSegmentor.extract_features）
  - `LitDFormerV2InfoNCE`（继承 BaseLitSeg，重写 training_step）
- 是否新增 model name：**yes** — `dformerv2_infonce`
- 是否修改 backbone/dataset/dataloader/decoder/base_lit.py：**NO**

## CrossAlignLoss 设计细节

- Projection head：纯 1×1 conv（不加 BN/ReLU，per Codex suggestion），将 c3/c4 特征投影到 infonce_proj_dim 维
- L2 normalize：在投影后做 channel-wise L2 normalize
- InfoNCE 计算：
  - 正样本对：同像素位置 (i,j) 的 RGB projection 和 Depth projection
  - 负样本对：同一 batch 内不同像素位置的 Depth projection
  - `sim(z_r, z_d) = z_r · z_d / tau`
  - `L = -log(exp(sim_pos/tau) / sum_k(exp(sim_k/tau)))`
- Stage weights：`infonce_stage_weights=0,0,1,1`（c3+c4 only）
- 默认超参：`lambda_infonce=0.01`, `infonce_proj_dim=64`, `infonce_tau=0.07`

## Smoke test 计划

1. `train.py --help` 检查新参数 `--lambda_infonce`, `--infonce_proj_dim`, `--infonce_tau`, `--infonce_stage_weights` 是否出现
2. `python -c "from src.models.mid_fusion import LitDFormerV2InfoNCE"` 确认可导入
3. 单独 smoke 目录（不混入正式 run01）：1 epoch forward/backward 测试，确认 loss 下降、无 NaN、无 shape mismatch

## 训练命令草案

```
cd /d "C:\Users\qintian\Desktop\qintian\final daima" && python train.py --model dformerv2_infonce --data_root "C:\Users\qintian\Desktop\qintian\data\NYUDepthv2" --dformerv2_pretrained "C:\Users\qintian\Desktop\qintian\dformer_work\checkpoints\pretrained\DFormerv2_Small_pretrained.pth" --batch_size 2 --max_epochs 50 --lr 6e-5 --num_workers 4 --early_stop_patience 30 --checkpoint_dir ".\checkpoints\dformerv2_infonce_run01" --lambda_infonce 0.01 --infonce_proj_dim 64 --infonce_tau 0.07 --infonce_stage_weights "0,0,1,1"
```

## 希望 Codex 审查的问题

1. InfoNCE 的负样本策略：用同 batch 内所有其他像素位置的 depth projection 作为负样本是否合理？还是应该用 batch 内其他 sample 的 depth projection？
2. temperature tau=0.07 是 SimCLR 默认值，对 RGB-D 特征是否合适？
3. lambda_infonce=0.01 是否太弱？InfoNCE loss 的量级通常比 seg loss 大还是小？
4. projection head 纯 1x1 conv 是否够？是否需要一个 hidden layer（1x1 → ReLU → 1x1）？

## ALLOW_CODE_CHANGE_BY_CODEX
no
