# RGB-D 下一步结构方向源码分析报告

## 1. 一句话结论

在当前 DFormerV2-S + ResNet-18 双流 mid-fusion 架构上，所有增量式融合/损失/解码器修改均告失败（20+ 实验，mIoU 停滞 0.507–0.517）。**最值得尝试的方向是 PrimKD 式知识蒸馏**（冻结 RGB-only teacher 引导 RGBD student），其次是 MixPrompt 式 stage-wise prompt 注入，再次是增强 DFormerV2 内部 geometry prior 的利用方式。

---

## 2. 当前代码可插入点

### 2.1 模型架构（`src/models/mid_fusion.py`）

| 插入点 | 位置 | 说明 |
|--------|------|------|
| `GatedFusion.forward()` | L16-27 | 替换或增强当前 `gate → sigmoid → refine` 融合逻辑 |
| `DFormerV2MidFusionSegmentor.extract_features()` | L69-98 | 双流特征提取后、decoder 前的融合区域 |
| `DFormerv2_S.forward(x, x_e)` | `dformerv2_encoder.py` L584 | RGB encoder 已接收 depth，可用于内部 prompt 注入 |

### 2.2 训练循环（`src/models/base_lit.py`）

| 插入点 | 位置 | 说明 |
|--------|------|------|
| `BaseLitSeg.training_step()` | L30-36 | 可添加 distillation loss |
| `BaseLitSeg.validation_step()` | L38-48 | 可添加 teacher-guided eval |

### 2.3 解码器（`src/models/decoder.py`）

| 插入点 | 位置 | 说明 |
|--------|------|------|
| `SimpleFPNDecoder.forward()` | L35-55 | 可在 FPN 后添加 prompt-guided refinement |

---

## 3. 方向排序（可行性 × 收益 × 论文故事性）

| 排名 | 方向 | 预估提升 | 工程量 | 论文故事 |
|------|------|----------|--------|----------|
| **1** | PrimKD 式知识蒸馏 | +1.5–3% | 中 | "teacher-guided cross-modal distillation" |
| **2** | MixPrompt 式 stage-wise prompt | +1–2% | 小 | "depth-as-prompt for frozen RGB encoder" |
| **3** | 增强 geometry prior 利用 | +0.5–1.5% | 小 | "explicit depth-aware feature calibration" |
| 4 | Reliability-aware gating | +0–1% | 中 | 已被实验否定 |
| 5 | DepthAnythingV2/PDDM 预训练 | +2–4% | 大 | 工程量过大，不适合本科毕设时间线 |

---

## 4. 第一推荐方向：PrimKD 式知识蒸馏

### 4.1 源码关键发现

**PrimKD 训练循环**（`ref_codes/PrimKD/train.py` L230-288）：

```python
# 双模型：student (RGBD) + teacher (RGB-only, frozen)
logits, rgbd_x, loss = model(imgs, modal_xs, gts)      # student
logits2, rgb_x, loss2 = model2(imgs, None, gts)          # teacher (frozen)

# KL divergence on logits
loss_rdkl = kl_calculator.compute_kl_divergence(logits2.detach(), logits) * alpha

# MSE feature alignment (select max/min loss stage)
feature_loss = distill_feature_maps(rgbd_x[i], rgb_x[j].detach()) * beta

loss = loss + loss_rdkl + feature_loss
```

**FeatureRectifyModule**（`ref_codes/PrimKD/models/net_utils.py` L49-77）：

```python
# 双向 rectification：channel weights + spatial weights
channel_weights = ChannelWeights(cat(x1, x2))  # avg+max pool → MLP → sigmoid
spatial_weights = SpatialWeights(cat(x1, x2))   # conv → sigmoid
out_x1 = x1 + λc * w_c[1] * x2 + λs * w_s[1] * x2
out_x2 = x2 + λc * w_c[0] * x1 + λs * w_s[0] * x1
```

### 4.2 为什么这个方向最适合当前项目

1. **直接解决核心矛盾**：当前 RGBD student 的 mIoU (0.517) 低于 RGB-only baseline。PrimKD 通过 KL loss 强制 student 的 logits 接近 teacher（RGB-only），防止 depth 融入后破坏 RGB 特征。
2. **不需要新 backbone**：teacher 使用相同 DFormerV2-S，只需加载 RGB-only 预训练权重。
3. **工程量可控**：新增 `LitDFormerV2MidFusionDistill` 类 + teacher 模型 + 两个 loss 项。
4. **论文故事清晰**："Cross-Modal Distillation for RGB-D Segmentation" — teacher 提供 RGB 信号的 "锚点"，student 学习在不破坏 RGB 的前提下利用 depth。

### 4.3 实现方案

```
teacher: DFormerV2-S (frozen, RGB-only pretrained)
student: DFormerV2-S + ResNet-18 + GatedFusion (现有架构)

Loss = CE(student_logits, gt)
     + α * KL(teacher_logits || student_logits)     # logit distillation
     + β * Σ MSE(student_features[i], teacher_features[i])  # feature alignment

α = 1.0, β = 0.1 (PrimKD 默认)
```

### 4.4 需要新增的代码

1. **`src/models/teacher_model.py`**：冻结的 RGB-only DFormerV2-S wrapper
2. **`src/models/mid_fusion.py`**：新增 `LitDFormerV2MidFusionDistill` 类
3. **`train.py`**：新增 `--teacher_checkpoint`, `--distill_alpha`, `--distill_beta` 参数
4. **`src/models/base_lit.py`**：`training_step` 中添加 distillation loss 计算

### 4.5 预期效果与风险

- **预期**：mIoU 提升 1.5–3%，从 0.517 提升到 0.530–0.545
- **风险**：teacher 质量依赖于 RGB-only 预训练权重；如果 RGB-only 本身就弱，teacher 的信号价值有限
- **验证方式**：先用现有 RGB-only checkpoint 作为 teacher，跑 1 个 run 看趋势

---

## 5. 第二推荐方向：MixPrompt 式 Stage-wise Prompt 注入

### 5.1 源码关键发现

**MixPrompt ADA 模块**（`ref_codes/MixPrompt/semseg/models/Mixprompt.py` L22-55）：

```python
class ADA(nn.Module):
    def __init__(self, dim, down_ratio=4, n=4):
        self.down_rgb = nn.Linear(dim, dim // down_ratio)  # 64→16
        self.down_x = nn.Linear(dim, dim // down_ratio)     # 64→16
        self.mixer_rgb = nn.Linear(n, n, bias=False)        # 4×4 mixing
        self.up = nn.Linear(dim // down_ratio, dim)          # 16→64

    def forward(self, rgb, x):
        rgb_sub = self.down_rgb(rgb).chunk(n, dim=-1)  # [B, N, 4, 4]
        x_sub = self.down_x(x).chunk(n, dim=-1)        # [B, N, 4, 4]
        prompt = self.up(mixer_rgb(rgb_sub) + mixer_x(x_sub))
        return prompt  # same shape as rgb
```

**DPLNet ADA 模块**（`ref_codes/DPLNet/RGBD/toolbox/models/DPLNet.py` L10-23）：

```python
class ADA(nn.Module):
    def __init__(self, dim):
        self.conv0_0 = nn.Linear(dim, dim // 4)  # depth→quarter
        self.conv0_1 = nn.Linear(dim, dim // 4)   # rgb→quarter
        self.conv = nn.Linear(dim // 4, dim)       # quarter→full

    def forward(self, p, x):  # p=depth, x=rgb
        return self.conv(self.conv0_0(p) + self.conv0_1(x))
```

**注入方式**（Mixprompt.py L123-126）：

```python
prompted = self.ada1(x1, d1)   # mix depth into prompt
prompted = self.prompt_norms1(prompted)
x1 = x1 + prompted             # residual add before transformer blocks
```

### 5.2 适配到当前架构的方案

DFormerV2 的 `RGBD_Block` 已经有 `x`（RGB tokens）和 `x_e`（depth）输入。可以在每个 stage 的 transformer blocks 前注入 prompt：

```python
# 在 DFormerv2_S.forward() 中，每个 layer 前：
prompt = self.ada[layer_idx](x, depth_tokens[layer_idx])
x = x + prompt  # residual prompt injection
x = self.layers[layer_idx](x, x_e)  # normal RGBD_Block forward
```

### 5.3 优势与风险

- **优势**：工程量极小（~50 行新代码），不改变现有 encoder 结构，depth 以 "prompt" 形式影响 RGB 而非直接融合
- **风险**：与现有 GeoPriorGen 的 depth-based attention mask 可能存在冗余；效果不确定
- **预期**：mIoU 提升 1–2%

---

## 6. 第三推荐方向：增强 Geometry Prior 利用

### 6.1 源码关键发现

**DFormerV2 已有 geometry prior**（`dformerv2_encoder.py` L116-170）：

```python
class GeoPriorGen(nn.Module):
    def generate_depth_decay(self, H, W, depth_map):
        # 计算 patch 间 depth 差异 → 可学习衰减
        depth_decay = pairwise_depth_diff * self.decay
        return depth_decay

    def forward(self, shape, depth_map, split_or_not):
        position = self._get_position(shape)         # 位置编码
        depth_decay = self.generate_depth_decay(...)  # depth 衰减
        return self.weight[0] * position + self.weight[1] * depth_decay
```

**当前使用方式**（`RGBD_Block.forward()`）：

```python
def forward(self, x, x_e):
    geo = self.Geo((h, w), x_e, split_or_not)  # geometry prior
    # geo 作为 attention mask 使用
    x = self.attn(x, geo_prior=geo)  # attention with depth decay
```

### 6.2 增强方案

当前 depth 仅通过 attention decay mask 间接影响特征。可以增加 **显式 depth-aware feature calibration**：

```python
# 在 RGBD_Block.forward() 中添加：
depth_scale = self.depth_gate(x_e)  # Conv2d → sigmoid
x = x * depth_scale  # channel-wise modulation based on depth
```

或者使用 PrimKD 的 FeatureRectifyModule 做双向 rectification。

### 6.3 优势与风险

- **优势**：改动最小（~20 行），直接增强已有的 depth 利用
- **风险**：与 GeoPriorGen 可能存在信息冗余；提升幅度可能有限
- **预期**：mIoU 提升 0.5–1.5%

---

## 7. 不建议继续投入的方向

### 7.1 Reliability-aware GatedFusion

**理由**：
- 当前 GatedFusion 已经是 `sigmoid gate → weighted sum → refine`，本质上就是一种 "reliability-aware" 融合
- 增加 reliability 估计（如 depth 估计误差图）需要额外监督信号或预训练 depth estimator
- 实验已证明：修改融合方式（CE+Dice、C4 PPM、FFT、InfoNCE 等）均无法稳定超越 baseline
- **结论**：融合层不是瓶颈，继续修改融合层属于 "在错误的地方找答案"

### 7.2 DepthAnythingV2 / PDDM 预训练

**理由**：
- DepthAnythingV2 基于 DINOv2-Large，单次推理就需要 >4GB VRAM
- 将其集成到当前 pipeline 需要：预处理 depth → 替换/增强 DepthEncoder → 重新对齐特征维度
- PDDM (EMSANet) 使用多任务学习框架，工程量极大
- **结论**：工程量过大，不适合本科毕设时间线（~2 个月）

### 7.3 继续 recipe tuning（学习率、loss、decoder 替换）

**理由**：
- 已完成 20+ 实验，覆盖 CE/Dice/CE+Dice、不同学习率、PPM、FFT、HiLo Enhance、InfoNCE 等
- 结果分布集中在 0.507–0.517，与 baseline (0.517±0.005) 无显著差异
- **结论**：recipe tuning 的边际收益已耗尽

---

## 8. 给 Codex 接力的任务清单

### Phase 1: PrimKD 式知识蒸馏（优先级最高）

- [ ] **T1.1** 创建 `src/models/teacher_model.py`：
  - `FrozenRGBTeacher` 类，加载 DFormerv2_S pretrained weights
  - `forward(x)` 返回 `(logits, [feat1, feat2, feat3, feat4])`
  - 所有参数 `requires_grad=False`

- [ ] **T1.2** 修改 `src/models/mid_fusion.py`：
  - 新增 `LitDFormerV2MidFusionDistill(LitDFormerV2MidFusion)` 类
  - `training_step` 中添加 KL loss + feature MSE loss
  - 新增 `teacher_checkpoint`, `distill_alpha`, `distill_beta` 参数

- [ ] **T1.3** 修改 `train.py`：
  - 添加 `--teacher_checkpoint`, `--distill_alpha`, `--distill_beta` CLI 参数
  - `build_model` 中处理 teacher 模型加载

- [ ] **T1.4** 运行实验：
  - Teacher: 使用现有 `dformerv2_rgb_only` pretrained checkpoint
  - Student: 现有 `dformerv2_mid_fusion` baseline 配置
  - 配置：`α=1.0, β=0.1, lr=6e-5, epochs=50, batch_size=2`
  - 跑 3 个 run，记录 best mIoU

### Phase 2: MixPrompt 式 Prompt 注入（如果 Phase 1 有效）

- [ ] **T2.1** 在 `dformerv2_encoder.py` 中添加 `StagePromptADA` 模块
- [ ] **T2.2** 在 `DFormerv2_S.forward()` 每个 stage 前注入 prompt
- [ ] **T2.3** 配置：4 个 stage 各一个 ADA，dim = embed_dims[i]
- [ ] **T2.4** 跑 3 个 run 验证

### Phase 3: 组合实验（如果 Phase 1+2 均有效）

- [ ] **T3.1** 将 PrimKD distillation + MixPrompt 组合
- [ ] **T3.2** 跑 5-run repeat，写入 `miou_list/` 目录
- [ ] **T3.3** 对比 baseline 5-run repeat (mean=0.5119)

### 实验记录规范

每个实验完成后，在 `miou_list/` 下创建 `{model}_{change}_{runN}.md`，格式参照 `dformerv2_mid_fusion_gate_baseline_repeat5_run01.md`。

---

## 附录：源码文件索引

| 方向 | 关键文件 | 路径 |
|------|----------|------|
| PrimKD distillation | train.py | `ref_codes/PrimKD/train.py` |
| PrimKD distillation | builder.py | `ref_codes/PrimKD/models/builder.py` |
| PrimKD distillation | net_utils.py | `ref_codes/PrimKD/models/net_utils.py` |
| MixPrompt | Mixprompt.py | `ref_codes/MixPrompt/semseg/models/Mixprompt.py` |
| MixPrompt | mspa.py | `ref_codes/MixPrompt/semseg/models/modules/mspa.py` |
| DPLNet | DPLNet.py | `ref_codes/DPLNet/RGBD/toolbox/models/DPLNet.py` |
| DFormerV2 | dformerv2_encoder.py | `src/models/dformerv2_encoder.py` |
| 当前融合 | mid_fusion.py | `src/models/mid_fusion.py` |
| 当前解码器 | decoder.py | `src/models/decoder.py` |
| PDDM/EMSANet | model.py | `ref_codes/PDDM/EMSANet/emsanet/model.py` |
