# 模型结构变更记录

本文档记录当前 RGB-D semantic segmentation 项目的结构主线、
模块改动和弃用方案。

它的目标不是写论文正文，而是让下一次实验能快速知道：
现在模型长什么样、为什么这样改、哪些旧方案不能再用。

## 当前主线

- 记录日期：2026-04-30
- 任务：RGB-D semantic segmentation
- 数据集：NYUDepthV2
- RGB branch：DINOv2-small
- Depth branch：Swin-Tiny
- Fusion：multi-level RGB-D mid-fusion
- 当前改进重点：Context-FPN、ResGamma、Depth Adapter、融合模块
- 当前可确认最好结果：Context-FPN ResGamma 7 runs，best mIoU about `0.3933`

## 模块状态

| 模块方向 | 当前状态 | 作用 | 记录重点 |
|---|---|---|---|
| RGB branch | 当前主线 | 使用 DINOv2-small 提取 RGB 语义特征。 | 记录是否冻结、输出层级、fusion 输入维度。 |
| Depth branch | 当前主线 | 使用 Swin-Tiny 提取 depth 结构特征。 | 记录是否经过 adapter，是否改善 mIoU。 |
| Multi-level mid-fusion | 当前主线 | 在多个尺度融合 RGB 与 depth 特征。 | 记录层级、融合方式、参数量和稳定性。 |
| Context-FPN | 正在改进 | 增强多尺度上下文和 decoder 侧表达。 | 记录是否提升 best mIoU，是否带来训练不稳定。 |
| ResGamma | 正在改进 | 调整残差分支贡献，控制融合强度。 | 记录 gamma 初始化、收敛变化和重复实验方差。 |
| Depth Adapter | 第一优先级 | 在融合前提升 depth branch 的可用性。 | 记录 adapter 结构、插入位置、mIoU 和类别表现。 |

## 弃用 / 不成立方案

| 方案 | 旧记录结果 | 状态 | 原因 | 使用限制 |
|---|---|---|---|---|
| Swin-B RGB + DINOv2-B Depth | about `0.48-0.49` mIoU | invalid / deprecated | 预训练模型太大，当前环境无法正常使用。 | 不作为有效版本、论文主线、baseline 或当前最好结果。 |

补充说明：

- 该方案曾出现在旧 README 中。
- 该方案不是当前代码主线下稳定可复现的结果。
- 旧记录 `0.48-0.49` 只能作为 deprecated/invalid 记录保留。
- 后续论文和实验表不得引用它作为有效结果。

## 结构改动记录规则

- 每次改动 fusion、Depth Adapter、Context-FPN、ResGamma、encoder 或 decoder 后，都要更新本文档。
- 新模块必须写清楚目的和预期优点，不能只写“加入某模块”。
- 如果模块失败，要记录失败现象、可能原因、是否继续保留。
- 如果模块被弃用，要放入“弃用 / 不成立方案”，不能从文档里直接消失。

## 2026-05-02 Teacher Code RGB Encoder Change

- Scope: `老师原版代码/src/models/encoder.py`.
- Change: replaced the teacher RGB ResNet-18 encoder with local DINOv2-small loaded from `C:\Users\qintian\Desktop\qintian\pretrained\dinov2_small`.
- Output contract: kept four feature maps and the original `[64, 128, 256, 512]` channel contract through 1x1 projection, so `MidFusionSegmentor`, `GatedFusion`, and `SimpleFPNDecoder` stay unchanged.
- Unchanged: `DepthEncoder` and `EarlyFusionEncoder` remain ResNet-18 based.
- Status: structure change only; no training or evaluation result is recorded here.

## 2026-05-02 Teacher Code RGB Encoder Source Change

- Scope: `老师原版代码/src/models/encoder.py`.
- Change: switched the RGB encoder from HuggingFace `Dinov2Model.from_pretrained(...)` to the local GitHub DINOv2 source tree at `C:\Users\qintian\Desktop\qintian\pretrained\dinov2-main`.
- Backbone: `dinov2_vits14(pretrained=False)`, so no downloaded pretrained weights are loaded from `pretrained/dinov2_small`.
- Output contract: still returns four projected feature maps with `[64, 128, 256, 512]` channels for the existing mid-fusion and FPN decoder.
- Shape handling: RGB input is padded to a multiple of DINOv2 patch size 14 before the local backbone, then projected feature maps are resized back to the original FPN scales.
- Status: structure/source change only; no training or evaluation result is recorded here.

## 2026-05-02 Teacher Code RGB Encoder Pretrained Restore

- Scope: `老师原版代码/src/models/encoder.py`.
- Change: restored RGB encoder loading from the local HuggingFace-format DINOv2-small directory `C:\Users\qintian\Desktop\qintian\pretrained\dinov2_small`.
- Backbone: `Dinov2Model.from_pretrained(..., local_files_only=True)`, using the downloaded pretrained `pytorch_model.bin` in that directory.
- Reason: the GitHub source-tree variant used `dinov2_vits14(pretrained=False)`, which trained from random initialization and produced lower validation mIoU (`0.2515` best in `dinov2_git_random_rgb_run01`).
- Status: structure/source restore only; no new training result is recorded here.

## 2026-05-02 Teacher Original Code RGB Encoder Restart

- Scope: `teacher's daima/src/models/encoder.py`.
- Change: starting from the teacher original code directory, replaced only `RGBEncoder` from ResNet-18 to local Hugging Face `Dinov2Model`.
- Local pretrained path: `C:\Users\qintian\Desktop\qintian\pretrained\dinov2_small`.
- Backbone details: DINOv2-small, hidden size `384`, patch size `14`, hidden state indices `(3, 6, 9, 12)`.
- Output contract: preserved four feature maps with the teacher original `[64, 128, 256, 512]` channel contract and original ResNet-style spatial scales.
- Unchanged: `DepthEncoder`, `EarlyFusionEncoder`, `GatedFusion`, `SimpleFPNDecoder`, `train.py`, `data_module.py`, loss, optimizer, and scheduler.
- Verification: `py_compile` passed for `src/models/encoder.py`; dummy forward with `rgb=(1,3,480,640)` and `depth=(1,1,480,640)` produced logits `(1,40,480,640)`.
- Status: structure replacement and dummy check only; no training or mIoU result is recorded here.

## 2026-05-02 Teacher Original Code RGB Encoder Restore

- Scope: `teacher's daima/src/models/encoder.py`.
- Change: restored `RGBEncoder` from local DINOv2-small back to the teacher original ResNet-18 implementation.
- Removed: `transformers.Dinov2Model`, `torch.nn.functional`, hidden-state reshaping, CLS handling, and 1x1 DINOv2 projection layers.
- Output contract: ResNet-18 four feature maps with `[64, 128, 256, 512]` channels and spatial shapes `(120,160)`, `(60,80)`, `(30,40)`, `(15,20)` for `480x640` input.
- Unchanged: `DepthEncoder`, `EarlyFusionEncoder`, `GatedFusion`, `SimpleFPNDecoder`, `train.py`, `data_module.py`, loss, optimizer, and scheduler.
- Verification: `py_compile` passed for `src/models/encoder.py`; dummy forward with `rgb=(1,3,480,640)` and `depth=(1,1,480,640)` produced logits `(1,40,480,640)`.
- Status: restore and dummy check only; no training result is recorded here.

## 2026-05-03 DFormerv2 Reliability Geometry Fusion

- Scope: `teacher's daima/src/models/fusion_blocks.py`, `teacher's daima/src/models/mid_fusion.py`, `teacher's daima/train.py`.
- Change: added `ReliabilityGeometryFusion` and registered `--model dformerv2_reliability_fusion` for ablation against the existing `dformerv2_mid_fusion`.
- Structure: keeps `DFormerv2_S` backbone, teacher original ResNet-18 `DepthEncoder`, and teacher original `SimpleFPNDecoder`; only replaces the four-stage fusion blocks in the new model.
- Fusion design: uses DFormerv2 feature as the main feature, ResNet depth feature as auxiliary geometry, raw depth as a lightweight geometry prior, and residual output `rgbd_feat + gamma * residual` with `gamma=1e-3`.
- Source attribution: `fusion_blocks.py` records inspiration from DFormer/DFormerv2, SA-Gate, TokenFusion, CMX, and ShapeConv.
- Unchanged: existing `dformerv2_mid_fusion`, original `GatedFusion`, DFormerv2 encoder body, `DepthEncoder`, decoder, DataModule, loss, and label protocol.
- Verification: `py_compile` passed for `src/models/fusion_blocks.py`, `src/models/mid_fusion.py`, and `train.py`.
- Status: structure change only; no training result is recorded here.

## 2026-05-03 DFormerv2 Attention Fusion V2

- Scope: `teacher's daima/src/models/fusion_blocks.py`, `teacher's daima/src/models/mid_fusion.py`, `teacher's daima/train.py`.
- Change: added `CrossModalReliabilityAttentionFusion` and registered `--model dformerv2_attention_fusion`.
- Structure: keeps `DFormerv2_S` backbone, teacher original ResNet-18 `DepthEncoder`, and teacher original `SimpleFPNDecoder`; only replaces the four-stage fusion blocks in the new model.
- Fusion design: adds explicit channel reliability attention, spatial reliability attention, cross-modal depth/prior rectification, raw depth geometry prior, and residual output `rgbd_feat + gamma * residual` with `gamma=1e-3`.
- Source attribution: `fusion_blocks.py` records inspiration from DFormer/DFormerv2, SA-Gate, TokenFusion, CMX, and ShapeConv.
- Unchanged: existing `dformerv2_mid_fusion`, existing `dformerv2_reliability_fusion`, original `GatedFusion`, DFormerv2 encoder body, `DepthEncoder`, decoder, DataModule, loss, and label protocol.
- Verification: `py_compile` passed for `src/models/fusion_blocks.py`, `src/models/mid_fusion.py`, and `train.py`.
- Status: structure change only; no training result is recorded here.

## 2026-05-03 DFormerv2 Clean Depth Fusion

- Scope: `teacher's daima/src/models/encoder.py`, `teacher's daima/src/models/mid_fusion.py`, `teacher's daima/train.py`.
- Change: added `CleanShapeDepthEncoder` and registered `--model dformerv2_clean_depth_fusion`.
- Structure: keeps `DFormerv2_S` backbone, original `GatedFusion`, and teacher original `SimpleFPNDecoder`; only replaces the depth branch in the new model.
- Depth design: cleans raw depth with `nan_to_num`, Sobel edge, local shape residual, confidence-gated clean depth, four-channel ResNet-18 input, and per-stage `DepthResidualRefineBlock` with edge prior residual links.
- Source attribution: implementation is inspired by DFormerv2 raw depth geometry prior, ShapeConv local geometry cues, SA-Gate noisy depth filtering, CMX rectification, and TokenFusion dynamic selection; no reference class was directly copied.
- Unchanged: existing `dformerv2_mid_fusion`, existing `dformerv2_attention_fusion`, original `DepthEncoder`, original `GatedFusion`, DFormerv2 encoder body, decoder, DataModule, loss, and label protocol.
- Verification: `py_compile` passed for `src/models/encoder.py`, `src/models/mid_fusion.py`, and `train.py`.
- Status: structure change only; no training result is recorded here.

## 2026-05-03 Clean Depth Shape-SA-Gate Refinement

- Scope: `teacher's daima/src/models/encoder.py`.
- Change: strengthened `CleanShapeDepthEncoder` without changing the model registry or decoder/fusion path.
- ShapeConv reference: borrowed the base/shape decomposition idea from `ShapeConv-master/rgbd_seg/models/utils/shape_conv.py`, but did not copy the full `ShapeConv2d` replacement layer.
- SA-Gate reference: adapted the lightweight `FilterLayer`, `FSP`, and two-way softmax gate pattern from `RGBD_Semantic_Segmentation_PyTorch-master/model/SA-Gate.nyu/net_util.py`.
- Depth design: uses multi-scale depth bases, learnable base/shape gains, reliability recalibration, and per-stage edge-guided feature separation/aggregation with small residual `gamma=1e-3`.
- Unchanged: existing `dformerv2_mid_fusion`, existing `dformerv2_attention_fusion`, DFormerv2 encoder body, original `GatedFusion`, decoder, DataModule, loss, and label protocol.
- Verification: `py_compile` passed for `src/models/encoder.py`, `src/models/mid_fusion.py`, and `train.py`.
- Status: structure refinement; first completed run is recorded in `docs/experiment_log.md` as `dformerv2_clean_depth_fusion_shape_sagate_run01`.

## 2026-05-03 Restore DFormerv2 Mid Fusion Stable Baseline

- Scope: `teacher's daima/src/models/encoder.py`, `teacher's daima/src/models/mid_fusion.py`, `teacher's daima/train.py`.
- Change: restored the active code entry to the stable `dformerv2_mid_fusion` baseline after the clean-depth single run showed no clear stable gain.
- Active model: `DFormerv2_S` backbone, original ResNet-18 `DepthEncoder`, original `GatedFusion`, and teacher original `SimpleFPNDecoder`.
- Removed from active code: experimental `dformerv2_reliability_fusion`, `dformerv2_attention_fusion`, `dformerv2_clean_depth_fusion`, and the unused `fusion_blocks.py` file.
- Preserved: TensorBoard records, mIoU markdown files, and experiment notes for the removed experimental branches.
- Verification: `py_compile` passed for `src/models/encoder.py`, `src/models/mid_fusion.py`, and `train.py`.
- Status: code is back to the stable `dformerv2_mid_fusion` baseline for continued experiments.

## 2026-05-03 DFormerv2 CMX Fusion

- Scope: `teacher's daima/src/models/fusion_blocks.py`, `teacher's daima/src/models/mid_fusion.py`, `teacher's daima/train.py`.
- Change: added pure CMX-style fusion as `--model dformerv2_cmx_fusion`.
- Main logic: replaces only the original fusion block with CMX-style feature rectification and feature fusion; keeps `DFormerv2_S`, original ResNet-18 `DepthEncoder`, and `SimpleFPNDecoder` unchanged.
- CMX source: adapted `FeatureRectifyModule` and `FeatureFusionModule` from `RGBX_Semantic_Segmentation-main/models/net_utils.py`.
- Preserved old entries: `dformerv2_mid_fusion`, `dformerv2_reliability_fusion`, `dformerv2_attention_fusion`, and `dformerv2_clean_depth_fusion` are registered.
- Verification: `py_compile` passed for `src/models/fusion_blocks.py`, `src/models/mid_fusion.py`, `train.py`, and `src/models/encoder.py`.
- Status: structure change only; no training result is recorded here.

## 2026-05-03 Deprecate DFormerv2 CMX Fusion

- Scope: `teacher's daima/src/models/fusion_blocks.py`, `teacher's daima/src/models/mid_fusion.py`, `teacher's daima/train.py`.
- Change: removed the pure CMX fusion branch from active code after `dformerv2_cmx_fusion_run01` underperformed.
- Removed: CMX-related classes in `fusion_blocks.py`, `DFormerV2CMXFusionSegmentor`, `LitDFormerV2CMXFusion`, and the `dformerv2_cmx_fusion` registry entry.
- Preserved: `dformerv2_mid_fusion`, `dformerv2_reliability_fusion`, `dformerv2_attention_fusion`, and `dformerv2_clean_depth_fusion`.
- Evidence: `dformerv2_cmx_fusion_run01` best `val/mIoU=0.503343`, lower than repeated `dformerv2_mid_fusion` mean best `0.513406`.
- Status: deprecated / negative result; do not continue this exact full CMX-style symmetric fusion branch.

## 2026-05-03 DFormerv2 PrimKD Primary-Guided Fusion

- Scope: `teacher's daima/src/models/fusion_blocks.py`, `teacher's daima/src/models/mid_fusion.py`, `teacher's daima/train.py`.
- Change: added PrimKD-inspired asymmetric primary-guided fusion as `--model dformerv2_primkd_fusion`.
- Main logic: keeps DFormerv2 features as the primary modality and uses ResNet-18 depth features only to generate selected auxiliary residual support.
- PrimKD source: adapted the 1x1 feature alignment idea from `PrimKD/models/decoders/MLPDecoder.py::DecoderHead_hint`, the two-layer 3x3 reconstruction block and `lambda_mask=0.75` masked generation idea from `PrimKD/models/decoders/MLPDecoder.py::DecoderHead_mask`, and the selective feature supervision idea from `PrimKD/train.py`.
- Adaptation boundary: no PrimKD teacher-student training loop, KL loss, MSE distillation loss, or random mask training branch was added.
- Preserved: `dformerv2_mid_fusion`, `dformerv2_reliability_fusion`, `dformerv2_attention_fusion`, and `dformerv2_clean_depth_fusion`.
- Verification: `py_compile` passed for `src/models/fusion_blocks.py`, `src/models/mid_fusion.py`, and `train.py`.
- Status: structure change only; no training result is recorded here.
