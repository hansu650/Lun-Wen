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
