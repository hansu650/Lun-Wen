# 论文笔记

本文档记录论文写作相关的事实和边界。

目标是避免把无效实验、旧 README 信息或未经确认的数字写进论文。

## 当前论文动机

RGB-D semantic segmentation 需要同时利用 RGB 的语义纹理信息和 depth 的空间结构信息。

当前项目的核心问题是：
在轻量化、当前环境可运行的前提下，如何让 depth branch 的信息真正参与多层融合，
而不是只增加参数或堆模块。

当前论文叙事应围绕三个点展开：

- DINOv2-small 提供较强 RGB 语义特征，但单独依赖 RGB 不足以利用 NYUDepthV2 的空间结构。
- Swin-Tiny 作为 depth branch 更符合当前环境限制。
- Multi-level RGB-D mid-fusion、Context-FPN、ResGamma、Depth Adapter 应服务于更有效的跨模态融合。

## 当前方法主线

- 任务：RGB-D semantic segmentation
- 数据集：NYUDepthV2
- RGB branch：DINOv2-small
- Depth branch：Swin-Tiny
- Fusion：multi-level RGB-D mid-fusion
- 当前重点模块：Context-FPN、ResGamma、Depth Adapter、融合模块
- 当前可确认最好结果：Context-FPN ResGamma，best mIoU about `0.3933`

## 可写成创新点的模块

| 模块 | 可写角度 | 需要证明的问题 |
|---|---|---|
| Depth Adapter | 在融合前校准或增强 depth features。 | 是否比直接融合 depth features 更好；是否提升 mIoU。 |
| Context-FPN | 在多尺度 decoder/fusion 路径中增强上下文表达。 | 是否提升整体 mIoU；是否对小物体、边界或场景结构有帮助。 |
| ResGamma | 用可控残差强度稳定融合模块贡献。 | 是否提升重复实验稳定性；是否减少无效 depth 分支干扰。 |
| 融合模块改进 | 围绕明确优点设计跨模态交互。 | 是否有清晰机制、消融收益和可解释失败结论。 |

## 消融实验设计

| 消融项 | 对照设置 | 观察指标 | 结论记录方式 |
|---|---|---|---|
| Depth Adapter | 无 adapter vs 加入 adapter | best mIoU、mean mIoU、重复实验方差、关键类别 IoU | 写清楚 adapter 是否让 depth branch 更有效。 |
| Context-FPN | 原 decoder/fusion 路径 vs Context-FPN | best mIoU、边界/小目标表现、训练稳定性 | 记录是否带来稳定收益，不能只看单次最高点。 |
| ResGamma | 无 gamma 控制 vs 加 ResGamma | best mIoU、重复实验稳定性、收敛曲线 | 记录 gamma 是否帮助稳定融合贡献。 |
| 融合模块 | 当前 fusion baseline vs 新 fusion 设计 | best mIoU、参数量、速度、重复实验稳定性 | 只有明确机制和收益时才进入论文主线。 |

## 不能写进论文结果的内容

- Swin-B RGB + DINOv2-B Depth 是 deprecated/invalid 记录。
- 旧 README 中的 `0.48-0.49` mIoU 是 deprecated/invalid 记录。
- 上述方案依赖的预训练模型太大，当前环境无法正常使用。
- 上述方案不属于当前可复现实验结果。
- 它不得作为论文结果、有效 baseline、当前最好结果或主线方案引用。
- 任何新结果必须有配置、日志、checkpoint，或用户明确确认。
- 只有满足上一条的新结果，才能进入 `docs/experiment_log.md` 的有效实验表。

## 写作边界

- 论文中的最好结果目前只能写 Context-FPN ResGamma，best mIoU about `0.3933`。
- 新模块如果没有明确实验支撑，只能写成待验证想法。
- 未经验证的模块不能写成已证明贡献。
- 失败尝试可以用于解释设计取舍，但必须标注为失败或弃用。

## 下一步写作需要补充的证据

| 证据 | 为什么需要 | 对应文档 |
|---|---|---|
| Context-FPN ResGamma 7 runs 的 checkpoint 路径和日志路径 | 支撑当前 best mIoU about `0.3933`。 | `docs/experiment_log.md` |
| Depth Adapter 的明确结构和插入位置 | 说明它如何增强 depth features。 | `docs/model_changes.md` |
| Depth Adapter 的消融结果 | 验证第一优先级方向是否提升 mIoU 和稳定性。 | `docs/experiment_log.md` |
| Context-FPN、ResGamma、融合模块的独立消融 | 区分各模块贡献，避免论文只写组合结果。 | `docs/experiment_log.md`、`docs/paper_notes.md` |
| 失败实验和 deprecated 方案说明 | 解释为什么不采用大模型旧记录，保护论文结果边界。 | `docs/model_changes.md`、`docs/experiment_log.md` |
