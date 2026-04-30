# 实验记录

本文档只记录可以跨会话继续使用的实验结论。

任何结果进入“有效实验记录”之前，必须有当前环境下的配置、日志、
checkpoint，或用户明确确认。

## 有效实验记录

| 日期 | 实验名称 | 配置 | 结果 | 状态 | 备注 |
|---|---|---|---|---|---|
| 2026-04-30 | Context-FPN ResGamma 7 runs | DINOv2-small + Swin-Tiny；NYUDepthV2；multi-level RGB-D mid-fusion | best mIoU about `0.3933` | 当前有效最好结果 | 用户已确认；后续新结果必须超过或解释该记录。 |

记录细节：

- RGB branch：DINOv2-small
- Depth branch：Swin-Tiny
- Dataset：NYUDepthV2
- Fusion：multi-level RGB-D mid-fusion
- 当前结论：Context-FPN ResGamma 7 runs 是当前可确认最好结果。

## 无效或弃用记录

| 记录日期 | 记录内容 | 旧记录结果 | 状态 | 原因 | 后续处理 |
|---|---|---|---|---|---|
| 2026-04-30 | 旧 README 中的 Swin-B RGB + DINOv2-B Depth | about `0.48-0.49` mIoU | 无效 / 不采用 / 不可复现 | 预训练模型过大，当前环境无法使用。 | 只能作为 deprecated/invalid 记录保留。 |

弃用说明：

- 该记录不属于当前代码主线下可复现实验结果。
- 该记录不能进入有效实验表。
- 该记录不能作为论文 baseline。
- 该记录不能作为当前最好结果引用。

## 每次实验必须补充的信息

- 日期：使用绝对日期，例如 `2026-04-30`。
- 实验名称：和 checkpoint/log 目录名保持一致。
- 模型配置：RGB branch、Depth branch、Fusion、decoder/head、改动模块。
- 训练配置：数据集、epoch、batch size、learning rate、关键开关。
- 结果：best mIoU、重复实验稳定性、是否超过当前最好结果。
- 证据：checkpoint 路径、log 路径、必要时附 eval 命令。
- 结论：成功、失败或待观察，并说明原因。
- 下一步：保留一个可以直接继续执行的实验方向。

## 记录原则

- 不把一次性猜测写成实验结论。
- 不把没有日志和 checkpoint 的数字写成当前最好结果。
- 失败实验也要记录，但要写清楚失败原因和是否值得继续。
- deprecated/invalid 记录必须和有效实验表分开，避免以后误用。
