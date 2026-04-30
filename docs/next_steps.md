# 下一步计划

本文档记录当前实验优先级。

每个方向都要围绕“为什么做、改哪里、怎么验证、怎么记录结论”推进，
避免为了堆模块而堆模块。

## 第一优先级：Depth Adapter

### 目的

提升 depth branch 在进入 multi-level RGB-D mid-fusion 前的特征质量。

目标是让 depth 信息更适合与 DINOv2-small 的 RGB 语义特征交互。

### 为什么值得做

当前主线使用 DINOv2-small + Swin-Tiny，depth branch 相对轻量。

直接融合可能无法充分利用 depth 的结构信息。Depth Adapter 是最直接、
最可控的改进点，可以验证“depth 特征预处理/校准”是否比继续堆 fusion
模块更有效。

### 可能需要改哪些代码文件

- `framework_download/src/models/encoder.py`
- `framework_download/src/models/mid_fusion.py`
- `framework_download/src/models/decoder.py`
- `framework_download/train.py`，仅当需要新增明确实验参数时再改
- `framework_download/eval.py`，仅当需要新增明确评估输出时再改

### 应该用什么指标验证

- best mIoU
- mean mIoU 和 7 runs 稳定性
- 每类 IoU，重点看依赖空间结构的类别
- 收敛曲线和训练稳定性
- 参数量和显存变化

### 成功/失败后怎么记录结论

- 成功：写入 `docs/experiment_log.md` 的有效实验记录。
- 成功：在 `docs/model_changes.md` 更新 Depth Adapter 结构和插入位置。
- 失败：写入 `docs/experiment_log.md` 的失败记录。
- 失败：说明是无提升、训练不稳定、过拟合，还是显存/速度不划算。
- 待观察：记录已跑配置和缺失证据，不提前写成论文贡献。

## 第二优先级：融合模块改进

### 目的

改进 multi-level RGB-D mid-fusion 的跨模态交互方式。

目标是让 RGB 和 depth 在多个尺度上更有效地互补。

### 为什么值得做

Fusion 是当前模型主线，但必须围绕明确优点设计。

可考虑的优点包括更稳定的门控、更清楚的跨模态注意力、
更低成本的局部交互，或者更可解释的残差贡献。

不要为了拼模块而拼模块。

### 可能需要改哪些代码文件

- `framework_download/src/models/mid_fusion.py`
- `framework_download/src/models/decoder.py`
- `framework_download/src/models/encoder.py`，仅当 fusion 输入层级或通道需要同步时再改
- `framework_download/train.py`，仅当需要新增明确实验参数时再改

### 应该用什么指标验证

- best mIoU
- 7 runs 的 mean mIoU 和方差
- 单次最高点是否可复现
- 参数量、显存占用、训练速度
- 与 Context-FPN、ResGamma、Depth Adapter 的组合收益

### 成功/失败后怎么记录结论

- 成功：记录模块机制、替换位置、对比实验和是否进入论文主线。
- 失败：记录失败原因，尤其是无明确机制、只增加复杂度、单次有效但不稳定。
- 所有结论都要同步到 `docs/experiment_log.md` 和 `docs/model_changes.md`。

## 第三优先级：论文叙事整理

### 目的

把当前实验主线整理成可写论文的逻辑。

重点包括动机、相关工作、方法设计、消融实验和无效方案边界。

### 为什么值得做

当前项目已经有明确主线和当前最好结果，但还需要把实验选择转成论文叙事。

论文不能只罗列模块，必须解释为什么选择 DINOv2-small + Swin-Tiny，
为什么优先做 Depth Adapter，为什么 deprecated 的大模型结果不能引用。

### 可能需要改哪些代码文件

- 通常不需要改代码。
- 如果论文叙事暴露出缺失消融，再回到明确代码文件做实验。
- 可能涉及 `framework_download/src/models/mid_fusion.py`。
- 可能涉及 `framework_download/src/models/decoder.py`。
- 可能涉及 `framework_download/src/models/encoder.py`。

### 应该用什么指标验证

- 消融表是否覆盖 Depth Adapter、Context-FPN、ResGamma、fusion 模块。
- 每个论文贡献是否有对应实验支撑。
- deprecated/invalid 记录是否被排除在论文结果之外。
- 当前最好结果是否仍是 Context-FPN ResGamma，best mIoU about `0.3933`。
- 除非新结果有配置、日志、checkpoint 或用户明确确认，否则不得替换当前最好结果。

### 成功/失败后怎么记录结论

- 成功：更新 `docs/paper_notes.md` 的动机、方法主线、创新点和消融实验设计。
- 失败：记录缺少哪些实验支撑，不能把未经验证的模块写成贡献。
- 如果论文叙事和实验记录冲突，优先修正 `docs/experiment_log.md` 和 `docs/model_changes.md`。
