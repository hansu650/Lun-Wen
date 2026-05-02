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

## 官方 DFormer 复现记录

| 日期 | 实验名称 | 配置 | 结果 | 状态 | 证据 |
|---|---|---|---|---|---|
| 2026-04-30 | Official DFormerv2-Small NYUDepthV2 eval clean environment | 官方 `DFormer/local_configs/NYUDepthv2/DFormerv2_S.py`；checkpoint `dformer_work/checkpoints/trained/DFormerv2_Small_NYU.pth`；dataset `dformer_work/datasets/NYUDepthv2`；独立 `dformer` conda 环境；torch `2.7.1+cu126`；单卡 RTX 4090；direct `python utils/eval.py` | mIoU `50.55`，mAcc `65.32`，mF1 `64.95` | 官方外部 baseline 已在干净环境复现 | `dformer_work/checkpoints/NYUDepthv2_DFormerv2_S_20260430-214400/log_2026_04_30_21_44_00.log` |

说明：

- 该记录是官方 DFormer 外部基线复现，不属于 `framework_download/` 当前自研模型主线。
- 该记录不能替换当前有效最好结果 `Context-FPN ResGamma 7 runs`。
- `qintian-rgbd` 临时补齐 DFormer 依赖后的验证结果为 mIoU `50.52`；随后已卸载临时依赖，保持 `qintian-rgbd` 干净。
- 最终官方复现证据以独立 `dformer` 环境日志为准。

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

## 2026-05-02 Teacher Original Code DINOv2 RGB Dummy Check

- Date: `2026-05-02`.
- Scope: `teacher's daima/`.
- Configuration: RGB branch replaced with local DINOv2-small only; depth branch remains teacher original ResNet-18; fusion remains `GatedFusion`; decoder remains `SimpleFPNDecoder`.
- Local pretrained path: `C:\Users\qintian\Desktop\qintian\pretrained\dinov2_small`.
- Check command: `D:\Anaconda\envs\qintian-rgbd\python.exe -m py_compile src\models\encoder.py`.
- Dummy input: `rgb=(1,3,480,640)`, `depth=(1,1,480,640)`.
- Observed RGB feature shapes: `(1,64,120,160)`, `(1,128,60,80)`, `(1,256,30,40)`, `(1,512,15,20)`.
- Observed model output: logits `(1,40,480,640)`.
- Evidence path: code file `teacher's daima/src/models/encoder.py`; dummy forward terminal output in this Codex run.
- Conclusion: compile and dummy forward passed; this is not a training run and records no mIoU result.
- Next step: run a real NYUDepthV2 training job from `teacher's daima/` if this clean RGB-only replacement is accepted.

## 2026-05-02 Teacher DINOv2 RGB Clean Run01

- Date: `2026-05-02`.
- Experiment name: `teacher_dinov2_rgb_clean_run01`.
- Code path: `teacher's daima/`.
- Configuration: local DINOv2-small RGB branch, teacher original ResNet-18 depth branch, teacher original `GatedFusion`, teacher original `SimpleFPNDecoder`.
- Training configuration: NYUDepthV2 folder data, `max_epochs=50`, `batch_size=2`, `lr=6e-5`, `num_workers=0`, GPU 1 device.
- Result: best `val/mIoU=0.318669` at epoch `46`.
- Evidence: checkpoint `teacher's daima/checkpoints/teacher_dinov2_rgb_clean_run01/mid_fusion-epoch=46-val/mIoU=0.3187.ckpt`; event log `teacher's daima/checkpoints/teacher_dinov2_rgb_clean_run01/lightning_logs/version_0/events.out.tfevents.1777729070.Administrator.18036.0`.
- Per-epoch record: `miou_list/teacher_dinov2_rgb_clean_run01.md`.
- Curve observation: reached `0.311150` by epoch `18`, about `97.64%` of the run best; later epochs mostly produced small oscillating gains.
- Conclusion: valid completed run, but not better than the current confirmed best `0.3933`; keep as clean teacher-code DINOv2 RGB baseline.
- Next step: use this as a sanity baseline only; stronger results still need the active `framework_download/` main line.

## 2026-05-02 V3 Gated Anchor ContextFPN ResGamma Confirm Run01

- Date: `2026-05-02`.
- Experiment name: `v3_gated_anchor_contextfpn_resgamma_confirm_run01`.
- Code path: `framework_download/`.
- Configuration: active mid-fusion main line with gated anchor, Context-FPN, and ResGamma naming from checkpoint directory.
- Training configuration: `data_root=C:\Users\qintian\Desktop\qintian\data\NYUDepthv2_matdepth`, `max_epochs=50`, `batch_size=2`, `lr=6e-5`, `num_workers=0`, GPU 1 device.
- Result: best `val/mIoU=0.381563` at epoch `41`.
- Evidence: checkpoint `framework_download/checkpoints/v3_gated_anchor_contextfpn_resgamma_confirm_run01/mid_fusion-epoch=41-val/mIoU=0.3816.ckpt`; event log `framework_download/checkpoints/v3_gated_anchor_contextfpn_resgamma_confirm_run01/lightning_logs/version_0/events.out.tfevents.1777544782.Administrator.39196.0`.
- Per-epoch record: `miou_list/v3_gated_anchor_contextfpn_resgamma_confirm_run01.md`.
- Curve observation: reached `0.363164` by epoch `14`, about `95.18%` of run best; reached `0.380078` by epoch `29`, about `99.61%` of run best; later epochs mostly fluctuated around the plateau.
- Conclusion: valid completed run, but below the current confirmed best `0.3933`; shows early plateau with only small late-epoch gains.
- Next step: do not claim improvement from this run; use repeated runs or protocol changes only if trying to beat the current confirmed best.

## 2026-05-02 Teacher Original ResNet Run01

- Date: `2026-05-02`.
- Experiment name: `teacher_resnet_original_run01`.
- Code path: `teacher's daima/`.
- Configuration: teacher original ResNet-18 RGB branch, teacher original ResNet-18 depth branch, teacher original `GatedFusion`, teacher original `SimpleFPNDecoder`.
- Training configuration: NYUDepthV2 folder data, `max_epochs=50`, `batch_size=2`, `lr=6e-5`, `num_workers=0`, GPU 1 device.
- Result: best `val/mIoU=0.368008` at epoch `41`.
- Evidence: checkpoint `teacher's daima/checkpoints/teacher_resnet_original_run01/mid_fusion-epoch=41-val/mIoU=0.3680.ckpt`; event log `teacher's daima/checkpoints/teacher_resnet_original_run01/lightning_logs/version_0/events.out.tfevents.1777731102.Administrator.14772.0`.
- Per-epoch record: `miou_list/teacher_resnet_original_run01.md`.
- Comparison: in the paired teacher-code runs, ResNet original best `0.368008` is higher than DINOv2 RGB clean best `0.318669` by `+0.049339` mIoU; ResNet is higher at every recorded epoch.
- Conclusion: valid completed run; teacher original ResNet baseline is clearly stronger than the clean DINOv2-only RGB replacement in this setup.
- Next step: keep the ResNet teacher baseline as the stronger clean reference; DINOv2 token backbone needs more protocol/backbone adaptation before it is a fair improvement candidate.
