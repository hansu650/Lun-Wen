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

## 2026-05-03 DFormerv2 Mid Fusion Pretrained Run01

- Date: `2026-05-03`.
- Experiment name: `dformerv2_mid_fusion_pretrained_run01`.
- Code path: `teacher's daima/`.
- Configuration: `DFormerv2_S` RGB-D encoder features, teacher original ResNet-18 `DepthEncoder`, teacher original `GatedFusion`, teacher original `SimpleFPNDecoder`.
- Pretrained: `C:\Users\qintian\Desktop\qintian\dformer_work\checkpoints\pretrained\DFormerv2_Small_pretrained.pth`.
- Training configuration: NYUDepthV2 folder data, `max_epochs=50`, `batch_size=2`, `lr=6e-5`, `num_workers=0`, GPU 1 device.
- Result: best `val/mIoU=0.507965` at epoch `41`.
- Evidence: checkpoint `teacher's daima/checkpoints/dformerv2_mid_fusion_pretrained_run01/dformerv2_mid_fusion-epoch=41-val/mIoU=0.5080.ckpt`; event log `teacher's daima/checkpoints/dformerv2_mid_fusion_pretrained_run01/lightning_logs/version_0/events.out.tfevents.1777735793.Administrator.33924.0`.
- Per-epoch record: `miou_list/dformerv2_mid_fusion_pretrained_run01.md`.
- Curve observation: validation mIoU reached `0.454100` by epoch `7`, `0.482066` by epoch `11`, `0.502551` by epoch `29`, and peaked at epoch `41`; later epochs stayed near `0.50` with fluctuation.
- Comparison: this run is much higher than teacher ResNet original best `0.368008` and DINOv2 RGB clean best `0.318669`; it also exceeds the previous documented `0.3933` confirmed best.
- Conclusion: valid completed run with clear checkpoint and TensorBoard evidence; `DFormerv2_S + ResNet18 depth + GatedFusion + SimpleFPNDecoder` is currently the strongest recorded result in this workspace.
- Next step: repeat this configuration for stability, then try the NYU-trained DFormerv2 checkpoint finetune command if confirming robustness.

## 2026-05-03 DFormerv2 Attention Fusion 5 Runs

- Date: `2026-05-03`.
- Experiment group: `dformerv2_attention_fusion_run01` to `dformerv2_attention_fusion_run05`.
- Code path: `teacher's daima/`.
- Configuration: `DFormerv2_S` backbone, teacher original ResNet-18 `DepthEncoder`, `CrossModalReliabilityAttentionFusion`, teacher original `SimpleFPNDecoder`.
- Pretrained: `C:\Users\qintian\Desktop\qintian\dformer_work\checkpoints\pretrained\DFormerv2_Small_pretrained.pth`.
- Training configuration: NYUDepthV2 folder data, `max_epochs=50`, `batch_size=2`, `lr=6e-5`, `num_workers=0`, GPU 1 device.
- Run01: best `val/mIoU=0.516979` at epoch `45`; per-epoch record `miou_list/dformerv2_attention_fusion_run01.md`.
- Run02: best `val/mIoU=0.507672` at epoch `46`; per-epoch record `miou_list/dformerv2_attention_fusion_run02.md`.
- Run03: best `val/mIoU=0.513333` at epoch `42`; per-epoch record `miou_list/dformerv2_attention_fusion_run03.md`.
- Run04: best `val/mIoU=0.518196` at epoch `49`; per-epoch record `miou_list/dformerv2_attention_fusion_run04.md`.
- Run05: best `val/mIoU=0.515997` at epoch `49`; per-epoch record `miou_list/dformerv2_attention_fusion_run05.md`.
- Summary: mean best `val/mIoU=0.514435`, std `0.004184`, min `0.507672`, max `0.518196`.
- Comparison: this improves over `dformerv2_mid_fusion_pretrained_run01` best `0.507965` by about `+0.006470` mean best and `+0.010231` max best.
- Conclusion: valid 5-run result; `dformerv2_attention_fusion` is currently the strongest recorded model variant in this workspace.
- Next step: keep this as the current best fusion direction; run one more controlled comparison with the same seed protocol if needed for paper stability claims.

## 2026-05-03 DFormerv2 Mid Fusion Repeat 4 Runs

- Date: `2026-05-03`.
- Experiment group: `dformerv2_mid_fusion_repeat_run01` to `dformerv2_mid_fusion_repeat_run04`.
- Code path: `teacher's daima/`.
- Configuration: `DFormerv2_S` backbone, teacher original ResNet-18 `DepthEncoder`, original `GatedFusion`, teacher original `SimpleFPNDecoder`.
- Pretrained: `C:\Users\qintian\Desktop\qintian\dformer_work\checkpoints\pretrained\DFormerv2_Small_pretrained.pth`.
- Training configuration: NYUDepthV2 folder data, `max_epochs=50`, `batch_size=2`, `lr=6e-5`, `num_workers=0`, `early_stop_patience=30`, GPU 1 device.
- Run01: best `val/mIoU=0.513287` at epoch `49`; per-epoch record `miou_list/dformerv2_mid_fusion_repeat_run01.md`.
- Run02: best `val/mIoU=0.509708` at epoch `38`; per-epoch record `miou_list/dformerv2_mid_fusion_repeat_run02.md`.
- Run03: best `val/mIoU=0.515470` at epoch `44`; per-epoch record `miou_list/dformerv2_mid_fusion_repeat_run03.md`.
- Run04: best `val/mIoU=0.515157` at epoch `46`; per-epoch record `miou_list/dformerv2_mid_fusion_repeat_run04.md`.
- Summary: mean best `val/mIoU=0.513406`, std `0.002647`, min `0.509708`, max `0.515470`.
- Comparison: `dformerv2_attention_fusion` 5-run mean best is `0.514435`, so attention fusion is higher by only `+0.001030` mean best; this difference is smaller than the run-to-run standard deviations.
- Summary document: `docs/dformerv2_mid_vs_attention_summary.md`.
- Conclusion: valid 4-run baseline; current attention fusion does not show a clearly stable improvement over the repeated GatedFusion baseline, although it has the highest single run `0.518196`.
- Next step: treat attention fusion as a small/uncertain gain unless more repeats or paired-seed comparisons show a larger margin.

## 2026-05-03 DFormerv2 Clean Depth Fusion Shape-SA-Gate Run01

- Date: `2026-05-03`.
- Experiment name: `dformerv2_clean_depth_fusion_shape_sagate_run01`.
- Code path: `teacher's daima/`.
- Configuration: `DFormerv2_S` backbone, ShapeConv/SA-Gate-inspired `CleanShapeDepthEncoder`, original `GatedFusion`, teacher original `SimpleFPNDecoder`.
- Pretrained: `C:\Users\qintian\Desktop\qintian\dformer_work\checkpoints\pretrained\DFormerv2_Small_pretrained.pth`.
- Training configuration: NYUDepthV2 folder data, `max_epochs=50`, `batch_size=2`, `lr=6e-5`, `num_workers=0`, `early_stop_patience=30`, GPU 1 device.
- Result: best `val/mIoU=0.515150` at epoch `43`; last epoch `val/mIoU=0.470769`.
- Best validation loss: `val/loss=1.032949` at epoch `6`.
- Evidence: event log `teacher's daima/checkpoints/dformerv2_clean_depth_fusion_shape_sagate_run01/lightning_logs/version_0/events.out.tfevents.1777798791.Administrator.41684.0`.
- Per-epoch record: `miou_list/dformerv2_clean_depth_fusion_shape_sagate_run01.md`.
- Comparison: this single run is close to the repeated `dformerv2_mid_fusion` baseline mean best `0.513406` and below/near the best baseline repeat `0.515470`; it is also near the `dformerv2_attention_fusion` mean best `0.514435`.
- Conclusion: valid completed run, but one run is not enough to claim improvement because the gain is inside the known run-to-run fluctuation range.
- Next step: run two more repeats with the same configuration before deciding whether this clean-depth branch is better than the original `DepthEncoder`.

## 2026-05-03 DFormerv2 CMX Fusion Run01

- Date: `2026-05-03`.
- Experiment name: `dformerv2_cmx_fusion_run01`.
- Code path: `teacher's daima/`.
- Configuration: `DFormerv2_S` backbone, original ResNet-18 `DepthEncoder`, CMX-style `DFormerCMXFusion`, teacher original `SimpleFPNDecoder`.
- Pretrained: `C:\Users\qintian\Desktop\qintian\dformer_work\checkpoints\pretrained\DFormerv2_Small_pretrained.pth`.
- Training configuration: NYUDepthV2 folder data, `max_epochs=50`, `batch_size=2`, `lr=6e-5`, `num_workers=0`, `early_stop_patience=30`, GPU 1 device.
- Result: best `val/mIoU=0.503343` at epoch `47`; last epoch `val/mIoU=0.499639`.
- Status: deprecated / negative result; this branch is not continued.
- Best validation loss: `val/loss=1.023529` at epoch `7`.
- Evidence: event log `teacher's daima/checkpoints/dformerv2_cmx_fusion_run01/lightning_logs/version_0/events.out.tfevents.1777805961.Administrator.25256.0`.
- Per-epoch record: `miou_list/dformerv2_cmx_fusion_run01.md`.
- Comparison: repeated `dformerv2_mid_fusion` baseline mean best is `0.513406`, so CMX run01 is lower by about `0.010063`; it is also below the weakest repeated baseline run `0.509708`.
- Curve observation: CMX reaches `0.498322` at epoch `20`, then mostly oscillates around `0.49-0.50`; late improvement is small and never enters the baseline repeat range.
- Conclusion: valid completed run; full CMX-style symmetric cross-modal fusion underperforms the repeated DFormerv2 mid-fusion baseline, so this branch is deprecated and will not be continued.
- Next step: do not spend more repeats on this exact CMX block unless the CMX implementation is changed in a targeted way; return to `dformerv2_mid_fusion` as stable baseline.

## 2026-05-03 DFormerv2 PrimKD Fusion Soft Run01

- Date: `2026-05-03`.
- Experiment name: `dformerv2_primkd_fusion_soft_run01`.
- Code path: `teacher's daima/`.
- Configuration: `DFormerv2_S` backbone, original ResNet-18 `DepthEncoder`, PrimKD-inspired soft primary-guided `PrimaryGuidedFusion`, teacher original `SimpleFPNDecoder`.
- Pretrained: `C:\Users\qintian\Desktop\qintian\dformer_work\checkpoints\pretrained\DFormerv2_Small_pretrained.pth`.
- Training configuration: NYUDepthV2 folder data, `max_epochs=50`, `batch_size=2`, `lr=6e-5`, `num_workers=4`, `early_stop_patience=30`, GPU 1 device.
- Result: best `val/mIoU=0.515757` at epoch `47`; last epoch `val/mIoU=0.505458`.
- Best validation loss: `val/loss=1.048144` at epoch `10`.
- Evidence: event log `teacher's daima/checkpoints/dformerv2_primkd_fusion_soft_run01/lightning_logs/version_0/events.out.tfevents.1777812820.Administrator.36692.0`; checkpoint `teacher's daima\checkpoints\dformerv2_primkd_fusion_soft_run01\dformerv2_primkd_fusion-epoch=47-val\mIoU=0.5158.ckpt`.
- Per-epoch record: `miou_list/dformerv2_primkd_fusion_soft_run01.md`.
- Comparison: repeated `dformerv2_mid_fusion` baseline mean best is `0.513406`; this single PrimKD-soft run is higher by `+0.002351`. The best repeated baseline run is `0.515470`; this run is higher by `+0.000287`.
- Comparison: `dformerv2_attention_fusion` 5-run mean best is `0.514435`; this run is higher by `+0.001322`, but lower than the best attention single run `0.518196` by `-0.002439`.
- Conclusion: valid completed run. The soft PrimKD-style primary-guided fusion is competitive and slightly above the repeated GatedFusion baseline mean in this single run, but the gain is still inside the previously observed run-to-run fluctuation, so it needs repeats before claiming stable improvement.
- Next step: run at least two more repeats with fresh checkpoint directories before deciding whether this branch replaces `dformerv2_mid_fusion` as the main fusion line.

## 2026-05-03 Restore Main Branch To DFormerv2 Mid Fusion

- Deprecated experimental branches in active code:
- `dformerv2_attention_fusion`: mean improvement was marginal compared with repeated `dformerv2_mid_fusion`.
- `dformerv2_clean_depth_fusion`: increased parameters but did not show a clear stable gain.
- `dformerv2_cmx_fusion`: underperformed the repeated baseline and was already marked as a negative result.
- `dformerv2_primkd_fusion`: single-run result was slightly above baseline mean, but not enough to justify extra active-code complexity without repeats.
- Main branch restored to: `dformerv2_mid_fusion`.
- Code boundary: experiment records and `miou_list` markdown files are retained; only inactive experimental model branches were removed from the main training path.

## 2026-05-04 DFormerv2 CMNeXt Fusion Run01

- Date: `2026-05-04`.
- Experiment name: `dformerv2_cmnext_fusion_run01`.
- Code path: `teacher's daima/`.
- Configuration: `DFormerv2_S` backbone, original ResNet-18 `DepthEncoder`, DELIVER / CMNeXt-inspired hub-guided `DFormerHubFusion`, teacher original `SimpleFPNDecoder`.
- Pretrained: `C:\Users\qintian\Desktop\qintian\dformer_work\checkpoints\pretrained\DFormerv2_Small_pretrained.pth`.
- Training configuration: NYUDepthV2 folder data, `max_epochs=50`, `batch_size=2`, `lr=6e-5`, `num_workers=4`, `early_stop_patience=30`, GPU 1 device.
- Result: best `val/mIoU=0.508904` at epoch `37`; last epoch `val/mIoU=0.504782`.
- Best validation loss: `val/loss=1.034672` at epoch `11`.
- Evidence: event log `teacher's daima/checkpoints/dformerv2_cmnext_fusion_run01/lightning_logs/version_0/events.out.tfevents.1777819430.Administrator.12636.0`; checkpoint `teacher's daima\checkpoints\dformerv2_cmnext_fusion_run01\dformerv2_cmnext_fusion-epoch=37-val\mIoU=0.5089.ckpt`.
- Per-epoch record: `miou_list/dformerv2_cmnext_fusion_run01.md`.
- Comparison: repeated `dformerv2_mid_fusion` baseline mean best is `0.513406`; this run is lower by `-0.004502`. The best repeated baseline run is `0.515470`; this run is lower by `-0.006566`.
- Comparison: `dformerv2_attention_fusion` mean best is `0.514435`; this run is lower by `-0.005531`. `dformerv2_primkd_fusion_soft_run01` best is `0.515757`; this run is lower by `-0.006853`.
- Comparison: this run is higher than the deprecated pure CMX run best `0.503343` by `+0.005561`, but still below the stable baseline range.
- Conclusion: valid completed run. The current DELIVER / CMNeXt-style hub fusion implementation does not improve over the stable `dformerv2_mid_fusion` baseline and should not replace the baseline in its current form.
- Next step: do not spend repeat budget on this exact block unless simplifying or changing the hub fusion design; return to `dformerv2_mid_fusion` as the stable reference.

## 2026-05-04 Deprecate DFormerv2 CMNeXt Fusion

- Experiment: `dformerv2_cmnext_fusion_run01`.
- Result: best `val/mIoU=0.508904`; last epoch `val/mIoU=0.504782`.
- Conclusion: DELIVER / CMNeXt-inspired hub-guided fusion underperformed the repeated DFormerv2 mid-fusion baseline, so this branch is deprecated.
- Code boundary: experiment records, checkpoints, and `miou_list/dformerv2_cmnext_fusion_run01.md` are retained; only the active model code path was restored to the clean baseline.
- Main branch restored to: `dformerv2_mid_fusion`.
