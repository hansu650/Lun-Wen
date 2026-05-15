# R027 Primary Residual Depth Injection

## Hypothesis

The main fusion problem may be feature replacement rather than depth signal quality. Preserve DFormerv2 primary features and inject external DepthEncoder information only through a zero-initialized residual branch.

## Implementation

- Branch: `exp/R027-primary-residual-depth-injection-v1`
- Model: `dformerv2_primary_residual_depth`
- Run: `R027_primary_residual_depth_run01`
- Code change: add `PrimaryResidualDepthInjection` and replace the four existing `GatedFusion` modules with `rgb_feat + residual(depth_proj, abs(rgb_feat - depth_proj))`.
- Initialization: the last residual `Conv2d` is zero-initialized, so initial fused features equal the DFormerv2 primary features.
- Kept unchanged: DFormerv2-S backbone level, DFormerv2-S pretrained loading, ResNet-18 DepthEncoder, SimpleFPNDecoder, CE loss, AdamW, batch size `2`, max epochs `50`, lr `6e-5`, num workers `4`, early-stop patience `30`, data split, loaders, augmentation, eval metric, and mIoU calculation.

## Evidence

- Status: full train completed with exit code `0`; `Trainer.fit` reached `max_epochs=50`.
- TensorBoard event: `final daima/checkpoints/R027_primary_residual_depth_run01/lightning_logs/version_0/events.out.tfevents.1778808666.Administrator.14000.0`
- Best checkpoint: `final daima/checkpoints/R027_primary_residual_depth_run01/dformerv2_primary_residual_depth-epoch=40-val_mIoU=0.5367.pt`
- Per-epoch mIoU: `final daima/miou_list/R027_primary_residual_depth_run01.md`

## Result

| Metric | Value |
|---|---:|
| recorded validation epochs | 50 |
| best val/mIoU | 0.536739 |
| best validation epoch | 41 |
| last val/mIoU | 0.505286 |
| last-5 mean val/mIoU | 0.519799 |
| last-10 mean val/mIoU | 0.522758 |
| best-to-last drop | 0.031453 |
| best val/loss | 0.964226 |
| best val/loss epoch | 9 |
| final train/loss_epoch | 0.122960 |

## Decision

R027 is a partial-positive but unstable result. The peak crosses `0.53`, but it is below R016 `0.541121` by `-0.004382`, below the final `0.56` goal, and the last-epoch value drops to `0.505286`.

Do not continue variants that replace the R016 `GatedFusion` path wholesale. The next highest-value experiment should preserve R016 `GatedFusion` at initialization and add only a zero-initialized correction on top.

## Forbidden-Change Check

R027 did not modify dataset split, eval metric, mIoU calculation, val/test loader behavior, data augmentation, optimizer, scheduler, batch size, max epochs, learning rate, worker count, early-stop setting, DFormerv2-S level, pretrained loading, pretrained DFormerv2 weights, pretrained DepthEncoder weights, SimpleFPNDecoder, checkpoint artifacts, dataset files, pretrained weights, or TensorBoard event files.
