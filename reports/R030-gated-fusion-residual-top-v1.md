# R030 GatedFusion Residual-Top

## Hypothesis

R027 showed that residual depth injection can create a useful peak, but replacing the proven R016 `GatedFusion` path was unstable. R030 preserves the R016 `GatedFusion` output and adds only a zero-initialized residual correction on top.

## Implementation

- Branch: `exp/R030-gated-fusion-residual-top-v1`
- Model: `dformerv2_gated_fusion_residual_top`
- Run: `R030_gated_fusion_residual_top_run01`
- Code change: add `GatedFusionResidualTop`, `DFormerV2GatedFusionResidualTopSegmentor`, and `LitDFormerV2GatedFusionResidualTop`.
- Fusion form: compute the original `GatedFusion` base, then return `base + residual(rgb_feat, depth_proj, base, abs(rgb_feat - depth_proj))`.
- Initialization: the last residual `Conv2d` is zero-initialized, so initial output equals the `GatedFusion` base.
- Kept unchanged: DFormerv2-S backbone level, DFormerv2-S pretrained loading, ResNet-18 DepthEncoder, SimpleFPNDecoder, CE loss, AdamW, batch size `2`, max epochs `50`, lr `6e-5`, num workers `4`, early-stop patience `30`, data split, loaders, augmentation, eval metric, and mIoU calculation.

## Evidence

- Status: full train completed with exit code `0`; `Trainer.fit` reached `max_epochs=50`.
- TensorBoard event: `final daima/checkpoints/R030_gated_fusion_residual_top_run01/lightning_logs/version_0/events.out.tfevents.1778813976.Administrator.33508.0`
- Best checkpoint: `final daima/checkpoints/R030_gated_fusion_residual_top_run01/dformerv2_gated_fusion_residual_top-epoch=41-val_mIoU=0.5365.pt`
- Per-epoch mIoU: `final daima/miou_list/R030_gated_fusion_residual_top_run01.md`

## Result

| Metric | Value |
|---|---:|
| recorded validation epochs | 50 |
| best val/mIoU | 0.536454 |
| best validation epoch | 42 |
| last val/mIoU | 0.529803 |
| last-5 mean val/mIoU | 0.506209 |
| last-10 mean val/mIoU | 0.511101 |
| best-to-last drop | 0.006651 |
| best val/loss | 0.975530 |
| best val/loss epoch | 13 |
| final train/loss_epoch | 0.057927 |

## Decision

R030 is partial-positive but below the corrected baseline. It crosses `0.53`, but is below R016 `0.541121` by `-0.004667`, below R027 peak `0.536739` by `-0.000285`, and below the final `0.56` goal.

The result argues against continuing all-stage residual-depth corrections as the next main path. The next round should pivot to a distinct low-risk baseline regularization hypothesis, especially SimpleFPN classifier dropout, rather than another residual-family variant.

## Forbidden-Change Check

R030 did not modify dataset split, eval metric, mIoU calculation, val/test loader behavior, data augmentation, optimizer, scheduler, batch size, max epochs, learning rate, worker count, early-stop setting, DFormerv2-S level, pretrained loading, pretrained DFormerv2 weights, pretrained DepthEncoder weights, SimpleFPNDecoder, checkpoint artifacts, dataset files, pretrained weights, or TensorBoard event files.
