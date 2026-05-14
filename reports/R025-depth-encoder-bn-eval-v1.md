# R025 DepthEncoder BN Eval

## Hypothesis

Small-batch BatchNorm drift inside the external ResNet-18 DepthEncoder may destabilize the strongest corrected mid-fusion path.

## Implementation

- Branch: `exp/R025-depth-encoder-bn-eval-v1`
- Model: `dformerv2_depth_encoder_bn_eval`
- Run: `R025_depth_encoder_bn_eval_run01`
- Code change: add a model entry that forces only DepthEncoder `BatchNorm2d` modules to eval mode during training.
- BN affine parameters remain trainable; no parameters are frozen and optimizer construction is unchanged.
- Fixed recipe preserved: batch size `2`, max epochs `50`, lr `6e-5`, num workers `4`, early-stop patience `30`, CE loss, AdamW, DFormerv2-S level, and existing DFormerv2-S pretrained loading.

## Evidence

- Status: full train completed with exit code `0`; `Trainer.fit` reached `max_epochs=50`.
- TensorBoard event: `final daima/checkpoints/R025_depth_encoder_bn_eval_run01/lightning_logs/version_0/events.out.tfevents.1778798018.Administrator.26772.0`
- Best checkpoint: `final daima/checkpoints/R025_depth_encoder_bn_eval_run01/dformerv2_depth_encoder_bn_eval-epoch=46-val_mIoU=0.5326.pt`
- Per-epoch mIoU: `final daima/miou_list/R025_depth_encoder_bn_eval_run01.md`

## Result

| Metric | Value |
|---|---:|
| recorded validation epochs | 50 |
| best val/mIoU | 0.532572 |
| best validation epoch | 47 |
| last val/mIoU | 0.496030 |
| last-5 mean val/mIoU | 0.520333 |
| last-10 mean val/mIoU | 0.517969 |
| best-to-last drop | 0.036541 |
| best val/loss | 0.961156 |
| best val/loss epoch | 9 |
| final train/loss_epoch | 0.095029 |

## Decision

R025 is a partial-positive peak but stability-negative result. It is above R024 `0.530186`, but below R022 `0.534332` and below R016 `0.541121`; the final drop is severe.

Do not build the next experiment on DepthEncoder BN eval. The next fixed-recipe experiment should test a different single stability hypothesis, preferably official-style initialization of local random modules (`GatedFusion` and `SimpleFPNDecoder` only).

## Forbidden-Change Check

R025 did not modify dataset split, eval metric, mIoU calculation, val/test loader behavior, data augmentation, optimizer, scheduler, batch size, max epochs, learning rate, worker count, early-stop setting, DFormerv2-S level, pretrained loading, DepthEncoder architecture, GatedFusion equations, SimpleFPNDecoder, checkpoint artifacts, dataset files, pretrained weights, or TensorBoard event files.
