# R024 Geometry-Primary Ham Decoder

## Hypothesis

Raw `DFormerv2_S(rgb, depth)` features may match the official Ham decoder contract better than the local post-backbone external DepthEncoder/GatedFusion stack.

## Implementation

- Branch: `exp/R024-geometry-primary-ham-decoder-v1`
- Model: `dformerv2_geometry_primary_ham_decoder`
- Run: `R024_geometry_primary_ham_decoder_run01`
- Code change: add a model entry for `DFormerv2_S(rgb, depth) -> OfficialHamDecoder`.
- No external ResNet-18 DepthEncoder or GatedFusion is instantiated.
- Existing `OfficialHamDecoder` is reused unchanged, including `Dropout2d(0.1)`.
- Fixed recipe preserved: batch size `2`, max epochs `50`, lr `6e-5`, num workers `4`, early-stop patience `30`, CE loss, AdamW, DFormerv2-S level, and the existing DFormerv2-S pretrained loading.

## Evidence

- Status: full train completed with exit code `0`; `Trainer.fit` reached `max_epochs=50`.
- TensorBoard event: `final daima/checkpoints/R024_geometry_primary_ham_decoder_run01/lightning_logs/version_0/events.out.tfevents.1778793121.Administrator.20368.0`
- Best checkpoint: `final daima/checkpoints/R024_geometry_primary_ham_decoder_run01/dformerv2_geometry_primary_ham_decoder-epoch=44-val_mIoU=0.5302.pt`
- Per-epoch mIoU: `final daima/miou_list/R024_geometry_primary_ham_decoder_run01.md`

## Result

| Metric | Value |
|---|---:|
| recorded validation epochs | 50 |
| best val/mIoU | 0.530186 |
| best validation epoch | 45 |
| last val/mIoU | 0.529383 |
| last-5 mean val/mIoU | 0.521843 |
| last-10 mean val/mIoU | 0.522327 |
| best-to-last drop | 0.000803 |
| best val/loss | 1.062941 |
| best val/loss epoch | 8 |
| final train/loss_epoch | 0.071329 |

## Decision

R024 is a stable positive structure diagnostic, but it is not a new best. It is above `0.53`, below R022 `0.534332` by `0.004146`, and below R016 `0.541121` by `0.010935`.

This suggests that the external DepthEncoder/GatedFusion path is not simply harmful. The next experiment should return to the stronger corrected mid-fusion path and test one stability hypothesis, not continue Ham decoder micro-fixes.

## Forbidden-Change Check

R024 did not modify dataset split, eval metric, mIoU calculation, val/test loader behavior, data augmentation, optimizer, scheduler, batch size, max epochs, learning rate, worker count, early-stop setting, DFormerv2-S level, pretrained loading, OfficialHamDecoder internals, checkpoint artifacts, dataset files, pretrained weights, or TensorBoard event files.
