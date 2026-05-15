# R031 SimpleFPN Classifier Dropout

## Hypothesis

R022 showed that adding the official classifier `Dropout2d(0.1)` to the Ham decoder improved that path. R031 tests whether the strongest corrected SimpleFPN path benefits from the same tiny classifier regularizer.

## Implementation

- Branch: `exp/R031-simplefpn-classifier-dropout-v1`
- Model: `dformerv2_simplefpn_classifier_dropout`
- Run: `R031_simplefpn_classifier_dropout_run01`
- Code change: add `SimpleFPNDecoderWithClassifierDropout`, which applies `Dropout2d(0.1)` immediately before the SimpleFPN classifier.
- Integration: add a separate `DFormerV2SimpleFPNClassifierDropoutSegmentor` / `LitDFormerV2SimpleFPNClassifierDropout` entry.
- Kept unchanged: baseline `dformerv2_mid_fusion`, DFormerv2-S backbone level, DFormerv2-S pretrained loading, ResNet-18 DepthEncoder, GatedFusion equations, CE loss, AdamW, batch size `2`, max epochs `50`, lr `6e-5`, num workers `4`, early-stop patience `30`, data split, loaders, augmentation, eval metric, and mIoU calculation.

## Evidence

- Status: full train completed with exit code `0`; `Trainer.fit` reached `max_epochs=50`.
- TensorBoard event: `final daima/checkpoints/R031_simplefpn_classifier_dropout_run01/lightning_logs/version_0/events.out.tfevents.1778819244.Administrator.21360.0`
- Best checkpoint: `final daima/checkpoints/R031_simplefpn_classifier_dropout_run01/dformerv2_simplefpn_classifier_dropout-epoch=39-val_mIoU=0.5315.pt`
- Per-epoch mIoU: `final daima/miou_list/R031_simplefpn_classifier_dropout_run01.md`

## Result

| Metric | Value |
|---|---:|
| recorded validation epochs | 50 |
| best val/mIoU | 0.531544 |
| best validation epoch | 40 |
| last val/mIoU | 0.525760 |
| last-5 mean val/mIoU | 0.508009 |
| last-10 mean val/mIoU | 0.507366 |
| best-to-last drop | 0.005784 |
| best val/loss | 0.971063 |
| best val/loss epoch | 11 |
| final train/loss_epoch | 0.064540 |

## Decision

R031 is negative relative to the corrected baseline. It crosses `0.53`, but is below R016 `0.541121` by `-0.009577`, below R030 `0.536454`, and below R027 `0.536739`.

Do not continue classifier-dropout variants on SimpleFPN. The next round should test a different SimpleFPN-specific hypothesis: control high-resolution `c1` detail contribution instead of adding generic dropout.

## Forbidden-Change Check

R031 did not modify dataset split, eval metric, mIoU calculation, val/test loader behavior, data augmentation, optimizer, scheduler, batch size, max epochs, learning rate, worker count, early-stop setting, DFormerv2-S level, pretrained loading, pretrained DFormerv2 weights, pretrained DepthEncoder weights, GatedFusion equations, checkpoint artifacts, dataset files, pretrained weights, or TensorBoard event files.
