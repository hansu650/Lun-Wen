# R026 Official-Style Local Module Initialization

## Hypothesis

The local random fusion/decoder modules may be under-initialized relative to the official DFormer decode head contract; applying official-style initialization only to `GatedFusion` and `SimpleFPNDecoder` may improve the corrected mid-fusion path.

## Implementation

- Branch: `exp/R026-official-init-local-modules-v1`
- Model: `dformerv2_official_init_local_modules`
- Run: `R026_official_init_local_modules_run01`
- Code change: add a separate model entry that initializes only `self.fusions` and `self.decoder`.
- Conv2d weights use Kaiming normal, `mode="fan_in"`, `nonlinearity="relu"`.
- BatchNorm2d modules use `eps=1e-3`, `momentum=0.1`, weight `1`, bias `0`.
- Pretrained DFormerv2 and ResNet DepthEncoder weights are untouched.
- Fixed recipe preserved: batch size `2`, max epochs `50`, lr `6e-5`, num workers `4`, early-stop patience `30`, CE loss, AdamW, DFormerv2-S level, and existing DFormerv2-S pretrained loading.

## Evidence

- Status: full train completed with exit code `0`; `Trainer.fit` reached `max_epochs=50`.
- TensorBoard event: `final daima/checkpoints/R026_official_init_local_modules_run01/lightning_logs/version_0/events.out.tfevents.1778803189.Administrator.35684.0`
- Best checkpoint: `final daima/checkpoints/R026_official_init_local_modules_run01/dformerv2_official_init_local_modules-epoch=32-val_mIoU=0.5079.pt`
- Per-epoch mIoU: `final daima/miou_list/R026_official_init_local_modules_run01.md`

## Result

| Metric | Value |
|---|---:|
| recorded validation epochs | 50 |
| best val/mIoU | 0.507906 |
| best validation epoch | 33 |
| last val/mIoU | 0.499770 |
| last-5 mean val/mIoU | 0.496476 |
| last-10 mean val/mIoU | 0.495483 |
| best-to-last drop | 0.008136 |
| best val/loss | 1.073346 |
| best val/loss epoch | 5 |
| final train/loss_epoch | 0.054818 |

## Decision

R026 is negative. It is far below R016 `0.541121`, R025 `0.532572`, and the fixed-recipe `0.53` threshold.

Do not continue local-init experiments. The next fixed-recipe experiment should test a fusion-form hypothesis: preserve DFormerv2 primary features and inject depth as a residual instead of replacing features through gated mixing.

## Forbidden-Change Check

R026 did not modify dataset split, eval metric, mIoU calculation, val/test loader behavior, data augmentation, optimizer, scheduler, batch size, max epochs, learning rate, worker count, early-stop setting, DFormerv2-S level, pretrained loading, pretrained DFormerv2 weights, pretrained DepthEncoder weights, GatedFusion equations, SimpleFPNDecoder topology, checkpoint artifacts, dataset files, pretrained weights, or TensorBoard event files.
