# R025 DepthEncoder BN Eval Run01 mIoU

- branch: `exp/R025-depth-encoder-bn-eval-v1`
- model: `dformerv2_depth_encoder_bn_eval`
- run: `R025_depth_encoder_bn_eval_run01`
- hypothesis: small-batch BatchNorm drift inside the external ResNet-18 DepthEncoder may destabilize the corrected mid-fusion path.
- implementation: only DepthEncoder `BatchNorm2d` modules are forced to eval mode during training; affine parameters remain trainable.
- status: full train completed with exit code `0`; `Trainer.fit` reached `max_epochs=50`.
- TensorBoard event: `checkpoints/R025_depth_encoder_bn_eval_run01/lightning_logs/version_0/events.out.tfevents.1778798018.Administrator.26772.0`
- best checkpoint: `checkpoints/R025_depth_encoder_bn_eval_run01/dformerv2_depth_encoder_bn_eval-epoch=46-val_mIoU=0.5326.pt`

## Summary

- recorded validation epochs: `50`
- best val/mIoU: `0.532572` at validation epoch `47`
- last val/mIoU: `0.496030`
- last-5 mean val/mIoU: `0.520333`
- last-10 mean val/mIoU: `0.517969`
- best-to-last drop: `0.036541`
- best val/loss: `0.961156` at validation epoch `9`
- final train/loss_epoch: `0.095029`
- comparison: above R024 `0.530186` by `0.002386`, below R022 `0.534332` by `0.001760`, and below R016 `0.541121` by `0.008549`.
- decision: partial-positive peak but stability negative. DepthEncoder BN eval does not solve late collapse and should not be used as the next base.

## Per-Epoch val/mIoU

| validation epoch | val/mIoU |
|---:|---:|
| 1 | 0.190535 |
| 2 | 0.246904 |
| 3 | 0.305333 |
| 4 | 0.364299 |
| 5 | 0.407080 |
| 6 | 0.429054 |
| 7 | 0.453302 |
| 8 | 0.480569 |
| 9 | 0.483143 |
| 10 | 0.484240 |
| 11 | 0.489301 |
| 12 | 0.468042 |
| 13 | 0.495702 |
| 14 | 0.499185 |
| 15 | 0.505787 |
| 16 | 0.502642 |
| 17 | 0.478523 |
| 18 | 0.489683 |
| 19 | 0.491947 |
| 20 | 0.501552 |
| 21 | 0.520351 |
| 22 | 0.516789 |
| 23 | 0.520880 |
| 24 | 0.458370 |
| 25 | 0.469421 |
| 26 | 0.505221 |
| 27 | 0.509868 |
| 28 | 0.527788 |
| 29 | 0.522723 |
| 30 | 0.493672 |
| 31 | 0.508177 |
| 32 | 0.520150 |
| 33 | 0.521483 |
| 34 | 0.526435 |
| 35 | 0.523921 |
| 36 | 0.523002 |
| 37 | 0.500063 |
| 38 | 0.492653 |
| 39 | 0.523989 |
| 40 | 0.518102 |
| 41 | 0.513533 |
| 42 | 0.491793 |
| 43 | 0.520506 |
| 44 | 0.524831 |
| 45 | 0.527359 |
| 46 | 0.531091 |
| 47 | 0.532572 |
| 48 | 0.528962 |
| 49 | 0.513011 |
| 50 | 0.496030 |
