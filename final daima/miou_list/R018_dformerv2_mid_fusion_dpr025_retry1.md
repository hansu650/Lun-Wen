# R018 dformerv2_mid_fusion_dpr025 retry1 mIoU

- Run: `R018_dformerv2_mid_fusion_dpr025_retry1`
- Branch: `exp/R018-droppath025-contract-v1`
- Model: `dformerv2_mid_fusion_dpr025`
- Hypothesis: official DFormerv2-S NYUDepthv2 `drop_path_rate=0.25` may improve the R016 corrected baseline.
- TensorBoard event: `final daima/checkpoints/R018_dformerv2_mid_fusion_dpr025_retry1/lightning_logs/version_0/events.out.tfevents.1778760450.Administrator.7836.0`
- Best checkpoint: `final daima/checkpoints/R018_dformerv2_mid_fusion_dpr025_retry1/dformerv2_mid_fusion_dpr025-epoch=45-val_mIoU=0.5263.pt`
- Exit evidence: `exit_code.txt` = 0; log contains `Trainer.fit stopped: max_epochs=50 reached`.

## Summary

| metric | value |
|---|---:|
| validation epochs | 50 |
| best val/mIoU | 0.526282 |
| best validation epoch | 46 |
| last val/mIoU | 0.522893 |
| last-5 mean val/mIoU | 0.512694 |
| last-10 mean val/mIoU | 0.513363 |
| best-to-last drop | 0.003389 |
| best val/loss | 0.948631 at validation epoch 10 |
| final train/loss_epoch | 0.078764 |

## Per-Epoch val/mIoU

| validation epoch | val/mIoU | val/loss | train/loss_epoch |
|---:|---:|---:|---:|
| 1 | 0.166860 | 1.634087 | 2.267387 |
| 2 | 0.219792 | 1.361841 | 1.613420 |
| 3 | 0.263238 | 1.243832 | 1.304015 |
| 4 | 0.307203 | 1.137319 | 1.088156 |
| 5 | 0.350226 | 1.065967 | 0.923411 |
| 6 | 0.414873 | 0.986981 | 0.769128 |
| 7 | 0.421140 | 0.998696 | 0.649667 |
| 8 | 0.456627 | 0.975190 | 0.582236 |
| 9 | 0.466458 | 0.952892 | 0.483880 |
| 10 | 0.474563 | 0.948631 | 0.430156 |
| 11 | 0.466032 | 0.971757 | 0.384802 |
| 12 | 0.484814 | 0.956977 | 0.348091 |
| 13 | 0.490740 | 0.954268 | 0.295246 |
| 14 | 0.470972 | 1.034859 | 0.284505 |
| 15 | 0.483458 | 1.021806 | 0.263399 |
| 16 | 0.473764 | 1.056884 | 0.225808 |
| 17 | 0.503868 | 0.969156 | 0.247834 |
| 18 | 0.499035 | 0.991505 | 0.221492 |
| 19 | 0.503707 | 1.009240 | 0.192596 |
| 20 | 0.480652 | 1.027877 | 0.196209 |
| 21 | 0.503008 | 1.044636 | 0.171968 |
| 22 | 0.517618 | 0.993170 | 0.157282 |
| 23 | 0.517786 | 1.001900 | 0.143493 |
| 24 | 0.501305 | 1.066886 | 0.138492 |
| 25 | 0.447537 | 1.259271 | 0.173983 |
| 26 | 0.504935 | 1.034288 | 0.186028 |
| 27 | 0.522851 | 1.025649 | 0.133728 |
| 28 | 0.506966 | 1.045698 | 0.116836 |
| 29 | 0.507355 | 1.050781 | 0.129322 |
| 30 | 0.519365 | 1.045519 | 0.110605 |
| 31 | 0.513567 | 1.077186 | 0.103322 |
| 32 | 0.497270 | 1.112135 | 0.138075 |
| 33 | 0.505308 | 1.112470 | 0.136063 |
| 34 | 0.503154 | 1.074437 | 0.130709 |
| 35 | 0.510158 | 1.089154 | 0.099203 |
| 36 | 0.523900 | 1.085350 | 0.087676 |
| 37 | 0.518157 | 1.104488 | 0.095353 |
| 38 | 0.518192 | 1.110840 | 0.086219 |
| 39 | 0.499680 | 1.163835 | 0.091371 |
| 40 | 0.495340 | 1.182661 | 0.094516 |
| 41 | 0.502950 | 1.170362 | 0.092183 |
| 42 | 0.508509 | 1.159234 | 0.135283 |
| 43 | 0.520085 | 1.137160 | 0.080891 |
| 44 | 0.514009 | 1.141097 | 0.074773 |
| 45 | 0.524602 | 1.117262 | 0.073304 |
| 46 | 0.526282 | 1.119689 | 0.067684 |
| 47 | 0.506116 | 1.195477 | 0.072130 |
| 48 | 0.489154 | 1.264337 | 0.115011 |
| 49 | 0.519027 | 1.152634 | 0.121117 |
| 50 | 0.522893 | 1.129430 | 0.078764 |

## Interpretation

R018 is a completed negative contract gate. It improves over the early old-contract baseline family but is below the R016 corrected baseline best 0.541121, so `drop_path_rate=0.25` should not become the active mainline for the current mid-fusion pipeline.
