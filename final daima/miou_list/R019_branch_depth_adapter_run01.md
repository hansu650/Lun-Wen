# R019 branch-specific depth adapter run01 mIoU

- Run: `R019_branch_depth_adapter_run01`
- Branch: `exp/R019-branch-depth-adapter-v1`
- Model: `dformerv2_branch_depth_adapter`
- Hypothesis: keep DFormerv2 geometry depth on the R016 official normalization while feeding the external ResNet-18 DepthEncoder a branch-specific `[0,1]` depth representation reconstructed inside the model.
- TensorBoard event: `final daima/checkpoints/R019_branch_depth_adapter_run01/lightning_logs/version_0/events.out.tfevents.1778765914.Administrator.27112.0`
- Best checkpoint: `final daima/checkpoints/R019_branch_depth_adapter_run01/dformerv2_branch_depth_adapter-epoch=45-val_mIoU=0.5325.pt`
- Exit evidence: `exit_code.txt` = 0; log contains `Trainer.fit stopped: max_epochs=50 reached`.

## Summary

| metric | value |
|---|---:|
| validation epochs | 50 |
| best val/mIoU | 0.532539 |
| best validation epoch | 46 |
| last val/mIoU | 0.495229 |
| last-5 mean val/mIoU | 0.509575 |
| last-10 mean val/mIoU | 0.518038 |
| best-to-last drop | 0.037311 |
| best val/loss | 0.958302 at validation epoch 8 |
| final train/loss_epoch | 0.067030 |

## Per-Epoch val/mIoU

| validation epoch | val/mIoU | val/loss | train/loss_epoch |
|---:|---:|---:|---:|
| 1 | 0.172123 | 1.660930 | 2.211444 |
| 2 | 0.253345 | 1.311970 | 1.524106 |
| 3 | 0.308081 | 1.204745 | 1.195306 |
| 4 | 0.351345 | 1.104686 | 0.978981 |
| 5 | 0.379635 | 1.082719 | 0.793252 |
| 6 | 0.419327 | 1.030983 | 0.663185 |
| 7 | 0.442757 | 1.010720 | 0.547914 |
| 8 | 0.470900 | 0.958302 | 0.462338 |
| 9 | 0.470701 | 1.004952 | 0.397220 |
| 10 | 0.486526 | 0.995659 | 0.348856 |
| 11 | 0.491364 | 0.962310 | 0.301124 |
| 12 | 0.493951 | 0.975683 | 0.259395 |
| 13 | 0.498461 | 0.990138 | 0.235991 |
| 14 | 0.488566 | 1.047459 | 0.215254 |
| 15 | 0.419749 | 1.272045 | 0.241079 |
| 16 | 0.504323 | 1.009349 | 0.248189 |
| 17 | 0.486624 | 1.042350 | 0.195640 |
| 18 | 0.494214 | 1.047544 | 0.175857 |
| 19 | 0.504921 | 1.037021 | 0.186773 |
| 20 | 0.507991 | 1.026742 | 0.145212 |
| 21 | 0.518417 | 1.022067 | 0.154179 |
| 22 | 0.496058 | 1.068381 | 0.141954 |
| 23 | 0.517707 | 1.047697 | 0.129871 |
| 24 | 0.519875 | 1.019248 | 0.111632 |
| 25 | 0.523626 | 1.055408 | 0.108868 |
| 26 | 0.503680 | 1.086791 | 0.123399 |
| 27 | 0.515927 | 1.074892 | 0.117287 |
| 28 | 0.509098 | 1.116242 | 0.113388 |
| 29 | 0.498407 | 1.150039 | 0.109686 |
| 30 | 0.519720 | 1.078100 | 0.105940 |
| 31 | 0.515166 | 1.068115 | 0.102329 |
| 32 | 0.524361 | 1.088878 | 0.082800 |
| 33 | 0.504468 | 1.176234 | 0.078341 |
| 34 | 0.515554 | 1.135990 | 0.094758 |
| 35 | 0.512019 | 1.172672 | 0.088972 |
| 36 | 0.471847 | 1.453389 | 0.081901 |
| 37 | 0.465067 | 1.250144 | 0.172249 |
| 38 | 0.511663 | 1.110706 | 0.123514 |
| 39 | 0.498271 | 1.155101 | 0.099091 |
| 40 | 0.518036 | 1.108049 | 0.079078 |
| 41 | 0.527238 | 1.122030 | 0.075505 |
| 42 | 0.531105 | 1.109163 | 0.062651 |
| 43 | 0.521230 | 1.150222 | 0.058244 |
| 44 | 0.523492 | 1.173142 | 0.056120 |
| 45 | 0.529436 | 1.170659 | 0.055874 |
| 46 | 0.532539 | 1.165628 | 0.056030 |
| 47 | 0.530474 | 1.174984 | 0.057130 |
| 48 | 0.484663 | 1.228133 | 0.134610 |
| 49 | 0.504972 | 1.215551 | 0.104294 |
| 50 | 0.495229 | 1.336034 | 0.067030 |

## Interpretation

R019 is a partial-positive original-method signal because it crosses 0.53 with a best val/mIoU of 0.532539, but it is below the R016 corrected baseline 0.541121 and has severe late instability with a best-to-last drop of 0.037311. Do not promote it over R016; use it to guide a more stable branch-specific depth adaptation design.
