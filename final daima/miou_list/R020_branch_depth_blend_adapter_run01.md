# R020 branch-specific depth blend adapter run01 mIoU

- Run: `R020_branch_depth_blend_adapter_run01`
- Branch: `exp/R020-depth-blend-adapter-v1`
- Model: `dformerv2_branch_depth_blend_adapter`
- Hypothesis: stabilize R019 by using a learnable convex blend between R016 official-normalized depth and reconstructed `[0,1]` depth for the external DepthEncoder.
- TensorBoard event: `final daima/checkpoints/R020_branch_depth_blend_adapter_run01/lightning_logs/version_0/events.out.tfevents.1778771221.Administrator.41764.0`
- Best checkpoint: `final daima/checkpoints/R020_branch_depth_blend_adapter_run01/dformerv2_branch_depth_blend_adapter-epoch=40-val_mIoU=0.5329.pt`
- Exit evidence: `exit_code.txt` = 0; log contains `Trainer.fit stopped: max_epochs=50 reached`.

## Summary

| metric | value |
|---|---:|
| validation epochs | 50 |
| best val/mIoU | 0.532924 |
| best validation epoch | 41 |
| last val/mIoU | 0.503238 |
| last-5 mean val/mIoU | 0.520456 |
| last-10 mean val/mIoU | 0.516804 |
| best-to-last drop | 0.029686 |
| best val/loss | 0.979484 at validation epoch 8 |
| final train/loss_epoch | 0.089993 |
| alpha first | 0.050022 |
| alpha last | 0.051455 |
| alpha min | 0.050022 |
| alpha max | 0.051455 |

## Per-Epoch Metrics

| validation epoch | val/mIoU | val/loss | train/loss_epoch | train/depth_blend_alpha |
|---:|---:|---:|---:|---:|
| 1 | 0.167865 | 1.705488 | 2.327032 | 0.050022 |
| 2 | 0.260989 | 1.333755 | 1.594921 | 0.050031 |
| 3 | 0.317767 | 1.175761 | 1.244015 | 0.050046 |
| 4 | 0.354962 | 1.166765 | 1.001339 | 0.050066 |
| 5 | 0.401099 | 1.036895 | 0.819067 | 0.050113 |
| 6 | 0.447038 | 0.986288 | 0.668157 | 0.050151 |
| 7 | 0.456842 | 0.984145 | 0.549249 | 0.050190 |
| 8 | 0.477555 | 0.979484 | 0.470830 | 0.050184 |
| 9 | 0.472634 | 0.987008 | 0.397139 | 0.050220 |
| 10 | 0.471639 | 1.035733 | 0.362759 | 0.050267 |
| 11 | 0.481543 | 0.991636 | 0.318461 | 0.050258 |
| 12 | 0.468483 | 1.042773 | 0.309100 | 0.050242 |
| 13 | 0.493985 | 1.001107 | 0.266803 | 0.050287 |
| 14 | 0.502379 | 1.013441 | 0.217209 | 0.050325 |
| 15 | 0.500190 | 1.025670 | 0.197909 | 0.050365 |
| 16 | 0.470465 | 1.060806 | 0.210455 | 0.050414 |
| 17 | 0.498191 | 1.023682 | 0.217106 | 0.050400 |
| 18 | 0.502503 | 1.038926 | 0.158622 | 0.050398 |
| 19 | 0.510869 | 1.021698 | 0.140088 | 0.050450 |
| 20 | 0.516735 | 1.038882 | 0.133319 | 0.050476 |
| 21 | 0.489296 | 1.111763 | 0.185969 | 0.050526 |
| 22 | 0.500819 | 1.072107 | 0.200876 | 0.050541 |
| 23 | 0.516687 | 1.057570 | 0.138995 | 0.050554 |
| 24 | 0.519850 | 1.016577 | 0.117764 | 0.050605 |
| 25 | 0.519513 | 1.034102 | 0.105047 | 0.050621 |
| 26 | 0.519672 | 1.081780 | 0.101734 | 0.050603 |
| 27 | 0.483589 | 1.164343 | 0.154092 | 0.050623 |
| 28 | 0.516904 | 1.097443 | 0.118149 | 0.050659 |
| 29 | 0.520679 | 1.124595 | 0.101034 | 0.050690 |
| 30 | 0.501852 | 1.098039 | 0.104392 | 0.050715 |
| 31 | 0.523662 | 1.096076 | 0.100340 | 0.050674 |
| 32 | 0.521141 | 1.105054 | 0.081527 | 0.050717 |
| 33 | 0.480731 | 1.278054 | 0.083600 | 0.050790 |
| 34 | 0.488834 | 1.180770 | 0.121763 | 0.050815 |
| 35 | 0.498715 | 1.146699 | 0.128227 | 0.050860 |
| 36 | 0.518191 | 1.109853 | 0.099381 | 0.050893 |
| 37 | 0.530118 | 1.109738 | 0.077870 | 0.050914 |
| 38 | 0.532237 | 1.114885 | 0.067322 | 0.050947 |
| 39 | 0.520890 | 1.124690 | 0.068033 | 0.051000 |
| 40 | 0.522931 | 1.143907 | 0.069340 | 0.050999 |
| 41 | 0.532924 | 1.153192 | 0.064507 | 0.051025 |
| 42 | 0.522108 | 1.164249 | 0.071691 | 0.051066 |
| 43 | 0.475089 | 1.221300 | 0.162419 | 0.051087 |
| 44 | 0.512119 | 1.151298 | 0.087155 | 0.051158 |
| 45 | 0.523515 | 1.155332 | 0.063425 | 0.051212 |
| 46 | 0.527078 | 1.155835 | 0.055585 | 0.051221 |
| 47 | 0.517356 | 1.196140 | 0.053716 | 0.051251 |
| 48 | 0.525414 | 1.188436 | 0.053422 | 0.051312 |
| 49 | 0.529194 | 1.199499 | 0.052520 | 0.051387 |
| 50 | 0.503238 | 1.201395 | 0.089993 | 0.051455 |

## Interpretation

R020 slightly improves the R019 best peak and is much more stable than R019 through epoch 49, but the final epoch still drops sharply. It remains below the R016 corrected baseline 0.541121, so it is not a new main result. The alpha trace stays near 0.05, showing the model did not learn to move strongly toward the hard `[0,1]` branch.
