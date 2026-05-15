# R027 Primary Residual Depth Injection Run01

- branch: `exp/R027-primary-residual-depth-injection-v1`
- model: `dformerv2_primary_residual_depth`
- run: `R027_primary_residual_depth_run01`
- checkpoint_dir: `final daima/checkpoints/R027_primary_residual_depth_run01`
- TensorBoard event: `final daima/checkpoints/R027_primary_residual_depth_run01/lightning_logs/version_0/events.out.tfevents.1778808666.Administrator.14000.0`
- best checkpoint: `final daima/checkpoints/R027_primary_residual_depth_run01/dformerv2_primary_residual_depth-epoch=40-val_mIoU=0.5367.pt`
- status: full train completed with exit code `0`; `Trainer.fit` reached `max_epochs=50`

## Summary

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

## Per-Epoch Validation Metrics

| Validation epoch | val/mIoU | val/loss |
|---:|---:|---:|
| 1 | 0.162684 | 1.722097 |
| 2 | 0.240028 | 1.362242 |
| 3 | 0.312528 | 1.206288 |
| 4 | 0.348188 | 1.123122 |
| 5 | 0.376705 | 1.066582 |
| 6 | 0.421864 | 1.056824 |
| 7 | 0.464285 | 0.988439 |
| 8 | 0.474836 | 0.967854 |
| 9 | 0.478858 | 0.964226 |
| 10 | 0.484110 | 0.999563 |
| 11 | 0.490517 | 0.981064 |
| 12 | 0.483027 | 1.021628 |
| 13 | 0.502062 | 1.012722 |
| 14 | 0.496885 | 0.994733 |
| 15 | 0.491762 | 1.018058 |
| 16 | 0.492808 | 1.049944 |
| 17 | 0.508509 | 1.003579 |
| 18 | 0.509170 | 1.037580 |
| 19 | 0.499004 | 1.123397 |
| 20 | 0.512955 | 1.095121 |
| 21 | 0.520478 | 1.057554 |
| 22 | 0.519871 | 1.084008 |
| 23 | 0.513321 | 1.115023 |
| 24 | 0.457726 | 1.408818 |
| 25 | 0.504924 | 1.159093 |
| 26 | 0.518991 | 1.103088 |
| 27 | 0.453414 | 1.296172 |
| 28 | 0.500042 | 1.148629 |
| 29 | 0.506432 | 1.102463 |
| 30 | 0.498254 | 1.139486 |
| 31 | 0.520853 | 1.100165 |
| 32 | 0.507950 | 1.127754 |
| 33 | 0.518229 | 1.116497 |
| 34 | 0.523946 | 1.095852 |
| 35 | 0.531663 | 1.100085 |
| 36 | 0.534310 | 1.121621 |
| 37 | 0.523439 | 1.137043 |
| 38 | 0.435233 | 1.694077 |
| 39 | 0.505493 | 1.169785 |
| 40 | 0.522514 | 1.146932 |
| 41 | 0.536739 | 1.131744 |
| 42 | 0.529275 | 1.155948 |
| 43 | 0.506366 | 1.207282 |
| 44 | 0.529295 | 1.166647 |
| 45 | 0.526908 | 1.158761 |
| 46 | 0.525457 | 1.164746 |
| 47 | 0.531610 | 1.176686 |
| 48 | 0.530568 | 1.196213 |
| 49 | 0.506075 | 1.289560 |
| 50 | 0.505286 | 1.262187 |

## Interpretation

R027 crosses the fixed-recipe `0.53` stage threshold but remains below the corrected R016 baseline best `0.541121` and finishes with a `0.031453` best-to-last drop. The result supports a real high-peak signal for primary-preserving residual depth injection, but replacing the proven R016 `GatedFusion` path is not stable enough to use as the next base.
