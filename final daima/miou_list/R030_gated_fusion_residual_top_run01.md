# R030 GatedFusion Residual-Top Run01

- branch: `exp/R030-gated-fusion-residual-top-v1`
- model: `dformerv2_gated_fusion_residual_top`
- run: `R030_gated_fusion_residual_top_run01`
- checkpoint_dir: `final daima/checkpoints/R030_gated_fusion_residual_top_run01`
- TensorBoard event: `final daima/checkpoints/R030_gated_fusion_residual_top_run01/lightning_logs/version_0/events.out.tfevents.1778813976.Administrator.33508.0`
- best checkpoint: `final daima/checkpoints/R030_gated_fusion_residual_top_run01/dformerv2_gated_fusion_residual_top-epoch=41-val_mIoU=0.5365.pt`
- status: full train completed with exit code `0`; `Trainer.fit` reached `max_epochs=50`

## Summary

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

## Per-Epoch Validation Metrics

| Validation epoch | val/mIoU | val/loss |
|---:|---:|---:|
| 1 | 0.174797 | 1.648973 |
| 2 | 0.252458 | 1.309061 |
| 3 | 0.302874 | 1.203599 |
| 4 | 0.347234 | 1.126258 |
| 5 | 0.397938 | 1.066335 |
| 6 | 0.443027 | 1.014459 |
| 7 | 0.459736 | 0.989582 |
| 8 | 0.463912 | 1.004606 |
| 9 | 0.457337 | 0.996788 |
| 10 | 0.444550 | 1.106902 |
| 11 | 0.471596 | 0.978982 |
| 12 | 0.465588 | 1.054331 |
| 13 | 0.492912 | 0.975530 |
| 14 | 0.505250 | 0.981273 |
| 15 | 0.484194 | 1.048502 |
| 16 | 0.492976 | 1.040033 |
| 17 | 0.497487 | 1.037482 |
| 18 | 0.512048 | 1.019642 |
| 19 | 0.510419 | 1.039733 |
| 20 | 0.511419 | 1.055387 |
| 21 | 0.496724 | 1.075164 |
| 22 | 0.511606 | 1.076404 |
| 23 | 0.509042 | 1.093806 |
| 24 | 0.522997 | 1.048353 |
| 25 | 0.499228 | 1.131073 |
| 26 | 0.513978 | 1.080923 |
| 27 | 0.500795 | 1.115320 |
| 28 | 0.505387 | 1.134064 |
| 29 | 0.492626 | 1.145668 |
| 30 | 0.518920 | 1.069017 |
| 31 | 0.529636 | 1.067055 |
| 32 | 0.530674 | 1.066779 |
| 33 | 0.528733 | 1.080362 |
| 34 | 0.524669 | 1.117497 |
| 35 | 0.531097 | 1.156590 |
| 36 | 0.488712 | 1.242564 |
| 37 | 0.513087 | 1.172543 |
| 38 | 0.513406 | 1.121329 |
| 39 | 0.508054 | 1.170241 |
| 40 | 0.520744 | 1.126044 |
| 41 | 0.529760 | 1.134977 |
| 42 | 0.536454 | 1.139586 |
| 43 | 0.509993 | 1.152595 |
| 44 | 0.516813 | 1.162556 |
| 45 | 0.486944 | 1.273651 |
| 46 | 0.493495 | 1.221074 |
| 47 | 0.489404 | 1.272513 |
| 48 | 0.500728 | 1.188674 |
| 49 | 0.517615 | 1.176792 |
| 50 | 0.529803 | 1.162641 |

## Interpretation

R030 preserves the R016 `GatedFusion` base at initialization and adds a zero-initialized residual correction. The run crosses `0.53` and ends higher than R027, but its best `0.536454` remains below R016 `0.541121` and below the final `0.56` goal. This suggests the residual-depth signal is real but insufficient in this all-stage top-correction form.
