# R016 Depth Normalization Official Baseline Retry1

## Summary

- Branch: `exp/R016-depth-norm-contract-v1`
- Model: `dformerv2_mid_fusion`
- Run: `R016_depth_norm_official_baseline_retry1`
- Hypothesis: after R015 aligns NYUDepthV2 label semantics, depth input should follow the official DFormer modal_x normalization contract: `raw / 255.0`, then `(x - 0.48) / 0.28`.
- Status: completed full train; `Trainer.fit` reached `max_epochs=50`; exit code `0`.
- TensorBoard event: `checkpoints/R016_depth_norm_official_baseline_retry1/lightning_logs/version_0/events.out.tfevents.1778745208.Administrator.36016.0`
- Best checkpoint: `checkpoints/R016_depth_norm_official_baseline_retry1/dformerv2_mid_fusion-epoch=48-val_mIoU=0.5411.pt`
- Recorded validation epochs: `50`
- Best val/mIoU: `0.541121` at validation epoch `49`
- Last val/mIoU: `0.527420`
- Last-5 mean val/mIoU: `0.535500`
- Last-10 mean val/mIoU: `0.524063`
- Best-to-last drop: `0.013702`
- Best val/loss: `0.978448` at validation epoch `14`
- Last val/loss: `1.211359`
- Final train/loss_epoch: `0.053537`
- Delta vs R015 official-label baseline best `0.537398`: `+0.003723`
- Contract note: compare this run only against R015 and later official-contract runs, not against old label-contract results.

## Per-Epoch Validation Metrics

| Val Epoch | Global Step | val/mIoU | val/loss |
|---:|---:|---:|---:|
| 1 | 396 | 0.179018 | 1.612541 |
| 2 | 793 | 0.245723 | 1.329458 |
| 3 | 1190 | 0.310877 | 1.175439 |
| 4 | 1587 | 0.365663 | 1.069930 |
| 5 | 1984 | 0.394522 | 1.046654 |
| 6 | 2381 | 0.439340 | 1.028850 |
| 7 | 2778 | 0.462156 | 0.993265 |
| 8 | 3175 | 0.469736 | 0.984289 |
| 9 | 3572 | 0.480372 | 1.015131 |
| 10 | 3969 | 0.476135 | 1.006973 |
| 11 | 4366 | 0.484214 | 1.012058 |
| 12 | 4763 | 0.481915 | 1.048197 |
| 13 | 5160 | 0.490102 | 1.029596 |
| 14 | 5557 | 0.508236 | 0.978448 |
| 15 | 5954 | 0.507050 | 1.006591 |
| 16 | 6351 | 0.504301 | 1.021487 |
| 17 | 6748 | 0.489938 | 1.073257 |
| 18 | 7145 | 0.506064 | 1.061074 |
| 19 | 7542 | 0.510928 | 1.056646 |
| 20 | 7939 | 0.513825 | 1.069642 |
| 21 | 8336 | 0.508816 | 1.077172 |
| 22 | 8733 | 0.507693 | 1.132512 |
| 23 | 9130 | 0.464279 | 1.228043 |
| 24 | 9527 | 0.475600 | 1.201767 |
| 25 | 9924 | 0.518130 | 1.096784 |
| 26 | 10321 | 0.522530 | 1.058416 |
| 27 | 10718 | 0.522476 | 1.085311 |
| 28 | 11115 | 0.533021 | 1.085067 |
| 29 | 11512 | 0.533857 | 1.085778 |
| 30 | 11909 | 0.537981 | 1.093806 |
| 31 | 12306 | 0.463357 | 1.285506 |
| 32 | 12703 | 0.512071 | 1.150658 |
| 33 | 13100 | 0.500597 | 1.158814 |
| 34 | 13497 | 0.527148 | 1.132308 |
| 35 | 13894 | 0.528191 | 1.113778 |
| 36 | 14291 | 0.535651 | 1.111421 |
| 37 | 14688 | 0.540963 | 1.120996 |
| 38 | 15085 | 0.538772 | 1.126550 |
| 39 | 15482 | 0.534594 | 1.136285 |
| 40 | 15879 | 0.535083 | 1.173313 |
| 41 | 16276 | 0.486855 | 1.279203 |
| 42 | 16673 | 0.503469 | 1.204238 |
| 43 | 17070 | 0.514029 | 1.168278 |
| 44 | 17467 | 0.524972 | 1.148931 |
| 45 | 17864 | 0.533803 | 1.173315 |
| 46 | 18261 | 0.536479 | 1.176281 |
| 47 | 18658 | 0.536035 | 1.173078 |
| 48 | 19055 | 0.536445 | 1.183411 |
| 49 | 19452 | 0.541121 | 1.209043 |
| 50 | 19849 | 0.527420 | 1.211359 |
