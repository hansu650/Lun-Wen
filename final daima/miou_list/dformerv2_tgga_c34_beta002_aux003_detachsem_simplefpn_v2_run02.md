# dformerv2_tgga_c34_beta002_aux003_detachsem_simplefpn_v2_run02

## Summary

- model: `dformerv2_tgga_c34_beta002_aux003_detachsem_simplefpn_v2`
- method: TGGA on DFormerV2 c3/c4 before external DepthEncoder + GatedFusion
- loss: `CE(final_logits, label) + 0.03 * CE(aux_c3, label) + 0.03 * CE(aux_c4, label)`
- training: `batch_size=2`, `max_epochs=50`, `lr=6e-5`, `num_workers=4`, `early_stop_patience=30`
- recorded validation epochs: `50`
- best val/mIoU: `0.517437` at epoch `49`
- last val/mIoU: `0.486566`
- best val/loss: `0.995109` at epoch `10`
- last val/loss: `1.387486`
- mean val/mIoU over last 10 epochs: `0.501329`
- population std val/mIoU over last 10 epochs: `0.015245`
- mean val/mIoU over last 5 epochs: `0.501959`
- best-epoch local 5-epoch window mean: `0.505654` over epochs `47-50`
- post-best mean val/mIoU: `0.486566`
- best-to-last val/mIoU drop: `0.030871`
- largest single-epoch val/mIoU drop: `-0.030871` from epoch `49` to epoch `50`
- final train/loss: `0.160672`
- final train/main_loss: `0.140603`
- final train/tgga_aux_loss_c3: `0.301237`
- final train/tgga_aux_loss_c4: `0.367719`
- final train/tgga_beta_c3: `0.039473`
- final train/tgga_beta_c4: `0.022773`
- final train/tgga_gate_c3_mean: `0.409689`
- final train/tgga_gate_c4_mean: `0.126628`
- final train/tgga_gate_c3_std: `0.346383`
- final train/tgga_gate_c4_std: `0.010253`
- comparison clean 10-run RGB-D baseline mean best: `0.517397`
- comparison clean 10-run RGB-D baseline std: `0.004901`
- comparison clean 10-run RGB-D baseline mean + 1 std: `0.522298`
- comparison clean 10-run RGB-D baseline best single: `0.524425`
- comparison PMAD logit-only w0.15 5-run mean best: `0.520795`
- comparison bounded class-context 5-run mean best: `0.515986`
- delta vs clean baseline mean: `+0.000040` (`+0.008` baseline std units)
- delta vs clean baseline mean + 1 std: `-0.004861`
- delta vs clean baseline best single: `-0.006988`
- delta vs PMAD w0.15 mean: `-0.003358`
- delta vs bounded class-context mean: `+0.001451`
- checkpoint: `checkpoints/dformerv2_tgga_c34_beta002_aux003_detachsem_simplefpn_v2_run02/dformerv2_tgga_c34_beta002_aux003_detachsem_simplefpn_v2-epoch=48-val_mIoU=0.5174.pt`
- tensorboard event: `checkpoints/dformerv2_tgga_c34_beta002_aux003_detachsem_simplefpn_v2_run02/lightning_logs/version_0/events.out.tfevents.1778578030.Administrator.31972.0`

## Late-Curve Check

- The best epoch is epoch 49, not the final epoch.
- Epoch 41-50 val/mIoU: `0.510171, 0.512682, 0.516483, 0.495031, 0.469128, 0.487177, 0.507587, 0.511026, 0.517437, 0.486566`
- Only one epoch is above the clean baseline mean (`0.517397`), and no epoch exceeds the baseline mean + 1 std (`0.522298`) or PMAD w0.15 5-run mean (`0.520795`).
- The late curve is highly oscillatory and ends with a sharp epoch 49 -> 50 collapse from `0.517437` to `0.486566`.
- Interpretation: run02 confirms the late instability observed in run01, but with a weaker peak. This run is essentially tied with the clean baseline mean on best mIoU and clearly worse on late stability.

## Per-Epoch Metrics

| epoch | global step | val/mIoU | val/loss |
|---:|---:|---:|---:|
| 1 | 396 | 0.159979 | 1.649599 |
| 2 | 793 | 0.212244 | 1.436775 |
| 3 | 1190 | 0.278345 | 1.264400 |
| 4 | 1587 | 0.321212 | 1.149468 |
| 5 | 1984 | 0.363342 | 1.125736 |
| 6 | 2381 | 0.407659 | 1.074422 |
| 7 | 2778 | 0.424220 | 1.072995 |
| 8 | 3175 | 0.455537 | 1.031358 |
| 9 | 3572 | 0.451868 | 1.040408 |
| 10 | 3969 | 0.475236 | 0.995109 |
| 11 | 4366 | 0.475235 | 1.028911 |
| 12 | 4763 | 0.458060 | 1.097741 |
| 13 | 5160 | 0.479167 | 1.060364 |
| 14 | 5557 | 0.487428 | 1.062658 |
| 15 | 5954 | 0.467394 | 1.115678 |
| 16 | 6351 | 0.470337 | 1.091897 |
| 17 | 6748 | 0.474329 | 1.117589 |
| 18 | 7145 | 0.486038 | 1.079106 |
| 19 | 7542 | 0.482659 | 1.145911 |
| 20 | 7939 | 0.455570 | 1.178581 |
| 21 | 8336 | 0.458549 | 1.177560 |
| 22 | 8733 | 0.499428 | 1.092006 |
| 23 | 9130 | 0.497948 | 1.111168 |
| 24 | 9527 | 0.503790 | 1.113846 |
| 25 | 9924 | 0.503610 | 1.126124 |
| 26 | 10321 | 0.505311 | 1.143976 |
| 27 | 10718 | 0.501903 | 1.179272 |
| 28 | 11115 | 0.498454 | 1.184593 |
| 29 | 11512 | 0.505490 | 1.177563 |
| 30 | 11909 | 0.515080 | 1.182031 |
| 31 | 12306 | 0.489188 | 1.238340 |
| 32 | 12703 | 0.467148 | 1.305787 |
| 33 | 13100 | 0.476795 | 1.196288 |
| 34 | 13497 | 0.480634 | 1.240535 |
| 35 | 13894 | 0.504169 | 1.179870 |
| 36 | 14291 | 0.480045 | 1.258668 |
| 37 | 14688 | 0.498289 | 1.204588 |
| 38 | 15085 | 0.503210 | 1.235997 |
| 39 | 15482 | 0.502925 | 1.227198 |
| 40 | 15879 | 0.505234 | 1.222095 |
| 41 | 16276 | 0.510171 | 1.249878 |
| 42 | 16673 | 0.512682 | 1.226846 |
| 43 | 17070 | 0.516483 | 1.231170 |
| 44 | 17467 | 0.495031 | 1.320824 |
| 45 | 17864 | 0.469128 | 1.344700 |
| 46 | 18261 | 0.487177 | 1.304071 |
| 47 | 18658 | 0.507587 | 1.257278 |
| 48 | 19055 | 0.511026 | 1.252727 |
| 49 | 19452 | 0.517437 | 1.278357 |
| 50 | 19849 | 0.486566 | 1.387486 |

## Conclusion

Run02 is not a stable improvement. Its best val/mIoU is only `+0.000040` above the clean baseline mean, it stays below PMAD w0.15's five-run mean, and it repeats the late-collapse pattern from run01. Together with run01, this suggests TGGA has a real but unstable late-stage signal rather than a reliable improvement.
