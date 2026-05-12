# dformerv2_tgga_c34_noaux_semgrad_beta002_simplefpn_v1_run01

## Summary

- model: `dformerv2_tgga_c34_noaux_semgrad_beta002_simplefpn_v1`
- method: TGGA on DFormerV2 c3/c4, no auxiliary CE, semantic cue trained only through final CE via gate path
- loss: `CE(final_logits, label)` only
- training: `batch_size=2`, `max_epochs=50`, `lr=6e-5`, `num_workers=4`, `early_stop_patience=30`
- recorded validation epochs: `50`
- best val/mIoU: `0.512152` at epoch `48`
- last val/mIoU: `0.492633`
- best val/loss: `1.032364` at epoch `15`
- last val/loss: `1.275699`
- mean val/mIoU over last 10 epochs: `0.501366`
- population std val/mIoU over last 10 epochs: `0.012195`
- mean val/mIoU over last 5 epochs: `0.498370`
- best-epoch local 5-epoch window mean: `0.498370` over epochs `46-50`
- post-best mean val/mIoU: `0.481050`
- best-to-last val/mIoU drop: `0.019520`
- largest single-epoch val/mIoU drop: `-0.053409` from epoch `37` to epoch `38`
- final train/loss: `0.187033`
- final train/main_loss: `0.187033`
- final train/tgga_aux_ce_c3_diag: `3.880328`
- final train/tgga_aux_ce_c4_diag: `3.911476`
- final train/tgga_beta_c3: `0.035324`
- final train/tgga_beta_c4: `0.025326`
- final train/tgga_gate_c3_mean: `0.474472`
- final train/tgga_gate_c4_mean: `0.230513`
- final train/tgga_gate_c3_std: `0.311781`
- final train/tgga_gate_c4_std: `0.135297`
- comparison clean 10-run RGB-D baseline mean best: `0.517397`
- comparison clean 10-run RGB-D baseline std: `0.004901`
- comparison clean 10-run RGB-D baseline mean + 1 std: `0.522298`
- comparison clean 10-run RGB-D baseline best single: `0.524425`
- comparison PMAD logit-only w0.15 5-run mean best: `0.520795`
- comparison original TGGA aux run01 best: `0.522206`
- comparison original TGGA aux run02 best: `0.517437`
- delta vs clean baseline mean: `-0.005245` (`-1.070` baseline std units)
- delta vs clean baseline mean + 1 std: `-0.010146`
- delta vs PMAD w0.15 mean: `-0.008643`
- delta vs original TGGA aux run01: `-0.010054`
- delta vs original TGGA aux run02: `-0.005285`
- epochs above clean baseline mean: `0/50`
- checkpoint: `checkpoints/dformerv2_tgga_c34_noaux_semgrad_beta002_simplefpn_v1_run01/dformerv2_tgga_c34_noaux_semgrad_beta002_simplefpn_v1-epoch=47-val_mIoU=0.5122.pt`
- tensorboard event: `checkpoints/dformerv2_tgga_c34_noaux_semgrad_beta002_simplefpn_v1_run01/lightning_logs/version_0/events.out.tfevents.1778587242.Administrator.39208.0`

## Late-Curve Check

- The best epoch is epoch 48, not the final epoch.
- Epoch 41-50 val/mIoU: `0.495870, 0.503853, 0.504099, 0.508151, 0.509840, 0.510360, 0.507238, 0.512152, 0.469468, 0.492633`
- The curve still has late instability: epoch 48 reaches the best point, epoch 49 drops sharply to `0.469468`, and epoch 50 only recovers to `0.492633`.
- Compared with original TGGA aux runs, removing aux CE lowers the peak substantially and does not eliminate late fluctuation.
- Diagnostic interpretation: auxiliary CE is not the only cause of collapse. TGGA gate/residual dynamics, especially the increasingly open c3 gate and now also a more open c4 gate, remain problematic.

## Per-Epoch Metrics

| epoch | global step | val/mIoU | val/loss |
|---:|---:|---:|---:|
| 1 | 396 | 0.172671 | 1.683213 |
| 2 | 793 | 0.229452 | 1.359233 |
| 3 | 1190 | 0.298782 | 1.232930 |
| 4 | 1587 | 0.348205 | 1.145880 |
| 5 | 1984 | 0.376082 | 1.107121 |
| 6 | 2381 | 0.422124 | 1.046338 |
| 7 | 2778 | 0.438552 | 1.059916 |
| 8 | 3175 | 0.452488 | 1.060095 |
| 9 | 3572 | 0.456054 | 1.061962 |
| 10 | 3969 | 0.419656 | 1.171489 |
| 11 | 4366 | 0.475744 | 1.040359 |
| 12 | 4763 | 0.463465 | 1.095581 |
| 13 | 5160 | 0.470134 | 1.092744 |
| 14 | 5557 | 0.485316 | 1.049382 |
| 15 | 5954 | 0.489281 | 1.032364 |
| 16 | 6351 | 0.482812 | 1.090944 |
| 17 | 6748 | 0.479990 | 1.134869 |
| 18 | 7145 | 0.489828 | 1.085855 |
| 19 | 7542 | 0.494119 | 1.088286 |
| 20 | 7939 | 0.489013 | 1.092127 |
| 21 | 8336 | 0.479191 | 1.140114 |
| 22 | 8733 | 0.486711 | 1.107043 |
| 23 | 9130 | 0.434030 | 1.250041 |
| 24 | 9527 | 0.489297 | 1.108465 |
| 25 | 9924 | 0.497098 | 1.137373 |
| 26 | 10321 | 0.500880 | 1.112727 |
| 27 | 10718 | 0.495403 | 1.144282 |
| 28 | 11115 | 0.499406 | 1.161039 |
| 29 | 11512 | 0.498003 | 1.165355 |
| 30 | 11909 | 0.502616 | 1.171885 |
| 31 | 12306 | 0.504988 | 1.156521 |
| 32 | 12703 | 0.490698 | 1.173529 |
| 33 | 13100 | 0.490587 | 1.148775 |
| 34 | 13497 | 0.502053 | 1.167703 |
| 35 | 13894 | 0.506139 | 1.186797 |
| 36 | 14291 | 0.485140 | 1.252164 |
| 37 | 14688 | 0.502977 | 1.145099 |
| 38 | 15085 | 0.449568 | 1.325716 |
| 39 | 15482 | 0.463541 | 1.247294 |
| 40 | 15879 | 0.491486 | 1.200864 |
| 41 | 16276 | 0.495870 | 1.178547 |
| 42 | 16673 | 0.503853 | 1.194561 |
| 43 | 17070 | 0.504099 | 1.217585 |
| 44 | 17467 | 0.508151 | 1.213201 |
| 45 | 17864 | 0.509840 | 1.227473 |
| 46 | 18261 | 0.510360 | 1.241082 |
| 47 | 18658 | 0.507238 | 1.248768 |
| 48 | 19055 | 0.512152 | 1.239474 |
| 49 | 19452 | 0.469468 | 1.346247 |
| 50 | 19849 | 0.492633 | 1.275699 |

## Conclusion

Negative diagnostic result. Removing auxiliary CE does not recover stability or performance: best val/mIoU is `0.512152`, no epoch beats the clean baseline mean, and late mIoU still collapses. The original TGGA peak appears to depend partly on aux CE, but the instability is not only an aux-loss conflict; TGGA gate/residual dynamics themselves are likely unsafe in this form.
