# R042 DiffPixel C3-to-C4 Cue Run01 mIoU Detail

- Run: `R042_diffpixel_c3toc4_cue_run01`
- Model: `dformerv2_diffpixel_c3toc4_cue`
- Branch: `exp/R042-diffpixel-c3toc4-cue-v1`
- TensorBoard event: `final daima/checkpoints/R042_diffpixel_c3toc4_cue_run01/lightning_logs/version_0/events.out.tfevents.1778887449.Administrator.10152.0`
- Best checkpoint: `final daima/checkpoints/R042_diffpixel_c3toc4_cue_run01/dformerv2_diffpixel_c3toc4_cue-epoch=42-val_mIoU=0.5307.pt`
- Status: completed full train, exit code `0`

## Summary

- Validation epochs: `50`
- Best val/mIoU: `0.530729` at validation epoch `43`
- Last val/mIoU: `0.458179`
- Last-5 mean val/mIoU: `0.505197`
- Last-10 mean val/mIoU: `0.510157`
- Best-to-last drop: `0.072551`
- Best val/loss: `0.957006` at validation epoch `10`
- Last val/loss: `1.417108`
- Final train/loss_epoch: `0.065262`

## Diagnostics

- c3-to-c4 cue_abs first/last: `0.018134` / `0.246822`
- c3-to-c4 diff_context_abs first/last: `0.765808` / `0.773092`
- c3-to-c4 gate_mean first/last: `0.500000` / `0.545177`
- c3-to-c4 gate_std first/last: `0.208145` / `0.215891`

## Per-Epoch val/mIoU

| Validation epoch | Global step | val/mIoU |
|---:|---:|---:|
| 1 | 396 | 0.179216 |
| 2 | 793 | 0.264999 |
| 3 | 1190 | 0.306072 |
| 4 | 1587 | 0.356016 |
| 5 | 1984 | 0.412732 |
| 6 | 2381 | 0.438822 |
| 7 | 2778 | 0.430962 |
| 8 | 3175 | 0.463600 |
| 9 | 3572 | 0.456953 |
| 10 | 3969 | 0.492739 |
| 11 | 4366 | 0.463797 |
| 12 | 4763 | 0.501676 |
| 13 | 5160 | 0.492980 |
| 14 | 5557 | 0.478378 |
| 15 | 5954 | 0.496638 |
| 16 | 6351 | 0.490246 |
| 17 | 6748 | 0.500521 |
| 18 | 7145 | 0.497926 |
| 19 | 7542 | 0.511538 |
| 20 | 7939 | 0.508726 |
| 21 | 8336 | 0.497802 |
| 22 | 8733 | 0.521582 |
| 23 | 9130 | 0.519822 |
| 24 | 9527 | 0.522238 |
| 25 | 9924 | 0.524098 |
| 26 | 10321 | 0.508705 |
| 27 | 10718 | 0.506475 |
| 28 | 11115 | 0.504038 |
| 29 | 11512 | 0.519127 |
| 30 | 11909 | 0.514131 |
| 31 | 12306 | 0.506710 |
| 32 | 12703 | 0.450029 |
| 33 | 13100 | 0.503305 |
| 34 | 13497 | 0.520831 |
| 35 | 13894 | 0.523235 |
| 36 | 14291 | 0.527953 |
| 37 | 14688 | 0.499767 |
| 38 | 15085 | 0.500962 |
| 39 | 15482 | 0.516647 |
| 40 | 15879 | 0.500174 |
| 41 | 16276 | 0.502738 |
| 42 | 16673 | 0.517243 |
| 43 | 17070 | 0.530729 |
| 44 | 17467 | 0.527811 |
| 45 | 17864 | 0.497059 |
| 46 | 18261 | 0.517289 |
| 47 | 18658 | 0.501041 |
| 48 | 19055 | 0.519576 |
| 49 | 19452 | 0.529901 |
| 50 | 19849 | 0.458179 |
