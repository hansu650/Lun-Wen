# R017 RGB/BGR Official Channel Contract

## Summary

- Branch: `exp/R017-rgb-bgr-contract-v1`
- Model: `dformerv2_mid_fusion`
- Run: `R017_rgb_bgr_official_contract`
- Hypothesis: after R015/R016 align official label and depth contracts, RGB channel order might also need to match the official DFormer NYUDepthV2 BGR input path.
- Status: completed full train; `Trainer.fit` reached `max_epochs=50`; exit code `0`.
- TensorBoard event: `checkpoints/R017_rgb_bgr_official_contract/lightning_logs/version_0/events.out.tfevents.1778750750.Administrator.38244.0`
- Best checkpoint: `checkpoints/R017_rgb_bgr_official_contract/dformerv2_mid_fusion-epoch=37-val_mIoU=0.5291.pt`
- Recorded validation epochs: `50`
- Best val/mIoU: `0.529090` at validation epoch `38`
- Last val/mIoU: `0.523078`
- Last-5 mean val/mIoU: `0.494251`
- Last-10 mean val/mIoU: `0.506949`
- Best-to-last drop: `0.006011`
- Best val/loss: `0.973518` at validation epoch `9`
- Last val/loss: `1.228286`
- Final train/loss_epoch: `0.063107`
- Delta vs R016 official-label-and-depth baseline best `0.541121`: `-0.012031`
- Contract note: compare this run only against R016 and later official-contract runs, not against old label-contract results.

## Per-Epoch Validation Metrics

| Val Epoch | Global Step | val/mIoU | val/loss |
|---:|---:|---:|---:|
| 1 | 396 | 0.176391 | 1.644496 |
| 2 | 793 | 0.257211 | 1.349380 |
| 3 | 1190 | 0.301426 | 1.212718 |
| 4 | 1587 | 0.354172 | 1.100591 |
| 5 | 1984 | 0.400238 | 1.047957 |
| 6 | 2381 | 0.426883 | 1.002181 |
| 7 | 2778 | 0.442170 | 0.983839 |
| 8 | 3175 | 0.452771 | 1.017991 |
| 9 | 3572 | 0.460346 | 0.973518 |
| 10 | 3969 | 0.476430 | 0.979476 |
| 11 | 4366 | 0.482702 | 0.981964 |
| 12 | 4763 | 0.486263 | 1.008390 |
| 13 | 5160 | 0.494418 | 0.993623 |
| 14 | 5557 | 0.471421 | 1.056922 |
| 15 | 5954 | 0.475974 | 1.074250 |
| 16 | 6351 | 0.492245 | 1.052305 |
| 17 | 6748 | 0.484026 | 1.035153 |
| 18 | 7145 | 0.503782 | 1.013935 |
| 19 | 7542 | 0.511651 | 1.023034 |
| 20 | 7939 | 0.506764 | 1.055075 |
| 21 | 8336 | 0.504041 | 1.084337 |
| 22 | 8733 | 0.510356 | 1.094065 |
| 23 | 9130 | 0.488855 | 1.090386 |
| 24 | 9527 | 0.508547 | 1.094468 |
| 25 | 9924 | 0.472827 | 1.186851 |
| 26 | 10321 | 0.461307 | 1.230871 |
| 27 | 10718 | 0.506249 | 1.095366 |
| 28 | 11115 | 0.515176 | 1.096661 |
| 29 | 11512 | 0.516615 | 1.111842 |
| 30 | 11909 | 0.511327 | 1.117102 |
| 31 | 12306 | 0.515197 | 1.121595 |
| 32 | 12703 | 0.519664 | 1.095388 |
| 33 | 13100 | 0.517850 | 1.126463 |
| 34 | 13497 | 0.516008 | 1.138427 |
| 35 | 13894 | 0.478175 | 1.218047 |
| 36 | 14291 | 0.509947 | 1.174981 |
| 37 | 14688 | 0.505819 | 1.169020 |
| 38 | 15085 | 0.529090 | 1.135901 |
| 39 | 15482 | 0.523454 | 1.162511 |
| 40 | 15879 | 0.519910 | 1.179200 |
| 41 | 16276 | 0.508808 | 1.199978 |
| 42 | 16673 | 0.511853 | 1.189394 |
| 43 | 17070 | 0.520578 | 1.213420 |
| 44 | 17467 | 0.528616 | 1.164990 |
| 45 | 17864 | 0.528378 | 1.169394 |
| 46 | 18261 | 0.455516 | 1.371004 |
| 47 | 18658 | 0.510153 | 1.214424 |
| 48 | 19055 | 0.473732 | 1.295249 |
| 49 | 19452 | 0.508774 | 1.194724 |
| 50 | 19849 | 0.523078 | 1.228286 |
