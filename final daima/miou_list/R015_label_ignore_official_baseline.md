# R015 Label/Ignore Official Baseline mIoU

- Run: `R015_label_ignore_official_baseline`
- Branch: `exp/R015-label-ignore-contract-v1`
- Model: `dformerv2_mid_fusion`
- Contract: official NYU label mapping reset, raw `0 -> 255` ignore and raw `1..40 -> 0..39`
- TensorBoard event: `final daima/checkpoints/R015_label_ignore_official_baseline/lightning_logs/version_0/events.out.tfevents.1778734783.Administrator.15996.0`
- Best checkpoint: `final daima/checkpoints/R015_label_ignore_official_baseline/dformerv2_mid_fusion-epoch=44-val_mIoU=0.5374.pt`
- Recorded validation epochs: `50`
- Best val/mIoU: `0.537398` at validation epoch `45`
- Last val/mIoU: `0.499418`
- Last-5 mean val/mIoU: `0.520010`
- Last-10 mean val/mIoU: `0.520691`
- Best-to-last drop: `0.037981`
- Best val/loss: `0.969897` at validation epoch `10`
- Last val/loss: `1.291720`
- Final train/loss_epoch: `0.093611`
- Exit code: `0`

R015 satisfies the fixed-recipe `0.53` stage target with real TensorBoard and checkpoint evidence. Because it resets label/ignore semantics, it establishes the new official-label baseline and should not be reported as a direct old-contract improvement over earlier runs.

| Epoch | val/mIoU | val/loss | train/loss_epoch |
|---:|---:|---:|---:|
| 1 | 0.167783 | 1.637273 | 2.240000 |
| 2 | 0.251912 | 1.337056 | 1.544917 |
| 3 | 0.303886 | 1.192700 | 1.216296 |
| 4 | 0.371014 | 1.110971 | 0.964845 |
| 5 | 0.417564 | 1.026556 | 0.790841 |
| 6 | 0.434774 | 1.009868 | 0.653410 |
| 7 | 0.466896 | 0.971217 | 0.557675 |
| 8 | 0.466748 | 0.984339 | 0.479007 |
| 9 | 0.461278 | 0.998361 | 0.395680 |
| 10 | 0.478814 | 0.969897 | 0.346911 |
| 11 | 0.484061 | 0.985572 | 0.324626 |
| 12 | 0.486452 | 1.061595 | 0.284624 |
| 13 | 0.500580 | 0.991659 | 0.245087 |
| 14 | 0.498167 | 1.000100 | 0.210549 |
| 15 | 0.496629 | 1.065974 | 0.226426 |
| 16 | 0.502293 | 1.008661 | 0.196379 |
| 17 | 0.500324 | 1.059132 | 0.193387 |
| 18 | 0.467774 | 1.191092 | 0.207216 |
| 19 | 0.500272 | 1.064690 | 0.181624 |
| 20 | 0.514546 | 1.031093 | 0.151086 |
| 21 | 0.523882 | 1.022078 | 0.124470 |
| 22 | 0.518174 | 1.058665 | 0.120053 |
| 23 | 0.520095 | 1.048855 | 0.110636 |
| 24 | 0.497150 | 1.110838 | 0.175842 |
| 25 | 0.524953 | 1.046785 | 0.141874 |
| 26 | 0.522216 | 1.109178 | 0.106967 |
| 27 | 0.513531 | 1.116469 | 0.098073 |
| 28 | 0.518024 | 1.125176 | 0.093004 |
| 29 | 0.490397 | 1.187788 | 0.118918 |
| 30 | 0.512095 | 1.112577 | 0.101661 |
| 31 | 0.517979 | 1.100530 | 0.086854 |
| 32 | 0.503665 | 1.188188 | 0.100373 |
| 33 | 0.471982 | 1.240730 | 0.131828 |
| 34 | 0.506342 | 1.126898 | 0.168297 |
| 35 | 0.520965 | 1.123177 | 0.105961 |
| 36 | 0.501414 | 1.170912 | 0.080467 |
| 37 | 0.502828 | 1.150173 | 0.093733 |
| 38 | 0.520641 | 1.145588 | 0.077536 |
| 39 | 0.534226 | 1.124773 | 0.067289 |
| 40 | 0.532002 | 1.158620 | 0.063504 |
| 41 | 0.522334 | 1.202455 | 0.075722 |
| 42 | 0.502360 | 1.166193 | 0.117986 |
| 43 | 0.518998 | 1.192833 | 0.083376 |
| 44 | 0.525763 | 1.184225 | 0.066586 |
| 45 | 0.537398 | 1.158207 | 0.060698 |
| 46 | 0.528022 | 1.204619 | 0.056915 |
| 47 | 0.530376 | 1.211396 | 0.058071 |
| 48 | 0.523453 | 1.237482 | 0.058223 |
| 49 | 0.518783 | 1.247728 | 0.067836 |
| 50 | 0.499418 | 1.291720 | 0.093611 |
