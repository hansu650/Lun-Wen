# dformerv2_geometry_primary_teacher_run01

## Settings

- model: `dformerv2_geometry_primary_teacher`
- purpose: Phase 0 geometry-primary teacher sanity check for PMAD / PrimKD logit distillation
- architecture: `DFormerv2_S(rgb, real_depth) + SimpleFPNDecoder`; no extra `DepthEncoder + GatedFusion` branch
- settings: `batch_size=2`, `max_epochs=50`, `lr=6e-5`, `num_workers=4`, `early_stop_patience=30`, `loss_type=ce`
- pretrained: `C:/Users/qintian/Desktop/qintian/dformer_work/checkpoints/pretrained/DFormerv2_Small_pretrained.pth`
- checkpoint: `checkpoints/dformerv2_geometry_primary_teacher_run01/dformerv2_geometry_primary_teacher-epoch=37-val_mIoU=0.5168.pt`

## Summary

- recorded validation epochs: `50`
- best val/mIoU: `0.516824` at epoch `38`
- last val/mIoU: `0.509223`
- best val/loss: `1.032507` at epoch `8`
- last val/loss: `1.263402`
- mean val/mIoU over last 10 epochs: `0.504379`
- teacher usability threshold: `0.515000`
- strong teacher threshold (baseline mean + 1 std): `0.522298`
- clean 10-run RGB-D baseline mean best: `0.517397`
- clean 10-run RGB-D baseline std: `0.004901`
- clean 10-run RGB-D baseline best single: `0.524425`
- constant-zero teacher best: `0.488489`
- delta vs teacher usability threshold: `0.001824`
- delta vs clean baseline mean: `-0.000573`
- delta vs clean baseline mean in std units: `-0.117`
- delta vs constant-zero teacher: `0.028335`

## Per-Epoch Metrics

| epoch | step | val/mIoU | val/loss | train/loss_epoch |
|---:|---:|---:|---:|---:|
| 1 | 396 | 0.150816 | 1.727539 | 2.307330 |
| 2 | 793 | 0.206841 | 1.435525 | 1.620199 |
| 3 | 1190 | 0.259072 | 1.264483 | 1.318987 |
| 4 | 1587 | 0.311002 | 1.157716 | 1.107618 |
| 5 | 1984 | 0.354407 | 1.118998 | 0.949291 |
| 6 | 2381 | 0.399315 | 1.074237 | 0.830693 |
| 7 | 2778 | 0.428614 | 1.055889 | 0.759322 |
| 8 | 3175 | 0.428460 | 1.032507 | 0.655597 |
| 9 | 3572 | 0.458297 | 1.037090 | 0.580888 |
| 10 | 3969 | 0.456524 | 1.046756 | 0.537235 |
| 11 | 4366 | 0.460912 | 1.043828 | 0.496768 |
| 12 | 4763 | 0.478960 | 1.045522 | 0.449913 |
| 13 | 5160 | 0.459175 | 1.094287 | 0.419051 |
| 14 | 5557 | 0.444520 | 1.206355 | 0.405124 |
| 15 | 5954 | 0.476841 | 1.079435 | 0.397215 |
| 16 | 6351 | 0.460373 | 1.098639 | 0.400885 |
| 17 | 6748 | 0.485236 | 1.093730 | 0.349330 |
| 18 | 7145 | 0.495619 | 1.107050 | 0.318979 |
| 19 | 7542 | 0.499759 | 1.088187 | 0.304619 |
| 20 | 7939 | 0.500325 | 1.119585 | 0.291020 |
| 21 | 8336 | 0.502021 | 1.137976 | 0.278315 |
| 22 | 8733 | 0.498961 | 1.131125 | 0.291214 |
| 23 | 9130 | 0.481701 | 1.197711 | 0.296555 |
| 24 | 9527 | 0.492950 | 1.175778 | 0.283380 |
| 25 | 9924 | 0.501042 | 1.156910 | 0.258182 |
| 26 | 10321 | 0.501489 | 1.169915 | 0.245053 |
| 27 | 10718 | 0.491106 | 1.171836 | 0.244135 |
| 28 | 11115 | 0.476930 | 1.237486 | 0.233897 |
| 29 | 11512 | 0.480318 | 1.277150 | 0.248837 |
| 30 | 11909 | 0.489259 | 1.208430 | 0.254023 |
| 31 | 12306 | 0.484098 | 1.273972 | 0.235425 |
| 32 | 12703 | 0.489610 | 1.203360 | 0.247327 |
| 33 | 13100 | 0.477748 | 1.250185 | 0.250225 |
| 34 | 13497 | 0.498002 | 1.179747 | 0.244244 |
| 35 | 13894 | 0.499253 | 1.203520 | 0.208130 |
| 36 | 14291 | 0.509178 | 1.187311 | 0.203418 |
| 37 | 14688 | 0.511947 | 1.207199 | 0.192611 |
| 38 | 15085 | 0.516824 | 1.187733 | 0.187653 |
| 39 | 15482 | 0.510435 | 1.222192 | 0.184874 |
| 40 | 15879 | 0.507239 | 1.238750 | 0.182635 |
| 41 | 16276 | 0.509036 | 1.267527 | 0.184712 |
| 42 | 16673 | 0.505791 | 1.226267 | 0.197160 |
| 43 | 17070 | 0.494900 | 1.255426 | 0.239211 |
| 44 | 17467 | 0.488159 | 1.251902 | 0.204064 |
| 45 | 17864 | 0.508364 | 1.226235 | 0.189768 |
| 46 | 18261 | 0.506608 | 1.257723 | 0.177795 |
| 47 | 18658 | 0.508926 | 1.263504 | 0.181708 |
| 48 | 19055 | 0.505785 | 1.262845 | 0.199589 |
| 49 | 19452 | 0.506994 | 1.283643 | 0.168156 |
| 50 | 19849 | 0.509223 | 1.263402 | 0.161813 |

## Interpretation

- The geometry-primary teacher passes the Phase 0 teacher sanity gate.
- Best val/mIoU `0.516824` is above the minimum teacher gate `0.515000` by `0.001824`.
- It is essentially tied with the full RGB-D baseline mean: delta `-0.000573` (`-0.117` std units).
- It improves over the failed constant-zero teacher by `0.028335`, confirming that real DFormerV2 geometry prior is necessary.
- Since this teacher is usable but not strong, use conservative PMAD settings such as `kd_weight=0.15`, `kd_temperature=4.0`.

## Decision

- Proceed to Phase 1 PMAD logit-only with this teacher checkpoint.
- Do not repeat teacher immediately; only repeat teacher if PMAD later shows a meaningful positive signal and teacher variance becomes important for the paper.
