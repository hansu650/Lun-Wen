# dformerv2_primkd_logit_only_w015_t4_run01

## Settings

- model: `dformerv2_primkd_logit_only`
- purpose: Phase 1 PMAD / PrimKD logit-only distillation with geometry-primary teacher
- student: `DFormerv2_S + ResNet-18 DepthEncoder + GatedFusion + SimpleFPNDecoder`
- teacher: `dformerv2_geometry_primary_teacher_run01`, `DFormerv2_S(rgb, real_depth) + SimpleFPNDecoder`, frozen during KD
- teacher checkpoint: `checkpoints/dformerv2_geometry_primary_teacher_run01/dformerv2_geometry_primary_teacher-epoch=37-val_mIoU=0.5168.pt`
- settings: `batch_size=2`, `max_epochs=50`, `lr=6e-5`, `num_workers=4`, `early_stop_patience=30`, `loss_type=ce`
- KD: `kd_weight=0.15`, `kd_temperature=4.0`, logit-only, no feature KD
- pretrained: `C:/Users/qintian/Desktop/qintian/dformer_work/checkpoints/pretrained/DFormerv2_Small_pretrained.pth`
- checkpoint: `checkpoints/dformerv2_primkd_logit_only_w015_t4_run01/dformerv2_primkd_logit_only-epoch=47-val_mIoU=0.5230.pt`

## Summary

- recorded validation epochs: `50`
- best val/mIoU: `0.522998` at epoch `48`
- last val/mIoU: `0.513176`
- best val/loss: `1.060515` at epoch `7`
- last val/loss: `1.203945`
- mean val/mIoU over last 10 epochs: `0.504683`
- final train/loss_epoch: `0.246043`
- final train/ce_loss_epoch: `0.172287`
- final train/kd_loss_epoch: `0.491706`
- clean 10-run RGB-D baseline mean best: `0.517397`
- clean 10-run RGB-D baseline std: `0.004901`
- clean 10-run RGB-D baseline mean + 1 std: `0.522298`
- clean 10-run RGB-D baseline best single: `0.524425`
- repeat5 RGB-D baseline mean best: `0.511893`
- geometry-primary teacher best: `0.516824`
- delta vs clean baseline mean: `0.005601`
- delta vs clean baseline mean in std units: `1.143`
- delta vs clean baseline mean + 1 std: `0.000700`
- delta vs clean baseline best single: `-0.001427`
- delta vs repeat5 mean: `0.011105`
- delta vs teacher best: `0.006174`

## Per-Epoch Metrics

| epoch | step | val/mIoU | val/loss | train/loss_epoch | train/ce_loss_epoch | train/kd_loss_epoch |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 396 | 0.169706 | 1.614423 | 3.417742 | 2.214170 | 8.023812 |
| 2 | 793 | 0.218149 | 1.325049 | 2.476783 | 1.535408 | 6.275834 |
| 3 | 1190 | 0.254014 | 1.234763 | 2.028919 | 1.241778 | 5.247612 |
| 4 | 1587 | 0.312488 | 1.142576 | 1.702981 | 1.035149 | 4.452214 |
| 5 | 1984 | 0.328137 | 1.131726 | 1.452239 | 0.877836 | 3.829355 |
| 6 | 2381 | 0.363469 | 1.084115 | 1.254460 | 0.758468 | 3.306611 |
| 7 | 2778 | 0.385713 | 1.060515 | 1.088166 | 0.659455 | 2.858071 |
| 8 | 3175 | 0.438439 | 1.070569 | 0.976411 | 0.598011 | 2.522668 |
| 9 | 3572 | 0.420725 | 1.146792 | 0.851348 | 0.524926 | 2.176149 |
| 10 | 3969 | 0.459172 | 1.106632 | 0.765607 | 0.476532 | 1.927166 |
| 11 | 4366 | 0.476247 | 1.093585 | 0.680795 | 0.428700 | 1.680631 |
| 12 | 4763 | 0.475820 | 1.101661 | 0.632867 | 0.404853 | 1.520088 |
| 13 | 5160 | 0.431706 | 1.299265 | 0.590582 | 0.383463 | 1.380791 |
| 14 | 5557 | 0.479768 | 1.138357 | 0.537140 | 0.351620 | 1.236804 |
| 15 | 5954 | 0.487484 | 1.166120 | 0.488528 | 0.323453 | 1.100498 |
| 16 | 6351 | 0.457961 | 1.253154 | 0.464597 | 0.311231 | 1.022444 |
| 17 | 6748 | 0.483136 | 1.192543 | 0.463904 | 0.312375 | 1.010192 |
| 18 | 7145 | 0.459926 | 1.257159 | 0.419493 | 0.285628 | 0.892434 |
| 19 | 7542 | 0.479842 | 1.200202 | 0.421566 | 0.288437 | 0.887525 |
| 20 | 7939 | 0.493760 | 1.198776 | 0.378995 | 0.260620 | 0.789164 |
| 21 | 8336 | 0.488693 | 1.223328 | 0.361384 | 0.251002 | 0.735880 |
| 22 | 8733 | 0.489317 | 1.224605 | 0.346149 | 0.240666 | 0.703221 |
| 23 | 9130 | 0.489434 | 1.227742 | 0.368849 | 0.257285 | 0.743766 |
| 24 | 9527 | 0.484872 | 1.218376 | 0.343633 | 0.240987 | 0.684305 |
| 25 | 9924 | 0.500070 | 1.213683 | 0.336571 | 0.234953 | 0.677452 |
| 26 | 10321 | 0.497186 | 1.222253 | 0.330280 | 0.232491 | 0.651922 |
| 27 | 10718 | 0.493326 | 1.242078 | 0.329309 | 0.230240 | 0.660456 |
| 28 | 11115 | 0.502126 | 1.183602 | 0.305495 | 0.214375 | 0.607466 |
| 29 | 11512 | 0.501799 | 1.205001 | 0.319268 | 0.223999 | 0.635125 |
| 30 | 11909 | 0.492299 | 1.238993 | 0.287537 | 0.203101 | 0.562912 |
| 31 | 12306 | 0.506989 | 1.209882 | 0.290495 | 0.204727 | 0.571786 |
| 32 | 12703 | 0.505043 | 1.215771 | 0.284732 | 0.201189 | 0.556953 |
| 33 | 13100 | 0.505007 | 1.211753 | 0.280429 | 0.198842 | 0.543913 |
| 34 | 13497 | 0.512621 | 1.220977 | 0.267642 | 0.190384 | 0.515056 |
| 35 | 13894 | 0.506030 | 1.219485 | 0.267264 | 0.189997 | 0.515117 |
| 36 | 14291 | 0.510637 | 1.195507 | 0.264235 | 0.187276 | 0.513062 |
| 37 | 14688 | 0.479711 | 1.360133 | 0.291077 | 0.205785 | 0.568612 |
| 38 | 15085 | 0.496764 | 1.269121 | 0.282315 | 0.199416 | 0.552665 |
| 39 | 15482 | 0.501791 | 1.242160 | 0.250520 | 0.177991 | 0.483526 |
| 40 | 15879 | 0.512020 | 1.217760 | 0.242797 | 0.172302 | 0.469970 |
| 41 | 16276 | 0.511911 | 1.243317 | 0.228421 | 0.162536 | 0.439236 |
| 42 | 16673 | 0.518025 | 1.219881 | 0.223697 | 0.159313 | 0.429224 |
| 43 | 17070 | 0.516973 | 1.221002 | 0.221551 | 0.158352 | 0.421326 |
| 44 | 17467 | 0.503748 | 1.232000 | 0.240931 | 0.171171 | 0.465068 |
| 45 | 17864 | 0.479737 | 1.377032 | 0.267857 | 0.189879 | 0.519859 |
| 46 | 18261 | 0.491097 | 1.303200 | 0.317915 | 0.223963 | 0.626344 |
| 47 | 18658 | 0.511940 | 1.243927 | 0.246800 | 0.174683 | 0.480779 |
| 48 | 19055 | 0.522998 | 1.207598 | 0.221812 | 0.157560 | 0.428346 |
| 49 | 19452 | 0.477229 | 1.309293 | 0.229163 | 0.162276 | 0.445910 |
| 50 | 19849 | 0.513176 | 1.203945 | 0.246043 | 0.172287 | 0.491706 |

## Interpretation

- PMAD logit-only with `kd_weight=0.15` gives a meaningful positive single-run signal.
- Best val/mIoU `0.522998` exceeds the clean baseline mean by `0.005601` (`1.143` std units).
- It also exceeds the strong-signal threshold baseline mean + 1 std by `0.000700`, but does not exceed the clean baseline best single run `0.524425`.
- The late epochs remain reasonably high: last mIoU is above the baseline mean, and last10 mean is near the best region, so this is healthier than many prior auxiliary-loss runs with sharp collapse.
- This is still a single run. It should be treated as a positive candidate, not a stable improvement claim, until repeated runs confirm the mean.

## Decision

- Proceed to repeat or narrow ablation according to the decision-value rule.
- Recommended next step: run `kd_weight=0.10` and/or `kd_weight=0.20` single-run ablations, then repeat the best setting for 3 runs if the signal remains above baseline mean + 1 std.
- Do not add feature KD yet; first confirm logit-only stability.
