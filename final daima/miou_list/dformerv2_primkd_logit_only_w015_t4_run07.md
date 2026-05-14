# dformerv2_primkd_logit_only_w015_t4_run07

- run_id: `R012-primkd-logit-w015-repeat-v2`
- model: `dformerv2_primkd_logit_only`
- branch: `exp/R012-primkd-logit-w015-repeat-v2`
- purpose: repeat the strongest PMAD logit-only setting (`kd_weight=0.15`, `kd_temperature=4.0`) as run07 after R010's high-tail partial positive result.
- student: `DFormerv2_S + ResNet-18 DepthEncoder + GatedFusion + SimpleFPNDecoder`
- teacher: frozen `dformerv2_geometry_primary_teacher_run01`
- teacher checkpoint: `checkpoints/dformerv2_geometry_primary_teacher_run01/dformerv2_geometry_primary_teacher-epoch=37-val_mIoU=0.5168.pt`
- settings: `batch_size=2`, `max_epochs=50`, `lr=6e-5`, `num_workers=4`, `early_stop_patience=30`, `loss_type=ce`
- KD settings: `kd_weight=0.15`, `kd_temperature=4.0`, logit-only, no feature KD, `--save_student_only`
- pretrained: `C:/Users/qintian/Desktop/qintian/dformer_work/checkpoints/pretrained/DFormerv2_Small_pretrained.pth`
- recorded validation epochs: `50`
- best val/mIoU: `0.516967` at epoch `43`
- last val/mIoU: `0.508205`
- last-5 mean val/mIoU: `0.496441`
- last-10 mean val/mIoU: `0.503120`
- best val/loss: `1.039913` at epoch `8`
- final train/loss: `0.216768`
- final train/ce_loss: `0.153825`
- final train/kd_loss: `0.419615`
- checkpoint: `checkpoints/dformerv2_primkd_logit_only_w015_t4_run07/dformerv2_primkd_logit_only-epoch=42-val_mIoU=0.5170.pt`
- TensorBoard event: `checkpoints/dformerv2_primkd_logit_only_w015_t4_run07/lightning_logs/version_0/events.out.tfevents.1778675198.Administrator.22916.0`
- process note: launched through `cmd.exe /c` hidden script with UTF-8 Python environment variables. Training completed with exit code `0` and `Trainer.fit stopped: max_epochs=50 reached`; no Windows/Rich teardown crash occurred.

## Comparison

- clean 10-run baseline mean best: `0.517397`
- clean 10-run baseline std: `0.004901`
- clean 10-run baseline mean + 1 std: `0.522298`
- clean 10-run baseline best single: `0.524425`
- prior PMAD w0.15/T4 five-run mean best: `0.520795`
- R010 PMAD run06_retry1 best: `0.527469`
- delta vs clean baseline mean: `-0.000430` (`-0.088` baseline std units)
- delta vs clean baseline mean + 1 std: `-0.005331`
- delta vs clean baseline best single: `-0.007458`
- delta vs prior PMAD five-run mean: `-0.003828`
- delta vs R010 run06_retry1: `-0.010502`
- gap to `0.53` goal: `-0.013033`
- updated PMAD w0.15/T4 seven-run mean best, using run01-run07: `0.521201`
- updated PMAD w0.15/T4 seven-run population std: `0.004148`
- runs above clean baseline mean: `5/7`
- runs above clean baseline mean + 1 std: `4/7`
- runs above clean baseline best single: `1/7`

## Epoch Metrics

| Epoch | Step | val/mIoU | val/loss | train/loss | train/ce | train/kd |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 396 | 0.150765 | 1.683608 | 3.506384 | 2.281791 | 8.163961 |
| 2 | 793 | 0.189665 | 1.369872 | 2.540107 | 1.581580 | 6.390180 |
| 3 | 1190 | 0.262752 | 1.209322 | 2.072455 | 1.275105 | 5.315676 |
| 4 | 1587 | 0.299352 | 1.153670 | 1.755874 | 1.073118 | 4.551710 |
| 5 | 1984 | 0.334212 | 1.101416 | 1.503173 | 0.917213 | 3.906401 |
| 6 | 2381 | 0.377008 | 1.083752 | 1.286811 | 0.781788 | 3.366815 |
| 7 | 2778 | 0.400305 | 1.092868 | 1.114730 | 0.676729 | 2.920009 |
| 8 | 3175 | 0.440584 | 1.039913 | 0.981318 | 0.600957 | 2.535733 |
| 9 | 3572 | 0.445619 | 1.061994 | 0.868351 | 0.534510 | 2.225613 |
| 10 | 3969 | 0.444191 | 1.122084 | 0.757634 | 0.468837 | 1.925314 |
| 11 | 4366 | 0.469862 | 1.081734 | 0.686588 | 0.432694 | 1.692623 |
| 12 | 4763 | 0.455313 | 1.146576 | 0.645753 | 0.413362 | 1.549276 |
| 13 | 5160 | 0.469619 | 1.129710 | 0.570791 | 0.368995 | 1.345306 |
| 14 | 5557 | 0.478709 | 1.138569 | 0.508141 | 0.331908 | 1.174886 |
| 15 | 5954 | 0.478053 | 1.168171 | 0.479116 | 0.318057 | 1.073723 |
| 16 | 6351 | 0.441471 | 1.273795 | 0.488818 | 0.329373 | 1.062967 |
| 17 | 6748 | 0.467579 | 1.193653 | 0.510836 | 0.345402 | 1.102896 |
| 18 | 7145 | 0.469856 | 1.208785 | 0.435736 | 0.296236 | 0.930001 |
| 19 | 7542 | 0.483388 | 1.192049 | 0.406998 | 0.279275 | 0.851481 |
| 20 | 7939 | 0.492845 | 1.204505 | 0.377490 | 0.260137 | 0.782358 |
| 21 | 8336 | 0.495148 | 1.209468 | 0.356610 | 0.247269 | 0.728939 |
| 22 | 8733 | 0.486344 | 1.174236 | 0.371727 | 0.259268 | 0.749723 |
| 23 | 9130 | 0.501229 | 1.198306 | 0.337934 | 0.235638 | 0.681975 |
| 24 | 9527 | 0.501062 | 1.195835 | 0.315917 | 0.221319 | 0.630656 |
| 25 | 9924 | 0.492949 | 1.218585 | 0.304054 | 0.214292 | 0.598413 |
| 26 | 10321 | 0.501305 | 1.213697 | 0.308442 | 0.216448 | 0.613295 |
| 27 | 10718 | 0.497328 | 1.247696 | 0.314115 | 0.221901 | 0.614763 |
| 28 | 11115 | 0.464460 | 1.273442 | 0.327542 | 0.229322 | 0.654795 |
| 29 | 11512 | 0.490319 | 1.223024 | 0.340090 | 0.238021 | 0.680456 |
| 30 | 11909 | 0.503829 | 1.234056 | 0.297674 | 0.209729 | 0.586297 |
| 31 | 12306 | 0.506618 | 1.242442 | 0.274838 | 0.194401 | 0.536246 |
| 32 | 12703 | 0.505447 | 1.237899 | 0.258221 | 0.183253 | 0.499789 |
| 33 | 13100 | 0.511042 | 1.232439 | 0.253648 | 0.180386 | 0.488410 |
| 34 | 13497 | 0.496988 | 1.255034 | 0.251140 | 0.178549 | 0.483941 |
| 35 | 13894 | 0.478685 | 1.265241 | 0.353799 | 0.249898 | 0.692672 |
| 36 | 14291 | 0.493959 | 1.242945 | 0.336217 | 0.235898 | 0.668789 |
| 37 | 14688 | 0.500898 | 1.256022 | 0.277691 | 0.194945 | 0.551636 |
| 38 | 15085 | 0.498982 | 1.236297 | 0.247340 | 0.175872 | 0.476452 |
| 39 | 15482 | 0.504916 | 1.270399 | 0.238136 | 0.169135 | 0.460010 |
| 40 | 15879 | 0.502930 | 1.251448 | 0.230521 | 0.164247 | 0.441825 |
| 41 | 16276 | 0.512109 | 1.254063 | 0.227006 | 0.161498 | 0.436723 |
| 42 | 16673 | 0.510743 | 1.251343 | 0.228194 | 0.162197 | 0.439979 |
| 43 | 17070 | 0.516967 | 1.254150 | 0.224690 | 0.160521 | 0.427793 |
| 44 | 17467 | 0.505332 | 1.261875 | 0.255157 | 0.180729 | 0.496182 |
| 45 | 17864 | 0.503838 | 1.248301 | 0.259143 | 0.184218 | 0.499502 |
| 46 | 18261 | 0.502238 | 1.253745 | 0.279058 | 0.196766 | 0.548614 |
| 47 | 18658 | 0.482113 | 1.259062 | 0.302919 | 0.212949 | 0.599799 |
| 48 | 19055 | 0.489091 | 1.300583 | 0.264540 | 0.185347 | 0.527952 |
| 49 | 19452 | 0.500560 | 1.227584 | 0.231331 | 0.163491 | 0.452268 |
| 50 | 19849 | 0.508205 | 1.234542 | 0.216768 | 0.153825 | 0.419615 |

## Conclusion

R012 is a negative repeat. The run completed cleanly with 50 validation epochs, but best val/mIoU `0.516967` is slightly below the clean 10-run baseline mean `0.517397`, below the prior PMAD w0.15/T4 five-run mean `0.520795`, far below R010's high-tail `0.527469`, and below the required `0.53`. This argues against more blind PMAD w0.15/T4 repeats as the next decision-value step.
