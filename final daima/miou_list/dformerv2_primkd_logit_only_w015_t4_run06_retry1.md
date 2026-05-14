# dformerv2_primkd_logit_only_w015_t4_run06_retry1

- run_id: `R010-primkd-logit-w015-repeat-v1`
- model: `dformerv2_primkd_logit_only`
- branch: `exp/R010-primkd-logit-w015-repeat-v1`
- purpose: repeat the strongest PMAD logit-only setting (`kd_weight=0.15`, `kd_temperature=4.0`) as run06 to test high-tail potential and improve stability evidence.
- student: `DFormerv2_S + ResNet-18 DepthEncoder + GatedFusion + SimpleFPNDecoder`
- teacher: frozen `dformerv2_geometry_primary_teacher_run01`
- teacher checkpoint: `checkpoints/dformerv2_geometry_primary_teacher_run01/dformerv2_geometry_primary_teacher-epoch=37-val_mIoU=0.5168.pt`
- settings: `batch_size=2`, `max_epochs=50`, `lr=6e-5`, `num_workers=4`, `early_stop_patience=30`, `loss_type=ce`
- KD settings: `kd_weight=0.15`, `kd_temperature=4.0`, logit-only, no feature KD, `--save_student_only`
- pretrained: `C:/Users/qintian/Desktop/qintian/dformer_work/checkpoints/pretrained/DFormerv2_Small_pretrained.pth`
- recorded validation epochs: `50`
- best val/mIoU: `0.527469` at epoch `49`
- last val/mIoU: `0.526316`
- last-5 mean val/mIoU: `0.519330`
- last-10 mean val/mIoU: `0.516229`
- best val/loss: `1.064507` at epoch `7`
- final train/loss: `0.209408`
- final train/ce_loss: `0.149212`
- final train/kd_loss: `0.401306`
- checkpoint: `checkpoints/dformerv2_primkd_logit_only_w015_t4_run06_retry1/dformerv2_primkd_logit_only-epoch=48-val_mIoU=0.5275.pt`
- TensorBoard event: `checkpoints/dformerv2_primkd_logit_only_w015_t4_run06_retry1/lightning_logs/version_0/events.out.tfevents.1778662831.Administrator.24204.0`
- process note: the first non-retry `run06` launch stopped during epoch 0 with a Windows `forrtl error (200): program aborting due to window-CLOSE event` and recorded no `val/mIoU`; this file uses only the completed `run06_retry1` evidence.

## Comparison

- clean 10-run baseline mean best: `0.517397`
- clean 10-run baseline std: `0.004901`
- clean 10-run baseline mean + 1 std: `0.522298`
- clean 10-run baseline best single: `0.524425`
- prior PMAD w0.15/T4 five-run mean best: `0.520795`
- prior PMAD w0.15/T4 best single: `0.524028`
- delta vs clean baseline mean: `+0.010072` (`+2.055` baseline std units)
- delta vs clean baseline mean + 1 std: `+0.005171`
- delta vs clean baseline best single: `+0.003044`
- delta vs prior PMAD five-run mean: `+0.006674`
- delta vs prior PMAD best single: `+0.003441`
- gap to `0.53` goal: `-0.002531`
- updated PMAD w0.15/T4 six-run mean best, using run01-run06: `0.521907`
- updated PMAD w0.15/T4 six-run population std: `0.004073`
- runs above clean baseline mean: `5/6`
- runs above clean baseline mean + 1 std: `4/6`
- runs above clean baseline best single: `1/6`

## Epoch Metrics

| Epoch | Step | val/mIoU | val/loss | train/loss | train/ce | train/kd |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 396 | 0.160527 | 1.698263 | 3.456682 | 2.240047 | 8.110890 |
| 2 | 793 | 0.228153 | 1.385481 | 2.494286 | 1.545010 | 6.328506 |
| 3 | 1190 | 0.264563 | 1.226706 | 2.034276 | 1.244285 | 5.266603 |
| 4 | 1587 | 0.311568 | 1.145780 | 1.732763 | 1.058034 | 4.498195 |
| 5 | 1984 | 0.338457 | 1.154059 | 1.466338 | 0.891100 | 3.834908 |
| 6 | 2381 | 0.369913 | 1.071340 | 1.281059 | 0.781178 | 3.332534 |
| 7 | 2778 | 0.415360 | 1.064507 | 1.110770 | 0.678501 | 2.881788 |
| 8 | 3175 | 0.418685 | 1.109739 | 0.983743 | 0.604734 | 2.526732 |
| 9 | 3572 | 0.447551 | 1.093258 | 0.878621 | 0.544383 | 2.228257 |
| 10 | 3969 | 0.450758 | 1.112298 | 0.762024 | 0.474654 | 1.915802 |
| 11 | 4366 | 0.466773 | 1.103989 | 0.676708 | 0.427034 | 1.664491 |
| 12 | 4763 | 0.476727 | 1.125291 | 0.620385 | 0.397434 | 1.486344 |
| 13 | 5160 | 0.473676 | 1.145482 | 0.570122 | 0.370365 | 1.331709 |
| 14 | 5557 | 0.483544 | 1.147064 | 0.521006 | 0.341967 | 1.193596 |
| 15 | 5954 | 0.487826 | 1.142791 | 0.527505 | 0.351327 | 1.174520 |
| 16 | 6351 | 0.489953 | 1.138963 | 0.453977 | 0.303680 | 1.001976 |
| 17 | 6748 | 0.489440 | 1.165338 | 0.426485 | 0.288878 | 0.917383 |
| 18 | 7145 | 0.489012 | 1.203800 | 0.415353 | 0.281882 | 0.889806 |
| 19 | 7542 | 0.488679 | 1.187821 | 0.383977 | 0.263012 | 0.806437 |
| 20 | 7939 | 0.486307 | 1.244128 | 0.384511 | 0.263589 | 0.806149 |
| 21 | 8336 | 0.481459 | 1.232076 | 0.384978 | 0.266368 | 0.790734 |
| 22 | 8733 | 0.503278 | 1.225786 | 0.399793 | 0.276480 | 0.822087 |
| 23 | 9130 | 0.504739 | 1.225109 | 0.340061 | 0.236622 | 0.689594 |
| 24 | 9527 | 0.516216 | 1.191387 | 0.319249 | 0.223099 | 0.641004 |
| 25 | 9924 | 0.513831 | 1.189883 | 0.322466 | 0.225839 | 0.644180 |
| 26 | 10321 | 0.506219 | 1.245136 | 0.305600 | 0.215674 | 0.599504 |
| 27 | 10718 | 0.513922 | 1.195937 | 0.320567 | 0.225221 | 0.635638 |
| 28 | 11115 | 0.514931 | 1.224314 | 0.293078 | 0.206007 | 0.580474 |
| 29 | 11512 | 0.493223 | 1.278941 | 0.301811 | 0.213515 | 0.588643 |
| 30 | 11909 | 0.504020 | 1.261904 | 0.314113 | 0.220713 | 0.622668 |
| 31 | 12306 | 0.506750 | 1.240594 | 0.281212 | 0.198920 | 0.548610 |
| 32 | 12703 | 0.515760 | 1.213961 | 0.266240 | 0.188167 | 0.520480 |
| 33 | 13100 | 0.518765 | 1.223801 | 0.255839 | 0.181412 | 0.496178 |
| 34 | 13497 | 0.511007 | 1.258085 | 0.266417 | 0.188830 | 0.517250 |
| 35 | 13894 | 0.468423 | 1.276214 | 0.375722 | 0.263229 | 0.749952 |
| 36 | 14291 | 0.506457 | 1.208994 | 0.319242 | 0.223652 | 0.637269 |
| 37 | 14688 | 0.515994 | 1.199842 | 0.263399 | 0.186633 | 0.511774 |
| 38 | 15085 | 0.518310 | 1.211662 | 0.241416 | 0.171925 | 0.463271 |
| 39 | 15482 | 0.520674 | 1.208361 | 0.232015 | 0.165583 | 0.442876 |
| 40 | 15879 | 0.520874 | 1.196316 | 0.230052 | 0.163360 | 0.444613 |
| 41 | 16276 | 0.524292 | 1.195466 | 0.225030 | 0.160195 | 0.432235 |
| 42 | 16673 | 0.522800 | 1.220580 | 0.223154 | 0.159171 | 0.426555 |
| 43 | 17070 | 0.521926 | 1.216193 | 0.224205 | 0.160465 | 0.424933 |
| 44 | 17467 | 0.521537 | 1.242552 | 0.220904 | 0.157932 | 0.419809 |
| 45 | 17864 | 0.475086 | 1.303573 | 0.354740 | 0.251061 | 0.691194 |
| 46 | 18261 | 0.502437 | 1.240061 | 0.306967 | 0.215174 | 0.611951 |
| 47 | 18658 | 0.514658 | 1.216788 | 0.259136 | 0.182881 | 0.508367 |
| 48 | 19055 | 0.525772 | 1.191746 | 0.224560 | 0.160060 | 0.430000 |
| 49 | 19452 | 0.527469 | 1.202548 | 0.212013 | 0.151106 | 0.406048 |
| 50 | 19849 | 0.526316 | 1.200338 | 0.209408 | 0.149212 | 0.401306 |

## Conclusion

R010 is a partial positive repeat, not a goal-completing run. It is the best recorded orchestration-loop single run and the first PMAD logit-only w0.15/T4 repeat above the clean baseline best single, but it remains below the required `0.53`. The updated six-run PMAD mean (`0.521907`) supports PMAD logit-only as a durable marginal-positive direction, not as a solved result.
