# dformerv2_freqfpn_decoder_run01

## Settings

- model: `dformerv2_freqfpn_decoder`
- purpose: R002 frequency-aware decoder top-down fusion
- architecture: `DFormerv2_S + ResNet-18 DepthEncoder + GatedFusion + FrequencyAwareFPNDecoder`
- decoder change: replace only the FPN top-down additions with `FrequencyAwareTopDownFuse`
- fixed recipe: `batch_size=2`, `max_epochs=50`, `lr=6e-5`, `num_workers=4`, `early_stop_patience=30`, `loss_type=ce`
- pretrained: `C:/Users/qintian/Desktop/qintian/dformer_work/checkpoints/pretrained/DFormerv2_Small_pretrained.pth`
- TensorBoard event: `checkpoints/dformerv2_freqfpn_decoder_run01/lightning_logs/version_0/events.out.tfevents.1778607762.Administrator.22656.0`
- best checkpoint: `checkpoints/dformerv2_freqfpn_decoder_run01/dformerv2_freqfpn_decoder-epoch=43-val_mIoU=0.5169.pt`
- process note: `Trainer.fit` reached `max_epochs=50`.

## Summary

- recorded validation epochs: `50`
- best val/mIoU: `0.516915` at epoch `44`
- last val/mIoU: `0.486524`
- last-5 mean val/mIoU: `0.498475`
- last-10 mean val/mIoU: `0.504222`
- best val/loss: `1.022916` at epoch `9`
- final train/loss: `0.174152`
- clean 10-run RGB-D baseline mean best: `0.517397`
- clean 10-run RGB-D baseline std: `0.004901`
- clean 10-run RGB-D baseline mean + 1 std: `0.522298`
- clean 10-run RGB-D baseline best single: `0.524425`
- PMAD logit-only w0.15/T4 five-run mean best: `0.520795`
- delta vs clean baseline mean: `-0.000482`
- delta vs clean baseline mean in std units: `-0.098`
- delta vs clean baseline mean + 1 std: `-0.005383`
- delta vs PMAD w0.15/T4 five-run mean: `-0.003880`

## Per-Epoch Metrics

| epoch | step | val/mIoU | val/loss | train/loss |
|---:|---:|---:|---:|---:|
| 1 | 396 | 0.157933 | 1.727083 | 2.290528 |
| 2 | 793 | 0.216690 | 1.394614 | 1.595115 |
| 3 | 1190 | 0.278240 | 1.234707 | 1.278652 |
| 4 | 1587 | 0.342581 | 1.128758 | 1.066937 |
| 5 | 1984 | 0.367607 | 1.090175 | 0.908409 |
| 6 | 2381 | 0.420043 | 1.052813 | 0.772613 |
| 7 | 2778 | 0.432394 | 1.058258 | 0.668456 |
| 8 | 3175 | 0.438000 | 1.086089 | 0.579214 |
| 9 | 3572 | 0.461420 | 1.022916 | 0.534215 |
| 10 | 3969 | 0.458408 | 1.035296 | 0.473096 |
| 11 | 4366 | 0.463395 | 1.033671 | 0.444155 |
| 12 | 4763 | 0.472275 | 1.033235 | 0.398819 |
| 13 | 5160 | 0.470170 | 1.035946 | 0.363448 |
| 14 | 5557 | 0.474307 | 1.067566 | 0.338680 |
| 15 | 5954 | 0.472034 | 1.051775 | 0.343280 |
| 16 | 6351 | 0.478071 | 1.067023 | 0.316728 |
| 17 | 6748 | 0.474505 | 1.067127 | 0.305344 |
| 18 | 7145 | 0.481487 | 1.061378 | 0.290228 |
| 19 | 7542 | 0.472661 | 1.098259 | 0.292198 |
| 20 | 7939 | 0.473823 | 1.141985 | 0.286255 |
| 21 | 8336 | 0.491041 | 1.074848 | 0.259947 |
| 22 | 8733 | 0.497852 | 1.064552 | 0.231130 |
| 23 | 9130 | 0.486895 | 1.120423 | 0.248110 |
| 24 | 9527 | 0.501111 | 1.103241 | 0.222517 |
| 25 | 9924 | 0.500803 | 1.117466 | 0.210819 |
| 26 | 10321 | 0.494188 | 1.142565 | 0.244200 |
| 27 | 10718 | 0.502640 | 1.090175 | 0.241280 |
| 28 | 11115 | 0.507642 | 1.129387 | 0.208201 |
| 29 | 11512 | 0.505421 | 1.117145 | 0.203303 |
| 30 | 11909 | 0.515610 | 1.129864 | 0.198217 |
| 31 | 12306 | 0.423367 | 1.310625 | 0.211887 |
| 32 | 12703 | 0.475093 | 1.207702 | 0.228175 |
| 33 | 13100 | 0.500944 | 1.159349 | 0.191273 |
| 34 | 13497 | 0.468024 | 1.234370 | 0.198285 |
| 35 | 13894 | 0.497711 | 1.201895 | 0.185925 |
| 36 | 14291 | 0.501397 | 1.186954 | 0.171471 |
| 37 | 14688 | 0.499537 | 1.196238 | 0.184031 |
| 38 | 15085 | 0.499892 | 1.186694 | 0.181689 |
| 39 | 15482 | 0.500413 | 1.192422 | 0.174656 |
| 40 | 15879 | 0.489740 | 1.201978 | 0.202220 |
| 41 | 16276 | 0.507330 | 1.155594 | 0.171112 |
| 42 | 16673 | 0.497845 | 1.199275 | 0.158114 |
| 43 | 17070 | 0.511875 | 1.206056 | 0.151291 |
| 44 | 17467 | 0.516915 | 1.180251 | 0.145505 |
| 45 | 17864 | 0.515881 | 1.217476 | 0.146466 |
| 46 | 18261 | 0.513671 | 1.234523 | 0.144131 |
| 47 | 18658 | 0.516334 | 1.204808 | 0.144046 |
| 48 | 19055 | 0.487358 | 1.284668 | 0.171394 |
| 49 | 19452 | 0.488486 | 1.264650 | 0.167857 |
| 50 | 19849 | 0.486524 | 1.299209 | 0.174152 |

## Interpretation

- Neutral/negative result. The run completed 50 validation epochs and peaked close to the clean 10-run baseline mean, but best val/mIoU remains below the clean baseline mean and below PMAD logit-only w0.15/T4 five-run mean.
- Late training is unstable: the best value occurs at epoch 44, while the final epoch falls to `0.486524`.
- This decoder-side frequency-aware top-down fusion should not be claimed as an improvement and should not be repeated unchanged.
