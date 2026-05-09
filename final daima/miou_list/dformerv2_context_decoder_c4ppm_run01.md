# dformerv2_context_decoder_c4ppm_run01

- model: `dformerv2_context_decoder`
- change: decoder-side C4 PPM context refinement before FPN lateral4
- settings: `batch_size=2`, `max_epochs=50`, `lr=6e-5`, `num_workers=4`, `early_stop_patience=30`, `loss_type=ce`
- pretrained: `C:/Users/qintian/Desktop/qintian/dformer_work/checkpoints/pretrained/DFormerv2_Small_pretrained.pth`
- checkpoint: `checkpoints/dformerv2_context_decoder_c4ppm_run01/dformerv2_context_decoder-epoch=49-val_mIoU=0.5073.pt`

## val/mIoU per epoch

| epoch | val/mIoU | val/loss |
|------:|---------:|---------:|
| 0 | 0.163379 | 1.705623 |
| 1 | 0.219430 | 1.408096 |
| 2 | 0.259182 | 1.276727 |
| 3 | 0.320460 | 1.178468 |
| 4 | 0.361350 | 1.103866 |
| 5 | 0.407433 | 1.063822 |
| 6 | 0.431457 | 1.073712 |
| 7 | 0.438946 | 1.081529 |
| 8 | 0.454363 | 1.055534 |
| 9 | 0.455125 | 1.082210 |
| 10 | 0.428948 | 1.106607 |
| 11 | 0.468984 | 1.042966 |
| 12 | 0.470033 | 1.051420 |
| 13 | 0.466159 | 1.112920 |
| 14 | 0.467564 | 1.087920 |
| 15 | 0.480843 | 1.059960 |
| 16 | 0.470623 | 1.081790 |
| 17 | 0.483556 | 1.084394 |
| 18 | 0.483616 | 1.085253 |
| 19 | 0.489112 | 1.094688 |
| 20 | 0.476531 | 1.159101 |
| 21 | 0.468721 | 1.150590 |
| 22 | 0.485404 | 1.128777 |
| 23 | 0.496490 | 1.110689 |
| 24 | 0.495887 | 1.122140 |
| 25 | 0.501436 | 1.123682 |
| 26 | 0.496835 | 1.157712 |
| 27 | 0.498177 | 1.154148 |
| 28 | 0.501039 | 1.157369 |
| 29 | 0.479442 | 1.197072 |
| 30 | 0.479988 | 1.178601 |
| 31 | 0.491159 | 1.147989 |
| 32 | 0.506079 | 1.157932 |
| 33 | 0.502170 | 1.159525 |
| 34 | 0.507170 | 1.159823 |
| 35 | 0.477156 | 1.275114 |
| 36 | 0.461655 | 1.310659 |
| 37 | 0.480953 | 1.253101 |
| 38 | 0.485551 | 1.239197 |
| 39 | 0.492727 | 1.216190 |
| 40 | 0.504428 | 1.205144 |
| 41 | 0.506422 | 1.227014 |
| 42 | 0.505209 | 1.228253 |
| 43 | 0.479130 | 1.291183 |
| 44 | 0.483639 | 1.270496 |
| 45 | 0.500476 | 1.217680 |
| 46 | 0.493138 | 1.257555 |
| 47 | 0.498982 | 1.260563 |
| 48 | 0.503747 | 1.261455 |
| 49 | 0.507293 | 1.266029 |

## Summary

- recorded validation epochs: 50
- best val/mIoU: 0.507293 at epoch 49
- last val/mIoU: 0.507293
- best val/loss: 1.042966 at epoch 11
- last val/loss: 1.266029
- mean val/mIoU over last 10 epochs: 0.498246
- comparison clean 10-run baseline mean best: 0.517397
- delta vs baseline mean best: -0.010104
- comparison clean 10-run baseline std: 0.004901
- delta in baseline std units: -2.062
- comparison clean 10-run baseline best single: 0.524425
- delta vs baseline best single: -0.017132
