# dformerv2_mid_fusion_ce_dice_w05_run01

- model: `dformerv2_mid_fusion`
- loss: `CE + 0.5 * Dice`
- settings: `batch_size=2`, `max_epochs=50`, `lr=6e-5`, `num_workers=4`, `early_stop_patience=30`, `loss_type=ce_dice`, `dice_weight=0.5`
- pretrained: `C:/Users/qintian/Desktop/qintian/dformer_work/checkpoints/pretrained/DFormerv2_Small_pretrained.pth`
- checkpoint: `checkpoints/dformerv2_mid_fusion_ce_dice_w05_run01/dformerv2_mid_fusion-epoch=41-val_mIoU=0.5070.pt`

## val/mIoU per epoch

| epoch | val/mIoU | val/loss |
|------:|---------:|---------:|
| 0 | 0.162833 | 1.683125 |
| 1 | 0.230539 | 1.422123 |
| 2 | 0.271271 | 1.268495 |
| 3 | 0.323881 | 1.172945 |
| 4 | 0.364818 | 1.139479 |
| 5 | 0.416663 | 1.071084 |
| 6 | 0.447775 | 1.037181 |
| 7 | 0.431892 | 1.086734 |
| 8 | 0.455044 | 1.070771 |
| 9 | 0.459116 | 1.078610 |
| 10 | 0.467042 | 1.067596 |
| 11 | 0.470201 | 1.085959 |
| 12 | 0.464034 | 1.114122 |
| 13 | 0.474356 | 1.086348 |
| 14 | 0.481609 | 1.084862 |
| 15 | 0.480077 | 1.115030 |
| 16 | 0.471902 | 1.119741 |
| 17 | 0.484148 | 1.105664 |
| 18 | 0.466911 | 1.157948 |
| 19 | 0.484161 | 1.136807 |
| 20 | 0.493384 | 1.108293 |
| 21 | 0.483745 | 1.186103 |
| 22 | 0.486707 | 1.138788 |
| 23 | 0.489289 | 1.164587 |
| 24 | 0.495118 | 1.184308 |
| 25 | 0.493516 | 1.217946 |
| 26 | 0.480603 | 1.268085 |
| 27 | 0.480686 | 1.314314 |
| 28 | 0.490708 | 1.294399 |
| 29 | 0.499852 | 1.282505 |
| 30 | 0.486651 | 1.362654 |
| 31 | 0.494599 | 1.303087 |
| 32 | 0.500253 | 1.346704 |
| 33 | 0.502035 | 1.386419 |
| 34 | 0.461256 | 1.507886 |
| 35 | 0.479291 | 1.453524 |
| 36 | 0.472275 | 1.543229 |
| 37 | 0.443662 | 1.669371 |
| 38 | 0.488199 | 1.490194 |
| 39 | 0.502783 | 1.503453 |
| 40 | 0.500367 | 1.585994 |
| 41 | 0.507000 | 1.621864 |
| 42 | 0.499150 | 1.670873 |
| 43 | 0.493861 | 1.699252 |
| 44 | 0.490867 | 1.714478 |
| 45 | 0.501879 | 1.712322 |
| 46 | 0.502486 | 1.741857 |
| 47 | 0.504386 | 1.729762 |
| 48 | 0.493297 | 1.838148 |
| 49 | 0.489077 | 1.830893 |

## Summary

- recorded validation epochs: 50
- best val/mIoU: 0.507000 at epoch 41
- last val/mIoU: 0.489077
- best val/loss: 1.037181 at epoch 6
- last val/loss: 1.830893
- mean val/mIoU over last 10 epochs: 0.498237
- comparison clean 10-run baseline mean best: 0.517397
- delta vs baseline mean best: -0.010397
- comparison clean 10-run baseline std: 0.004901
- delta in baseline std units: -2.121
- comparison clean 10-run baseline best single: 0.524425
- delta vs baseline best single: -0.017425
