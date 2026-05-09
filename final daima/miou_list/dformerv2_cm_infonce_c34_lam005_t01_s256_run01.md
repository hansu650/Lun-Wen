# dformerv2_cm_infonce_c34_lam005_t01_s256_run01

## Model
- model name: `dformerv2_cm_infonce`
- run name: `dformerv2_cm_infonce_c34_lam005_t01_s256_run01`
- change: training-only c3+c4 one-way depth-to-primary InfoNCE contrastive auxiliary loss

## Hyperparameters
- batch_size: 2
- max_epochs: 50
- lr: 6e-5
- num_workers: 4
- early_stop_patience: 30
- lambda_contrast: 0.005
- contrast_temperature: 0.1
- contrast_proj_dim: 64
- contrast_sample_points: 256
- contrast_stage_weights: 0,0,1,1
- pretrained: DFormerv2_Small_pretrained.pth

## Per-Epoch val/mIoU
| Epoch | val/mIoU |
|-------|----------|
| 0     | 0.160815 |
| 1     | 0.239241 |
| 2     | 0.286328 |
| 3     | 0.312848 |
| 4     | 0.358828 |
| 5     | 0.405604 |
| 6     | 0.417133 |
| 7     | 0.448275 |
| 8     | 0.469821 |
| 9     | 0.473591 |
| 10    | 0.445870 |
| 11    | 0.469113 |
| 12    | 0.470321 |
| 13    | 0.478676 |
| 14    | 0.444590 |
| 15    | 0.473530 |
| 16    | 0.472689 |
| 17    | 0.485030 |
| 18    | 0.476042 |
| 19    | 0.483691 |
| 20    | 0.468941 |
| 21    | 0.484595 |
| 22    | 0.457070 |
| 23    | 0.492425 |
| 24    | 0.498241 |
| 25    | 0.495179 |
| 26    | 0.499864 |
| 27    | 0.502800 |
| 28    | 0.497439 |
| 29    | 0.456364 |
| 30    | 0.495685 |
| 31    | 0.502865 |
| 32    | 0.503755 |
| 33    | 0.505397 |
| 34    | 0.499681 |
| 35    | 0.485003 |
| 36    | 0.455054 |
| 37    | 0.498167 |
| 38    | 0.498402 |
| 39    | 0.503385 |
| 40    | 0.507993 |
| 41    | 0.510188 |
| 42    | 0.498718 |
| 43    | 0.497813 |
| 44    | 0.501132 |
| 45    | 0.503188 |
| 46    | 0.514461 |
| 47    | 0.513062 |
| 48    | 0.513106 |
| 49    | 0.498469 |

## Per-Epoch train/contrast_loss
| Epoch | contrast_loss |
|-------|---------------|
| 0     | 5.574991      |
| 1     | 4.668571      |
| 2     | 4.338335      |
| 3     | 4.150106      |
| 4     | 4.023189      |
| 5     | 3.939373      |
| 6     | 3.851986      |
| 7     | 3.792480      |
| 8     | 3.737252      |
| 9     | 3.673117      |
| 10    | 3.651408      |
| 11    | 3.619910      |
| 12    | 3.568615      |
| 13    | 3.539943      |
| 14    | 3.519456      |
| 15    | 3.491404      |
| 16    | 3.447163      |
| 17    | 3.420231      |
| 18    | 3.436309      |
| 19    | 3.419297      |
| 20    | 3.362830      |
| 21    | 3.366087      |
| 22    | 3.322124      |
| 23    | 3.316959      |
| 24    | 3.276968      |
| 25    | 3.261430      |
| 26    | 3.250741      |
| 27    | 3.223850      |
| 28    | 3.206922      |
| 29    | 3.220863      |
| 30    | 3.310349      |
| 31    | 3.219730      |
| 32    | 3.178733      |
| 33    | 3.158403      |
| 34    | 3.151742      |
| 35    | 3.152401      |
| 36    | 3.181286      |
| 37    | 3.222354      |
| 38    | 3.148084      |
| 39    | 3.147282      |
| 40    | 3.105277      |
| 41    | 3.077235      |
| 42    | 3.093623      |
| 43    | 3.111173      |
| 44    | 3.134023      |
| 45    | 3.074948      |
| 46    | 3.044694      |
| 47    | 3.055024      |
| 48    | 3.046888      |
| 49    | 3.089162      |

## Summary
- best val/mIoU: 0.514461 at epoch 46
- last val/mIoU: 0.498469 at epoch 49
- contrast loss: 5.575 → 3.089 (45% drop, active learning)
- comparison baseline mean best: 0.517397
- delta vs baseline mean: -0.002936
- comparison baseline std: 0.004901
- delta in baseline std units: -0.599
- comparison baseline best single: 0.524425
- delta vs baseline best single: -0.009964
- late collapse: epoch 46 (0.5145) → epoch 49 (0.4985), drop 0.016
- conclusion: NEGATIVE. InfoNCE loss is active and converging, but does not improve validation mIoU above baseline mean.
