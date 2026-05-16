# R045 c3/c4 Zero-Init Modality Adapter mIoU Detail

- Branch: `exp/R045-c34-zero-init-modality-adapter-v1`
- Model: `dformerv2_c34_zero_init_modality_adapter`
- Run: `R045_c34_zero_init_modality_adapter_run01`
- TensorBoard event: `checkpoints/R045_c34_zero_init_modality_adapter_run01/lightning_logs/version_0/events.out.tfevents.1778907691.Administrator.38920.0`
- Best checkpoint: `checkpoints/R045_c34_zero_init_modality_adapter_run01/dformerv2_c34_zero_init_modality_adapter-epoch=47-val_mIoU=0.5315.pt`
- Full train: completed 50 validation epochs; `Trainer.fit` reached `max_epochs=50`.

## Summary

- Best val/mIoU: `0.531454` at validation epoch `48`
- Last val/mIoU: `0.505130`
- Last-5 mean val/mIoU: `0.511191`
- Last-10 mean val/mIoU: `0.509930`
- Best-to-last drop: `0.026324`
- Best val/loss: `0.981750` at validation epoch `8`
- Last val/loss: `1.161006`
- Final train/loss_epoch: `0.151835`

## Adapter Diagnostics

- `rgb_c3_adapter_delta_abs` first/last/max: `0.008236` / `0.043839` / `0.043839`
- `rgb_c4_adapter_delta_abs` first/last/max: `0.010011` / `0.040954` / `0.040954`
- `depth_c3_adapter_delta_abs` first/last/max: `0.005683` / `0.101323` / `0.108361`
- `depth_c4_adapter_delta_abs` first/last/max: `0.018327` / `0.098307` / `0.098307`

## Command

```powershell
cd "C:\Users\qintian\Desktop\qintian\final daima"

$env:PYTHONUTF8="1"
$env:PYTHONIOENCODING="utf-8"

& "D:\Anaconda\envs\qintian-rgbd\python.exe" train.py `
  --model dformerv2_c34_zero_init_modality_adapter `
  --data_root "C:\Users\qintian\Desktop\qintian\data\NYUDepthv2" `
  --num_classes 40 `
  --batch_size 2 `
  --max_epochs 50 `
  --lr 6e-5 `
  --num_workers 4 `
  --early_stop_patience 30 `
  --accelerator gpu `
  --devices 1 `
  --dformerv2_pretrained "C:\Users\qintian\Desktop\qintian\dformer_work\checkpoints\pretrained\DFormerv2_Small_pretrained.pth" `
  --loss_type ce `
  --checkpoint_dir ".\checkpoints\R045_c34_zero_init_modality_adapter_run01"
```

## Per-Epoch Metrics

| validation epoch | TB step | val/mIoU | val/loss | train/loss_epoch |
|---:|---:|---:|---:|---:|
| 1 | 396 | 0.170664 | 1.655170 | 2.308698 |
| 2 | 793 | 0.251103 | 1.373693 | 1.597124 |
| 3 | 1190 | 0.312049 | 1.200997 | 1.251093 |
| 4 | 1587 | 0.342231 | 1.124553 | 1.003645 |
| 5 | 1984 | 0.402299 | 1.063119 | 0.837359 |
| 6 | 2381 | 0.408328 | 1.045578 | 0.697780 |
| 7 | 2778 | 0.447341 | 1.017917 | 0.575082 |
| 8 | 3175 | 0.462025 | 0.981750 | 0.481164 |
| 9 | 3572 | 0.456515 | 0.993326 | 0.417900 |
| 10 | 3969 | 0.476160 | 0.993660 | 0.364682 |
| 11 | 4366 | 0.472218 | 1.014933 | 0.319511 |
| 12 | 4763 | 0.475877 | 1.055833 | 0.294161 |
| 13 | 5160 | 0.487085 | 0.999680 | 0.252642 |
| 14 | 5557 | 0.498819 | 1.003297 | 0.221597 |
| 15 | 5954 | 0.481234 | 1.058266 | 0.222712 |
| 16 | 6351 | 0.501775 | 1.060188 | 0.217181 |
| 17 | 6748 | 0.504966 | 1.032583 | 0.201741 |
| 18 | 7145 | 0.510775 | 1.031380 | 0.165731 |
| 19 | 7542 | 0.499625 | 1.064218 | 0.143312 |
| 20 | 7939 | 0.496199 | 1.085284 | 0.142470 |
| 21 | 8336 | 0.503783 | 1.072561 | 0.137928 |
| 22 | 8733 | 0.508030 | 1.098257 | 0.126711 |
| 23 | 9130 | 0.486128 | 1.121405 | 0.181261 |
| 24 | 9527 | 0.500375 | 1.097700 | 0.155888 |
| 25 | 9924 | 0.511998 | 1.079485 | 0.124300 |
| 26 | 10321 | 0.515902 | 1.087724 | 0.099444 |
| 27 | 10718 | 0.519725 | 1.110232 | 0.096096 |
| 28 | 11115 | 0.511557 | 1.098937 | 0.102017 |
| 29 | 11512 | 0.474757 | 1.148003 | 0.184249 |
| 30 | 11909 | 0.504239 | 1.108291 | 0.128435 |
| 31 | 12306 | 0.519859 | 1.097813 | 0.094307 |
| 32 | 12703 | 0.517058 | 1.117691 | 0.088826 |
| 33 | 13100 | 0.523040 | 1.115891 | 0.077817 |
| 34 | 13497 | 0.524632 | 1.121677 | 0.074083 |
| 35 | 13894 | 0.522926 | 1.123701 | 0.070024 |
| 36 | 14291 | 0.484661 | 1.228672 | 0.093344 |
| 37 | 14688 | 0.510842 | 1.174942 | 0.124070 |
| 38 | 15085 | 0.523497 | 1.160063 | 0.092694 |
| 39 | 15482 | 0.518025 | 1.153666 | 0.091611 |
| 40 | 15879 | 0.517033 | 1.166507 | 0.074886 |
| 41 | 16276 | 0.514130 | 1.200061 | 0.065422 |
| 42 | 16673 | 0.521334 | 1.197784 | 0.070084 |
| 43 | 17070 | 0.469446 | 1.432609 | 0.071283 |
| 44 | 17467 | 0.511877 | 1.249712 | 0.102360 |
| 45 | 17864 | 0.526561 | 1.207144 | 0.070082 |
| 46 | 18261 | 0.527321 | 1.186541 | 0.060505 |
| 47 | 18658 | 0.530378 | 1.203632 | 0.055730 |
| 48 | 19055 | 0.531454 | 1.206175 | 0.056339 |
| 49 | 19452 | 0.461673 | 1.374749 | 0.118876 |
| 50 | 19849 | 0.505130 | 1.161006 | 0.151835 |
