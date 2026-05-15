# R034 MASG Gated Fusion Run01

## Summary

- Branch: `exp/R034-masg-gated-fusion-v1`
- Model: `dformerv2_masg_fusion`
- Run: `R034_masg_gated_fusion_run01`
- Hypothesis: gate-only stop-gradient on the depth projection in `GatedFusion` may reduce late instability while preserving the depth value path gradient.
- Status: completed full train; `Trainer.fit` reached `max_epochs=50`; exit code `0`.
- TensorBoard event: `checkpoints/R034_masg_gated_fusion_run01/lightning_logs/version_0/events.out.tfevents.1778839940.Administrator.26056.0`
- Best checkpoint: `checkpoints/R034_masg_gated_fusion_run01/dformerv2_masg_fusion-epoch=39-val_mIoU=0.5393.pt`
- Recorded validation epochs: `50`
- Best val/mIoU: `0.539322` at validation epoch `40`
- Last val/mIoU: `0.518738`
- Last-5 mean val/mIoU: `0.504633`
- Last-10 mean val/mIoU: `0.512033`
- Best-to-last drop: `0.020584`
- Best val/loss: `0.966439` at validation epoch `8`
- Last val/loss: `1.230551`
- Final train/loss_epoch: `0.059559`
- Delta vs R016 corrected baseline best `0.541121`: `-0.001799`
- Decision: negative/unstable relative to R016; do not promote R034 MASG to active mainline.

## Command

```powershell
cd "C:\Users\qintian\Desktop\qintian\final daima"

$env:PYTHONUTF8="1"
$env:PYTHONIOENCODING="utf-8"

& "D:\Anaconda\envs\qintian-rgbd\python.exe" train.py `
  --model dformerv2_masg_fusion `
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
  --checkpoint_dir ".\checkpoints\R034_masg_gated_fusion_run01"
```

## Per-Epoch Validation Metrics

| Val Epoch | Global Step | val/mIoU | val/loss |
|---:|---:|---:|---:|
| 1 | 396 | 0.169274 | 1.704312 |
| 2 | 793 | 0.233276 | 1.324021 |
| 3 | 1190 | 0.311256 | 1.187242 |
| 4 | 1587 | 0.333621 | 1.111330 |
| 5 | 1984 | 0.396819 | 1.059491 |
| 6 | 2381 | 0.434335 | 1.020355 |
| 7 | 2778 | 0.441574 | 1.006475 |
| 8 | 3175 | 0.478632 | 0.966439 |
| 9 | 3572 | 0.459103 | 1.022653 |
| 10 | 3969 | 0.461753 | 1.029641 |
| 11 | 4366 | 0.486784 | 0.983905 |
| 12 | 4763 | 0.472730 | 1.056751 |
| 13 | 5160 | 0.503793 | 0.987255 |
| 14 | 5557 | 0.495466 | 1.008286 |
| 15 | 5954 | 0.510370 | 1.001069 |
| 16 | 6351 | 0.513790 | 1.000322 |
| 17 | 6748 | 0.475628 | 1.114879 |
| 18 | 7145 | 0.506006 | 1.017208 |
| 19 | 7542 | 0.516880 | 1.037128 |
| 20 | 7939 | 0.504767 | 1.048488 |
| 21 | 8336 | 0.514772 | 1.041761 |
| 22 | 8733 | 0.516255 | 1.071303 |
| 23 | 9130 | 0.509934 | 1.097114 |
| 24 | 9527 | 0.511507 | 1.113064 |
| 25 | 9924 | 0.509067 | 1.096935 |
| 26 | 10321 | 0.524531 | 1.067144 |
| 27 | 10718 | 0.526617 | 1.087230 |
| 28 | 11115 | 0.534583 | 1.062570 |
| 29 | 11512 | 0.530597 | 1.091398 |
| 30 | 11909 | 0.533914 | 1.116921 |
| 31 | 12306 | 0.508816 | 1.182726 |
| 32 | 12703 | 0.522937 | 1.150425 |
| 33 | 13100 | 0.509534 | 1.147948 |
| 34 | 13497 | 0.506148 | 1.169619 |
| 35 | 13894 | 0.517591 | 1.144997 |
| 36 | 14291 | 0.524453 | 1.132661 |
| 37 | 14688 | 0.528924 | 1.137939 |
| 38 | 15085 | 0.533102 | 1.138088 |
| 39 | 15482 | 0.534259 | 1.129172 |
| 40 | 15879 | 0.539322 | 1.135599 |
| 41 | 16276 | 0.535438 | 1.161992 |
| 42 | 16673 | 0.524742 | 1.174655 |
| 43 | 17070 | 0.506301 | 1.210415 |
| 44 | 17467 | 0.504590 | 1.262963 |
| 45 | 17864 | 0.526088 | 1.179670 |
| 46 | 18261 | 0.499053 | 1.270381 |
| 47 | 18658 | 0.486214 | 1.330321 |
| 48 | 19055 | 0.506802 | 1.192004 |
| 49 | 19452 | 0.512358 | 1.188610 |
| 50 | 19849 | 0.518738 | 1.230551 |
