# R037 DGL Minimal Run01

## Summary

- Branch: exp/R037-dgl-minimal-v1
- Model: dformerv2_dgl_minimal
- Run: R037_dgl_minimal_run01
- Hypothesis: DGL-style gradient disentanglement can reduce multimodal optimization conflict by preventing the fused CE from updating the primary/depth encoders, while primary/depth aux CE trains the encoders during training only.
- Status: completed full train; `Trainer.fit` reached `max_epochs=50`; exit code `0`.
- TensorBoard event: `checkpoints\R037_dgl_minimal_run01\lightning_logs\version_0\events.out.tfevents.1778857094.Administrator.1736.0`
- Best checkpoint: `checkpoints\R037_dgl_minimal_run01\dformerv2_dgl_minimal-epoch=41-val_mIoU=0.5347.pt`
- Recorded validation epochs: `50`
- Best val/mIoU: `0.534656` at validation epoch `42`
- Last val/mIoU: `0.530153`
- Last-5 mean val/mIoU: `0.526926`
- Last-10 mean val/mIoU: `0.526304`
- Best-to-last drop: `0.004503`
- Best val/loss: `0.949518` at validation epoch `12`
- Last val/loss: `1.118480`
- Final train/loss_epoch: `0.056264`
- DGL aux weight: `0.03`
- Primary aux CE first/last: `2.388133` / `0.066309`
- Depth aux CE first/last: `2.517628` / `0.131319`
- Delta vs R016 corrected baseline best `0.541121`: `-0.006465`
- Decision: stable but below R016; do not promote as active mainline.

## Command

```powershell
cd "C:\Users\qintian\Desktop\qintian\final daima"

$env:PYTHONUTF8="1"
$env:PYTHONIOENCODING="utf-8"

& "D:\Anaconda\envs\qintian-rgbd\python.exe" train.py `
  --model dformerv2_dgl_minimal `
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
  --checkpoint_dir ".\checkpoints\R037_dgl_minimal_run01"
```

## Per-Epoch Validation Metrics

| Val Epoch | Global Step | val/mIoU | val/loss |
|---:|---:|---:|---:|
| 1 | 396 | 0.168151 | 1.643394 |
| 2 | 793 | 0.222327 | 1.387170 |
| 3 | 1190 | 0.277816 | 1.273263 |
| 4 | 1587 | 0.328981 | 1.106765 |
| 5 | 1984 | 0.385803 | 1.057517 |
| 6 | 2381 | 0.436324 | 1.007783 |
| 7 | 2778 | 0.456839 | 0.966863 |
| 8 | 3175 | 0.458132 | 0.989397 |
| 9 | 3572 | 0.470148 | 0.970705 |
| 10 | 3969 | 0.476660 | 0.967025 |
| 11 | 4366 | 0.487687 | 0.950516 |
| 12 | 4763 | 0.491272 | 0.949518 |
| 13 | 5160 | 0.489945 | 0.972901 |
| 14 | 5557 | 0.504994 | 0.965295 |
| 15 | 5954 | 0.507790 | 0.957521 |
| 16 | 6351 | 0.497930 | 0.978720 |
| 17 | 6748 | 0.489188 | 1.005595 |
| 18 | 7145 | 0.501186 | 0.979692 |
| 19 | 7542 | 0.513755 | 0.988447 |
| 20 | 7939 | 0.506105 | 0.991909 |
| 21 | 8336 | 0.477968 | 1.048142 |
| 22 | 8733 | 0.510289 | 0.987705 |
| 23 | 9130 | 0.508798 | 0.998641 |
| 24 | 9527 | 0.478733 | 1.108854 |
| 25 | 9924 | 0.488050 | 1.093614 |
| 26 | 10321 | 0.512293 | 1.008596 |
| 27 | 10718 | 0.510847 | 1.013530 |
| 28 | 11115 | 0.515363 | 1.033340 |
| 29 | 11512 | 0.517418 | 1.029903 |
| 30 | 11909 | 0.518339 | 1.016350 |
| 31 | 12306 | 0.503060 | 1.066111 |
| 32 | 12703 | 0.521818 | 1.049482 |
| 33 | 13100 | 0.496965 | 1.112357 |
| 34 | 13497 | 0.515528 | 1.058190 |
| 35 | 13894 | 0.525833 | 1.074518 |
| 36 | 14291 | 0.503585 | 1.074160 |
| 37 | 14688 | 0.511054 | 1.071178 |
| 38 | 15085 | 0.533393 | 1.042109 |
| 39 | 15482 | 0.507258 | 1.097484 |
| 40 | 15879 | 0.501001 | 1.079515 |
| 41 | 16276 | 0.526710 | 1.034850 |
| 42 | 16673 | 0.534656 | 1.051022 |
| 43 | 17070 | 0.530682 | 1.078685 |
| 44 | 17467 | 0.530887 | 1.065587 |
| 45 | 17864 | 0.505476 | 1.137269 |
| 46 | 18261 | 0.516628 | 1.096419 |
| 47 | 18658 | 0.525697 | 1.099665 |
| 48 | 19055 | 0.530081 | 1.096634 |
| 49 | 19452 | 0.532070 | 1.120638 |
| 50 | 19849 | 0.530153 | 1.118480 |
