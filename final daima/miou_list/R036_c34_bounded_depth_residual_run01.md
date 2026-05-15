# R036 C34 Bounded Depth Residual Run01

## Summary

- Branch: `exp/R036-c34-bounded-depth-residual-v1`
- Model: `dformerv2_c34_bounded_depth_residual`
- Run: `R036_c34_bounded_depth_residual_run01`
- Hypothesis: c3/c4-only, zero-initialized, low-amplitude bounded depth residual after the proven R016 GatedFusion output may recover residual-depth signal without all-stage instability.
- Status: completed full train; `Trainer.fit` reached `max_epochs=50`; exit code `0`.
- TensorBoard event: `checkpoints/R036_c34_bounded_depth_residual_run01/lightning_logs/version_0/events.out.tfevents.1778851092.Administrator.20952.0`
- Best checkpoint: `checkpoints/R036_c34_bounded_depth_residual_run01/dformerv2_c34_bounded_depth_residual-epoch=43-val_mIoU=0.5398.pt`
- Recorded validation epochs: `50`
- Best val/mIoU: `0.539790` at validation epoch `44`
- Last val/mIoU: `0.528882`
- Last-5 mean val/mIoU: `0.516304`
- Last-10 mean val/mIoU: `0.521443`
- Best-to-last drop: `0.010908`
- Best val/loss: `0.950258` at validation epoch `8`
- Last val/loss: `1.208208`
- Final train/loss_epoch: `0.052817`
- c3 residual alpha first/last: `0.025097` / `0.026970`
- c4 residual alpha first/last: `0.025034` / `0.025553`
- Delta vs R016 corrected baseline best `0.541121`: `-0.001331`
- Decision: partial-positive below R016; do not promote as active mainline.

## Command

```powershell
cd "C:\Users\qintian\Desktop\qintian\final daima"

$env:PYTHONUTF8="1"
$env:PYTHONIOENCODING="utf-8"

& "D:\Anaconda\envs\qintian-rgbd\python.exe" train.py `
  --model dformerv2_c34_bounded_depth_residual `
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
  --checkpoint_dir ".\checkpoints\R036_c34_bounded_depth_residual_run01"
```

## Per-Epoch Validation Metrics

| Val Epoch | Global Step | val/mIoU | val/loss |
|---:|---:|---:|---:|
| 1 | 396 | 0.166644 | 1.627236 |
| 2 | 793 | 0.260703 | 1.332259 |
| 3 | 1190 | 0.323621 | 1.141765 |
| 4 | 1587 | 0.346400 | 1.079173 |
| 5 | 1984 | 0.403198 | 1.044407 |
| 6 | 2381 | 0.433291 | 0.986051 |
| 7 | 2778 | 0.464084 | 0.982872 |
| 8 | 3175 | 0.482719 | 0.950258 |
| 9 | 3572 | 0.483256 | 0.952819 |
| 10 | 3969 | 0.470843 | 1.020493 |
| 11 | 4366 | 0.490999 | 0.973297 |
| 12 | 4763 | 0.494154 | 1.000209 |
| 13 | 5160 | 0.494209 | 0.998186 |
| 14 | 5557 | 0.496038 | 1.001643 |
| 15 | 5954 | 0.488705 | 1.068895 |
| 16 | 6351 | 0.511418 | 1.013710 |
| 17 | 6748 | 0.498353 | 1.055141 |
| 18 | 7145 | 0.468877 | 1.078078 |
| 19 | 7542 | 0.506285 | 1.042380 |
| 20 | 7939 | 0.512038 | 1.046751 |
| 21 | 8336 | 0.511262 | 1.057358 |
| 22 | 8733 | 0.504446 | 1.064456 |
| 23 | 9130 | 0.509399 | 1.072267 |
| 24 | 9527 | 0.514443 | 1.079740 |
| 25 | 9924 | 0.512349 | 1.099479 |
| 26 | 10321 | 0.499426 | 1.128174 |
| 27 | 10718 | 0.481167 | 1.153434 |
| 28 | 11115 | 0.508030 | 1.104224 |
| 29 | 11512 | 0.491697 | 1.144493 |
| 30 | 11909 | 0.495330 | 1.148127 |
| 31 | 12306 | 0.515653 | 1.091286 |
| 32 | 12703 | 0.511328 | 1.141202 |
| 33 | 13100 | 0.522318 | 1.112480 |
| 34 | 13497 | 0.526432 | 1.118934 |
| 35 | 13894 | 0.514139 | 1.198328 |
| 36 | 14291 | 0.525868 | 1.126611 |
| 37 | 14688 | 0.498640 | 1.205247 |
| 38 | 15085 | 0.469963 | 1.321077 |
| 39 | 15482 | 0.512442 | 1.164979 |
| 40 | 15879 | 0.512812 | 1.189018 |
| 41 | 16276 | 0.517324 | 1.171700 |
| 42 | 16673 | 0.507178 | 1.251508 |
| 43 | 17070 | 0.532901 | 1.128109 |
| 44 | 17467 | 0.539790 | 1.139830 |
| 45 | 17864 | 0.535716 | 1.151958 |
| 46 | 18261 | 0.493625 | 1.215188 |
| 47 | 18658 | 0.504998 | 1.201053 |
| 48 | 19055 | 0.528787 | 1.152801 |
| 49 | 19452 | 0.525226 | 1.189851 |
| 50 | 19849 | 0.528882 | 1.208208 |
