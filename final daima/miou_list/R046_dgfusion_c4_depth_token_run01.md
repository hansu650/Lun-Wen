# R046 DGFusion c4 Depth-Token mIoU Detail

- Branch: `exp/R046-dgfusion-c4-depth-token-v1`
- Model: `dformerv2_dgfusion_c4_depth_token`
- Run: `R046_dgfusion_c4_depth_token_run01`
- TensorBoard event: `checkpoints/R046_dgfusion_c4_depth_token_run01/lightning_logs/version_0/events.out.tfevents.1778913804.Administrator.33260.0`
- Best checkpoint: `checkpoints/R046_dgfusion_c4_depth_token_run01/dformerv2_dgfusion_c4_depth_token-epoch=43-val_mIoU=0.5318.pt`
- Full train: completed 50 validation epochs; `Trainer.fit` reached `max_epochs=50`.

## Summary

- Best val/mIoU: `0.531838` at validation epoch `44`
- Last val/mIoU: `0.527239`
- Last-5 mean val/mIoU: `0.510100`
- Last-10 mean val/mIoU: `0.514172`
- Best-to-last drop: `0.004599`
- Best val/loss: `0.961911` at validation epoch `10`
- Last val/loss: `1.208670`
- Final train/loss_epoch: `0.054207`

## Depth-Token Diagnostics

- `c4_token_delta_abs` first/last/min/max: `0.009067` / `0.273799` / `0.009067` / `0.289744`
- `c4_token_affinity_mean` first/last/min/max: `0.623793` / `0.287895` / `0.287302` / `0.623793`
- `c4_token_affinity_std` first/last/min/max: `0.026211` / `0.028371` / `0.026211` / `0.048227`

## Command

```powershell
cd "C:\Users\qintian\Desktop\qintian\final daima"

$env:PYTHONUTF8="1"
$env:PYTHONIOENCODING="utf-8"

& "D:\Anaconda\envs\qintian-rgbd\python.exe" train.py `
  --model dformerv2_dgfusion_c4_depth_token `
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
  --checkpoint_dir ".\checkpoints\R046_dgfusion_c4_depth_token_run01"
```

## Per-Epoch Metrics

| validation epoch | TB step | val/mIoU | val/loss | train/loss_epoch |
|---:|---:|---:|---:|---:|
| 1 | 396 | 0.177535 | 1.615803 | 2.260942 |
| 2 | 793 | 0.266692 | 1.324298 | 1.545819 |
| 3 | 1190 | 0.307133 | 1.194807 | 1.212064 |
| 4 | 1587 | 0.335982 | 1.101277 | 0.977617 |
| 5 | 1984 | 0.400329 | 1.013749 | 0.802469 |
| 6 | 2381 | 0.440674 | 1.013229 | 0.659322 |
| 7 | 2778 | 0.450591 | 0.983978 | 0.555423 |
| 8 | 3175 | 0.470606 | 0.984633 | 0.475864 |
| 9 | 3572 | 0.485652 | 0.972856 | 0.392998 |
| 10 | 3969 | 0.494633 | 0.961911 | 0.335057 |
| 11 | 4366 | 0.494830 | 0.968639 | 0.296832 |
| 12 | 4763 | 0.489310 | 1.004068 | 0.266919 |
| 13 | 5160 | 0.500221 | 0.991881 | 0.240099 |
| 14 | 5557 | 0.490074 | 1.062207 | 0.216796 |
| 15 | 5954 | 0.457569 | 1.155351 | 0.226776 |
| 16 | 6351 | 0.494169 | 1.045651 | 0.236568 |
| 17 | 6748 | 0.496839 | 1.053441 | 0.180445 |
| 18 | 7145 | 0.497167 | 1.047068 | 0.204458 |
| 19 | 7542 | 0.506620 | 1.025600 | 0.162310 |
| 20 | 7939 | 0.516500 | 1.025061 | 0.138922 |
| 21 | 8336 | 0.511124 | 1.065998 | 0.124596 |
| 22 | 8733 | 0.508652 | 1.084798 | 0.135071 |
| 23 | 9130 | 0.482133 | 1.159865 | 0.154651 |
| 24 | 9527 | 0.485274 | 1.103390 | 0.172843 |
| 25 | 9924 | 0.514560 | 1.056235 | 0.122312 |
| 26 | 10321 | 0.517449 | 1.071938 | 0.102141 |
| 27 | 10718 | 0.502418 | 1.164010 | 0.096043 |
| 28 | 11115 | 0.518768 | 1.102520 | 0.108773 |
| 29 | 11512 | 0.514474 | 1.112010 | 0.103995 |
| 30 | 11909 | 0.510728 | 1.128525 | 0.099128 |
| 31 | 12306 | 0.504515 | 1.157768 | 0.114335 |
| 32 | 12703 | 0.502807 | 1.163611 | 0.106954 |
| 33 | 13100 | 0.504753 | 1.143976 | 0.133099 |
| 34 | 13497 | 0.507039 | 1.151558 | 0.100767 |
| 35 | 13894 | 0.523890 | 1.113695 | 0.082190 |
| 36 | 14291 | 0.522798 | 1.133297 | 0.082076 |
| 37 | 14688 | 0.518084 | 1.155415 | 0.078840 |
| 38 | 15085 | 0.510401 | 1.203997 | 0.092173 |
| 39 | 15482 | 0.514765 | 1.169476 | 0.094689 |
| 40 | 15879 | 0.498998 | 1.198822 | 0.095148 |
| 41 | 16276 | 0.509556 | 1.238312 | 0.077188 |
| 42 | 16673 | 0.523560 | 1.198385 | 0.069900 |
| 43 | 17070 | 0.529855 | 1.192749 | 0.061975 |
| 44 | 17467 | 0.531838 | 1.167615 | 0.059384 |
| 45 | 17864 | 0.496411 | 1.312941 | 0.089907 |
| 46 | 18261 | 0.515092 | 1.207805 | 0.083601 |
| 47 | 18658 | 0.459985 | 1.463246 | 0.078066 |
| 48 | 19055 | 0.522483 | 1.209206 | 0.070260 |
| 49 | 19452 | 0.525700 | 1.196884 | 0.058091 |
| 50 | 19849 | 0.527239 | 1.208670 | 0.054207 |
