# R035 Gate Balance Regularizer Run01

## Summary

- Branch: `exp/R035-gate-balance-reg-v1`
- Model: `dformerv2_gate_balance_reg`
- Run: `R035_gate_balance_reg_run01`
- Hypothesis: a tiny training-only regularizer on global GatedFusion gate means may reduce modality bias and late instability without changing inference architecture.
- Status: completed full train; `Trainer.fit` reached `max_epochs=50`; exit code `0`.
- TensorBoard event: `checkpoints/R035_gate_balance_reg_run01/lightning_logs/version_0/events.out.tfevents.1778845626.Administrator.38684.0`
- Best checkpoint: `checkpoints/R035_gate_balance_reg_run01/dformerv2_gate_balance_reg-epoch=37-val_mIoU=0.5295.pt`
- Recorded validation epochs: `50`
- Best val/mIoU: `0.529498` at validation epoch `38`
- Last val/mIoU: `0.521308`
- Last-5 mean val/mIoU: `0.506682`
- Last-10 mean val/mIoU: `0.510080`
- Best-to-last drop: `0.008190`
- Best val/loss: `0.966273` at validation epoch `11`
- Last val/loss: `1.177837`
- Final train/loss_epoch: `0.068845`
- Delta vs R016 corrected baseline best `0.541121`: `-0.011623`
- Delta vs 0.53 stage threshold: `-0.000502`
- Decision: negative; the regularizer improves no main metric and drops below the `0.53` stage threshold.

## Gate Stats

- c1 gate mean first/last: `0.499128` / `0.494539`; std first/last: `0.202534` / `0.195780`
- c2 gate mean first/last: `0.499932` / `0.496786`; std first/last: `0.209182` / `0.208731`
- c3 gate mean first/last: `0.499839` / `0.495702`; std first/last: `0.209284` / `0.209883`
- c4 gate mean first/last: `0.499718` / `0.501463`; std first/last: `0.207818` / `0.207205`

## Command

```powershell
cd "C:\Users\qintian\Desktop\qintian\final daima"

$env:PYTHONUTF8="1"
$env:PYTHONIOENCODING="utf-8"

& "D:\Anaconda\envs\qintian-rgbd\python.exe" train.py `
  --model dformerv2_gate_balance_reg `
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
  --checkpoint_dir ".\checkpoints\R035_gate_balance_reg_run01"
```

## Per-Epoch Validation Metrics

| Val Epoch | Global Step | val/mIoU | val/loss |
|---:|---:|---:|---:|
| 1 | 396 | 0.168630 | 1.690726 |
| 2 | 793 | 0.241349 | 1.341716 |
| 3 | 1190 | 0.316828 | 1.170002 |
| 4 | 1587 | 0.369785 | 1.064499 |
| 5 | 1984 | 0.403042 | 1.051516 |
| 6 | 2381 | 0.425768 | 1.011981 |
| 7 | 2778 | 0.451337 | 0.995444 |
| 8 | 3175 | 0.458911 | 1.000940 |
| 9 | 3572 | 0.457952 | 1.028617 |
| 10 | 3969 | 0.471975 | 1.018225 |
| 11 | 4366 | 0.494020 | 0.966273 |
| 12 | 4763 | 0.466470 | 1.027253 |
| 13 | 5160 | 0.498508 | 1.007928 |
| 14 | 5557 | 0.472928 | 1.075109 |
| 15 | 5954 | 0.492189 | 1.039282 |
| 16 | 6351 | 0.485414 | 1.094866 |
| 17 | 6748 | 0.499120 | 1.069382 |
| 18 | 7145 | 0.503128 | 1.028236 |
| 19 | 7542 | 0.508433 | 1.021445 |
| 20 | 7939 | 0.468944 | 1.197849 |
| 21 | 8336 | 0.510385 | 1.053249 |
| 22 | 8733 | 0.498461 | 1.061627 |
| 23 | 9130 | 0.515087 | 1.079188 |
| 24 | 9527 | 0.488622 | 1.097210 |
| 25 | 9924 | 0.518210 | 1.064396 |
| 26 | 10321 | 0.516267 | 1.072342 |
| 27 | 10718 | 0.510094 | 1.120871 |
| 28 | 11115 | 0.508618 | 1.114049 |
| 29 | 11512 | 0.523833 | 1.132002 |
| 30 | 11909 | 0.493983 | 1.149225 |
| 31 | 12306 | 0.473333 | 1.200963 |
| 32 | 12703 | 0.486639 | 1.178622 |
| 33 | 13100 | 0.498939 | 1.142540 |
| 34 | 13497 | 0.512929 | 1.144088 |
| 35 | 13894 | 0.516830 | 1.130668 |
| 36 | 14291 | 0.520112 | 1.142315 |
| 37 | 14688 | 0.520006 | 1.128525 |
| 38 | 15085 | 0.529498 | 1.139127 |
| 39 | 15482 | 0.472620 | 1.260829 |
| 40 | 15879 | 0.499013 | 1.231585 |
| 41 | 16276 | 0.513327 | 1.204846 |
| 42 | 16673 | 0.501060 | 1.181380 |
| 43 | 17070 | 0.521019 | 1.176120 |
| 44 | 17467 | 0.510840 | 1.190645 |
| 45 | 17864 | 0.521144 | 1.171600 |
| 46 | 18261 | 0.498172 | 1.228219 |
| 47 | 18658 | 0.502538 | 1.201010 |
| 48 | 19055 | 0.498139 | 1.264498 |
| 49 | 19452 | 0.513255 | 1.192237 |
| 50 | 19849 | 0.521308 | 1.177837 |
