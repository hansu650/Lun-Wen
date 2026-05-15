# R035 Gate Balance Regularizer V1

## Summary

- Branch: `exp/R035-gate-balance-reg-v1`
- Model: `dformerv2_gate_balance_reg`
- Run: `R035_gate_balance_reg_run01`
- Status: `completed_negative_below_053`
- Hypothesis: a training-only gate balance regularizer can reduce modality-bias-driven instability while leaving inference unchanged.
- Code change: added a separate model entry that logs gates and adds `0.01 * mean((gate_mean - 0.5)^2)` during training only; the baseline `dformerv2_mid_fusion` remains unchanged.
- Full train: completed 50 validation epochs with exit code `0`.

## Evidence

- TensorBoard event: `final daima/checkpoints/R035_gate_balance_reg_run01/lightning_logs/version_0/events.out.tfevents.1778845626.Administrator.38684.0`
- Best checkpoint: `final daima/checkpoints/R035_gate_balance_reg_run01/dformerv2_gate_balance_reg-epoch=37-val_mIoU=0.5295.pt`
- mIoU detail: `final daima/miou_list/R035_gate_balance_reg_run01.md`

## Metrics

- Best val/mIoU: `0.529498` at validation epoch `38`
- Last val/mIoU: `0.521308`
- Last-5 mean val/mIoU: `0.506682`
- Last-10 mean val/mIoU: `0.510080`
- Best-to-last drop: `0.008190`
- Best val/loss: `0.966273` at validation epoch `11`
- Last val/loss: `1.177837`
- Final train/loss_epoch: `0.068845`
- Delta vs R016 best `0.541121`: `-0.011623`
- Delta vs 0.53 stage threshold: `-0.000502`

## Gate Stats

- c1 gate mean first/last: `0.499128` / `0.494539`; std first/last: `0.202534` / `0.195780`
- c2 gate mean first/last: `0.499932` / `0.496786`; std first/last: `0.209182` / `0.208731`
- c3 gate mean first/last: `0.499839` / `0.495702`; std first/last: `0.209284` / `0.209883`
- c4 gate mean first/last: `0.499718` / `0.501463`; std first/last: `0.207818` / `0.207205`

## Decision

R035 is negative. It is more stable than R034 by best-to-last drop, but it suppresses peak mIoU to `0.529498`, below both the `0.53` stage threshold and R016. Do not continue gate-balance lambda tuning. Pivot to a structurally distinct direction, with c3/c4 bounded low-amplitude depth residual as the next highest decision-value candidate.

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
