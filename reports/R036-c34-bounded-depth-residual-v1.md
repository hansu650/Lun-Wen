# R036 C34 Bounded Depth Residual V1

## Summary

- Branch: `exp/R036-c34-bounded-depth-residual-v1`
- Model: `dformerv2_c34_bounded_depth_residual`
- Run: `R036_c34_bounded_depth_residual_run01`
- Status: `completed_partial_positive_below_corrected_baseline`
- Hypothesis: restrict residual depth correction to c3/c4, bound its amplitude, and zero-initialize it so the model starts exactly from the R016 GatedFusion base.
- Code change: added a separate model entry with c1/c2 original `GatedFusion`, c3/c4 `GatedFusionC34BoundedDepthResidual`; baseline `dformerv2_mid_fusion` remains unchanged.
- Full train: completed 50 validation epochs with exit code `0`.

## Evidence

- TensorBoard event: `final daima/checkpoints/R036_c34_bounded_depth_residual_run01/lightning_logs/version_0/events.out.tfevents.1778851092.Administrator.20952.0`
- Best checkpoint: `final daima/checkpoints/R036_c34_bounded_depth_residual_run01/dformerv2_c34_bounded_depth_residual-epoch=43-val_mIoU=0.5398.pt`
- mIoU detail: `final daima/miou_list/R036_c34_bounded_depth_residual_run01.md`

## Metrics

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
- Delta vs R016 best `0.541121`: `-0.001331`

## Decision

R036 is partial-positive below the corrected baseline. It beats R034 and has a usable final epoch, but it does not exceed R016. The small alpha increase suggests c3/c4 residual carries weak signal, but not enough to justify promoting or continuing the same bounded-residual micro-family.

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
