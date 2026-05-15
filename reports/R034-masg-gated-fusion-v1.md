# R034 MASG Gated Fusion V1

## Summary

- Branch: `exp/R034-masg-gated-fusion-v1`
- Model: `dformerv2_masg_fusion`
- Run: `R034_masg_gated_fusion_run01`
- Status: `completed_negative_unstable_below_corrected_baseline`
- Hypothesis: detach only the depth projection used to compute the GatedFusion gate, while keeping the RGB gate path and depth value path trainable, to test whether gate-depth gradient coupling causes R016 late instability.
- Code change: added a separate MASG model entry; the baseline `dformerv2_mid_fusion` remains unchanged.
- Full train: completed 50 validation epochs with exit code `0`.

## Evidence

- TensorBoard event: `final daima/checkpoints/R034_masg_gated_fusion_run01/lightning_logs/version_0/events.out.tfevents.1778839940.Administrator.26056.0`
- Best checkpoint: `final daima/checkpoints/R034_masg_gated_fusion_run01/dformerv2_masg_fusion-epoch=39-val_mIoU=0.5393.pt`
- mIoU detail: `final daima/miou_list/R034_masg_gated_fusion_run01.md`

## Metrics

- Best val/mIoU: `0.539322` at validation epoch `40`
- Last val/mIoU: `0.518738`
- Last-5 mean val/mIoU: `0.504633`
- Last-10 mean val/mIoU: `0.512033`
- Best-to-last drop: `0.020584`
- Best val/loss: `0.966439` at validation epoch `8`
- Last val/loss: `1.230551`
- Final train/loss_epoch: `0.059559`
- Delta vs R016 best `0.541121`: `-0.001799`

## Decision

R034 is negative relative to the corrected baseline. It stays below R016 and has a larger best-to-last drop than R016, so depth-only gate stop-gradient does not solve the late-instability problem. Do not merge the MASG model into active mainline; keep the result as evidence that the dominant instability is not fixed by detaching the depth projection only in gate computation.

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

## Review Notes

- Fixed recipe was preserved: batch size `2`, max epochs `50`, lr `6e-5`, workers `4`, early-stop patience `30`, CE loss, unchanged dataset/eval/loader/split.
- No checkpoint, TensorBoard event, dataset, pretrained weight, ref code, or large log should be staged.
- Next direction: stop MASG micro-search and pivot to a distinct high-value hypothesis such as modality-balance regularization or bounded high-stage depth residual.
