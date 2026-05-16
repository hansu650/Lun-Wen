# R044 Conditioned C34 Bounded Residual

- Branch: `exp/R044-conditioned-c34-bounded-residual-v1`
- Model: `dformerv2_conditioned_c34_bounded_residual`
- Run: `R044_conditioned_c34_bounded_residual_run01`
- Evidence: `final daima/miou_list/R044_conditioned_c34_bounded_residual_run01.md`

## Hypothesis

R036's c3/c4 bounded depth residual may underuse image-specific context because alpha is a static stage/channel parameter. R044 tests whether conditioning c3/c4 residual amplitude on DFormerv2 c4 global pooled features can selectively open residual depth signal while preserving the original GatedFusion base.

This was intended as one hypothesis: conditioned residual amplitude. It did not change the decoder, loss, data pipeline, evaluation, or fixed training recipe.

## Implementation

- Added a separate model entry `dformerv2_conditioned_c34_bounded_residual`.
- Preserved c1/c2 original `GatedFusion`.
- Replaced only c3/c4 with a wrapper that computes `base = GatedFusion(rgb, depth)` and adds `alpha * residual`.
- `alpha` is bounded by `alpha_max=0.05` and generated from DFormerv2 c4 global pooled features through a zero-initialized channel head.
- The residual final projection is zero-initialized.
- Logged c3/c4 alpha and residual diagnostics.

## Fixed Recipe Check

R044 did not modify dataset split, evaluation metric, mIoU calculation, val/test loader behavior, data augmentation, optimizer, scheduler, batch size, max_epochs, learning rate, num_workers, early stop, DFormerv2-S level, pretrained loading path, DepthEncoder structure, SimpleFPNDecoder, checkpoint artifacts, datasets, pretrained weights, or TensorBoard event logs.

## Command

```powershell
cd "C:\Users\qintian\Desktop\qintian\final daima"

$env:PYTHONUTF8="1"
$env:PYTHONIOENCODING="utf-8"

& "D:\Anaconda\envs\qintian-rgbd\python.exe" train.py `
  --model dformerv2_conditioned_c34_bounded_residual `
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
  --checkpoint_dir ".\checkpoints\R044_conditioned_c34_bounded_residual_run01"
```

## Evidence

- Full train status: completed with exit code `0`; `Trainer.fit` reached `max_epochs=50`.
- Recorded validation epochs: `50`
- Best val/mIoU: `0.535663` at validation epoch `49`
- Last val/mIoU: `0.520020`
- Last-5 mean val/mIoU: `0.521228`
- Last-10 mean val/mIoU: `0.519138`
- Best-to-last drop: `0.015643`
- Best val/loss: `0.955417` at validation epoch `10`
- Last val/loss: `1.279686`
- Final train/loss_epoch: `0.051134`
- TensorBoard event: `final daima/checkpoints/R044_conditioned_c34_bounded_residual_run01/lightning_logs/version_0/events.out.tfevents.1778900799.Administrator.11996.0`
- Best checkpoint: `final daima/checkpoints/R044_conditioned_c34_bounded_residual_run01/dformerv2_conditioned_c34_bounded_residual-epoch=48-val_mIoU=0.5357.pt`
- Saved command: `final daima/checkpoints/R044_conditioned_c34_bounded_residual_run01/run_r044.cmd`

## Diagnostics

| Metric | First | Last | Max |
|---|---:|---:|---:|
| c3 alpha mean | 0.027802 | 0.049420 | 0.049420 |
| c3 alpha max | 0.030972 | 0.049999 | 0.049999 |
| c3 residual_abs | 0.066489 | 0.737136 | 0.737921 |
| c4 alpha mean | 0.025361 | 0.027943 | 0.028668 |
| c4 alpha max | 0.026435 | 0.044898 | 0.045183 |
| c4 residual_abs | 0.088054 | 0.708959 | 0.708959 |

## Comparison

- Below R016 corrected baseline `0.541121` by `-0.005458`.
- Below R036 c3/c4 bounded residual `0.539790` by `-0.004127`.
- Below R041 DiffPixel c4 cue `0.537098` by `-0.001435`.
- Slightly above R043 depth geometry c4 cue `0.535592` by `+0.000071`, not meaningful.

## Decision

R044 is a negative/diagnostic result below the corrected baseline. The c3 residual branch saturates near its `0.05` cap, residual magnitude grows steadily, and the final best remains below R016/R036/R041 with a best-to-last drop above the `0.015` instability tripwire.

Do not promote the R044 code as active mainline. Archive the implementation under `final daima/feiqi/failed_experiments_r044_20260516/` and remove the registry entry after recording evidence. Do not continue alpha-bound or hidden-size micro-search.

Next direction should avoid another c3/c4 conditioned-residual tweak unless it tests a clearly different mechanism. The most defensible follow-up is a distinct adapter or modality-balance hypothesis, not a repeat of R044.

