# R045 c3/c4 Zero-Init Modality Adapter

- Branch: `exp/R045-c34-zero-init-modality-adapter-v1`
- Model: `dformerv2_c34_zero_init_modality_adapter`
- Run: `R045_c34_zero_init_modality_adapter_run01`
- Status: `completed_negative_adapter_below_corrected_baseline`

## Hypothesis

R016 may need lightweight high-stage modality adaptation before GatedFusion rather than depth input conversion or output residual correction. R045 tests c3/c4-only zero-initialized bottleneck adapters on both DFormerv2 and aligned DepthEncoder features before the original GatedFusion modules.

## Implementation

- Added an independent model entry `dformerv2_c34_zero_init_modality_adapter`.
- c1/c2 stay exactly on the original R016 `GatedFusion` path.
- c3/c4 DFormerv2 and DepthEncoder features pass through separate `Conv1x1 -> ReLU -> Conv1x1` adapters before the existing `GatedFusion` modules.
- The final adapter convolution is zero-initialized, so the smoke test starts exactly from the R016 feature stream.
- Logged adapter delta magnitudes for `rgb_c3`, `rgb_c4`, `depth_c3`, and `depth_c4`.
- Did not change dataset split, eval, mIoU, loaders, augmentation, optimizer, scheduler, batch size, max epochs, lr, workers, early stopping, DFormerv2-S level, pretrained loading, loss, `GatedFusion` class, or `SimpleFPNDecoder`.

## Smoke Test

- `py_compile train.py src\models\mid_fusion.py`: passed.
- `train.py --help`: passed and exposed `dformerv2_c34_zero_init_modality_adapter`.
- Random tensor forward/backward: logits shape `(1, 40, 128, 128)`; initial adapter deltas were all `0.0`; zero-init final conv received nonzero gradient.

## Evidence

- TensorBoard event: `final daima/checkpoints/R045_c34_zero_init_modality_adapter_run01/lightning_logs/version_0/events.out.tfevents.1778907691.Administrator.38920.0`
- Best checkpoint: `final daima/checkpoints/R045_c34_zero_init_modality_adapter_run01/dformerv2_c34_zero_init_modality_adapter-epoch=47-val_mIoU=0.5315.pt`
- Saved command: `final daima/checkpoints/R045_c34_zero_init_modality_adapter_run01/run_r045.ps1`
- mIoU detail: `final daima/miou_list/R045_c34_zero_init_modality_adapter_run01.md`

## Metrics

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

## Decision

R045 is negative relative to the corrected R016 baseline. It crosses `0.53`, but remains below R016 `0.541121` by `-0.009667`, below R036 `0.539790` by `-0.008336`, and below R041 `0.537098` by `-0.005644`. The late drop is `0.026324`, so the adapter path does not stabilize late training.

The adapter deltas clearly opened, especially `depth_c3_adapter_delta_abs` and `depth_c4_adapter_delta_abs`, but this did not translate into a stronger fixed-recipe peak. Do not tune adapter reduction/scale/stages. Archive the code under `final daima/feiqi/failed_experiments_r045_20260516/` and pivot to a distinct R046 hypothesis.

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
