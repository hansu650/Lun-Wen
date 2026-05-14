# R019 Branch-Specific Depth Adapter

- Branch: `exp/R019-branch-depth-adapter-v1`
- Model: `dformerv2_branch_depth_adapter`
- Run: `R019_branch_depth_adapter_run01`
- Status: partial-positive original-method signal, below corrected baseline

## Hypothesis

R016 made the DFormerv2 geometry branch healthier by feeding official DFormer-normalized depth, but the same normalized tensor is also sent to the external ResNet-18 DepthEncoder. R019 tests whether separating the depth representation inside the model helps: DFormerv2 keeps the R016 official depth, while the external DepthEncoder receives a reconstructed `[0,1]` depth map.

## Implementation

- Added `DFormerV2BranchDepthAdapterSegmentor`.
- Added `LitDFormerV2BranchDepthAdapter`.
- Registered `dformerv2_branch_depth_adapter`.
- DFormerv2 branch: unchanged, receives R016 normalized depth.
- DepthEncoder branch: receives `torch.clamp(depth * 0.28 + 0.48, min=0.0, max=1.0)`.
- `dformerv2_mid_fusion` default behavior is unchanged.

No dataset split, dataloader, augmentation, eval metric, mIoU calculation, loss, optimizer, scheduler, batch size, epoch count, learning rate, worker count, early stopping, DFormerv2-S level, pretrained loading, DepthEncoder architecture, GatedFusion, or SimpleFPNDecoder was changed.

## Smoke Test

- `python -m compileall -q train.py src/models` passed.
- `train.py --help` listed `dformerv2_branch_depth_adapter`.
- CUDA smoke passed:
  - DFormer depth range: `[-1.714286, 1.857143]`
  - DepthEncoder adapter range: `[0.000000, 1.000000]`
  - DepthEncoder adapter mean/std: `0.495325 / 0.305991`
  - logits shape: `(2, 40, 480, 640)`
  - labels preserved as train ids plus `255`.

## Full Train Command

```powershell
cd "C:\Users\qintian\Desktop\qintian_worktrees\nyu056-mainline\final daima"
$env:PYTHONUTF8="1"
$env:PYTHONIOENCODING="utf-8"

& "D:\Anaconda\envs\qintian-rgbd\python.exe" train.py `
  --model dformerv2_branch_depth_adapter `
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
  --checkpoint_dir ".\checkpoints\R019_branch_depth_adapter_run01"
```

## Result

- Exit code: `0`
- Validation epochs: `50`
- Best val/mIoU: `0.532539` at validation epoch `46`
- Last val/mIoU: `0.495229`
- Last-5 mean val/mIoU: `0.509575`
- Last-10 mean val/mIoU: `0.518038`
- Best-to-last drop: `0.037311`
- Best val/loss: `0.958302` at validation epoch `8`
- Final train/loss_epoch: `0.067030`
- TensorBoard event: `final daima/checkpoints/R019_branch_depth_adapter_run01/lightning_logs/version_0/events.out.tfevents.1778765914.Administrator.27112.0`
- Best checkpoint: `final daima/checkpoints/R019_branch_depth_adapter_run01/dformerv2_branch_depth_adapter-epoch=45-val_mIoU=0.5325.pt`
- mIoU detail: `final daima/miou_list/R019_branch_depth_adapter_run01.md`

## Interpretation

R019 crosses the fixed-recipe 0.53 threshold, but it is below the current corrected baseline R016 (`0.541121`) by `0.008582`. The final epoch collapses to `0.495229`, so this is not a stable improvement and not a new main result.

The useful signal is conceptual: branch-specific depth representation can produce a high peak, but simple `[0,1]` reconstruction is unstable. The next step should not be a blind repeat. It should stabilize or soften the branch-specific adapter, or compare against an official Ham parity audit if the goal is still to quantify the DFormer reference gap.
