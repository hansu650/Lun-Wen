# R020 Branch-Specific Depth Blend Adapter

- Branch: `exp/R020-depth-blend-adapter-v1`
- Model: `dformerv2_branch_depth_blend_adapter`
- Run: `R020_branch_depth_blend_adapter_run01`
- Status: partial-positive stabilization signal, below corrected baseline

## Hypothesis

R019 showed that branch-specific depth representation can produce a useful high peak, but hard-switching the external DepthEncoder input to reconstructed `[0,1]` depth caused severe late collapse. R020 tests whether a learnable global convex blend can stabilize that signal while staying close to the R016 corrected baseline input distribution.

## Implementation

- Added `DFormerV2BranchDepthBlendAdapterSegmentor`.
- Added `LitDFormerV2BranchDepthBlendAdapter`.
- Registered `dformerv2_branch_depth_blend_adapter`.
- DFormerv2 geometry branch receives the original R016 official-normalized depth.
- DepthEncoder receives `(1 - alpha) * depth + alpha * depth01`, where `depth01 = clamp(depth * 0.28 + 0.48, 0, 1)`.
- `alpha = sigmoid(depth_blend_logit)` is a single learnable scalar initialized to `0.05`.
- Logged `train/depth_blend_alpha`.

No dataset split, dataloader, augmentation, eval metric, mIoU calculation, loss, optimizer, scheduler, batch size, epoch count, learning rate, worker count, early stopping, DFormerv2-S level, pretrained loading, DepthEncoder architecture, GatedFusion, or SimpleFPNDecoder was changed.

## Smoke Test

- `python -m compileall -q train.py src/models` passed.
- `train.py --help` listed `dformerv2_branch_depth_blend_adapter`.
- CUDA smoke passed:
  - initial alpha: `0.050000`
  - DFormer depth range: `[-1.714286, 1.857143]`
  - reconstructed depth01 range: `[0.000000, 1.000000]`
  - blended DepthEncoder input range: `[-1.628571, 1.814286]`
  - blended DepthEncoder input mean/std: `-0.004069 / 0.926482`
  - logits shape: `(2, 40, 480, 640)`

## Full Train Command

```powershell
cd "C:\Users\qintian\Desktop\qintian_worktrees\nyu056-mainline\final daima"
$env:PYTHONUTF8="1"
$env:PYTHONIOENCODING="utf-8"

& "D:\Anaconda\envs\qintian-rgbd\python.exe" train.py `
  --model dformerv2_branch_depth_blend_adapter `
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
  --checkpoint_dir ".\checkpoints\R020_branch_depth_blend_adapter_run01"
```

## Result

- Exit code: `0`
- Validation epochs: `50`
- Best val/mIoU: `0.532924` at validation epoch `41`
- Last val/mIoU: `0.503238`
- Last-5 mean val/mIoU: `0.520456`
- Last-10 mean val/mIoU: `0.516804`
- Best-to-last drop: `0.029686`
- Best val/loss: `0.979484` at validation epoch `8`
- Final train/loss_epoch: `0.089993`
- Alpha first/last: `0.050022` / `0.051455`
- Alpha min/max: `0.050022` / `0.051455`
- TensorBoard event: `final daima/checkpoints/R020_branch_depth_blend_adapter_run01/lightning_logs/version_0/events.out.tfevents.1778771221.Administrator.41764.0`
- Best checkpoint: `final daima/checkpoints/R020_branch_depth_blend_adapter_run01/dformerv2_branch_depth_blend_adapter-epoch=40-val_mIoU=0.5329.pt`
- mIoU detail: `final daima/miou_list/R020_branch_depth_blend_adapter_run01.md`

## Interpretation

R020 slightly improves the R019 peak (`0.532924` vs `0.532539`) and is more stable through the late epochs, but it remains below the R016 corrected baseline `0.541121` by `0.008197`. It is not a new main result.

The alpha trace stays near `0.05`, so the model did not learn to move strongly toward R019's hard `[0,1]` depth branch. The remaining late drop suggests the instability is not only caused by the hard input switch; the next experiment should either target late stability directly or use a richer branch adapter than a global scalar blend.
