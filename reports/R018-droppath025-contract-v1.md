# R018 DropPath 0.25 Contract Gate

- Branch: `exp/R018-droppath025-contract-v1`
- Model tested: `dformerv2_mid_fusion_dpr025`
- Run used for official result: `R018_dformerv2_mid_fusion_dpr025_retry1`
- Status: completed negative contract gate

## Hypothesis

Official DFormerv2-S NYUDepthv2 uses `drop_path_rate = 0.25`; the local DFormerv2-S wrapper defaulted to `0.1`. After R015/R016 corrected label and depth contracts, testing the official drop path rate may close part of the remaining gap toward the DFormerv2-S reference result.

## Evidence for the Change

- Official config: `C:\Users\qintian\Desktop\qintian\ref_codes\DFormer\local_configs\NYUDepthv2\DFormerv2_S.py` sets `C.drop_path_rate = 0.25`.
- Local encoder: `final daima/src/models/dformerv2_encoder.py` defaults `drop_path_rate=0.1`, and `DFormerv2_S(pretrained=False, **kwargs)` forwards kwargs into `dformerv2(...)`.

## Implementation

- Temporarily added `drop_path_rate` passthrough to `DFormerV2MidFusionSegmentor`.
- Temporarily registered `dformerv2_mid_fusion_dpr025` as a separate model name.
- Preserved `dformerv2_mid_fusion` default behavior.
- Did not change dataset split, label mapping, depth normalization, RGB path, data augmentation, loader behavior, eval metric, mIoU calculation, loss, optimizer, scheduler, batch size, epoch count, learning rate, worker count, early stopping, DFormerv2-S level, pretrained path, DepthEncoder, GatedFusion, or SimpleFPNDecoder.

Because the full train was negative, the active code change was not retained. The failed diff is archived in `final daima/feiqi/failed_experiments_r014_plus_20260514/R018_droppath025_contract.md`.

## Smoke Test

- `python -m compileall -q train.py src/models` passed.
- `train.py --help` listed `dformerv2_mid_fusion_dpr025`.
- CUDA smoke confirmed:
  - DropPath module count: `29`
  - max drop path: `0.25`
  - last drop path: `0.25`
  - logits shape: `(2, 40, 480, 640)`
  - labels preserved as train ids plus `255`.

## Full Train Command

```powershell
cd "C:\Users\qintian\Desktop\qintian_worktrees\nyu056-mainline\final daima"
$env:PYTHONUTF8="1"
$env:PYTHONIOENCODING="utf-8"

& "D:\Anaconda\envs\qintian-rgbd\python.exe" train.py `
  --model dformerv2_mid_fusion_dpr025 `
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
  --checkpoint_dir ".\checkpoints\R018_dformerv2_mid_fusion_dpr025_retry1"
```

The first foreground launch reached 42 validation epochs but hung after the command wrapper timed out and the stdout/progress pipe stopped advancing. It is partial process evidence only and is excluded from the result. Retry1 was launched through a `.bat` wrapper and completed normally.

## Result

- Exit code: `0`
- Validation epochs: `50`
- Best val/mIoU: `0.526282` at validation epoch `46`
- Last val/mIoU: `0.522893`
- Last-5 mean val/mIoU: `0.512694`
- Last-10 mean val/mIoU: `0.513363`
- Best-to-last drop: `0.003389`
- Best val/loss: `0.948631` at validation epoch `10`
- Final train/loss_epoch: `0.078764`
- TensorBoard event: `final daima/checkpoints/R018_dformerv2_mid_fusion_dpr025_retry1/lightning_logs/version_0/events.out.tfevents.1778760450.Administrator.7836.0`
- Best checkpoint: `final daima/checkpoints/R018_dformerv2_mid_fusion_dpr025_retry1/dformerv2_mid_fusion_dpr025-epoch=45-val_mIoU=0.5263.pt`
- mIoU detail: `final daima/miou_list/R018_dformerv2_mid_fusion_dpr025_retry1.md`

## Interpretation

R018 is below the R016 corrected baseline `0.541121` by `0.014839` and below R010 PMAD logit-only `0.527469` by `0.001187`. The official `drop_path_rate=0.25` setting is therefore not promoted for the current local mid-fusion pipeline.

This result is a useful negative contract gate: label and depth contract alignment were beneficial, BGR and drop path 0.25 were not. The current corrected baseline remains R016.

## Next Direction

If the loop continues below `0.56`, stop baseline-contract micro-gates unless a new mismatch is found. The next highest-value direction is to decide between:

- official Ham decoder parity audit, if we still need to quantify official structure gap;
- corrected-contract PMAD teacher refresh;
- branch-specific depth input adapter that feeds DFormer-normalized depth to the geometry branch while giving the external DepthEncoder a better-suited representation.
