# R016 Depth Normalization Official Contract

## Dry Check

- Branch: `exp/R016-depth-norm-contract-v1`
- Model: `dformerv2_mid_fusion`
- Official completed run: `R016_depth_norm_official_baseline_retry1`
- Hypothesis: after R015 aligns the NYU label/ignore contract, depth input should also follow the official DFormer modal_x contract: `raw / 255.0`, then `(x - 0.48) / 0.28`.
- Official source: `ref_codes/DFormer/utils/dataloader/dataloader.py` normalizes `modal_x` with `[0.48, 0.48, 0.48]` / `[0.28, 0.28, 0.28]` when `x_is_single_channel=True`; `ref_codes/DFormer/utils/transforms.py` implements normalize as `/255.0`, subtract mean, divide std.
- Local minimal change: in `final daima/src/data_module.py`, treat `depth` as an Albumentations `mask` target so RGB ImageNet Normalize no longer touches depth, then manually apply DFormer depth normalization.
- No model, decoder, backbone, split, label mapping, metric, mIoU, optimizer, scheduler, batch size, epoch count, lr, num_workers, early stopping, or pretrained-loading logic is changed.

## Smoke Evidence

- `py_compile`: passed for `train.py`, `src/data_module.py`, and `src/models/*.py`.
- `train.py --help`: passed.
- Real batch stats after the change:
  - `rgb`: shape `(2, 3, 480, 640)`, ImageNet-normalized.
  - `depth`: shape `(2, 1, 480, 640)`, min `-1.714286`, max `1.857143`, matching the DFormer depth range.
  - `label`: shape `(2, 480, 640)`, canonical train IDs plus `255`.
- Forward sanity on CUDA passed:
  - logits shape `(2, 40, 480, 640)`
  - CE loss `3.711953`

## Full Train Command For Official Retry1 Result

```powershell
cd "C:\Users\qintian\Desktop\qintian_worktrees\nyu056-mainline\final daima"
$env:PYTHONUTF8="1"
$env:PYTHONIOENCODING="utf-8"

& "D:\Anaconda\envs\qintian-rgbd\python.exe" train.py `
  --model dformerv2_mid_fusion `
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
  --checkpoint_dir ".\checkpoints\R016_depth_norm_official_baseline_retry1"
```

## Result

Retry1 completed a full 50-epoch train with exit code `0`; `Trainer.fit` reached `max_epochs=50`.

- TensorBoard event: `checkpoints/R016_depth_norm_official_baseline_retry1/lightning_logs/version_0/events.out.tfevents.1778745208.Administrator.36016.0`
- Best checkpoint: `checkpoints/R016_depth_norm_official_baseline_retry1/dformerv2_mid_fusion-epoch=48-val_mIoU=0.5411.pt`
- Recorded validation epochs: `50`
- Best val/mIoU: `0.541121` at validation epoch `49`
- Last val/mIoU: `0.527420`
- Last-5 mean val/mIoU: `0.535500`
- Last-10 mean val/mIoU: `0.524063`
- Best-to-last drop: `0.013702`
- Best val/loss: `0.978448` at validation epoch `14`
- Last val/loss: `1.211359`
- Final train/loss_epoch: `0.053537`
- Evidence table: `final daima/miou_list/R016_depth_norm_official_baseline_retry1.md`

Comparison against the R015 official-label baseline:

- R015 best val/mIoU: `0.537398`
- R016 retry1 best val/mIoU: `0.541121`
- Delta: `+0.003723`

Interpretation:

- R016 is a positive official-contract alignment result and becomes the strongest current full-train run.
- This is not a novel method contribution. It aligns the local baseline with the official DFormer modal_x/depth preprocessing contract.
- The run is still below the final `0.56` goal and shows residual late instability, so the loop should continue from this official-label-and-depth baseline.

## Process Note

The first launch `R016_depth_norm_official_baseline` was interrupted after 47 validation epochs by `forrtl error (200): program aborting due to window-CLOSE event` because the empty command window was closed. It had a partial best val/mIoU `0.536203`, but it is not used as the official result because it did not complete 50 epochs. Retry1 is the valid full-train evidence.
