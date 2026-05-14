# R015 Label/Ignore Official Contract Baseline Reset

## Hypothesis

The current NYUDepthV2 label contract is misaligned with the official DFormer NYU recipe. Official DFormer maps raw label `0` to `255` ignore and raw labels `1..40` to train ids `0..39`. R015 tests this contract reset as a new baseline coordinate system.

## Scope

- Branch: `exp/R015-label-ignore-contract-v1`
- Model: `dformerv2_mid_fusion`
- Run: `R015_label_ignore_official_baseline`
- Main code changes:
  - `final daima/src/data_module.py`: map raw labels `0 -> 255`, `1..40 -> 0..39` after augmentation and tensor conversion.
  - `final daima/src/utils/metrics.py`: remove the old dynamic `min>=1` label shift and treat metrics input as train-id labels plus `255` ignore.
- No model, backbone, decoder, optimizer, scheduler, batch size, max epoch, lr, data split, augmentation, or loader-size change.

## Contract Boundary

This is not directly comparable to the old `0.517397` baseline as a simple improvement claim. It resets the label/ignore contract to match official DFormer semantics. Use it as the new official-label baseline for later experiments.

## Fixed Recipe

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
  --checkpoint_dir ".\checkpoints\R015_label_ignore_official_baseline"
```

## Status

Smoke tests passed; full train pending.

## Smoke Status

- `py_compile`: passed for `train.py`, `src/data_module.py`, `src/utils/metrics.py`, and `src/models/*.py`.
- `train.py --help`: passed; active choices are clean main entries and do not include the failed R014 model.
- label unit check: raw `[0, 1, 2, 40, 255, -1, 41]` maps to `[255, 0, 1, 39, 255, 255, 255]`; `sanitize_labels()` leaves this canonical tensor unchanged.
- real train batch check: labels had min `0`, max `255`, ignore count `51693`, and class-39 count `29120`, confirming raw class 40 is not dropped.
- real forward smoke: `dformerv2_mid_fusion` loaded DFormerv2-S pretrained weights, produced logits `(2, 40, 480, 640)`, and CE loss `3.827293`.
