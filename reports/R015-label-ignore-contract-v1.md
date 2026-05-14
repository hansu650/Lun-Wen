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

## Result

R015 completed the fixed-recipe full train with exit code `0`.

- Recorded validation epochs: `50`
- Best val/mIoU: `0.537398` at validation epoch `45`
- Last val/mIoU: `0.499418`
- Last-5 mean val/mIoU: `0.520010`
- Last-10 mean val/mIoU: `0.520691`
- Best-to-last drop: `0.037981`
- Best val/loss: `0.969897` at validation epoch `10`
- Last val/loss: `1.291720`
- Final train/loss_epoch: `0.093611`
- TensorBoard event: `final daima/checkpoints/R015_label_ignore_official_baseline/lightning_logs/version_0/events.out.tfevents.1778734783.Administrator.15996.0`
- Best checkpoint: `final daima/checkpoints/R015_label_ignore_official_baseline/dformerv2_mid_fusion-epoch=44-val_mIoU=0.5374.pt`
- mIoU details: `final daima/miou_list/R015_label_ignore_official_baseline.md`

## Interpretation

R015 satisfies the fixed-recipe `0.53` stage target with real TensorBoard and checkpoint evidence. The useful discovery is not PMAD/TGGA stacking; it is that the project needed an official NYU label/ignore contract reset before judging the DFormerv2-S gap.

This run still has late instability: best-to-last drop is `0.037981`, and the last epoch falls to `0.499418`. Therefore R015 should be treated as the new official-label baseline and a strong contract-alignment signal, not as a final stable `0.56` solution.

## Next Direction

Continue on the official-label baseline. The next highest decision-value fixed-recipe candidate is official depth normalization contract alignment, tested as one isolated hypothesis. Do not return to PMAD/TGGA threshold or c3 gate micro-search.

## Status

Full train complete; post-run audit pending.

## Smoke Status

- `py_compile`: passed for `train.py`, `src/data_module.py`, `src/utils/metrics.py`, and `src/models/*.py`.
- `train.py --help`: passed; active choices are clean main entries and do not include the failed R014 model.
- label unit check: raw `[0, 1, 2, 40, 255, -1, 41]` maps to `[255, 0, 1, 39, 255, 255, 255]`; `sanitize_labels()` leaves this canonical tensor unchanged.
- real train batch check: labels had min `0`, max `255`, ignore count `51693`, and class-39 count `29120`, confirming raw class 40 is not dropped.
- real forward smoke: `dformerv2_mid_fusion` loaded DFormerv2-S pretrained weights, produced logits `(2, 40, 480, 640)`, and CE loss `3.827293`.
