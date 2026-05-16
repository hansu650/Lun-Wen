# R054 GeomPrompt-Lite

## Summary

R054 tested whether the corrected R016 input-geometry contract still lacks a small segmentation-aware depth correction. The experiment added a zero-initialized, bounded, model-internal `DepthGeometryPrompt` before both DFormerv2-S geometry attention and the external DepthEncoder.

The run completed 50 validation epochs. It reached best val/mIoU `0.532737` at validation epoch `50`, with last val/mIoU `0.532737` and best-to-last drop `0.000000`.

## Implementation

- Added independent model entry `dformerv2_geomprompt_lite`.
- Added `DepthGeometryPrompt` with a tiny RGB-D prompt network and bounded scalar `alpha_max=0.10`.
- Prompt form: `prompted_depth = depth + alpha * tanh(prompt(cat(rgb, depth)))`.
- Initialized `alpha_logit=0`, so the initial depth update is exactly zero and initial logits matched the active `dformerv2_mid_fusion` output in smoke testing.
- Fed `prompted_depth` into both `DFormerv2_S(rgb, prompted_depth)` and `DepthEncoder(prompted_depth)`.
- Kept DFormerv2-S, pretrained loading, DepthEncoder structure, four-stage `GatedFusion`, SimpleFPNDecoder, CE loss, data module, metric, and fixed recipe unchanged.
- Logged `train/depth_prompt_alpha`, `train/depth_prompt_raw_abs`, and `train/depth_prompt_update_abs`.
- After the below-R016 result, the implementation is archived under `final daima/feiqi/failed_experiments_r054_20260517/` and should not remain in the active registry.

## Dry-Check

- `py_compile` passed for `train.py` and `src/models/mid_fusion.py`.
- `train.py --help` exposed `dformerv2_geomprompt_lite`.
- CUDA forward/backward smoke passed with logits shape `(1, 40, 96, 128)` and finite CE loss.
- Initial logits matched `dformerv2_mid_fusion` exactly in eval smoke (`max_diff=0.0`) because prompt alpha started at zero.
- DFormerv2 pretrained load stats stayed `loaded_keys=774, missing_keys=6, unexpected_keys=11`.
- Static code reviewer: pass.
- Reproducer: pass.

## Fixed Train Command

```powershell
cd "C:\Users\qintian\Desktop\qintian_worktrees\qintian_exp_R054\final daima"
$env:PYTHONUTF8="1"
$env:PYTHONIOENCODING="utf-8"
& "D:\Anaconda\envs\qintian-rgbd\python.exe" train.py `
  --model dformerv2_geomprompt_lite `
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
  --checkpoint_dir ".\checkpoints\R054_geomprompt_lite_run01"
```

## Evidence

- TensorBoard event: `final daima/checkpoints/R054_geomprompt_lite_run01/lightning_logs/version_0/events.out.tfevents.1778966173.Administrator.9928.0`
- Best checkpoint: `final daima/checkpoints/R054_geomprompt_lite_run01/dformerv2_geomprompt_lite-epoch=49-val_mIoU=0.5327.pt`
- mIoU detail: `final daima/miou_list/R054_geomprompt_lite_run01.md`
- Full train completed 50 validation epochs; `Trainer.fit` reached `max_epochs=50`.

## Metrics

| Metric | Value |
|---|---:|
| best val/mIoU | 0.532737 |
| best validation epoch | 50 |
| last val/mIoU | 0.532737 |
| last-5 mean val/mIoU | 0.522253 |
| last-10 mean val/mIoU | 0.516237 |
| best-to-last drop | 0.000000 |
| last val/loss | 1.204177 |
| final train/loss_epoch | 0.054752 |
| depth_prompt_alpha last | 0.000777 |
| prompt_update_abs last | 0.000259 |
| prompt_raw_abs last | 0.333468 |

## Comparison

- vs R016 corrected baseline `0.541121`: `-0.008384`
- vs R036 c3/c4 bounded residual `0.539790`: `-0.007053`
- vs R034 MASG `0.539322`: `-0.006585`
- vs R053 OCR-Lite `0.536867`: `-0.004130`
- vs R050 c4 geometry-primary bypass `0.533066`: `-0.000329`
- vs R024 geometry-primary Ham `0.530186`: `+0.002551`

## Decision

R054 is negative below the corrected mainline. It crosses `0.53` and ends exactly at its best epoch, but the prompt update stayed tiny and the peak is far below R016. This rejects the exact GeomPrompt-Lite path as an active mainline candidate and argues against alpha/hidden-size prompt micro-tuning.

Next candidates should either run a high-decision corrected R016 repeat to estimate whether the `0.541` peak is high-tail variance, or move to a distinct representation hypothesis such as depth-stem/shape-conv geometry extraction. Do not combine prompt tuning with another mechanism.
