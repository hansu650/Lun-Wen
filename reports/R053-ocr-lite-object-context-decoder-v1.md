# R053 OCR-Lite Object Context Decoder

## Summary

R053 tested whether a lightweight OCR-style object-context decoder can improve the R016 `SimpleFPNDecoder` ceiling without changing the DFormerv2-S encoder, DepthEncoder, GatedFusion, loss, data, or fixed training recipe.

The run completed 50 validation epochs. It reached best val/mIoU `0.536867` at validation epoch `49`, with last val/mIoU `0.522340` and best-to-last drop `0.014527`.

## Implementation

- Added independent model entry `dformerv2_ocr_lite_decoder`.
- Replaced only the decoder with `OCRLiteDecoder`.
- Kept DFormerv2-S, pretrained loading, DepthEncoder, four-stage `GatedFusion`, CE loss, data module, metric, and fixed recipe unchanged.
- `OCRLiteDecoder` keeps SimpleFPN lateral/top-down construction, then gathers class prototypes from a dense prior classifier with class-wise spatial softmax.
- Pixel-to-class attention refines p1 through a zero-initialized residual context update.
- No auxiliary CE, no hard masks, no mmcv/mmseg dependency, no teacher, and no extra loss.
- After the below-R016 result, the implementation is archived under `final daima/feiqi/failed_experiments_r053_20260517/` and should not remain in the active registry.

## Dry-Check

- `py_compile` passed for `train.py`, `src/models/decoder.py`, and `src/models/mid_fusion.py`.
- `train.py --help` exposed `dformerv2_ocr_lite_decoder`.
- CUDA forward/backward smoke passed with logits shape `(1, 40, 96, 128)` and finite CE loss.
- DFormerv2 pretrained load stats stayed `loaded_keys=774, missing_keys=6, unexpected_keys=11`.
- Static code reviewer: pass.
- Reproducer: pass.

## Fixed Train Command

```powershell
cd "C:\Users\qintian\Desktop\qintian_worktrees\qintian_exp_R053\final daima"
$env:PYTHONUTF8="1"
$env:PYTHONIOENCODING="utf-8"
& "D:\Anaconda\envs\qintian-rgbd\python.exe" train.py `
  --model dformerv2_ocr_lite_decoder `
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
  --checkpoint_dir ".\checkpoints\R053_ocr_lite_object_context_run01"
```

## Evidence

- TensorBoard event: `final daima/checkpoints/R053_ocr_lite_object_context_run01/lightning_logs/version_0/events.out.tfevents.1778959970.Administrator.39208.0`
- Best checkpoint: `final daima/checkpoints/R053_ocr_lite_object_context_run01/dformerv2_ocr_lite_decoder-epoch=48-val_mIoU=0.5369.pt`
- mIoU detail: `final daima/miou_list/R053_ocr_lite_object_context_run01.md`
- Full train completed 50 validation epochs; `Trainer.fit` reached `max_epochs=50`.

## Metrics

| Metric | Value |
|---|---:|
| best val/mIoU | 0.536867 |
| best validation epoch | 49 |
| last val/mIoU | 0.522340 |
| last-5 mean val/mIoU | 0.520132 |
| last-10 mean val/mIoU | 0.523362 |
| best-to-last drop | 0.014527 |
| val/loss at best mIoU | 1.217654 |
| last val/loss | 1.270678 |
| final train/loss_epoch | 0.057302 |
| OCR context update abs first / last / max | 0.768868 / 0.909595 / 0.909595 |
| OCR prior entropy first / last | 9.806945 / 9.703479 |

## Comparison

- vs R016 corrected baseline `0.541121`: `-0.004254`
- vs R036 c3/c4 bounded residual `0.539790`: `-0.002923`
- vs R034 MASG `0.539322`: `-0.002455`
- vs R049 norm-eval diagnostic `0.537890`: `-0.001023`
- vs R041 DiffPixel c4 cue `0.537098`: `-0.000231`
- vs R052 c3-only residual `0.535289`: `+0.001578`

## Decision

R053 is a partial positive below the corrected baseline. The OCR context branch clearly opened, but the mIoU plateau stayed below R016 and the late drop nearly reached the instability tripwire. This rejects OCR-Lite as an active mainline candidate and argues against immediate OCR width/context/dropout micro-tuning.

Next candidates should move away from decoder context micro-variants. Stronger options are an input-geometry prompt/contract experiment or a carefully justified repeat of the corrected R016 mainline if the goal is to estimate high-tail variance before adding another mechanism.
