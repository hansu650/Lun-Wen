# R055 R016 Corrected Repeat

## Summary

R055 repeated the corrected R016 mainline under the identical fixed recipe. It intentionally made no model, data, loss, optimizer, scheduler, or evaluation changes.

The run completed 50 validation epochs. It reached best val/mIoU `0.531952` at validation epoch `46`, with last val/mIoU `0.521925` and best-to-last drop `0.010027`.

## Implementation

- Model: `dformerv2_mid_fusion`.
- No code changes.
- Kept DFormerv2-S, DFormerv2 pretrained loading, DepthEncoder, four-stage `GatedFusion`, SimpleFPNDecoder, CE loss, data module, metric, and fixed recipe unchanged.
- This is a baseline calibration repeat, not a new method.

## Dry-Check

- `py_compile` passed for `train.py`, `src/models/mid_fusion.py`, `src/models/decoder.py`, and `src/data_module.py`.
- `train.py --help` exposed `dformerv2_mid_fusion`.
- CUDA forward/backward smoke passed with logits shape `(1, 40, 96, 128)` and finite CE loss.
- DFormerv2 pretrained load stats stayed `loaded_keys=774, missing_keys=6, unexpected_keys=11`.
- Static code reviewer: pass.
- Reproducer note: use explicit `D:\Anaconda\envs\qintian-rgbd\python.exe` from `final daima`; the run did so.

## Fixed Train Command

```powershell
cd "C:\Users\qintian\Desktop\qintian_worktrees\qintian_exp_R055\final daima"
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
  --checkpoint_dir ".\checkpoints\R055_r016_corrected_repeat_run01"
```

## Evidence

- TensorBoard event: `final daima/checkpoints/R055_r016_corrected_repeat_run01/lightning_logs/version_0/events.out.tfevents.1778972303.Administrator.13316.0`
- Best checkpoint: `final daima/checkpoints/R055_r016_corrected_repeat_run01/dformerv2_mid_fusion-epoch=45-val_mIoU=0.5320.pt`
- mIoU detail: `final daima/miou_list/R055_r016_corrected_repeat_run01.md`
- Full train completed 50 validation epochs; `Trainer.fit` reached `max_epochs=50`.
- `val/mIoU_global` matches `val/mIoU` for all 50 validation epochs.

## Metrics

| Metric | Value |
|---|---:|
| best val/mIoU | 0.531952 |
| best validation epoch | 46 |
| last val/mIoU | 0.521925 |
| last-5 mean val/mIoU | 0.509650 |
| last-10 mean val/mIoU | 0.512840 |
| best-to-last drop | 0.010027 |
| best val/loss | 0.956022 |
| best val/loss epoch | 7 |
| val/loss at best mIoU | 1.192035 |
| last val/loss | 1.209070 |
| final train/loss_epoch | 0.056808 |

## Comparison

- vs R016 corrected baseline `0.541121`: `-0.009169`
- vs R036 c3/c4 bounded residual `0.539790`: `-0.007838`
- vs R034 MASG `0.539322`: `-0.007370`
- vs R053 OCR-Lite `0.536867`: `-0.004915`
- vs R054 GeomPrompt-Lite `0.532737`: `-0.000785`
- vs clean baseline best single `0.524425`: `+0.007527`

## Decision

R055 does not confirm R016 as a reproducible `0.541` mainline level. It is a valid full-train corrected-repeat result, but it lands below R016/R036/R034 and close to R054. Treat R016 as a valid historical best checkpoint and likely high-tail anchor, not as a stable expectation for every corrected repeat.

The result strengthens the case for trying a genuinely distinct depth-branch representation experiment next, rather than continuing prompt/residual/decoder micro-variants. The current highest-decision R056 candidate is a narrow HDBFormer/LDFormer-style replacement of the external ResNet-18 DepthEncoder.

## Claim Boundary

- Do not claim R055 as an architecture improvement; it has no new architecture.
- Do not claim R016 peak reproducibility from this repeat.
- It is fair to say that a no-code corrected repeat reached best val/mIoU `0.531952`, below R016 by `0.009169`, which supports recalibrating R016 as a high-tail single-run anchor.
