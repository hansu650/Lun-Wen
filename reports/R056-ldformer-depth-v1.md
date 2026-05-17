# R056 LDFormer Depth

## Summary

R056 tested whether the external ResNet-18 depth encoder is the limiting depth-representation contract by replacing it with a thin HDBFormer/LDFormer-style depthwise-separable depth branch.

The run completed 50 validation epochs. It reached best val/mIoU `0.522759` at validation epoch `44`, with last val/mIoU `0.518073` and best-to-last drop `0.004686`.

## Implementation

- Added independent model entry `dformerv2_ldformer_depth`.
- Replaced only the external `DepthEncoder` with `LDFormerDepthEncoder`.
- `LDFormerDepthEncoder` used local-compatible channels `[1, 32, 64, 128, 256, 512]` and returned four feature maps with channels `[64, 128, 256, 512]`.
- Each depth stage used `DWConv3x3 -> BN -> ReLU -> PWConv1x1 -> BN -> ReLU -> MaxPool2d(2)`, inspired by HDBFormer LDFormer.
- Kept DFormerv2-S, DFormerv2 pretrained loading, four original `GatedFusion` blocks, SimpleFPNDecoder, CE loss, data module, metric, and fixed recipe unchanged.
- Did not copy HDBFormer, MIIM, LIFormer, 3-channel depth replication, decoder changes, auxiliary loss, or training recipe.
- After the below-R016 result, the implementation is archived under `final daima/feiqi/failed_experiments_r056_20260517/` and should not remain in the active registry.

## Paper/Code Boundary

- Paper: HDBFormer, arXiv `2504.13579`.
- Official repo: `https://github.com/Weishuobin/HDBFormer`.
- Official LDFormer uses a five-stage depthwise-separable conv pyramid and returns stages after dropping the first `/2` output in the full HDBFormer system.
- R056 is LDFormer-inspired and local-contract compatible; it is not an HDBFormer reproduction.

## Dry-Check

- `py_compile` passed for `train.py`, `src/models/encoder.py`, and `src/models/mid_fusion.py`.
- `train.py --help` exposed `dformerv2_ldformer_depth`.
- CUDA forward/backward smoke passed with logits shape `(1, 40, 96, 128)` and finite CE loss.
- Smoke verified depth, DFormerv2, and fused feature shapes all matched `[64, 128, 256, 512]` stage channels.
- LDFormer depth branch gradients were nonzero.
- DFormerv2 pretrained load stats stayed `loaded_keys=774, missing_keys=6, unexpected_keys=11`.
- Static code reviewer: pass.
- Reproducer: pass.

## Fixed Train Command

```powershell
cd "C:\Users\qintian\Desktop\qintian_worktrees\qintian_exp_R056\final daima"
$env:PYTHONUTF8="1"
$env:PYTHONIOENCODING="utf-8"
& "D:\Anaconda\envs\qintian-rgbd\python.exe" train.py `
  --model dformerv2_ldformer_depth `
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
  --checkpoint_dir ".\checkpoints\R056_ldformer_depth_run01"
```

## Evidence

- TensorBoard event: `final daima/checkpoints/R056_ldformer_depth_run01/lightning_logs/version_0/events.out.tfevents.1778977669.Administrator.18304.0`
- Best checkpoint: `final daima/checkpoints/R056_ldformer_depth_run01/dformerv2_ldformer_depth-epoch=43-val_mIoU=0.5228.pt`
- mIoU detail: `final daima/miou_list/R056_ldformer_depth_run01.md`
- Full train completed 50 validation epochs; `Trainer.fit` reached `max_epochs=50`.
- `val/mIoU_global` matches `val/mIoU` for all 50 validation epochs.

## Metrics

| Metric | Value |
|---|---:|
| best val/mIoU | 0.522759 |
| best validation epoch | 44 |
| last val/mIoU | 0.518073 |
| last-5 mean val/mIoU | 0.516419 |
| last-10 mean val/mIoU | 0.517597 |
| best-to-last drop | 0.004686 |
| best val/loss | 1.008049 |
| best val/loss epoch | 8 |
| val/loss at best mIoU | 1.208968 |
| last val/loss | 1.222330 |
| final train/loss_epoch | 0.080565 |

## Comparison

- vs R016 corrected baseline `0.541121`: `-0.018362`
- vs R055 corrected repeat `0.531952`: `-0.009193`
- vs R054 GeomPrompt-Lite `0.532737`: `-0.009978`
- vs R053 OCR-Lite `0.536867`: `-0.014108`
- vs clean baseline best single `0.524425`: `-0.001666`

## Decision

R056 is negative. The lightweight LDFormer-style external depth branch is stable but loses too much capacity compared with the pretrained ResNet-18 depth encoder under the fixed recipe. This rejects the exact lightweight depth-branch replacement as an active mainline candidate.

Do not tune LDFormer width, add MIIM, or add decoder/loss changes as immediate micro-variants. The next round should use the result as evidence that the external depth branch still benefits from pretrained capacity, then choose a distinct high-decision hypothesis rather than another from-scratch depth encoder swap.
