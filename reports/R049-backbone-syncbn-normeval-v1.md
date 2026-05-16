# R049 Backbone SyncBN Norm-Eval

## Hypothesis

DFormerv2's local `norm_eval=True` contract is incomplete because the inherited train hook freezes `nn.BatchNorm2d` but not `nn.SyncBatchNorm`. Freezing DFormerv2 backbone SyncBN running stats may reduce batch-size-2 late instability without changing the fixed recipe.

## Implementation

- Added independent model entry `dformerv2_backbone_syncbn_normeval`.
- Set only `rgb_encoder` `BatchNorm2d` / `SyncBatchNorm` modules to eval during training.
- Kept DepthEncoder, GatedFusion, SimpleFPNDecoder, loss, data, eval, and fixed training recipe unchanged.
- After the negative result, the code was removed from the active registry and archived under `final daima/feiqi/failed_experiments_r049_20260516/`.

## Evidence

- TensorBoard event: `final daima/checkpoints/R049_backbone_syncbn_normeval_run01/lightning_logs/version_0/events.out.tfevents.1778935513.Administrator.38220.0`
- Best checkpoint: `final daima/checkpoints/R049_backbone_syncbn_normeval_run01/dformerv2_backbone_syncbn_normeval-epoch=41-val_mIoU=0.5379.pt`
- mIoU detail: `final daima/miou_list/R049_backbone_syncbn_normeval_run01.md`
- Training completed with `Trainer.fit` reaching `max_epochs=50`.

## Results

| Metric | Value |
|---|---:|
| recorded validation epochs | 50 |
| best val/mIoU | 0.537890 |
| best epoch | 41 |
| last val/mIoU | 0.517793 |
| last-5 mean | 0.507274 |
| last-10 mean | 0.515344 |
| best-to-last drop | 0.020097 |
| best val/loss | 0.965282 |
| last val/loss | 1.193775 |

## Comparison

- vs R016 corrected baseline `0.541121`: `-0.003231`
- vs R036 bounded residual `0.539790`: `-0.001900`
- vs R041 DiffPixel c4 cue `0.537098`: `+0.000792`

## Decision

R049 is negative as a corrected-baseline improvement. It crosses `0.53` at the best checkpoint but remains below R016 and has a larger late drop than R016. Do not promote the model or continue norm-freeze micro-variants.

Next direction: R050 should test c4 geometry-primary bypass, asking whether external DepthEncoder c4 fusion is harming DFormerv2's already geometry-conditioned high-level representation.
