# R050 C4 Geometry-Primary Bypass

## Hypothesis

DFormerv2 c4 is already conditioned by raw depth through geometry self-attention, so the external ResNet-18 DepthEncoder c4 fusion might add high-level mismatch. R050 tests whether keeping c1-c3 original GatedFusion while bypassing external c4 depth fusion improves the corrected R016 mainline.

## Implementation

- Added independent model entry `dformerv2_c4_geometry_primary_bypass`.
- Kept DFormerv2-S, DepthEncoder, SimpleFPNDecoder, CE loss, data, eval, and fixed training recipe unchanged.
- Used exactly three `GatedFusion` blocks for c1-c3.
- Passed raw DFormerv2 c4 feature directly to the unchanged SimpleFPNDecoder.
- After the negative result, the code was removed from the active registry and archived under `final daima/feiqi/failed_experiments_r050_20260516/`.

## Evidence

- TensorBoard event: `final daima/checkpoints/R050_c4_geometry_primary_bypass_run01/lightning_logs/version_0/events.out.tfevents.1778941335.Administrator.29296.0`
- Best checkpoint: `final daima/checkpoints/R050_c4_geometry_primary_bypass_run01/dformerv2_c4_geometry_primary_bypass-epoch=48-val_mIoU=0.5331.pt`
- mIoU detail: `final daima/miou_list/R050_c4_geometry_primary_bypass_run01.md`
- Training completed with `Trainer.fit` reaching `max_epochs=50`.

## Results

| Metric | Value |
|---|---:|
| recorded validation epochs | 50 |
| best val/mIoU | 0.533066 |
| best epoch | 49 |
| last val/mIoU | 0.526781 |
| last-5 mean | 0.523733 |
| last-10 mean | 0.517746 |
| best-to-last drop | 0.006285 |
| best val/loss | 0.961802 |
| last val/loss | 1.279369 |

## Comparison

- vs R016 corrected baseline `0.541121`: `-0.008055`
- vs R036 bounded residual `0.539790`: `-0.006724`
- vs R049 SyncBN norm-eval `0.537890`: `-0.004824`
- vs R041 DiffPixel c4 cue `0.537098`: `-0.004032`

## Decision

R050 is negative as a corrected-baseline improvement. It crosses `0.53`, but it underperforms every stronger post-R016 candidate and does not support removing the external DepthEncoder c4 fusion path.

Next direction: do not extend this to c3+c4 bypass. The next experiment should keep R016's useful c4 fusion and test a distinct hypothesis, preferably c4 gate conditioning/rectification or another paper-code-supported high-stage mechanism that does not remove the calibrated path.
