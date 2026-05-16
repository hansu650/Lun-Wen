# R052 C3 Bounded Depth Residual

## Summary

R052 isolates the strongest partial-positive residual result R036 by keeping c1, c2, and c4 on the original R016 `GatedFusion` path and adding the zero-initialized bounded depth residual only at c3.

## Implementation

- Added independent model entry `dformerv2_c3_bounded_depth_residual`.
- Replaced only `self.fusions[2]` with `GatedFusionC3BoundedDepthResidual`.
- Kept c1, c2, and c4 as original `GatedFusion`.
- Residual form: `base + alpha * residual([depth_proj, abs(rgb-depth_proj)])`, with `alpha_max=0.05`.
- The final residual projection was zero-initialized, so the initial c3 wrapper output exactly matched the base `GatedFusion` output.
- Logged `train/c3_residual_alpha` and `train/c3_residual_abs`.
- After the negative result, the implementation is archived under `final daima/feiqi/failed_experiments_r052_20260517/` and should not remain in the active registry.

## Evidence

- TensorBoard event: `final daima/checkpoints/R052_c3_bounded_depth_residual_run01/lightning_logs/version_0/events.out.tfevents.1778953571.Administrator.43316.0`
- Best checkpoint: `final daima/checkpoints/R052_c3_bounded_depth_residual_run01/dformerv2_c3_bounded_depth_residual-epoch=30-val_mIoU=0.5353.pt`
- mIoU detail: `final daima/miou_list/R052_c3_bounded_depth_residual_run01.md`
- Full train completed 50 validation epochs; `Trainer.fit` reached `max_epochs=50`.

## Metrics

| Metric | Value |
|---|---:|
| best val/mIoU | 0.535289 |
| best validation epoch | 31 |
| last val/mIoU | 0.515195 |
| last-5 mean val/mIoU | 0.521096 |
| last-10 mean val/mIoU | 0.520507 |
| best-to-last drop | 0.020095 |
| best val/loss | 0.962395 |
| best val/loss validation epoch | 8 |
| last val/loss | 1.231749 |
| final train/loss_epoch | 0.114518 |
| c3 residual alpha first / last | 0.025108 / 0.026631 |
| c3 residual abs first / last / max | 0.070006 / 1.037385 / 1.076362 |

## Comparison

- vs R016 corrected baseline `0.541121`: `-0.005832`
- vs R036 c3/c4 bounded residual `0.539790`: `-0.004501`
- vs R051 c4 query-conditioned gate `0.536702`: `-0.001413`
- vs R050 c4 geometry-primary bypass `0.533066`: `+0.002223`

## Decision

R052 is negative below the corrected baseline and below R036. The c3 residual path opens smoothly but only weakly, while the residual magnitude grows to about `1.04` and late validation remains unstable. This rejects the idea that R036's useful signal came from c3 alone.

Do not promote `dformerv2_c3_bounded_depth_residual` and do not continue c3-only alpha/stage micro-search. The next experiment should pivot away from residual-stage isolation, with OCR-Lite decoder context or another non-residual representation mechanism as the current best fallback.
