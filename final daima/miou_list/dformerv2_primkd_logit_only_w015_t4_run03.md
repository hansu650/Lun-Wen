# dformerv2_primkd_logit_only_w015_t4_run03

## Summary

- model: `dformerv2_primkd_logit_only`
- purpose: PMAD / PrimKD logit-only KD with w=0.15 T=4.0 run03.
- student: `DFormerv2_S + ResNet-18 DepthEncoder + GatedFusion + SimpleFPNDecoder`
- teacher: frozen `dformerv2_geometry_primary_teacher_run01`, `DFormerv2_S(rgb, real_depth) + SimpleFPNDecoder`
- teacher checkpoint: `checkpoints/dformerv2_geometry_primary_teacher_run01/dformerv2_geometry_primary_teacher-epoch=37-val_mIoU=0.5168.pt`
- KD settings: `kd_weight=0.15`, `kd_temperature=4.0`, logit-only, no feature KD
- recorded validation epochs: unknown (TensorBoard validation log incomplete; data from checkpoint filename only)
- best val/mIoU: `0.522600` at epoch 41 (from checkpoint filename)
- last val/mIoU: unknown
- best val/loss: unknown
- last val/loss: unknown
- mean val/mIoU over last 10 epochs: unknown
- final train/loss: unknown
- final train/ce_loss: unknown
- final train/kd_loss: unknown
- comparison clean 10-run RGB-D baseline mean best: `0.517397`
- comparison clean 10-run RGB-D baseline std: `0.004901`
- comparison clean 10-run RGB-D baseline mean + 1 std: `0.522298`
- comparison clean 10-run RGB-D baseline best single: `0.524425`
- delta vs clean baseline mean: `+0.005203` (`+1.062` baseline std units)
- delta vs clean baseline mean + 1 std: `+0.000302`
- delta vs clean baseline best single: `-0.001825`
- checkpoint: `checkpoints/dformerv2_primkd_logit_only_w015_t4_run03/dformerv2_primkd_logit_only-epoch=41-val_mIoU=0.5226.pt`
- note: per-epoch TensorBoard validation data unavailable due to incomplete log cleanup; summary derived from checkpoint filename.
- conclusion: positive result. Best val/mIoU exceeds baseline mean + 1 std. This is consistent with run01 and run04, confirming w=0.15 PMAD as a marginal positive candidate.

## Per-Epoch Metrics

Not available (TensorBoard validation log incomplete). See checkpoint filename for best epoch/mIoU.
