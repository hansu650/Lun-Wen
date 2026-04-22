# version_001_stable_baseline_mid_fusion

## Role

This snapshot records the first stable mid-fusion baseline version that has
already been repeated multiple times and can be used as the main rollback point.

## Baseline Structure

- encoder: unchanged `Swin-B + DINOv2-B`
- c1: `GatedFusion`
- c2: `GatedFusion`
- c3: `DepthAwareLocalRefineFusion`
- c4: `GatedFusion + DepthPromptTokenBlock`
- decoder: unchanged `SimpleFPNDecoder`

## Included Code

- `train.py`
- `eval.py`
- `infer.py`
- `src/`
- `scripts/`
- the 10-run result summary in `results/`

## Why This Snapshot Is Important

- it is the current stable baseline
- later experiments should be compared against this version
- if a new structure performs worse, we can roll back to this folder directly

## Related Result Summary

See:

- `results/swin_dino_mid_stable_baseline_10runs.md`

This result summary shows that the current stable baseline is reproducible and
its performance is stably around the `48%~49%` mIoU range.
