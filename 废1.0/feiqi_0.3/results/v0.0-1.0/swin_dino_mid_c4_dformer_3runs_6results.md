# swin_dino_mid_c4_dformer_3runs_6results

## Experiment Goal

Evaluate a **single-point replacement** experiment on the stable baseline:

- keep `encoder` unchanged
- keep `decoder` unchanged
- keep `c1 = GatedFusion`
- keep `c2 = GatedFusion`
- keep `c3 = DepthAwareLocalRefineFusion`
- only replace the old `c4` prompt block with a **DFormerv2-style geometry-aware prompt token block**

This experiment is intended to test whether adding a DFormerv2-style geometry prior
to the c4 token interaction can improve the current stable baseline.

---

## Model Difference Compared with Stable Baseline

Stable baseline c4:

- `fused_c4 = GatedFusion(...)`
- `fused_c4 = DepthPromptTokenBlock(fused_c4, depth_c4)`

This experiment c4:

- `fused_c4 = GatedFusion(...)`
- `fused_c4 = GeometryAwarePromptTokenBlock(fused_c4, depth_c4)`

The rest of the model is unchanged.

---

## Collected Results

Three runs were executed, and the checkpoint folders contain a total of 6 saved
checkpoints. They are listed below as-is.

| Run | Epoch | val/mIoU | Checkpoint |
| --- | --- | --- | --- |
| run01 | 29 | 0.4754 | `mid_fusion-epoch=29-val_mIoU=0.4754.ckpt` |
| run01 | 41 | 0.4762 | `mid_fusion-epoch=41-val_mIoU=0.4762.ckpt` |
| run02 | 11 | 0.4779 | `mid_fusion-epoch=11-val_mIoU=0.4779.ckpt` |
| run02 | 27 | 0.4812 | `mid_fusion-epoch=27-val_mIoU=0.4812.ckpt` |
| run03 | 07 | 0.4705 | `mid_fusion-epoch=07-val_mIoU=0.4705.ckpt` |
| run03 | 32 | 0.4797 | `mid_fusion-epoch=32-val_mIoU=0.4797.ckpt` |

---

## Best Result from Each Run

| Run | Best Epoch | Best val/mIoU |
| --- | --- | --- |
| run01 | 41 | 0.4762 |
| run02 | 27 | 0.4812 |
| run03 | 32 | 0.4797 |

Best-of-run statistics:

- mean best val/mIoU: `0.4790`
- min best val/mIoU: `0.4762`
- max best val/mIoU: `0.4812`

---

## Statistics Over All 6 Saved Results

- mean val/mIoU: `0.4768`
- median val/mIoU: `0.4770`
- best val/mIoU: `0.4812`
- worst val/mIoU: `0.4705`
- standard deviation: `0.0034`
- range: `0.0107`

---

## Comparison with Current Stable Baseline

Current stable baseline (10-run summary):

- average val/mIoU: `0.4828`
- best val/mIoU: `0.4900`

This DFormerv2-style c4 replacement experiment:

- best-of-run mean: `0.4790`
- best single observed result: `0.4812`

Difference:

- vs stable baseline mean: `-0.0038`
- vs stable baseline best: `-0.0088`

---

## Interpretation

The current evidence suggests:

1. This c4 geometry-aware replacement is **not clearly better** than the current stable baseline.
2. The experiment can occasionally approach the stable baseline, but it does not
   stably exceed it.
3. Based on these 3 runs, the DFormerv2-style c4 block is better treated as an
   exploratory variant rather than the new main baseline.

In short:

`c4 geometry-aware prompt block did not show stable improvement over the current baseline`

---

## Practical Conclusion

For now, the recommended main baseline should still remain:

- c1: `GatedFusion`
- c2: `GatedFusion`
- c3: `DepthAwareLocalRefineFusion`
- c4: `GatedFusion + original DepthPromptTokenBlock`
- decoder: `SimpleFPNDecoder`

This DFormerv2-style c4 experiment should be recorded as:

- a valid single-point replacement study
- but not a structure that has clearly surpassed the baseline

---

## Suggested Reporting Wording

You can report it like this:

> I tested a single-point c4 replacement where the original prompt token block
> was replaced by a DFormerv2-style geometry-aware prompt block, while all other
> parts of the model remained unchanged. Across 3 runs, the best-per-run average
> mIoU is `47.90%`, which is still below the current stable baseline average of
> `48.28%`. This suggests that the geometry-aware c4 replacement does not yet
> provide stable improvement over the existing prompt-based baseline.
