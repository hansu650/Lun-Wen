# swin_dino_mid_stable_baseline_10runs

## Experiment Goal

Verify whether the current best mid-fusion baseline is stable and reproducible by
running the same experiment 10 times with the same training configuration.

Current fixed baseline structure:

- encoder: unchanged `Swin-B + DINOv2-B`
- c1: `GatedFusion`
- c2: `GatedFusion`
- c3: `DepthAwareLocalRefineFusion`
- c4: `GatedFusion + DepthPromptTokenBlock`
- decoder: unchanged `SimpleFPNDecoder`
- train / eval / infer flow: unchanged

The only purpose of this experiment is:

- measure run-to-run variance
- check whether the current best structure is truly reproducible

---

## Training Setup

- model: `mid_fusion`
- data root:
  `C:\Users\qintian\Desktop\qintian\data\NYUDepthv2`
- max epochs: `50`
- batch size: `2`
- learning rate: `1e-4`
- num workers: `0`
- accelerator: `gpu`
- device: `1`

All 10 runs used the same configuration. The only difference is the checkpoint
directory name.

---

## 10-Run Results

| Run | Best Epoch | Best val/mIoU | Checkpoint |
| --- | --- | --- | --- |
| run01 | 19 | 0.4820 | `mid_fusion-epoch=19-val_mIoU=0.4820.ckpt` |
| run02 | 14 | 0.4810 | `mid_fusion-epoch=14-val_mIoU=0.4810.ckpt` |
| run03 | 30 | 0.4798 | `mid_fusion-epoch=30-val_mIoU=0.4798.ckpt` |
| run04 | 46 | 0.4759 | `mid_fusion-epoch=46-val_mIoU=0.4759.ckpt` |
| run05 | 24 | 0.4866 | `mid_fusion-epoch=24-val_mIoU=0.4866.ckpt` |
| run06 | 17 | 0.4875 | `mid_fusion-epoch=17-val_mIoU=0.4875.ckpt` |
| run07 | 18 | 0.4869 | `mid_fusion-epoch=18-val_mIoU=0.4869.ckpt` |
| run08 | 21 | 0.4900 | `mid_fusion-epoch=21-val_mIoU=0.4900.ckpt` |
| run09 | 21 | 0.4803 | `mid_fusion-epoch=21-val_mIoU=0.4803.ckpt` |
| run10 | 25 | 0.4782 | `mid_fusion-epoch=25-val_mIoU=0.4782.ckpt` |

---

## Summary Statistics

- mean val/mIoU: `0.4828`
- median val/mIoU: `0.4815`
- best val/mIoU: `0.4900`
- worst val/mIoU: `0.4759`
- standard deviation: `0.0044`
- range: `0.0141`

Additional observations:

- `7 / 10` runs are at or above `0.4800`
- `4 / 10` runs are at or above `0.4850`

---

## Interpretation

This 10-run experiment supports the following conclusions:

1. The current mid-fusion baseline is reproducible.
   - The model does not rely on one lucky run to get a strong result.

2. The real performance level is around `48.3% mIoU`.
   - A single run may go up or down by a few tenths of a point.
   - The mean result is more reliable than any single isolated run.

3. The structure is stable enough to be used as the main baseline for later comparison.
   - average performance: `48.28%`
   - best observed performance: `49.00%`

In short:

`current stable baseline ≈ 48.3% mIoU, with best run reaching 49.0%`

---

## Why This Result Matters

This experiment is important because it confirms that:

- the current best structure is not accidental
- the observed gain is reproducible
- future module experiments should be compared against this 10-run baseline,
  not against a single lucky checkpoint

This gives a cleaner comparison standard for later experiments.

---

## Suggested Reporting Wording

You can report it like this:

> I repeated the current stable baseline experiment 10 times under the same
> training configuration. The average validation mIoU is `48.28%`, and the
> best run reaches `49.00%`. This indicates that the current structure has
> good reproducibility, and its true performance is stably in the `48%~49%`
> range rather than depending on a single lucky run.

---

## Current Baseline Definition

The baseline confirmed by this 10-run experiment is:

- c1: `GatedFusion`
- c2: `GatedFusion`
- c3: `DepthAwareLocalRefineFusion`
- c4: `GatedFusion + DepthPromptTokenBlock`
- decoder: `SimpleFPNDecoder`

This should be treated as the main reference baseline for future ablation
experiments.
