# swin_dino_mid_b2_mcads_decoder_summary

## Experiment Goal

Evaluate whether replacing only the final decoder/head stage with an
MCADS-inspired decoder core can improve the current `v1.0` stable baseline.

The stable baseline remains:

- encoder: unchanged `Swin-B + DINOv2-B`
- c1: `GatedFusion`
- c2: `GatedFusion`
- c3: `DepthAwareLocalRefineFusion`
- c4: `GatedFusion + DepthPromptTokenBlock`
- decoder: `SimpleFPNDecoder`

This B2 experiment changes only the decoder tail.

---

## Trial Groups

### Group A: original MCADS core transplant

Experiment prefix:

- `swin_dino_mid_b2_mcads_decoder_run01`
- `swin_dino_mid_b2_mcads_decoder_run02`
- `swin_dino_mid_b2_mcads_decoder_run03`

Result:

- all 3 runs failed during sanity check
- failure reason: CUDA out of memory
- root cause: the transplanted residual linear attention was applied directly on
  the `p1` feature map, which produced a very large `N x N` attention matrix

Conclusion:

- the original MCADS core block cannot be directly used at the current `p1`
  resolution under the current hardware setting

### Group B: reduced-complexity MCADS head

Experiment prefix:

- `swin_dino_mid_b2_mcads_decoder_reduced_run01`
- `swin_dino_mid_b2_mcads_decoder_reduced_run02`
- `swin_dino_mid_b2_mcads_decoder_reduced_run03`
- `swin_dino_mid_b2_mcads_decoder_reduced_run04`
- `swin_dino_mid_b2_mcads_decoder_reduced_run05`
- `swin_dino_mid_b2_mcads_decoder_reduced_run06`

In this version, the MCADS residual linear attention was kept, but attention was
computed on a reduced spatial map before being upsampled back.

---

## Reduced Version Results

| Run | Best val/mIoU | Best Checkpoint |
| --- | --- | --- |
| run01 | 0.4806 | `mid_fusion-epoch=27-val_mIoU=0.4806.ckpt` |
| run02 | 0.4834 | `mid_fusion-epoch=23-val_mIoU=0.4834.ckpt` |
| run03 | 0.4885 | `mid_fusion-epoch=25-val_mIoU=0.4885.ckpt` |
| run04 | 0.4809 | `mid_fusion-epoch=26-val_mIoU=0.4809.ckpt` |
| run05 | 0.4814 | `mid_fusion-epoch=18-val_mIoU=0.4814.ckpt` |
| run06 | 0.4902 | `mid_fusion-epoch=23-val_mIoU=0.4902.ckpt` |

---

## Reduced Version Statistics

- mean val/mIoU: `0.4842`
- median val/mIoU: `0.4824`
- best val/mIoU: `0.4902`
- worst val/mIoU: `0.4806`
- standard deviation: `0.0038`
- range: `0.0096`

---

## Comparison with v1.0 Stable Baseline

The `v1.0` stable baseline 10-run summary is:

- mean val/mIoU: `0.4828`
- best val/mIoU: `0.4900`
- standard deviation: `0.0044`

Comparison:

- MCADS reduced mean: `0.4842`
- baseline mean: `0.4828`
- difference in mean: `+0.0014`

- MCADS reduced best: `0.4902`
- baseline best: `0.4900`
- difference in best: `+0.0002`

Interpretation:

1. The original MCADS block is too heavy when directly transplanted to the final
   high-resolution decoder stage.
2. After reducing attention complexity, the MCADS-style head becomes trainable.
3. The reduced MCADS head is not dramatically better than the stable baseline,
   but its average result is slightly higher.
4. The gain is small, so this should be viewed as a marginal improvement rather
   than a decisive breakthrough.

---

## Overall Conclusion

Across all 9 attempts:

- the first 3 runs show that direct transplantation of the original MCADS core
  is not feasible under the current setup
- the later 6 reduced-complexity runs show that an adapted MCADS-style decoder
  head can work and may provide a small improvement over the stable baseline

At the moment, the safest summary is:

> The original MCADS core decoder block cannot be directly used at the current
> FPN output resolution because of memory cost. After reducing the attention
> complexity, the MCADS-style decoder head becomes trainable and achieves an
> average mIoU of `48.42%`, slightly above the stable baseline average of
> `48.28%`. However, the improvement is small, so this variant should be treated
> as a promising but not yet decisively superior decoder-side modification.

---

## Suggested Reporting Wording

You can report it like this:

> I tested an MCADS-inspired decoder-side enhancement while keeping the encoder
> and fusion backbone unchanged. The original core block was too memory-intensive
> to run directly at the final FPN stage. After reducing the attention
> complexity, the adapted MCADS-style decoder head became trainable. Across 6
> valid runs, it achieved an average validation mIoU of `48.42%`, compared with
> `48.28%` for the stable v1.0 baseline. This suggests that decoder-side
> refinement may still have some room for improvement, but the current gain is
> modest.
