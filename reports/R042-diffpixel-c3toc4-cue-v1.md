# R042 DiffPixel C3-to-C4 Cue V1

## Setup

- Branch: `exp/R042-diffpixel-c3toc4-cue-v1`
- Model: `dformerv2_diffpixel_c3toc4_cue`
- Run: `R042_diffpixel_c3toc4_cue_run01`
- Hypothesis: a c3 RGB-depth differential cue propagated to c4 can condition high-level `GatedFusion` better than R041's c4-only local difference, without changing c3 outputs or adding output residuals.
- Paper/code basis: DiffPixelFormer (`arXiv:2511.13047`, official repo `github.com/gongyan1/DiffPixelFormer`) motivates differential RGB-D cueing. R042 ports only the cross-stage c3-to-c4 cue idea, not the full framework.

## Evidence

- TensorBoard event: `final daima/checkpoints/R042_diffpixel_c3toc4_cue_run01/lightning_logs/version_0/events.out.tfevents.1778887449.Administrator.10152.0`
- Best checkpoint: `final daima/checkpoints/R042_diffpixel_c3toc4_cue_run01/dformerv2_diffpixel_c3toc4_cue-epoch=42-val_mIoU=0.5307.pt`
- Saved launch command: `final daima/checkpoints/R042_diffpixel_c3toc4_cue_run01/run_r042.cmd`
- Exit code: `0`
- mIoU detail: `final daima/miou_list/R042_diffpixel_c3toc4_cue_run01.md`

## Result

- Validation epochs: `50`
- Best val/mIoU: `0.530729` at validation epoch `43`
- Last val/mIoU: `0.458179`
- Last-5 mean val/mIoU: `0.505197`
- Last-10 mean val/mIoU: `0.510157`
- Best-to-last drop: `0.072551`
- Best val/loss: `0.957006` at validation epoch `10`
- Last val/loss: `1.417108`
- Final train/loss_epoch: `0.065262`

## Diagnostics

- c3-to-c4 cue_abs first/last: `0.018134` / `0.246822`
- c3-to-c4 diff_context_abs first/last: `0.765808` / `0.773092`
- c3-to-c4 gate_mean first/last: `0.500000` / `0.545177`
- c3-to-c4 gate_std first/last: `0.208145` / `0.215891`

## Decision

R042 is negative. It is below R041 `0.537098`, R036 `0.539790`, and R016 `0.541121`, and the final epoch collapses to `0.458179`. The c3 cue branch opens, but propagating c3 disagreement into c4 appears to amplify mid-level noise rather than stabilize high-level fusion.

Do not promote this exact module to active mainline. Archive the implementation under `feiqi`. The next experiment should avoid c3-propagated differential cues and pivot to a different hypothesis, preferably explicit geometry/normal-like cues or decoder/refinement stability.
