# R041 DiffPixel C4 Cue V1

## Setup

- Branch: `exp/R041-diffpixel-c4-cue-v1`
- Model: `dformerv2_diffpixel_c4_cue`
- Run: `R041_diffpixel_c4_cue_run01`
- Hypothesis: DiffPixelFormer-style RGB-depth differential cue can improve c4 gate decisions by explicitly modeling local modality disagreement before fusion, without adding an output residual.
- Paper/code basis: DiffPixelFormer (`arXiv:2511.13047`, official repo `github.com/gongyan1/DiffPixelFormer`), especially the differential cue path that uses `x0 - x1` to condition cross-modal interaction. R041 ports only the minimal c4 gate-logit correction idea.

## Evidence

- TensorBoard event: `final daima/checkpoints/R041_diffpixel_c4_cue_run01/lightning_logs/version_0/events.out.tfevents.1778881132.Administrator.43760.0`
- Best checkpoint: `final daima/checkpoints/R041_diffpixel_c4_cue_run01/dformerv2_diffpixel_c4_cue-epoch=43-val_mIoU=0.5371.pt`
- Saved launch command: `final daima/checkpoints/R041_diffpixel_c4_cue_run01/run_r041.cmd`
- Exit code: `0`
- mIoU detail: `final daima/miou_list/R041_diffpixel_c4_cue_run01.md`

## Result

- Validation epochs: `50`
- Best val/mIoU: `0.537098` at validation epoch `44`
- Last val/mIoU: `0.529552`
- Last-5 mean val/mIoU: `0.507803`
- Last-10 mean val/mIoU: `0.516635`
- Best-to-last drop: `0.007546`
- Best val/loss: `0.974793` at validation epoch `12`
- Last val/loss: `1.197305`
- Final train/loss_epoch: `0.071763`

## Diagnostics

- DiffPixel c4 gate_mean first/last: `0.500861` / `0.544060`
- DiffPixel c4 gate_std first/last: `0.208216` / `0.221571`
- DiffPixel c4 diff_gate_abs first/last: `0.015047` / `0.253500`
- DiffPixel c4 diff_context_abs first/last: `1.039935` / `0.954306`

## Decision

R041 is a partial positive signal but not an improvement over the corrected baseline. It crosses `0.53` and beats R040/R038/R037, showing that explicit c4 differential cue modeling is more valuable than the c4 low-rank prompt and sparse sampler variants. However, best `0.537098` is below R036 `0.539790` and R016 `0.541121`, and the last-5/last-10 means show late-window instability.

Do not promote this exact c4-only module to active mainline. Archive the implementation under `feiqi`. The next decision should either strengthen the same differential-cue principle in a distinct way, or pivot back to a higher-capacity/stability design that can exceed R016.
