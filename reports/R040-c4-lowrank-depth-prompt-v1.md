# R040 C4 Low-Rank Depth Prompt V1

## Setup

- Branch: `exp/R040-c4-lowrank-depth-prompt-v1`
- Model: `dformerv2_c4_lowrank_depth_prompt`
- Run: `R040_c4_lowrank_depth_prompt_run01`
- Hypothesis: MixPrompt-style low-rank c4 depth prompt can condition the DFormerv2 c4 primary feature before the original `GatedFusion`, testing whether depth is more useful as fusion-before prompt conditioning than as output-side residual correction.
- Paper/code basis: MixPrompt (`NeurIPS 2025`, official repo `github.com/xiaoshideta/MixPrompt`), especially the `ADA` low-rank prompt mixing idea. The implementation ports only the minimal c4 prompt idea, not the full framework.

## Evidence

- TensorBoard event: `final daima/checkpoints/R040_c4_lowrank_depth_prompt_run01/lightning_logs/version_0/events.out.tfevents.1778875233.Administrator.19400.0`
- Best checkpoint: `final daima/checkpoints/R040_c4_lowrank_depth_prompt_run01/dformerv2_c4_lowrank_depth_prompt-epoch=36-val_mIoU=0.5279.pt`
- Saved launch command: `final daima/checkpoints/R040_c4_lowrank_depth_prompt_run01/run_r040.cmd`
- Exit code: `0`
- mIoU detail: `final daima/miou_list/R040_c4_lowrank_depth_prompt_run01.md`

## Result

- Validation epochs: `50`
- Best val/mIoU: `0.527946` at validation epoch `37`
- Last val/mIoU: `0.524679`
- Last-5 mean val/mIoU: `0.509687`
- Last-10 mean val/mIoU: `0.508256`
- Best-to-last drop: `0.003267`
- Best val/loss: `0.950243` at validation epoch `9`
- Last val/loss: `1.172903`
- Final train/loss_epoch: `0.054379`

## Diagnostics

- C4 prompt_abs first/last: `0.030488` / `0.014483`
- C4 prompt_raw_abs first/last: `0.156883` / `0.013475`
- C4 prompt_gate_mean first/last: `0.499766` / `0.501595`
- C4 prompt_gate_std first/last: `0.207389` / `0.206496`

## Decision

R040 is negative below the corrected R016 baseline. The low-rank c4 prompt does not show R039-style gate explosion and recovers at the final epoch, but last-5/last-10 means remain low. The best `0.527946` remains below the stage threshold `0.53`, far below R036 `0.539790`, and far below R016 `0.541121`.

Do not tune prompt rank/down-ratio/c4 scale. Archive the code under `feiqi` and pivot to a distinct hypothesis, with DiffPixelFormer-style c4 differential cue as the next highest-value candidate.
