# R039 MIIM C4 Lite V1

## Setup

- Branch: `exp/R039-miim-c4-lite-v1`
- Model: `dformerv2_miim_c4_lite`
- Run: `R039_miim_c4_lite_run01`
- Hypothesis: HDBFormer/MIIM-style global-local c4 interaction can improve high-level RGB-depth fusion over R038 sparse sampling while preserving c1-c3 `GatedFusion`, `SimpleFPNDecoder`, CE loss, and the fixed recipe.
- Paper/code basis: HDBFormer (`arXiv:2504.13579`, official repo `github.com/Weishuobin/HDBFormer`), especially `models/MIIM.py`.

## Evidence

- TensorBoard event: `final daima/checkpoints/R039_miim_c4_lite_run01/lightning_logs/version_0/events.out.tfevents.1778869284.Administrator.3304.0`
- Best checkpoint: `final daima/checkpoints/R039_miim_c4_lite_run01/dformerv2_miim_c4_lite-epoch=40-val_mIoU=0.5341.pt`
- Saved launch command: `final daima/checkpoints/R039_miim_c4_lite_run01/run_r039.cmd`
- Exit code: `0`
- mIoU detail: `final daima/miou_list/R039_miim_c4_lite_run01.md`

## Result

- Validation epochs: `50`
- Best val/mIoU: `0.534131` at validation epoch `41`
- Last val/mIoU: `0.509767`
- Last-5 mean val/mIoU: `0.511286`
- Last-10 mean val/mIoU: `0.513125`
- Best-to-last drop: `0.024364`
- Best val/loss: `0.960918` at validation epoch `9`
- Last val/loss: `1.160239`
- Final train/loss_epoch: `0.126658`

## Diagnostics

- MIIM c4 alpha first/last: `0.002511` / `0.003592`
- MIIM c4 gate_mean first/last: `0.507850` / `0.895789`
- MIIM c4 gate_std first/last: `0.094741` / `0.298645`
- MIIM c4 update_abs first/last: `0.303729` / `0.392390`

## Decision

R039 is negative below the corrected R016 baseline. The MIIM-lite branch is active and crosses `0.53`, but the best `0.534131` is below R016 `0.541121` and R036 `0.539790`, and the late drop `0.024364` is too large.

Do not tune MIIM alpha/channel as a micro-search. Archive the code under `feiqi` and pivot to a distinct hypothesis such as low-rank prompt-style depth guidance or differential cue modeling.
