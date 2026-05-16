# Active Status

## Current Mainline

- Active fixed-recipe baseline: `dformerv2_mid_fusion`.
- Corrected-label/reference result: R016 best val/mIoU `0.541121` at validation epoch `49`, last `0.527420`.
- Current stage: R048 has been recorded as a stable decoder-refinement diagnostic below R016/R036/R041; continue the Goal-Driven loop toward `0.56` with a distinct R049 hypothesis.
- Active registry should stay clean: only mainline/teaching entries and reusable stable entries belong in `train.py`.

## Active Model Entries

### `dformerv2_mid_fusion`

- Status: active mainline.
- Role: DFormerv2-S + DepthEncoder + four-stage `GatedFusion` + `SimpleFPNDecoder`.
- Best current evidence: R016 best `0.541121`, last `0.527420`.

### `dformerv2_ham_decoder`

- Status: retained as a stable comparison entry.
- Best current evidence: R022 best=last `0.534332`.
- Boundary: stable but below R016; do not present as the strongest model.

### `dformerv2_geometry_primary_ham_decoder`

- Status: retained as a geometry-primary/teacher-style comparison entry.
- Best current evidence: R024 best `0.530186`, last `0.529383`.
- Boundary: stable but low ceiling.

### `early` / `mid_fusion`

- Status: legacy teaching baselines retained by the training script.
- Boundary: not part of the current DFormerv2-S mainline claim.

## Archived / Deprecated

- R041 DiffPixel c4 cue: partial positive below R016; code archived under `feiqi/failed_experiments_r041_20260516/`.
- R042 DiffPixel c3-to-c4 cue: negative and unstable below R016; code archived under `feiqi/failed_experiments_r042_20260516/`.
- R043 DepthGeo c4 cue: partial geometry-cue signal below R016; code archived under `feiqi/failed_experiments_r043_20260516/`.
- R044 conditioned c3/c4 bounded residual: negative below R016; code archived under `feiqi/failed_experiments_r044_20260516/`.
- R045 c3/c4 zero-init modality adapter: negative below R016; code archived under `feiqi/failed_experiments_r045_20260516/`.
- R046 DGFusion c4 depth-token: negative below R016; code archived under `feiqi/failed_experiments_r046_20260516/`.
- R047 GatedFusion local GroupNorm: negative and unstable below R016; code archived under `feiqi/failed_experiments_r047_20260516/`.
- R048 refined FPN decoder: stable but below R016/R036/R041; code archived under `feiqi/failed_experiments_r048_20260516/`.
- R040 c4 low-rank depth prompt: negative below `0.53`; code archived under `feiqi/failed_experiments_r040_20260516/`.
- R039 MIIM c4 lite, R038 DSCF c4 lite, R037 DGL minimal, R036 bounded residual, R034 MASG, and R035 gate-balance regularizer: not promoted to active registry.
- PMAD/TGGA small-parameter search, hard filtering KD, global FreqFPN/FFT, Lovasz/OHEM/Focal/CE+Dice, LMLP decoder, SimpleFPN classifier dropout, Ham-logit scalar tuning, all-stage residual, c1/c2 residual, official local init, and DepthEncoder BN eval remain deprecated unless a future branch restores them for reproduction only.

## Registry Policy

- Failed or partial experimental code must be archived under `feiqi/` and removed from active `train.py` registry after recording evidence.
- Evidence records stay in `docs/`, `miou_list/`, `reports/`, `metrics/`, and `experiments/`.
- Do not cite archived modules as active baselines.
- Do not claim a method reaches the goal without full-train TensorBoard/log/checkpoint-backed best val/mIoU.
