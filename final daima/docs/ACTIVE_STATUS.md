# Active Status

## Active Models

### dformerv2_mid_fusion

- Status: clean main baseline.
- Result: clean 10-run mean best mIoU `0.517397`, std `0.004901`, best single `0.524425`.
- Claim: main comparison baseline.

### dformerv2_tgga_c4only_beta002_aux003_detachsem_simplefpn_v1

- Status: active only as the c4-safe TGGA unit needed for the controlled R014 PMAD+TGGA experiment.
- Result: R004 c4-only best val/mIoU `0.522849` at epoch 42, last `0.509320`.
- Claim: promising diagnostic signal, not a stable improvement and not a goal-completing method.

### dformerv2_geometry_primary_teacher

- Status: active because PMAD/R014 depends on it.
- Result: teacher checkpoint `dformerv2_geometry_primary_teacher-epoch=37-val_mIoU=0.5168.pt`.
- Claim: teacher dependency, not a standalone paper improvement.

### dformerv2_primkd_logit_only

- Status: active PMAD logit-only branch.
- Result: R010 best `0.527469`, last `0.526316`; R012 repeat best `0.516967`, last `0.508205`.
- Claim: strongest partial-positive branch so far, but high variance and still below `0.53`.

## Archived / Deprecated

- TGGA c3/c4 and weak-c3 variants: archived after R004/R005 because c3 remained unsafe.
- R013 LMLP decoder: archived after best `0.517981`, last `0.490231`.
- R011 Lovasz, hard-pixel losses, boundary/confidence KD, correct/entropy KD, feature hint, global FreqFPN, LightHam/CMX, FFT/HiLo/depth FFT, DGBF, CGPC, SGBR-Lite, CGCD/ClassContext, context decoder/PPM, and CE+Dice: negative, unstable, or not part of the active mainline.
- Archived code snapshots live under `feiqi/`; recorded evidence remains in `docs/`, `miou_list/`, `reports/`, `metrics/`, and `experiments/`.

## Registry Policy

- Default `train.py` registry exposes only active models and small legacy teaching baselines.
- Failed experimental code must not be imported by `train.py` unless a future reproduction branch explicitly restores it.
- Do not cite archived modules as active baselines.
- Do not claim a method reaches the goal without a full-train TensorBoard/log/checkpoint-backed best val/mIoU.
