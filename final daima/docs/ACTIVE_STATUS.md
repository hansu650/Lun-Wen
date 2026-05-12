# Active Status

## Active Models

### dformerv2_mid_fusion

- Status: clean main baseline.
- Result: clean 10-run mean best mIoU `0.517397`, std `0.004901`, best single `0.524425`.
- Claim: main comparison baseline.

### dformerv2_tgga_c34_beta002_aux003_detachsem_simplefpn_v2

- Status: active / pending repeat.
- Run01 result: best val/mIoU `0.522206` at epoch 48; last val/mIoU `0.489865`.
- Interpretation: promising but unstable single-run signal.
- Claim: cannot claim stable improvement until run02/run03.

### dformerv2_geometry_primary_teacher

- Status: active because PMAD depends on it.
- Claim: teacher baseline, not a strong standalone result.

### dformerv2_primkd_logit_only

- Status: active.
- Result: PMAD logit-only w0.15 T4 five-run mean around `0.520795`.
- Claim: marginal positive repeat signal, not strong improvement.

## Archived / Deprecated

- DGBF: negative.
- CGPC: negative; loss decreases but mIoU does not improve.
- SGBR-Lite: negative.
- CGCD / ClassContext: seed-sensitive, five-run mean below baseline.
- FFT / HiLo / depth FFT: unstable or negative.
- Context decoder / PPM: archived.
- CE+Dice / DGBF loss recipe: not active.

## Registry Policy

- Default `train.py` registry exposes only active models and small legacy teaching baselines.
- Archived modules live under `feiqi/` or git history and are not imported by `train.py`.
- Do not cite archived modules as active baselines.
- Do not claim TGGA is stable until repeat runs are complete.
