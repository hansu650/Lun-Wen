# feiqi archived experimental modules

This folder stores failed, unstable, or deprecated experimental modules.

These modules are not part of the default training path and must not be imported by `train.py` unless explicitly reproducing old ablations.

Archived after cleanup:
- DGBF: negative.
- CGPC: negative; loss decreases but mIoU does not improve.
- SGBR-Lite: negative; run01 best around `0.510159`.
- CGCD / ClassContextFPNDecoder: seed-sensitive; five-run mean below the clean baseline.
- Context decoder / PPM: not the current main line.
- FFT freq enhance / FFT HiLo: unstable.
- Depth FFT select: negative.
- CE+Dice / DGBF loss recipe: not active.

Current active models:
- `dformerv2_mid_fusion`
- `dformerv2_tgga_c34_beta002_aux003_detachsem_simplefpn_v2`
- `dformerv2_geometry_primary_teacher`
- `dformerv2_primkd_logit_only`

TGGA status:
- run01 best val/mIoU `0.522206` at epoch 48.
- run01 final val/mIoU `0.489865`.
- promising but unstable single-run signal.
- not yet a stable improvement claim.

Implementation note:
- The default `train.py` registry no longer imports archived experimental models.
- Archived decoder blocks moved to `feiqi/models/archived_decoders.py`.
- Some archived loss implementations remain in `src/losses/` for historical compatibility, but default `train.py` exposes only `--loss_type ce`.
