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
- 2026-05-13 orchestration-loop failed variants are archived under `experiments_20260513/`:
  - R001 PMAD boundary/confidence KD, best val/mIoU `0.511646`.
  - R002 frequency-aware FPN decoder, best val/mIoU `0.516915`.
  - R003 correct-and-entropy PMAD KD, best val/mIoU `0.516597`.
  - R005 TGGA weak-c3 + c4, best val/mIoU `0.518253`.
- 2026-05-14 mainline cleanup snapshots are archived under `failed_experiments_r001_r013_20260514/`:
  - TGGA c3/c4, no-aux, and weak-c3 implementation snapshot.
  - pre-cleanup `train.py` registry snapshot.
  - depth FFT select, FFT frequency enhance, and FFT HiLo modules.
  - R013 LMLP remains a documented negative experiment and is not part of the active registry.

Current active models:
- `dformerv2_mid_fusion`
- `dformerv2_tgga_c4only_beta002_aux003_detachsem_simplefpn_v1`
- `dformerv2_geometry_primary_teacher`
- `dformerv2_primkd_logit_only`

TGGA status:
- original c3/c4 run01 best val/mIoU `0.522206` at epoch 48, final `0.489865`.
- R004 c4-only best val/mIoU `0.522849` at epoch 42, final `0.509320`.
- R005 weak-c3 + c4 best val/mIoU `0.518253` at epoch 43, final `0.514908`.
- R004 is the strongest current diagnostic signal, but no TGGA variant reaches the `0.53` goal or supports a stable improvement claim.

Implementation note:
- The default `train.py` registry no longer imports archived experimental models, including TGGA c3/weak-c3 variants.
- Archived decoder blocks moved to `feiqi/models/archived_decoders.py`.
- Some archived loss implementations remain in `src/losses/` for historical compatibility, but default `train.py` exposes only `--loss_type ce`.
