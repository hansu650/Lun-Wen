# Cleanup Notes

## 2026-05-12 cleanup/archive-failed-modules

This cleanup narrows the active training path without changing any recorded experimental result.

## 2026-05-14 cleanup/nyu056-mainline

This cleanup keeps only the useful R014-ready paths in the active registry before the next full train.

## What Was Moved Or Archived

- `src/models/depth_fft_select.py`, `src/models/freq_enhance.py`, and `src/models/fft_hilo_enhance.py` were moved to `feiqi/failed_experiments_r001_r013_20260514/`.
- The pre-cleanup TGGA c3/c4 and weak-c3 implementation snapshot was archived as `feiqi/failed_experiments_r001_r013_20260514/tgga_adapter_c3_variants_pre_cleanup.py`.
- The pre-cleanup registry snapshot was archived as `feiqi/failed_experiments_r001_r013_20260514/train_registry_pre_cleanup.py`.

## Active Registry After This Cleanup

- `dformerv2_mid_fusion`
- `dformerv2_tgga_c4only_beta002_aux003_detachsem_simplefpn_v1`
- `dformerv2_geometry_primary_teacher`
- `dformerv2_primkd_logit_only`
- legacy `early` and `mid_fusion`

## Disconnected From Active Training

- TGGA c3/c4 and weak-c3 variants
- R013 LMLP decoder
- R011 Lovasz and other hard-loss directions
- PMAD hard filters and feature-hint variants
- global frequency/FFT/HiLo/depth FFT paths

No checkpoint, mIoU record, TensorBoard log, report, metrics row, or experiment summary was deleted.

## What Was Fully Moved

- TGGA active code was moved out of `src/models/mid_fusion.py` into `src/models/tgga_adapter.py`.
- `src/models/mid_fusion.py` now keeps the clean baseline and legacy ResNet mid-fusion models only.
- Archived decoder blocks were moved out of `src/models/decoder.py` into `feiqi/models/archived_decoders.py`.
- `src/models/decoder.py` now contains only `SimpleFPNDecoder` for the active training path.

## What Was Disconnected From Default Training

The following branches are no longer imported by default `train.py` and no longer appear in `--model` choices:

- `dformerv2_sgbr_decoder`
- `dformerv2_class_context_decoder`
- `dformerv2_context_decoder`
- `dformerv2_depth_fft_select`
- `dformerv2_fft_freq_enhance`
- `dformerv2_fft_hilo_enhance`

The following CLI options are no longer exposed by default:

- DGBF options
- CGPC options
- SGBR options
- class-context options
- FFT / HiLo options

## Why Some Files Remain In Place

- `src/losses/dgbf_loss.py` and `src/losses/cgpc_loss.py` remain in the source tree because `BaseLitSeg` still has historical support code. Default `train.py` exposes only `--loss_type ce`, so these losses are not reachable from active training unless a future reproduction path is intentionally restored.
- This conservative choice avoids accidentally changing checkpoint compatibility, historical imports, or archived experiment reproducibility during the active registry cleanup.

## No Result Changed

No checkpoint, mIoU record, TensorBoard log, or experiment summary was deleted or rewritten as a new result.
