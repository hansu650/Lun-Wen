# Cleanup Notes

## 2026-05-12 cleanup/archive-failed-modules

This cleanup narrows the active training path without changing any recorded experimental result.

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
