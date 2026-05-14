# Failed Experiment Code Archive R001-R013

This archive keeps code snapshots that should not stay in the active training registry after the R013 pause.

## Archived Files

- `tgga_adapter_c3_variants_pre_cleanup.py`: pre-cleanup TGGA implementation containing c3/c4, no-aux, c4-only, and weak-c3 variants.
- `train_registry_pre_cleanup.py`: pre-cleanup active registry snapshot.
- `depth_fft_select.py`: inactive depth FFT selection module.
- `freq_enhance.py`: inactive FFT frequency enhancement module.
- `fft_hilo_enhance.py`: inactive FFT HiLo enhancement module.

## Active Replacement

The active code keeps only:

- `dformerv2_mid_fusion`
- `dformerv2_tgga_c4only_beta002_aux003_detachsem_simplefpn_v1`
- `dformerv2_geometry_primary_teacher`
- `dformerv2_primkd_logit_only`

R014 should reuse the active PMAD logit-only and TGGA c4-only pieces. Do not restore the archived c3/weak-c3 or global frequency modules unless a future reproduction branch explicitly asks for them.
