# 2026-05-13 Orchestration Loop Archive

This folder archives code snapshots from the R001-R005 goal-driven loop after pausing the experiment search.

These files are not imported by the active training path. They are kept only for reproducing or inspecting failed/diagnostic variants.

## Archived Snapshots

- `train_registry_r001_r005_before_cleanup.py`: registry snapshot before cleanup, including R001-R003 failed model entries and R004/R005 diagnostic entries.
- `decoder_with_r002_freqfpn.py`: decoder snapshot containing the R002 `FrequencyAwareFPNDecoder` implementation.
- `mid_fusion_with_r002_freqfpn.py`: mid-fusion snapshot containing the R002 `DFormerV2FreqFPNDecoderSegmentor` / Lightning wrapper.
- `primkd_failed_variants_r001_r003.py`: PMAD snapshot containing R001 boundary/confidence KD and R003 correct-and-entropy KD variants.

## Result Boundary

- R001 PMAD boundary/confidence KD: best val/mIoU `0.511646`, negative.
- R002 frequency-aware FPN decoder: best val/mIoU `0.516915`, negative/neutral.
- R003 correct-and-entropy PMAD KD: best val/mIoU `0.516597`, near-baseline but negative.
- R004 TGGA c4-only: best val/mIoU `0.522849`, strongest loop signal but below `0.53`.
- R005 TGGA weak-c3 + c4: best val/mIoU `0.518253`, worse than R004.

Active code should keep the clean baseline, TGGA diagnostic variants already present on `main`, geometry-primary teacher, and PMAD logit-only. Do not re-import these archived variants without a new explicit experiment decision.
