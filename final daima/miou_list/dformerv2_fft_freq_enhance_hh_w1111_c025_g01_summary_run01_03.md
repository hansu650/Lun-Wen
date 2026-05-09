# dformerv2_fft_freq_enhance_hh_w1111_c025_g01 3-run summary

## Settings

- model: `dformerv2_fft_freq_enhance`
- cutoff_ratio: 0.25, gamma_init: 0.1
- All other settings identical to clean baseline

## Results

| Run | Best val/mIoU | Best Epoch | Last val/mIoU | vs Baseline Mean |
|-----|--------------|------------|---------------|-----------------|
| run01 | 0.522688 | 41 | 0.514651 | +0.005291 |
| run02 | 0.5159 | 38 | 0.443 | -0.001497 |
| run03 | 0.5145 | 42 | 0.489 | -0.002897 |

## Statistics

- 3-run mean best val/mIoU: **0.517696**
- 3-run population std: 0.003664
- 3-run best single: 0.522688 (run01)
- 3-run worst single: 0.5145 (run03)

## Baseline Comparison

- clean 10-run baseline mean best: 0.517397
- 3-run mean delta vs baseline mean: **+0.000299**
- clean 10-run baseline population std: 0.004901
- 3-run mean in baseline std units: **+0.061**
- clean 10-run baseline best single: 0.524425

## Interpretation

The 3-run mean 0.517696 is essentially identical to the clean baseline mean 0.517397 (delta +0.000299, only 0.06 std). The first run's 0.522688 was a high-variance outlier — run02 and run03 both fell below the baseline mean. The direction cannot be claimed as an improvement.

## Decision

FFT freq_enhance (cutoff=0.25, gamma=0.1) is **not a stable improvement** over the GatedFusion baseline. The initial positive signal was statistical noise. This direction should be deprioritized.
