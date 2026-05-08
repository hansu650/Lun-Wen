# dformerv2_ms_freqcov aggressive sweep summary

- included runs: `7`
- complete 50-epoch runs: `7`
- clean baseline mean best val/mIoU: `0.517397`
- clean baseline population std: `0.004901`
- clean baseline best single run: `0.524425`
- sweep mean best val/mIoU: `0.515697`
- sweep population std best val/mIoU: `0.005143`
- best sweep run: `dformerv2_ms_freqcov_run01` with `0.520539` at epoch `50`

| rank | run | lambda_freq | weights | epochs | best val/mIoU | best epoch | last val/mIoU | delta vs baseline mean | train freq first -> last |
|---:|---|---:|---|---:|---:|---:|---:|---:|---|
| 1 | `dformerv2_ms_freqcov_run01` | 0.01 | `1,1,1,1` | 50 | 0.520539 | 50 | 0.520539 | +0.003142 | 0.00137723 -> 1.99331e-06 |
| 2 | `dformerv2_ms_freqcov_stage234_lam1_run01` | 1.0 | `0,0.5,1,2` | 50 | 0.520060 | 40 | 0.505907 | +0.002663 | 0.00206328 -> 2.01261e-07 |
| 3 | `dformerv2_ms_freqcov_lam1_run01` | 1.0 | `1,1,1,1` | 50 | 0.518769 | 45 | 0.512625 | +0.001372 | 0.00125115 -> 1.08741e-07 |
| 4 | `dformerv2_ms_freqcov_lam01_run01` | 0.1 | `1,1,1,1` | 50 | 0.516227 | 40 | 0.499021 | -0.001170 | 0.00137805 -> 4.62699e-07 |
| 5 | `dformerv2_ms_freqcov_stage3412_run01` | 0.1 | `0.5,1,1,2` | 50 | 0.515543 | 44 | 0.496894 | -0.001854 | 0.00195182 -> 5.3826e-07 |
| 6 | `dformerv2_ms_freqcov_stage3412_lam1_run01` | 1.0 | `0.5,1,1,2` | 50 | 0.514508 | 50 | 0.514508 | -0.002889 | 0.00167554 -> 1.62117e-07 |
| 7 | `dformerv2_ms_freqcov_stage3412_lam2_run01` | 2.0 | `0.5,1,1,2` | 50 | 0.504229 | 44 | 0.502087 | -0.013168 | 0.00159147 -> 1.48021e-07 |

## Conclusion

All listed freqcov runs completed 50 validation epochs. The default weak run remains the best freqcov setting in this sweep, while the most aggressive lambda 2.0 setting is clearly negative. The sweep does not yet establish a stable improvement over the clean 10-run GatedFusion baseline because no setting exceeds the baseline's best single run and only two single runs are above the baseline mean.
