# dformerv2_sagate_fusion Summary

| run | best epoch | best val/mIoU | last val/mIoU | best val/loss epoch | best val/loss | epochs |
|---|---:|---:|---:|---:|---:|---:|
| dformerv2_sagate_fusion_run01 | 50 | 0.522717 | 0.522717 | 10 | 0.999145 | 50 |
| dformerv2_sagate_fusion_run02 | 40 | 0.520692 | 0.510791 | 13 | 1.027782 | 50 |
| dformerv2_sagate_fusion_run03 | 39 | 0.507363 | 0.499340 | 9 | 1.034039 | 50 |
| dformerv2_sagate_fusion_run04 | 45 | 0.510646 | 0.500069 | 9 | 1.032801 | 50 |
| dformerv2_sagate_fusion_run05 | 29 | 0.504661 | 0.495424 | 12 | 1.040354 | 50 |

- mean best val/mIoU: 0.513216
- population std best val/mIoU: 0.007214
- sample std best val/mIoU: 0.008066
- baseline dformerv2_mid_fusion repeat mean best val/mIoU: 0.513406
- mean delta vs baseline: -0.000190

## Conclusion

SA-Gate-style one-way fusion is a valid stronger candidate than the repeated DFormerv2 mid-fusion baseline in these five runs. It also keeps the branch lighter than original GatedFusion. Use it as a candidate result, but still report it with five-run mean/std rather than a single best run.
