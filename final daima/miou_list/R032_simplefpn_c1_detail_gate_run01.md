# R032 SimpleFPN C1 Detail Gate Run01

- branch: `exp/R032-simplefpn-c1-detail-gate-v1`
- model: `dformerv2_simplefpn_c1_detail_gate`
- run: `R032_simplefpn_c1_detail_gate_run01`
- checkpoint_dir: `final daima/checkpoints/R032_simplefpn_c1_detail_gate_run01`
- TensorBoard event: `final daima/checkpoints/R032_simplefpn_c1_detail_gate_run01/lightning_logs/version_0/events.out.tfevents.1778824352.Administrator.34648.0`
- best checkpoint: `final daima/checkpoints/R032_simplefpn_c1_detail_gate_run01/dformerv2_simplefpn_c1_detail_gate-epoch=49-val_mIoU=0.5366.pt`
- status: full train completed with exit code `0`; `Trainer.fit` reached `max_epochs=50`

## Summary

| Metric | Value |
|---|---:|
| recorded validation epochs | 50 |
| best val/mIoU | 0.536603 |
| best validation epoch | 50 |
| last val/mIoU | 0.536603 |
| last-5 mean val/mIoU | 0.505390 |
| last-10 mean val/mIoU | 0.509657 |
| best-to-last drop | 0.000000 |
| best val/loss | 0.965559 |
| best val/loss epoch | 9 |
| final train/loss_epoch | 0.061732 |
| c1 detail alpha first | 0.998994 |
| c1 detail alpha last | 0.998770 |

## Per-Epoch Validation Metrics

| Validation epoch | val/mIoU | val/loss | c1 detail alpha |
|---:|---:|---:|---:|
| 1 | 0.178511 | 1.600230 | 0.998994 |
| 2 | 0.251570 | 1.337185 | 0.998973 |
| 3 | 0.311508 | 1.207198 | 0.998963 |
| 4 | 0.363638 | 1.090304 | 0.998942 |
| 5 | 0.391861 | 1.067994 | 0.998933 |
| 6 | 0.434786 | 1.003489 | 0.998924 |
| 7 | 0.448725 | 0.999765 | 0.998911 |
| 8 | 0.474851 | 0.974144 | 0.998902 |
| 9 | 0.478323 | 0.965559 | 0.998897 |
| 10 | 0.472439 | 1.001912 | 0.998881 |
| 11 | 0.486315 | 0.988076 | 0.998879 |
| 12 | 0.487226 | 1.009106 | 0.998873 |
| 13 | 0.493911 | 1.024117 | 0.998871 |
| 14 | 0.492306 | 1.029309 | 0.998865 |
| 15 | 0.485740 | 1.041062 | 0.998861 |
| 16 | 0.452723 | 1.158310 | 0.998850 |
| 17 | 0.500742 | 1.043802 | 0.998848 |
| 18 | 0.512207 | 1.017896 | 0.998842 |
| 19 | 0.513733 | 1.048882 | 0.998841 |
| 20 | 0.493081 | 1.086787 | 0.998840 |
| 21 | 0.504267 | 1.078881 | 0.998838 |
| 22 | 0.486277 | 1.086711 | 0.998833 |
| 23 | 0.508965 | 1.050002 | 0.998832 |
| 24 | 0.514938 | 1.093350 | 0.998820 |
| 25 | 0.508978 | 1.065353 | 0.998819 |
| 26 | 0.524018 | 1.072522 | 0.998818 |
| 27 | 0.521159 | 1.095132 | 0.998813 |
| 28 | 0.510451 | 1.143899 | 0.998812 |
| 29 | 0.527563 | 1.095319 | 0.998811 |
| 30 | 0.488065 | 1.187554 | 0.998810 |
| 31 | 0.494207 | 1.155741 | 0.998810 |
| 32 | 0.524148 | 1.105985 | 0.998808 |
| 33 | 0.525168 | 1.098304 | 0.998807 |
| 34 | 0.516354 | 1.195273 | 0.998802 |
| 35 | 0.513749 | 1.157081 | 0.998802 |
| 36 | 0.505077 | 1.186371 | 0.998794 |
| 37 | 0.511959 | 1.183784 | 0.998789 |
| 38 | 0.517887 | 1.133614 | 0.998789 |
| 39 | 0.519907 | 1.148321 | 0.998787 |
| 40 | 0.527334 | 1.155404 | 0.998785 |
| 41 | 0.493720 | 1.263661 | 0.998781 |
| 42 | 0.523386 | 1.169918 | 0.998780 |
| 43 | 0.521770 | 1.197670 | 0.998779 |
| 44 | 0.527872 | 1.212313 | 0.998779 |
| 45 | 0.502879 | 1.264741 | 0.998779 |
| 46 | 0.493916 | 1.248193 | 0.998777 |
| 47 | 0.488836 | 1.372786 | 0.998776 |
| 48 | 0.484702 | 1.324080 | 0.998772 |
| 49 | 0.522892 | 1.216150 | 0.998771 |
| 50 | 0.536603 | 1.195689 | 0.998770 |

## Interpretation

R032 crosses `0.53` and ends at its best epoch, but remains below R016 `0.541121`. The c1 detail alpha barely moves from `0.998994` to `0.998770`, so this exact baseline-near c1 gate provides little evidence that high-resolution c1 detail strength is the missing factor.
