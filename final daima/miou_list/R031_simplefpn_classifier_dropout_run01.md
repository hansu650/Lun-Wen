# R031 SimpleFPN Classifier Dropout Run01

- branch: `exp/R031-simplefpn-classifier-dropout-v1`
- model: `dformerv2_simplefpn_classifier_dropout`
- run: `R031_simplefpn_classifier_dropout_run01`
- checkpoint_dir: `final daima/checkpoints/R031_simplefpn_classifier_dropout_run01`
- TensorBoard event: `final daima/checkpoints/R031_simplefpn_classifier_dropout_run01/lightning_logs/version_0/events.out.tfevents.1778819244.Administrator.21360.0`
- best checkpoint: `final daima/checkpoints/R031_simplefpn_classifier_dropout_run01/dformerv2_simplefpn_classifier_dropout-epoch=39-val_mIoU=0.5315.pt`
- status: full train completed with exit code `0`; `Trainer.fit` reached `max_epochs=50`

## Summary

| Metric | Value |
|---|---:|
| recorded validation epochs | 50 |
| best val/mIoU | 0.531544 |
| best validation epoch | 40 |
| last val/mIoU | 0.525760 |
| last-5 mean val/mIoU | 0.508009 |
| last-10 mean val/mIoU | 0.507366 |
| best-to-last drop | 0.005784 |
| best val/loss | 0.971063 |
| best val/loss epoch | 11 |
| final train/loss_epoch | 0.064540 |

## Per-Epoch Validation Metrics

| Validation epoch | val/mIoU | val/loss |
|---:|---:|---:|
| 1 | 0.173816 | 1.649754 |
| 2 | 0.242113 | 1.367635 |
| 3 | 0.292178 | 1.224437 |
| 4 | 0.336489 | 1.132183 |
| 5 | 0.395931 | 1.060706 |
| 6 | 0.431663 | 1.001865 |
| 7 | 0.460412 | 0.974322 |
| 8 | 0.453464 | 1.028878 |
| 9 | 0.466446 | 1.012664 |
| 10 | 0.481706 | 0.990962 |
| 11 | 0.497909 | 0.971063 |
| 12 | 0.493387 | 1.011814 |
| 13 | 0.492047 | 1.009419 |
| 14 | 0.484807 | 1.021253 |
| 15 | 0.493533 | 1.013823 |
| 16 | 0.470477 | 1.127752 |
| 17 | 0.502939 | 1.022474 |
| 18 | 0.504839 | 1.036227 |
| 19 | 0.498030 | 1.071569 |
| 20 | 0.489990 | 1.139179 |
| 21 | 0.509620 | 1.063160 |
| 22 | 0.521458 | 1.055420 |
| 23 | 0.516370 | 1.054452 |
| 24 | 0.489780 | 1.138501 |
| 25 | 0.473747 | 1.210893 |
| 26 | 0.509930 | 1.103840 |
| 27 | 0.511844 | 1.088995 |
| 28 | 0.519228 | 1.087418 |
| 29 | 0.515347 | 1.094275 |
| 30 | 0.525952 | 1.098570 |
| 31 | 0.513210 | 1.147056 |
| 32 | 0.458190 | 1.301150 |
| 33 | 0.504262 | 1.117482 |
| 34 | 0.502648 | 1.153035 |
| 35 | 0.516843 | 1.116089 |
| 36 | 0.522143 | 1.114385 |
| 37 | 0.523668 | 1.120650 |
| 38 | 0.529407 | 1.115359 |
| 39 | 0.520187 | 1.152185 |
| 40 | 0.531544 | 1.147586 |
| 41 | 0.519665 | 1.143313 |
| 42 | 0.514662 | 1.209832 |
| 43 | 0.497170 | 1.254511 |
| 44 | 0.489906 | 1.285179 |
| 45 | 0.512211 | 1.193591 |
| 46 | 0.516245 | 1.190351 |
| 47 | 0.470131 | 1.404186 |
| 48 | 0.511786 | 1.200930 |
| 49 | 0.516123 | 1.199249 |
| 50 | 0.525760 | 1.185384 |

## Interpretation

R031 tests a separate SimpleFPN classifier-dropout entry while keeping the R016 baseline model unchanged. The run crosses `0.53`, but the peak `0.531544` is below R016 `0.541121`, R027 `0.536739`, R030 `0.536454`, and R022 `0.534332`. SimpleFPN classifier dropout is therefore not the same useful parity fix that it was for the Ham decoder.
