# R033 SimpleFPN Ham Logit Fusion Run01

- branch: `exp/R033-simplefpn-ham-logit-fusion-v1`
- model: `dformerv2_simplefpn_ham_logit_fusion`
- run: `R033_simplefpn_ham_logit_fusion_run01`
- checkpoint_dir: `final daima/checkpoints/R033_simplefpn_ham_logit_fusion_run01`
- TensorBoard event: `final daima/checkpoints/R033_simplefpn_ham_logit_fusion_run01/lightning_logs/version_0/events.out.tfevents.1778829755.Administrator.10268.0`
- best checkpoint: `final daima/checkpoints/R033_simplefpn_ham_logit_fusion_run01/dformerv2_simplefpn_ham_logit_fusion-epoch=48-val_mIoU=0.5330.pt`
- exit code: `0`
- fixed recipe: batch_size `2`, max_epochs `50`, lr `6e-5`, num_workers `4`, early_stop_patience `30`, loss `ce`, DFormerv2-S pretrained path unchanged.

## Summary

| Metric | Value |
|---|---:|
| recorded validation epochs | 50 |
| best val/mIoU | 0.533020 |
| best validation epoch | 49 |
| last val/mIoU | 0.528883 |
| last-5 mean val/mIoU | 0.527628 |
| last-10 mean val/mIoU | 0.519951 |
| best-to-last drop | 0.004137 |
| best val/loss | 0.977882 |
| best val/loss epoch | 9 |
| final train/loss_epoch | 0.052197 |
| ham logit alpha first | 0.050669 |
| ham logit alpha last | 0.090593 |

## Per-Epoch Validation Metrics

| Validation epoch | val/mIoU | val/loss | train/ham_logit_alpha |
|---:|---:|---:|---:|
| 1 | 0.172871 | 1.541051 | 0.050669 |
| 2 | 0.266937 | 1.241868 | 0.051949 |
| 3 | 0.338004 | 1.124357 | 0.053096 |
| 4 | 0.387940 | 1.015735 | 0.054210 |
| 5 | 0.421037 | 1.013760 | 0.055283 |
| 6 | 0.450348 | 0.983706 | 0.056327 |
| 7 | 0.457062 | 0.995837 | 0.057337 |
| 8 | 0.463168 | 1.007939 | 0.058341 |
| 9 | 0.479778 | 0.977882 | 0.059283 |
| 10 | 0.473346 | 1.014141 | 0.060226 |
| 11 | 0.492488 | 0.994928 | 0.061152 |
| 12 | 0.457100 | 1.116567 | 0.062072 |
| 13 | 0.478564 | 1.077595 | 0.062982 |
| 14 | 0.495239 | 1.065080 | 0.063879 |
| 15 | 0.492133 | 1.022799 | 0.064798 |
| 16 | 0.492901 | 1.033992 | 0.065711 |
| 17 | 0.508523 | 1.042024 | 0.066614 |
| 18 | 0.510112 | 1.037305 | 0.067488 |
| 19 | 0.517538 | 1.066454 | 0.068408 |
| 20 | 0.478469 | 1.147665 | 0.069284 |
| 21 | 0.487589 | 1.113113 | 0.070232 |
| 22 | 0.505977 | 1.092857 | 0.071194 |
| 23 | 0.507092 | 1.109039 | 0.072105 |
| 24 | 0.521623 | 1.106172 | 0.072971 |
| 25 | 0.531301 | 1.076916 | 0.073845 |
| 26 | 0.522047 | 1.114837 | 0.074685 |
| 27 | 0.492357 | 1.185226 | 0.075529 |
| 28 | 0.474760 | 1.245417 | 0.076469 |
| 29 | 0.498670 | 1.136829 | 0.077266 |
| 30 | 0.505172 | 1.150057 | 0.078110 |
| 31 | 0.505780 | 1.192165 | 0.078932 |
| 32 | 0.510092 | 1.180243 | 0.079644 |
| 33 | 0.504611 | 1.205236 | 0.080359 |
| 34 | 0.514211 | 1.156933 | 0.081029 |
| 35 | 0.520082 | 1.156790 | 0.081645 |
| 36 | 0.527582 | 1.134695 | 0.082232 |
| 37 | 0.518808 | 1.172588 | 0.082851 |
| 38 | 0.523021 | 1.159126 | 0.083478 |
| 39 | 0.520824 | 1.228312 | 0.084125 |
| 40 | 0.525416 | 1.190034 | 0.084856 |
| 41 | 0.521993 | 1.229107 | 0.085581 |
| 42 | 0.518860 | 1.209439 | 0.086311 |
| 43 | 0.486699 | 1.321077 | 0.086799 |
| 44 | 0.513150 | 1.214431 | 0.087192 |
| 45 | 0.520664 | 1.206041 | 0.087754 |
| 46 | 0.525960 | 1.214634 | 0.088304 |
| 47 | 0.527896 | 1.200132 | 0.088842 |
| 48 | 0.522383 | 1.237917 | 0.089459 |
| 49 | 0.533020 | 1.203147 | 0.090017 |
| 50 | 0.528883 | 1.240081 | 0.090593 |

## Decision

R033 crosses `0.53`, but it remains below R016 `0.541121` by `-0.008101`, below R015 `0.537398`, and below R027/R032/R030. The Ham logit residual alpha increased from `0.050669` to `0.090593`, so the branch was used, but the added Ham logits did not improve the corrected SimpleFPN peak. Treat R033 as partial-positive evidence for Ham complementarity but reject it as an active mainline improvement.
