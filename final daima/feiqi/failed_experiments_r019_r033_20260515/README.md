# Failed/Non-Mainline Experiment Code Archive: R019-R033

This folder archives experiment code that crossed useful diagnostic thresholds in some cases, but should not remain in the active training registry after the R033 cleanup.

The active mainline keeps:

- `dformerv2_mid_fusion`: corrected R016 baseline, best val/mIoU `0.541121`.
- `dformerv2_ham_decoder`: R022 Ham reference, best val/mIoU `0.534332`, stable but below R016.
- `dformerv2_geometry_primary_ham_decoder`: R024 structure diagnostic, best val/mIoU `0.530186`, stable but below R016.

Archived or removed from the active registry:

- R019 `dformerv2_branch_depth_adapter`: best `0.532539`, late collapse.
- R020 `dformerv2_branch_depth_blend_adapter`: best `0.532924`, below R016.
- R025 `dformerv2_depth_encoder_bn_eval`: best `0.532572`, late collapse.
- R026 `dformerv2_official_init_local_modules`: best `0.507906`, negative.
- R027 `dformerv2_primary_residual_depth`: best `0.536739`, high peak but unstable.
- R030 `dformerv2_gated_fusion_residual_top`: best `0.536454`, below R016/R027.
- R031 `dformerv2_simplefpn_classifier_dropout`: best `0.531544`, below R016.
- R032 `dformerv2_simplefpn_c1_detail_gate`: best `0.536603`, stable but below R016 and alpha barely moved.
- R033 `dformerv2_simplefpn_ham_logit_fusion`: best `0.533020`, below R016 and does not justify Ham-logit scalar tuning.
- Older PMAD/TGGA/Freq/LMLP-style branches remain evidence-only unless explicitly revived from reports.

Evidence remains in `docs/`, `miou_list/`, `reports/`, `metrics/`, and `experiments/`. Checkpoints and TensorBoard event files are not archived here.
