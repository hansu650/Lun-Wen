# Paper Notes

## 2026-05-06 DFormer-guided Depth Adapter Simple Candidate

- New candidate branch is `dformerv2_guided_depth_adapter_simple`.
- Motivation: Full++ `dformerv2_guided_depth_comp_fusion` had a strong single run but a negative five-run mean, so this branch isolates Part 1 and removes rectification and attention aggregation complexity.
- Design: reuse DFormerv2-guided stage-wise depth adaptation, then add the adapted depth through a minimal primary-preserving residual adapter.
- Final feature remains `f_i = p_i + gamma * delta_i`, where `delta_i` is generated only from guided depth and `abs(p_i - d_i')`.
- This branch avoids Full++ Part 2 and Part 3, DGC-AF++, CSG, GRM-ARD, `GatedFusion`, token attention, support/conflict logic, relation selection, and geometry warping.
- Run01-run03 result: mean best val/mIoU `0.514621`, above repeated baseline mean `0.513406` by `0.001215`.
- Run01-run06 result: mean best val/mIoU `0.512316`, below repeated baseline mean `0.513406` by `0.001090`, but above Full++ mean `0.511379` by `0.000937` and above DGC-AF++ mean `0.511418` by `0.000898`.
- Conclusion: near-baseline but negative repeated-run result. This branch is healthier than Full++ and DGC-AF++, but it should not be claimed as a stable paper improvement over the repeated DFormerv2 mid-fusion baseline.

## 2026-05-06 DFormer-guided Depth Rectification and Complementary Fusion Candidate

- New candidate branch is `dformerv2_guided_depth_comp_fusion`.
- Motivation: repeated DGC-AF++ runs did not beat the repeated DFormerv2 mid-fusion baseline, while CSG and GRM-ARD were negative; the new branch moves depth control earlier than late residual compensation.
- Design: DFormerv2 primary features guide stage-wise depth adaptation, then asymmetric complementary rectification keeps primary changes tiny, and attention complementary aggregation adds only a small depth-derived residual.
- Source-code motivation: PGDENet stage-wise depth enhancement, CMX feature rectification before fusion, ACNet attentive additive fusion, SGACNet/ESANet SE-style encoder-stage fusion, and DFormerV2 depth-as-geometry guidance.
- The final output remains primary-preserving: `f_i = p_i + small complementary residual`.
- The branch avoids DGC-AF++, CSG, GRM-ARD, `GatedFusion`, full token attention, geometry warping, and symmetric `g * primary + (1 - g) * depth` fusion.
- Run01-run05 repeated result: mean best val/mIoU `0.511379`, below repeated baseline mean `0.513406` by `0.002027`; essentially tied with DGC-AF++ mean `0.511418` but still slightly lower by `0.000039`.
- Conclusion: negative repeated-run result. This branch should be kept as a meaningful ablation, not as a paper improvement.

## 2026-05-04 DGC-AF Plus CSG Candidate

- New candidate branch is `dformerv2_dgc_af_plus_csg`.
- Motivation: keep the DGC-AF++ main branch, but inject high-level c4 semantic context into each stage's residual generation so low-level fusion can be guided by global semantics.
- Design: derive a global semantic context from DFormerv2 c4, project it into four stage-wise `semantic_gate`s, and use those gates to produce `p_guided` inside the DGC-AF++ relation/support/conflict/residual path.
- The final feature remains primary-preserving: `out = primary_feat + residual`.
- This differs from GRM-ARD because it does not add residual mixture, budget, ARD suppression, or multi-candidate residual arbitration.
- The branch avoids `GatedFusion`, full `QK^T` attention, token flattening, `HW x HW` attention, `grid_sample`, flow warp, deformable attention, GRM, and ARD.
- Run02 result: best val/mIoU `0.506402`, below repeated baseline mean `0.513406` and below DGC-AF++ run01 `0.513584`.
- Conclusion: negative result. Cross-stage semantic guidance should not be used as the paper main branch unless a later lighter ablation isolates a useful semantic cue.

## 2026-05-04 DGC-AF Plus GRM-ARD Candidate

- New candidate branch is `dformerv2_dgc_af_plus_grm_ard`.
- Motivation: DGC-AF++ is the first recent DFormerv2-primary residual branch to slightly exceed the repeated baseline mean in one run, so this branch deepens residual control rather than adding ordinary full fusion.
- Source-code inspiration checked: CAFuser condition-aware fusion and learnable adapter ratios, Mul-VMamba feature rectification/filtering, CMFormer multi-scale correction/global-local residual fusion, and prior GeminiFusion/KTB/PrimKD/CMX notes.
- Design: keep DGC-AF++ depth relation preparation, add depthwise multi-scale relation context, primary-guided residual anchor, guided/support/detail residual mixture, residual budget gate, and soft adaptive bad residual suppression.
- The branch avoids `GatedFusion`, full `QK^T` attention, token flattening, `HW x HW` attention, `grid_sample`, flow warp, deformable attention, stochastic hard drop, hard threshold, and top-k.
- Run01 result: best val/mIoU `0.507743`, below repeated baseline mean `0.513406` and below DGC-AF++ run01 `0.513584`.
- Conclusion: negative result. This heavier residual-control design should not be used as the paper main branch unless later ablations isolate a useful subcomponent.

## 2026-05-04 DGC-AF Plus Candidate

- New candidate branch is `dformerv2_dgc_af_plus`.
- Motivation: DGC-AF Full reached near-baseline quality but did not exceed the repeated baseline mean, so this branch explores stronger source-inspired mechanisms without changing the DFormerv2 encoder.
- Design: asymmetric depth rectification, Gemini-style noise-vs-depth relation selection, support/conflict decomposition, soft sparse support selection, and a small conflict/detail correction residual.
- The branch avoids `GatedFusion`, full `QK^T` token attention, token flattening, `HW x HW` attention, `grid_sample`, flow warp, and deformable attention.
- Run01 result: best val/mIoU `0.513584`, slightly above repeated baseline mean `0.513406` by `0.000178`.
- Run01-run04 repeated result: mean best val/mIoU `0.511418`, below repeated baseline mean `0.513406` by `0.001988`.
- Conclusion: negative repeated-run result. DGC-AF++ can be kept as a meaningful ablation because it is safer than the failed heavier variants, but it should not be claimed as a stable improvement over the repeated DFormerv2 mid-fusion baseline.

## 2026-05-04 DGC-AF Full Candidate

- New candidate branch is `dformerv2_dgc_af_full`.
- Motivation: extend PG-SparseComp while preserving the safer primary-residual design. DFormerv2 primary features guide depth cleaning, pixel-wise cross attention, support/conflict decomposition, and soft sparse residual compensation.
- The branch avoids `GatedFusion`, full `QK^T` token attention, and DFormerv2 geometry self-attention.
- Run01 result: best val/mIoU `0.512766`, below repeated baseline mean `0.513406` by `0.000640`, but above PG-SparseComp run01 `0.511478`.
- Conclusion: this is the best recent primary-preserving residual run, but not a verified paper improvement. It can motivate repeated-run validation, not a claim of superiority.

## 2026-05-04 Primary-Guided Sparse Depth Compensation Candidate

- New candidate branch is `dformerv2_pg_sparse_comp_fusion`.
- Motivation: after the failed gated co-attention residual branch, preserve DFormerv2 primary features and add depth only through a small reliability-weighted residual.
- The branch uses spatial reliability, channel reliability, and a depth/difference residual adapter, but no `GatedFusion`, full token attention, or geometry self-attention.
- Run01 result: best val/mIoU `0.511478`, below repeated baseline mean `0.513406`.
- Conclusion: this branch is safer than the deprecated gated co-attention branch and reaches near-baseline quality, but it is not a current paper improvement. It can be discussed only as a neutral/negative ablation unless repeated runs improve the mean.

## 2026-05-04 SA-Gate One-Way Fusion Candidate

- Current clean baseline remains `dformerv2_mid_fusion`.
- New candidate branch is `dformerv2_sagate_fusion`.
- The branch borrows SA-Gate `FilterLayer/FSP` channel recalibration, but keeps the fusion asymmetric.
- DFormerv2 features are treated as the primary depth-aware representation.
- DepthEncoder features are treated as auxiliary support and are added only through a small residual path.
- Five-run result: mean best val/mIoU is `0.513216`, slightly below the repeated baseline mean `0.513406`.
- Conclusion: do not treat this branch as a stable improvement. The best single run is high, but the five-run variance is also high.

## 2026-05-04 SA-Gate Token Selection Fusion Candidate

- New candidate branch is `dformerv2_sagate_token_fusion`.
- The change keeps SA-Gate `FilterLayer/FSP` as the main fusion skeleton.
- The only conceptual replacement is the unstable simple spatial gate, replaced by TokenFusion-style dynamic token scoring.
- This is still DFormerv2-primary fusion: the selection score is generated from DFormerv2 primary features, and depth support is residual only.
- Selector input includes DFormerv2 primary, depth auxiliary support, and their absolute difference, so the gate can assess whether the depth support is locally useful.
- This branch has no valid mIoU result yet, so it must not be reported as an improved result until a real checkpoint/log supports it.
- Result note for `dformerv2_sagate_token_fusion_run01`: best val/mIoU `0.509558`, below baseline mean `0.513406`; not a stable improvement.

## 2026-05-04 SA-Gate Ablations Deprecated

- `dformerv2_sagate_fusion` is deprecated because its five-run mean best val/mIoU is `0.513216`, below the repeated baseline mean `0.513406`.
- `dformerv2_sagate_token_fusion` is deprecated because run01 best val/mIoU is `0.509558`.
- Main active branch is restored to `dformerv2_mid_fusion`.
- These ablations can be cited only as negative records, not as improved paper results.

## 2026-05-04 Gated Co-Attention Residual Fusion Candidate

- New candidate branch is `dformerv2_gated_coattn_res_fusion`.
- Motivation: repeated baseline gate statistics show original GatedFusion learns a stable near-50/50 RGB-DFormer/depth balance, so the baseline gate should not be replaced.
- Design: preserve original GatedFusion and add a small CANet-style channel/spatial co-attention correction after the fused base feature.
- CANet is used only as theoretical motivation because there is no reliable official RGB-D CANet implementation in the current workspace.
- Local code references were used only for ideas: ACNet attentive additive fusion, SGACNet channel/spatial attention blocks, and FAFNet correction/alignment style.
- This branch has no valid mIoU result yet, so it must not be reported as an improved result until a real checkpoint/log supports it.
- Result note for `dformerv2_gated_coattn_res_fusion_run01`: manually stopped after 20 recorded validation epochs; best recorded val/mIoU `0.483357`, far below baseline mean `0.513406`; do not continue as a main branch.

## 2026-05-04 Gated Co-Attention Residual Fusion Deprecated

- `dformerv2_gated_coattn_res_fusion` is deprecated because it was manually stopped after 20 recorded validation epochs with best val/mIoU `0.483357`.
- This is far below the repeated `dformerv2_mid_fusion` baseline mean `0.513406`.
- Main active branch is restored to `dformerv2_mid_fusion`.
- This ablation can be cited only as a negative record, not as an improved paper result.
