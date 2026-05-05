# Experiment Log

## 2026-05-05 dformerv2_dgc_af_plus run01-run04 summary

- model: `dformerv2_dgc_af_plus`
- change: DGC-AF++ with asymmetric depth rectification, Gemini-style noise-vs-depth relation selection, support/conflict decomposition, soft sparse support selection, and small detail correction residual.
- settings: `batch_size=2`, `max_epochs=50`, `lr=6e-5`, `num_workers=4`, `early_stop_patience=30`
- pretrained: `C:/Users/qintian/Desktop/qintian/dformer_work/checkpoints/pretrained/DFormerv2_Small_pretrained.pth`
- completed runs: `4`
- dformerv2_dgc_af_plus_run01: best val/mIoU `0.513584` at recorded epoch `50`, last `0.513584`, best val/loss `1.034468` at recorded epoch `8`
- dformerv2_dgc_af_plus_run02: best val/mIoU `0.511645` at recorded epoch `48`, last `0.505630`, best val/loss `1.039455` at recorded epoch `10`
- dformerv2_dgc_af_plus_run03: best val/mIoU `0.510157` at recorded epoch `50`, last `0.510157`, best val/loss `1.045118` at recorded epoch `10`
- dformerv2_dgc_af_plus_run04: best val/mIoU `0.510288` at recorded epoch `50`, last `0.510288`, best val/loss `1.053412` at recorded epoch `9`
- mean best val/mIoU: `0.511418`
- population std best val/mIoU: `0.001379`
- comparison baseline mean best: `0.513406`
- mean delta vs baseline mean: `-0.001988`
- evidence: `miou_list/dformerv2_dgc_af_plus_run01.md`, `miou_list/dformerv2_dgc_af_plus_run02.md`, `miou_list/dformerv2_dgc_af_plus_run03.md`, `miou_list/dformerv2_dgc_af_plus_run04.md`, and `miou_list/dformerv2_dgc_af_plus_summary_run01_04.md`
- conclusion: negative repeated-run result. The original run01 was a high single run, but the four-run mean does not beat the repeated DFormerv2 mid-fusion baseline.
- next step: do not claim DGC-AF++ as a stable paper improvement. Either return to the stronger repeated baseline or design a smaller targeted ablation that specifically improves late-epoch stability without adding global residual controllers.

## 2026-05-05 dformerv2_dgc_af_plus_csg_run02

- model: `dformerv2_dgc_af_plus_csg`
- change: DGC-AF++ with cross-stage semantic guidance from DFormerv2 c4 global context injected into each stage's residual generation.
- settings: `batch_size=2`, `max_epochs=50`, `lr=6e-5`, `num_workers=4`, `early_stop_patience=30`
- pretrained: `C:/Users/qintian/Desktop/qintian/dformer_work/checkpoints/pretrained/DFormerv2_Small_pretrained.pth`
- recorded validation epochs: `50`
- best val/mIoU: `0.506402` at recorded epoch `50`
- last val/mIoU: `0.506402`
- best val/loss: `1.054549` at recorded epoch `8`
- comparison baseline mean best: `0.513406`
- delta vs baseline mean: `-0.007004`
- comparison DGC-AF++ run01 best: `0.513584`
- delta vs DGC-AF++ run01: `-0.007182`
- comparison GRM-ARD run01 best: `0.507743`
- delta vs GRM-ARD run01: `-0.001341`
- partial retry note: `dformerv2_dgc_af_plus_csg_run01` recorded 33 validation epochs and reached best recorded val/mIoU `0.506047`; it is not used as the final result.
- evidence: `miou_list/dformerv2_dgc_af_plus_csg_run02.md`
- conclusion: negative result. Cross-stage c4 semantic guidance did not improve DGC-AF++; it underperforms the repeated baseline mean, DGC-AF++ run01, DGC-AF Full run01, and PG-SparseComp run01.
- next step: do not promote CSG as the main branch. Keep `dformerv2_dgc_af_plus` as the current best self-designed candidate and prioritize repeated DGC-AF++ runs or a lighter ablation of semantic guidance.

## 2026-05-04 dformerv2_dgc_af_plus_grm_ard_run01

- model: `dformerv2_dgc_af_plus_grm_ard`
- change: DGC-AF++ with depthwise multi-scale relation context, primary-guided residual anchor, guided/support/detail residual mixture, residual budget gate, and soft adaptive bad residual suppression.
- settings: `batch_size=2`, `max_epochs=50`, `lr=6e-5`, `num_workers=4`, `early_stop_patience=30`
- pretrained: `C:/Users/qintian/Desktop/qintian/dformer_work/checkpoints/pretrained/DFormerv2_Small_pretrained.pth`
- recorded validation epochs: `50`
- best val/mIoU: `0.507743` at recorded epoch `45`
- last val/mIoU: `0.501644`
- best val/loss: `1.043510` at recorded epoch `10`
- comparison baseline mean best: `0.513406`
- delta vs baseline mean: `-0.005663`
- comparison DGC-AF++ run01 best: `0.513584`
- delta vs DGC-AF++ run01: `-0.005841`
- comparison DGC-AF Full run01 best: `0.512766`
- delta vs DGC-AF Full run01: `-0.005023`
- comparison PG-SparseComp run01 best: `0.511478`
- delta vs PG-SparseComp run01: `-0.003735`
- evidence: `miou_list/dformerv2_dgc_af_plus_grm_ard_run01.md`
- conclusion: negative result. The heavier GRM/ARD residual-control chain underperforms DGC-AF++, DGC-AF Full, PG-SparseComp, and the repeated baseline mean.
- next step: do not promote this branch. Keep `dformerv2_dgc_af_plus` as the current best single-run candidate; if debugging GRM/ARD later, first ablate mixture gate, budget gate, and ARD separately.

## 2026-05-04 dformerv2_dgc_af_plus_run01

- model: `dformerv2_dgc_af_plus`
- change: DGC-AF++ with asymmetric depth rectification, Gemini-style noise-vs-depth relation selection, support/conflict decomposition, soft sparse support selection, and small detail correction residual.
- settings: `batch_size=2`, `max_epochs=50`, `lr=6e-5`, `num_workers=4`, `early_stop_patience=30`
- pretrained: `C:/Users/qintian/Desktop/qintian/dformer_work/checkpoints/pretrained/DFormerv2_Small_pretrained.pth`
- recorded validation epochs: `50`
- best val/mIoU: `0.513584` at recorded epoch `50`
- last val/mIoU: `0.513584`
- best val/loss: `1.034468` at recorded epoch `8`
- comparison baseline mean best: `0.513406`
- delta vs baseline mean: `+0.000178`
- comparison DGC-AF Full run01 best: `0.512766`
- delta vs DGC-AF Full run01: `+0.000818`
- comparison PG-SparseComp run01 best: `0.511478`
- delta vs PG-SparseComp run01: `+0.002105`
- comparison SA-Gate five-run mean best: `0.513216`
- delta vs SA-Gate five-run mean: `+0.000368`
- evidence: `miou_list/dformerv2_dgc_af_plus_run01.md`
- conclusion: promising single-run result. It is the first recent DFormerv2-primary residual compensation variant to slightly exceed the repeated baseline mean, but the margin is too small to claim stable improvement without repeated runs.
- next step: run repeated seeds for `dformerv2_dgc_af_plus`; if the mean remains above `0.513406`, promote it to the main candidate branch, otherwise analyze whether the epoch-50 jump is seed variance.

## 2026-05-04 dformerv2_dgc_af_full_run01

- model: `dformerv2_dgc_af_full`
- change: DFormerv2-guided cyclic attention fusion with residual-safe depth cleaning, pixel-wise cross attention, support/conflict decomposition, and soft sparse residual compensation.
- settings: `batch_size=2`, `max_epochs=50`, `lr=6e-5`, `num_workers=4`, `early_stop_patience=30`
- pretrained: `C:/Users/qintian/Desktop/qintian/dformer_work/checkpoints/pretrained/DFormerv2_Small_pretrained.pth`
- recorded validation epochs: `50`
- best val/mIoU: `0.512766` at recorded epoch `47`
- last val/mIoU: `0.477881`
- best val/loss: `1.036298` at recorded epoch `7`
- comparison baseline mean best: `0.513406`
- delta vs baseline mean: `-0.000640`
- comparison PG-SparseComp run01 best: `0.511478`
- delta vs PG-SparseComp run01: `+0.001288`
- comparison gated co-attention run01 partial best: `0.483357`
- delta vs gated co-attention partial: `+0.029409`
- evidence: `miou_list/dformerv2_dgc_af_full_run01.md`
- conclusion: near-baseline and slightly better than PG-SparseComp run01, but still below the repeated baseline mean. Do not claim as a valid improvement without repeated runs.
- next step: run repeated seeds before changing architecture further; if repeats remain close but below baseline, inspect whether the residual gate opens too late or whether late-epoch overfitting causes the last-epoch drop.

## 2026-05-04 dformerv2_pg_sparse_comp_fusion_run01

- model: `dformerv2_pg_sparse_comp_fusion`
- change: DFormerv2 primary-guided sparse depth residual compensation without `GatedFusion`.
- settings: `batch_size=2`, `max_epochs=50`, `lr=6e-5`, `num_workers=4`, `early_stop_patience=30`
- pretrained: `C:/Users/qintian/Desktop/qintian/dformer_work/checkpoints/pretrained/DFormerv2_Small_pretrained.pth`
- recorded validation epochs: `50`
- best val/mIoU: `0.511478` at recorded epoch `44`
- last val/mIoU: `0.471010`
- best val/loss: `1.028691` at recorded epoch `10`
- comparison baseline mean best: `0.513406`
- delta vs baseline mean: `-0.001928`
- evidence: `miou_list/dformerv2_pg_sparse_comp_fusion_run01.md`
- conclusion: near-baseline but not improved. The primary-guided sparse residual branch is much healthier than the deprecated gated co-attention residual branch, but this single run should be treated as neutral/negative until repeated runs prove a higher mean.
- next step: do not claim as paper improvement. If continuing this direction, first inspect reliability/gamma behavior or run repeats before adding heavier attention.

## 2026-05-04 dformerv2_sagate_fusion 5-run summary

- model: `dformerv2_sagate_fusion`
- pretrained: `C:/Users/qintian/Desktop/qintian/dformer_work/checkpoints/pretrained/DFormerv2_Small_pretrained.pth`
- data_root: `C:/Users/qintian/Desktop/qintian/data/NYUDepthv2`
- settings: `batch_size=2`, `max_epochs=50`, `lr=6e-5`, `num_workers=4`, `early_stop_patience=30`
- dformerv2_sagate_fusion_run01: best val/mIoU `0.522717` at epoch `50`, best val/loss `0.999145` at epoch `10`
- dformerv2_sagate_fusion_run02: best val/mIoU `0.520692` at epoch `40`, best val/loss `1.027782` at epoch `13`
- dformerv2_sagate_fusion_run03: best val/mIoU `0.507363` at epoch `39`, best val/loss `1.034039` at epoch `9`
- dformerv2_sagate_fusion_run04: best val/mIoU `0.510646` at epoch `45`, best val/loss `1.032801` at epoch `9`
- dformerv2_sagate_fusion_run05: best val/mIoU `0.504661` at epoch `29`, best val/loss `1.040354` at epoch `12`
- mean best val/mIoU: `0.513216`
- population std best val/mIoU: `0.007214`
- comparison baseline mean best: `0.513406`
- mean delta vs baseline: `-0.000190`
- evidence: `miou_list/dformerv2_sagate_fusion_run01.md` through `miou_list/dformerv2_sagate_fusion_run05.md`, plus `miou_list/dformerv2_sagate_fusion_summary.md`
- conclusion: SA-Gate-style one-way fusion is lighter, but its five-run mean does not beat the repeated DFormerv2 mid-fusion baseline. Do not treat it as a stable improvement.

## 2026-05-04 dformerv2_sagate_token_fusion_run01

- model: `dformerv2_sagate_token_fusion`
- change: SA-Gate `FilterLayer/FSP` plus TokenFusion-style soft selector using `primary_feat`, `aux_support`, and `abs(primary-aux)`.
- settings: `batch_size=2`, `max_epochs=50`, `lr=6e-5`, `num_workers=4`, `early_stop_patience=30`
- best val/mIoU: `0.509558` at epoch `48`
- last val/mIoU: `0.501725`
- best val/loss: `1.048825` at epoch `7`
- comparison baseline mean best: `0.513406`
- delta vs baseline mean: `-0.003848`
- comparison SA-Gate V1 mean best: `0.513216`
- delta vs SA-Gate V1 mean: `-0.003658`
- evidence: `miou_list/dformerv2_sagate_token_fusion_run01.md`
- conclusion: negative result. The TokenFusion-style selector did not improve the stable baseline and should not be continued in this exact form.

## 2026-05-04 dformerv2_gated_coattn_res_fusion_run01 partial

- model: `dformerv2_gated_coattn_res_fusion`
- status: manually terminated before 50 epochs
- settings: `batch_size=2`, `max_epochs=50`, `lr=6e-5`, `num_workers=4`, `early_stop_patience=30`
- recorded validation epochs: `20`
- best recorded val/mIoU: `0.483357` at recorded epoch `17`
- last recorded val/mIoU: `0.480325`
- best recorded val/loss: `1.007498` at recorded epoch `11`
- comparison baseline mean best: `0.513406`
- delta vs baseline mean: `-0.030049`
- evidence: `miou_list/dformerv2_gated_coattn_res_fusion_run01_partial.md`
- conclusion: negative partial result. The co-attention residual correction branch is far below baseline by epoch 20 and was stopped early.

## 2026-05-04 dformerv2_gated_coattn_res_fusion deprecated

- model: `dformerv2_gated_coattn_res_fusion`
- status: manually terminated after 20 validation epochs
- best recorded val/mIoU: `0.483357`
- comparison baseline mean best: `0.513406`
- delta vs baseline mean: `-0.030049`
- conclusion: GatedFusion + co-attention residual correction severely underperformed the repeated DFormerv2 mid-fusion baseline, so this branch is deprecated.
- evidence: `miou_list/dformerv2_gated_coattn_res_fusion_run01_partial.md`
