# Paper Notes

## 2026-05-15 R033 SimpleFPN Ham Logit Fusion Boundary

- R033 completed 50 validation epochs with best val/mIoU `0.533020` at validation epoch `49`; final val/mIoU is `0.528883`.
- It crosses `0.53`, but remains below the corrected R016 baseline `0.541121` by `-0.008101` and below R027/R032/R030.
- Ham logit alpha increased from `0.050669` to `0.090593`, so the auxiliary Ham logit branch was used, but logits-level residual fusion did not improve the SimpleFPN peak.
- Paper boundary: do not claim SimpleFPN+Ham logit fusion as an improvement over the corrected baseline. It is a partial-positive diagnostic showing Ham context alone is not sufficient when injected only at logits.
- Strategic implication: avoid scalar Ham-logit tuning. If using Ham again, it should be a more mechanistic feature-level question; otherwise prioritize high-stage residual or stability-contract diagnostics.

## 2026-05-15 R032 SimpleFPN C1 Detail Gate Boundary

- R032 completed 50 validation epochs with best val/mIoU `0.536603` at validation epoch `50`; final val/mIoU is also `0.536603`.
- It crosses `0.53` and is near R027/R030, but remains below the corrected R016 baseline `0.541121` by `-0.004518`.
- The c1 detail alpha barely changes from `0.998994` to `0.998770`, so the near-baseline c1 strength parameter is not an active explanatory mechanism in this run.
- Paper boundary: do not claim c1 detail gating as an improvement over the corrected baseline. It can be described as a partial-positive decoder-detail diagnostic.
- Strategic implication: stop c1-gate tuning. The next hypothesis should test head-level complementarity between SimpleFPN detail and Ham semantic context.

## 2026-05-15 R031 SimpleFPN Classifier Dropout Boundary

- R031 completed 50 validation epochs with best val/mIoU `0.531544` at validation epoch `40`; final val/mIoU is `0.525760`.
- It crosses `0.53`, but is below the corrected R016 baseline `0.541121`, R027 `0.536739`, R030 `0.536454`, and R022 `0.534332`.
- Paper boundary: do not claim SimpleFPN classifier dropout as a method improvement or as a useful transfer of the Ham dropout parity fix.
- Method implication: the R022 gain was specific to the Ham classifier/head path; the current SimpleFPN bottleneck is not solved by generic classifier dropout.
- Strategic implication: move to a different SimpleFPN-specific mechanism, especially controlling high-resolution `c1` detail contribution, rather than tuning dropout rate.

## 2026-05-15 R030 GatedFusion Residual-Top Boundary

- R030 completed 50 validation epochs with best val/mIoU `0.536454` at validation epoch `42`; final val/mIoU is `0.529803`.
- It crosses `0.53` and recovers better at the last epoch than R027, but remains below the corrected R016 baseline `0.541121` by `-0.004667`.
- Paper boundary: do not claim all-stage residual-top correction as the main method or as an improvement over the corrected baseline.
- Method implication: preserving R016 `GatedFusion` avoids the worst replacement-fusion instability, but residual-depth correction on top is still insufficient to reach the 0.56 target.
- Strategic implication: stop the residual-family sequence for now. The next fixed-recipe experiment should be a distinct, low-risk regularization hypothesis on the current strongest path, such as SimpleFPN classifier dropout.

## 2026-05-15 R027 Primary Residual Depth Injection Boundary

- R027 completed 50 validation epochs with best val/mIoU `0.536739` at validation epoch `41`; final val/mIoU is `0.505286`.
- It crosses the fixed-recipe `0.53` stage threshold and is above R022 `0.534332`, but remains below the corrected R016 baseline `0.541121` by `-0.004382`.
- The best-to-last drop is `0.031453`, with last-5 mean `0.519799`, so late stability remains poor.
- Paper boundary: do not claim R027 as the main method or as an improvement over the corrected baseline. It can be cited only as a partial-positive fusion-form diagnostic.
- Method implication: primary-preserving residual depth has useful peak signal, but replacing the existing R016 `GatedFusion` path removes too much of the strongest corrected baseline behavior.
- Strategic implication: the next fixed-recipe experiment should preserve R016 `GatedFusion` and add a zero-initialized residual correction on top, rather than tuning R027 or repeating all-stage replacement variants.

## 2026-05-15 R026 Official-Style Init Boundary

- R026 completed 50 validation epochs with best val/mIoU `0.507906` at validation epoch `33`; final val/mIoU is `0.499770`.
- It is far below R016 `0.541121`, R025 `0.532572`, and the fixed-recipe `0.53` stage threshold.
- Paper boundary: do not claim official-style init as helpful. It should be logged as a negative contract/stability check.
- Method implication: the performance gap is not solved by Kaiming/BN init parity of local fusion/decoder modules. The current default initialization appears better for this pipeline.
- Strategic implication: stop local-init experiments. Next fixed-recipe work should target fusion form: preserve DFormerv2 primary features and add depth as a residual rather than replacing features through gated mixing.

## 2026-05-15 R025 DepthEncoder BN Eval Boundary

- R025 completed 50 validation epochs with best val/mIoU `0.532572` at validation epoch `47`; final val/mIoU dropped to `0.496030`.
- It is slightly above R024 `0.530186`, but below R022 `0.534332` and below the corrected R016 baseline `0.541121`.
- The best-to-last drop is `0.036541`, so the original late-instability problem is not solved.
- Paper boundary: do not claim DepthEncoder BN eval as a stable improvement. It can be cited only as a negative/partial diagnostic showing that small-batch BN drift is not the main cause of the corrected pipeline gap.
- Strategic implication: do not build R026 on BN eval. The next fixed-recipe experiment should target initialization of local random modules or another single structural stability issue.

## 2026-05-15 R024 Raw DFormerv2 + Ham Boundary

- R024 completed 50 validation epochs with best val/mIoU `0.530186` at validation epoch `45`; final val/mIoU is `0.529383`.
- It is stable and crosses `0.53`, but remains below R022 `0.534332` and the corrected R016 baseline `0.541121`.
- Paper boundary: R024 is a structure-contract diagnostic, not a main contribution. Do not claim official-style Ham alone solves the gap to `0.56`.
- Method implication: removing the external DepthEncoder/GatedFusion stack hurts peak mIoU relative to R016/R022, so the local fusion path contains useful signal. The remaining problem is likely fusion/depth-branch stability or initialization, not simply removing fusion.
- Strategic implication: stop Ham micro-fixes. The next original-method experiment should target the stronger corrected mid-fusion path, preferably a single stability hypothesis such as external DepthEncoder BatchNorm eval/freeze or an identity-start residual fusion.

## 2026-05-15 R023 Corrected Teacher Gate Boundary

- R023 completed 50 validation epochs with best val/mIoU `0.524498` at validation epoch `43`; final val/mIoU is `0.507023`.
- It is below the corrected R016 baseline `0.541121` by `-0.016623` and below the `0.53` PMAD-teacher gate, with best-to-last drop `0.017475`.
- Paper boundary: R023 is a teacher-quality gate, not a proposed method or improvement claim. Do not cite it as evidence that the geometry-primary teacher improves the pipeline.
- Method implication: corrected-contract PMAD should not be run from this teacher checkpoint; a weak teacher would likely anchor the student below R016.
- Strategic implication: pivot from KD to structure-contract isolation. The next highest-value experiment is raw `DFormerv2_S + OfficialHamDecoder` without the external DepthEncoder/GatedFusion stack, to test whether the local post-backbone fusion is breaking the official feature contract.

## 2026-05-15 R022 Ham Dropout Parity Boundary

- R022 completed 50 validation epochs with best val/mIoU `0.534332` at validation epoch `50`; final val/mIoU is also `0.534332`.
- It improves R021 no-dropout Ham by `+0.006979` and R020 by `+0.001408`, but it remains below the corrected R016 baseline `0.541121` by `-0.006790`.
- Paper boundary: the official classifier dropout is a useful parity fix, not a novel contribution by itself. Do not claim Ham decoder parity beats the corrected baseline.
- Method implication: a c2-c4 Ham decoder with classifier dropout is a viable partial-positive architecture variant, but it does not solve the 0.56 target under the fixed recipe.
- Strategic implication: stop Ham micro-fixes for now. The next paper-relevant step should refresh the geometry-primary teacher under the corrected R015/R016 contract, then use that result to decide whether corrected PMAD is still alive.

## 2026-05-15 R021 LightHam-Like Decoder Boundary

- R021 completed 50 validation epochs with best val/mIoU `0.527353` at validation epoch `39`, and final val/mIoU `0.501377`.
- It is below the corrected R016 baseline `0.541121` by `-0.013768` and below R020 `0.532924` by `-0.005571`.
- The implementation tests a self-contained c2-c4 LightHam-like decoder on top of the local DFormerv2 + DepthEncoder + GatedFusion fused features.
- Audit boundary: R021 is not strict official Ham parity because it omits official `BaseDecodeHead.cls_seg()` `Dropout2d(0.1)`. Do not claim it as an exact reproduction of DFormer Ham.
- Paper boundary: do not claim Ham decoder improves the corrected baseline. This result instead supports that simply replacing SimpleFPN with a heavier Ham-like c2-c4 decoder is not sufficient and may overfit the local fused features.
- Strategic implication: run one minimal R022 dropout parity fix to close the identified implementation gap. If R022 still fails to beat R016/R020, stop Ham decoder work and pivot to corrected-contract PMAD teacher refresh.

## 2026-05-15 R020 Branch-Specific Depth Blend Adapter Boundary

- R020 completed 50 validation epochs with best val/mIoU `0.532924` at validation epoch `41`, but final val/mIoU dropped to `0.503238`.
- It slightly improves the R019 peak (`0.532539`) and has a better late-window profile than R019, but it is still below the current corrected baseline R016 `0.541121` by `-0.008197`.
- The learned global alpha stays near the initialization (`0.050022` to `0.051455`), so the model did not move strongly toward the hard `[0,1]` DepthEncoder branch.
- Paper boundary: do not claim R020 as a new main result or stable improvement. It supports the narrative that branch-specific depth representation has a real but unstable signal; the simple global blend is not enough to close the gap to `0.56`.
- Current corrected baseline remains R016, best val/mIoU `0.541121`.
- Strategic implication: future work should avoid blind repeats of R019/R020. The next method should target late stability or use a richer per-stage/per-pixel depth adapter; official Ham parity remains useful only if the aim is reference-gap diagnosis rather than originality.

## 2026-05-14 R019 Branch-Specific Depth Adapter Boundary

- R019 completed 50 validation epochs with best val/mIoU `0.532539` at validation epoch `46`, but final val/mIoU dropped to `0.495229`.
- It is above the fixed-recipe `0.53` threshold, but below the current corrected baseline R016 `0.541121` by `-0.008582`.
- Unlike R015/R016/R017/R018, R019 is an original method direction: it separates the depth representation used by DFormerv2 geometry attention from the representation used by the external ResNet-18 DepthEncoder.
- Paper boundary: do not claim R019 as the best result or as a stable improvement. It can be described as an exploratory partial-positive signal showing that branch-specific depth representation matters, but the simple `[0,1]` reconstruction is not stable enough.
- The current corrected baseline remains R016: official label + official depth normalization + local RGB input + local drop path default, best val/mIoU `0.541121`.
- Strategic implication: future original-method work should stabilize the branch-specific depth path or design a learned/lightweight adapter; blind repeats of the same R019 are low value because the best-to-last drop is `0.037311`.

## 2026-05-14 R018 DropPath 0.25 Contract Negative Boundary

- R018 retry1 completed 50 validation epochs with best val/mIoU `0.526282` at validation epoch `46`, below the R016 corrected baseline `0.541121` by `-0.014839`.
- The tested change was official DFormerv2-S NYUDepthV2 `drop_path_rate=0.25`; it is a baseline-contract gate, not a method contribution.
- Result interpretation: unlike R015 label mapping and R016 depth normalization, raising the local DFormerv2-S drop path rate from `0.1` to `0.25` does not help this mid-fusion adaptation.
- Paper boundary: do not report `drop_path_rate=0.25` as part of the corrected local baseline and do not claim it as a proposed method. It can be mentioned only as an attempted official-contract check that failed.
- Current corrected baseline remains R016: official label + official depth normalization + local RGB input + local DFormerv2-S drop path default, best val/mIoU `0.541121`.
- Strategic implication: the remaining gap to `0.56` is unlikely to be solved by more small official-contract gates unless another concrete mismatch is found. The next paper-relevant step should either quantify official Ham decoder parity or start an original corrected-contract method such as refreshed PMAD teacher/student or branch-specific depth input adaptation.

## 2026-05-14 R017 RGB/BGR Contract Negative Boundary

- R017 completed 50 validation epochs with best val/mIoU `0.529090` at validation epoch `38`, below the R016 corrected baseline `0.541121` by `-0.012031`.
- The tested change was official DFormer NYUDepthV2 BGR input behavior. It is a baseline-contract gate, not a method contribution.
- Result interpretation: despite official DFormer using BGR for NYUDepthV2, the current local DFormerv2 + DepthEncoder + GatedFusion + SimpleFPN adaptation performs better with the R016 RGB input path.
- Paper boundary: do not report BGR as part of the corrected baseline and do not claim it as a proposed method. It can be mentioned only as an attempted contract check that failed.
- Current corrected baseline remains R016: official label + official depth normalization + local RGB input, best val/mIoU `0.541121`.
- Next baseline-contract gate should be DFormerv2-S `drop_path_rate=0.25`; after that, freeze the corrected baseline and move to original fusion/stability innovations if still below `0.56`.

## 2026-05-14 R016 Official Depth Normalization Result Boundary

- R016 retry1 completed 50 validation epochs with best val/mIoU `0.541121` at validation epoch `49`, improving over the R015 official-label baseline `0.537398` by `+0.003723`.
- This is the strongest current full-train result, but it is still an official-contract baseline correction, not a new method contribution.
- What R016 borrows from DFormer: the modal_x/depth preprocessing protocol `raw / 255.0`, then mean `[0.48, 0.48, 0.48]` and std `[0.28, 0.28, 0.28]`.
- Paper phrasing boundary: write this as official DFormer preprocessing alignment for NYUDepthV2 and cite DFormer. Do not claim label remapping or depth normalization as a proposed method.
- The corrected baseline is now `dformerv2_mid_fusion` with official label and depth contracts, best val/mIoU `0.541121`.
- The remaining paper gap to `0.56` is about `0.018879`; subsequent novelty must be built on top of this corrected baseline.
- The first R016 launch was interrupted by a Windows/Intel `forrtl` window-close event after 47 validation epochs and is not the official result.

## 2026-05-14 R016 Depth Normalization Contract Boundary

- R015 established the official-label baseline coordinate system with best val/mIoU `0.537398`, but late instability remains and the result is still below the final `0.56` target.
- R016 is the next contract-alignment run: official DFormer uses modal_x/depth normalization with mean `[0.48, 0.48, 0.48]` and std `[0.28, 0.28, 0.28]`, after raw `/255.0`.
- Local R016 changes only depth preprocessing: depth no longer receives RGB ImageNet Normalize through Albumentations; RGB normalization, label mapping, split, loader behavior, model, optimizer, scheduler, batch, epoch, lr, and evaluation remain unchanged.
- Interpret R016 only against the R015 official-label baseline. It is not a new architecture and should not be mixed with old label-contract runs as a direct gain claim.
- If R016 improves or stabilizes late validation, the next paper direction remains official DFormerv2-S contract alignment. If it regresses, keep R015 as the official-label baseline and investigate the next highest contract mismatch.

## 2026-05-14 R015 Official Label Contract Boundary

- R014 failed to reach `0.53`, so the loop pivots away from PMAD/TGGA micro-search.
- The next experiment is a contract-reset baseline, not a new architecture: official DFormer NYU label mapping `0 -> 255 ignore`, `1..40 -> 0..39`.
- R015 completed 50 validation epochs with best val/mIoU `0.537398` at validation epoch `45` and last val/mIoU `0.499418`.
- This satisfies the fixed-recipe `0.53` stage target with TensorBoard/checkpoint evidence, but it changes label/ignore semantics and historical mIoU comparability. Report it as the first official-label baseline, not as a direct improvement over the old `0.517397` baseline.
- Because the best-to-last drop is `0.037981`, R015 is not a final stable `0.56` result. The paper direction should continue from official contract alignment: first depth normalization, then RGB/BGR channel contract, before revisiting architecture.
- Later experiments should compare against R015 and not mix old-contract and official-contract numbers in one leaderboard claim.

## 2026-05-14 Mainline Cleanup Boundary

- Active mainline is now restricted to the clean baseline, PMAD logit-only, geometry-primary teacher, and TGGA c4-only reuse unit.
- TGGA c3/c4, weak-c3 TGGA, R013 LMLP, hard filtering KD, global frequency/FFT modules, hard losses, and plain decoder swaps are archived or inactive.
- This cleanup does not change any metric claim. It only prevents negative or unstable experiment code from being treated as active method candidates.
- R014 remains the next controlled fixed-recipe experiment: PMAD logit-only w0.15/T4 plus TGGA c4-only. If R014 fails, the next paper-relevant direction should be DFormerv2-S official structure/geometry-prior contract alignment, not more PMAD/TGGA parameter search.

## 2026-05-13 LMLP Decoder Boundary

- `dformerv2_lmlp_decoder_run01` completed 50 validation epochs.
- Setting: DFormer/SegFormer-style c2-c4 LMLP decoder head with the same DFormerv2_S, DepthEncoder, GatedFusion, CE loss, data/eval path, and fixed training recipe.
- Best val/mIoU is `0.517981` at epoch 41; last val/mIoU is `0.490231`.
- Clean ten-run baseline mean best is `0.517397`, std `0.004901`, mean + 1 std `0.522298`, best single `0.524425`.
- R010 PMAD run06_retry1 best is `0.527469`; R004 TGGA c4-only best is `0.522849`.
- Delta vs clean baseline mean is `+0.000584`, equal to `+0.119` baseline std units.
- Delta vs baseline mean + 1 std is `-0.004317`; delta vs R010 is `-0.009488`; delta vs R004 is `-0.004868`.
- Best-to-last drop is `0.027750`, so the run has meaningful late instability.
- Interpretation: **weak near-baseline, negative for the goal.** LMLP is not a stable improvement and should not be promoted as the main decoder.
- Paper boundary: cite only as a decoder negative/neutral ablation. It can support the claim that simply replacing SimpleFPN with a SegFormer-style c2-c4 MLP head is insufficient under the fixed recipe.
- Strategic implication: plain decoder-head replacement is unlikely to close the gap to `0.53` by itself. Stronger next directions likely need either a carefully bounded c4 reliability/fusion mechanism or a different teacher/distillation approach; any broader training recipe/backbone change would require explicit contract change.

## 2026-05-13 PMAD Logit-Only Run07 Repeat Boundary

- `dformerv2_primkd_logit_only_w015_t4_run07` completed 50 validation epochs.
- Setting: PMAD logit-only KD with `kd_weight=0.15`, `kd_temperature=4.0`, frozen geometry-primary teacher checkpoint `dformerv2_geometry_primary_teacher-epoch=37-val_mIoU=0.5168.pt`, and student-only checkpoint saving.
- Best val/mIoU is `0.516967` at epoch 43; last val/mIoU is `0.508205`.
- Clean ten-run baseline mean best is `0.517397`, std `0.004901`, mean + 1 std `0.522298`, best single `0.524425`.
- Prior PMAD logit-only w0.15/T4 five-run mean best was `0.520795`; R010 run06_retry1 best was `0.527469`.
- Delta vs clean baseline mean is `-0.000430`, equal to `-0.088` baseline std units.
- Delta vs prior PMAD five-run mean is `-0.003828`; delta vs R010 run06_retry1 is `-0.010502`.
- Gap to the `0.53` goal is `0.013033`.
- Updated PMAD w0.15/T4 seven-run mean best is `0.521201` with population std `0.004148`; `5/7` runs exceed the clean baseline mean, `4/7` exceed baseline mean + 1 std, and only `1/7` exceeds the clean baseline best single.
- Interpretation: **negative repeat.** R010 remains a real high-tail partial positive, but R012 does not reproduce that peak and falls slightly below the clean baseline mean.
- Paper boundary: PMAD logit-only w0.15/T4 should be described as a marginal and high-variance candidate, not a stable improvement and not a goal-completing method. Do not claim the project target is met.
- Strategic implication: stop blind repeats of the same PMAD setting. Future PMAD work needs a qualitatively different mechanism; otherwise the next experiment should pivot to a distinct decoder/head hypothesis.

## 2026-05-13 PMAD Logit-Only Run06 Repeat Boundary

- `dformerv2_primkd_logit_only_w015_t4_run06_retry1` completed 50 validation epochs.
- Setting: PMAD logit-only KD with `kd_weight=0.15`, `kd_temperature=4.0`, frozen geometry-primary teacher checkpoint `dformerv2_geometry_primary_teacher-epoch=37-val_mIoU=0.5168.pt`, and student-only checkpoint saving.
- Best val/mIoU is `0.527469` at epoch 49; last val/mIoU is `0.526316`.
- Clean ten-run baseline mean best is `0.517397`, std `0.004901`, mean + 1 std `0.522298`, best single `0.524425`.
- Prior PMAD logit-only w0.15/T4 five-run mean best was `0.520795`, with prior best single `0.524028`.
- Delta vs clean baseline mean is `+0.010072`, equal to `+2.055` baseline std units.
- Delta vs clean baseline best single is `+0.003044`; delta vs prior PMAD five-run mean is `+0.006674`; delta vs prior PMAD best single is `+0.003441`.
- Gap to the `0.53` goal is `0.002531`.
- Updated PMAD w0.15/T4 six-run mean best is `0.521907` with population std `0.004073`; `5/6` runs exceed the clean baseline mean and `4/6` exceed baseline mean + 1 std.
- Interpretation: **strong partial positive, not success.** The repeat confirms PMAD logit-only as the most durable marginal-positive direction so far, but it still does not satisfy the goal.
- Paper boundary: PMAD logit-only can be described as a useful ablation/candidate and this run can be cited as the best single repeat, but do not claim the project target is met or that the method is stable at `>=0.53`.
- Strategic implication: further PMAD filtering/feature-hint variants have already weakened the signal; the next experiment should pivot to a distinct training-only loss or other minimal hypothesis rather than another c4 feature KD/gate.

## 2026-05-13 TGGA Weak-C3 + C4 Boundary

- `dformerv2_tgga_c34_weakc3_beta001_c4beta002_aux003_detachsem_v1_run01` completed 50 validation epochs.
- Setting: TGGA weak c3 with `beta_init=0.01`, `beta_max=0.05`, `gate_bias_init=-3.0`, plus c4 with `beta_init=0.02`, `beta_max=0.1`, `gate_bias_init=-2.0`, aux CE weight `0.03` on both TGGA heads.
- Best val/mIoU is `0.518253` at epoch 43; last val/mIoU is `0.514908`.
- Clean ten-run baseline mean best is `0.517397`, std `0.004901`, mean + 1 std `0.522298`, best single `0.524425`.
- PMAD logit-only w0.15/T4 five-run mean best remains `0.520795`.
- R004 TGGA c4-only best is `0.522849`.
- Delta vs clean baseline mean is `+0.000856`, equal to `+0.175` baseline std units.
- Delta vs baseline mean + 1 std is `-0.004045`; delta vs PMAD mean is `-0.002542`; delta vs R004 c4-only is `-0.004596`.
- Diagnostic: final c3 gate opens substantially despite the weaker setup (`gate_c3_mean=0.293138`, `gate_c3_std=0.331622`), while c4 stays conservative (`gate_c4_mean=0.131140`, `gate_c4_std=0.012837`).
- Interpretation: **not a useful improvement.** Weak-c3 is only barely above the clean baseline mean and is clearly worse than c4-only TGGA.
- Paper boundary: do not cite weak-c3 TGGA as an improvement. It can be cited only as a negative TGGA diagnostic showing that reintroducing c3, even conservatively, does not recover R004's stronger signal.
- Strategic implication: pause the TGGA branch for external review. The current durable lesson is that c4-only is the safer TGGA signal and c3 remains risky.

## 2026-05-13 TGGA C4-Only Diagnostic Boundary

- `dformerv2_tgga_c4only_beta002_aux003_detachsem_simplefpn_v1_run01` completed 50 validation epochs.
- Setting: TGGA only on c4 with `beta_c4_init=0.02`, `aux_weight=0.03`, detached semantic cue, `DFormerv2_S + ResNet-18 DepthEncoder + GatedFusion + SimpleFPNDecoder`.
- Best val/mIoU is `0.522849` at epoch 42; last val/mIoU is `0.509320`.
- Clean ten-run baseline mean best is `0.517397`, std `0.004901`, mean + 1 std `0.522298`, best single `0.524425`.
- PMAD logit-only w0.15/T4 five-run mean best remains `0.520795`.
- Delta vs clean baseline mean is `+0.005452`, equal to `+1.112` baseline std units.
- Delta vs clean baseline mean + 1 std is `+0.000551`; delta vs clean baseline best single is `-0.001576`.
- Delta vs PMAD w0.15/T4 mean is `+0.002054`.
- Diagnostic: final `gate_c4_mean=0.130742`, `gate_c4_std=0.014394`, and `tgga_beta_c4=0.022874`; the c4 gate is conservative but gradually opens.
- Late-curve caveat remains: best epoch 42 falls to final `0.509320`, a best-to-last drop of `0.013529`.
- Interpretation: **strong single-run diagnostic signal, not a success claim.** Removing c3 improves the TGGA diagnostic relative to recent failed loop runs and slightly exceeds baseline mean + 1 std, but it does not reach `0.53` and does not beat the best single clean baseline.
- Paper boundary: do not cite c4-only TGGA as a stable improvement. It can be cited only as a promising diagnostic showing that c4-level semantic/geometry calibration is safer than the original c3/c4 high-resolution gate.
- Strategic implication: the next TGGA test should be a narrow follow-up around c4-safe calibration, not a repeat of failed PMAD filtering or the R002 frequency-aware decoder.

## 2026-05-13 Correct-and-Entropy-Selective PMAD Boundary

- `dformerv2_primkd_correct_entropy_w015_t4_h025_run01` completed 50 validation epochs.
- Setting: PMAD logit-only KD with `kd_weight=0.15`, `kd_temperature=4.0`, normalized teacher entropy threshold `0.25`, and KD only where `teacher_argmax == label` during training.
- Best val/mIoU is `0.516597` at epoch 50; last val/mIoU is also `0.516597`.
- Clean ten-run baseline mean best is `0.517397`, std `0.004901`, mean + 1 std `0.522298`.
- PMAD logit-only w0.15/T4 five-run mean best remains `0.520795`.
- Delta vs clean baseline mean is `-0.000800`, equal to `-0.163` baseline std units.
- Delta vs PMAD w0.15/T4 mean is `-0.004198`.
- Diagnostic: final `kd_mask_ratio` is `0.910636`, much more selective than R001's `0.998182`; final `kd_teacher_selected_acc` is `1.000000`.
- Interpretation: **near-baseline but negative result.** The trust gate is meaningful, but it does not improve over the clean baseline or the original PMAD repeated mean.
- Paper boundary: do not cite `dformerv2_primkd_correct_entropy` as an improvement. It can be cited only as a negative PMAD filtering ablation showing that removing teacher-wrong, high-entropy pixels does not preserve the original PMAD gain.
- Strategic implication: stricter PMAD pixel filtering is unlikely to be the path to `>=0.53`; future PMAD work should require a different mechanism, not another small threshold variant.

## 2026-05-13 Frequency-Aware FPN Decoder Boundary

- `dformerv2_freqfpn_decoder_run01` completed 50 validation epochs.
- Setting: same DFormerv2_S, DepthEncoder, and GatedFusion baseline path, but `SimpleFPNDecoder` is replaced by a frequency-aware top-down FPN decoder in this separate model entry.
- Best val/mIoU is `0.516915` at epoch 44; last val/mIoU is `0.486524`.
- Clean ten-run baseline mean best is `0.517397`, std `0.004901`, mean + 1 std `0.522298`.
- PMAD logit-only w0.15/T4 five-run mean best remains `0.520795`.
- Delta vs clean baseline mean is `-0.000482`, equal to `-0.098` baseline std units.
- Delta vs PMAD w0.15/T4 mean is `-0.003880`.
- Interpretation: **neutral/negative result.** The decoder nearly matches the baseline mean at its best checkpoint but does not improve it and has a large late drop.
- Paper boundary: do not cite `dformerv2_freqfpn_decoder` as an improvement. It can be cited only as a neutral/negative decoder ablation showing that this FreqFusion-style top-down approximation is insufficient.
- Strategic implication: do not repeat this exact decoder unchanged; the next loop should pivot to a different hypothesis rather than further tune this decoder without a clearer diagnostic.

## 2026-05-13 PMAD Boundary/Confidence-Selective KD Boundary

- `dformerv2_primkd_boundary_conf_w015_t4_run01` completed 50 validation epochs.
- Setting: PMAD logit-only KD with `kd_weight=0.15`, `kd_temperature=4.0`, confidence threshold `0.40`, confidence power `1.5`, and semantic-boundary boost `1.0`.
- Best val/mIoU is `0.511646` at epoch 50; last val/mIoU is also `0.511646`.
- Clean ten-run baseline mean best is `0.517397`, std `0.004901`, mean + 1 std `0.522298`.
- PMAD logit-only w0.15/T4 five-run mean best remains `0.520795`.
- Delta vs clean baseline mean is `-0.005751`, equal to `-1.173` baseline std units.
- Delta vs PMAD w0.15/T4 mean is `-0.009149`.
- Diagnostic: `kd_mask_ratio` stayed near `0.998` and final `kd_mask_ratio` is `0.998182`; the threshold `0.40` did not meaningfully filter uncertain pixels.
- Interpretation: **negative result.** This exact boundary/confidence KD setting weakens PMAD rather than stabilizing it.
- Paper boundary: do not cite `dformerv2_primkd_boundary_conf` as an improvement. It can be cited only as a negative PMAD refinement showing that weak confidence weighting plus boundary boosting does not improve the repeated PMAD signal.
- Strategic implication: if continuing PMAD, the next test must be genuinely selective, not a near-all-pixel mask; otherwise the loop should pivot to another high-value direction.

## 2026-05-12 TGGA No-Aux Run01 Boundary

- `dformerv2_tgga_c34_noaux_semgrad_beta002_simplefpn_v1_run01` completed 50 validation epochs.
- Setting: TGGA c3/c4 with final CE only; no auxiliary CE; semantic cue trained through final CE via the gate path.
- Best val/mIoU is `0.512152` at epoch 48; last val/mIoU is `0.492633`.
- Clean ten-run baseline mean best is `0.517397`, std `0.004901`, mean + 1 std `0.522298`.
- Delta vs clean baseline mean is `-0.005245`, equal to `-1.070` baseline std units.
- Delta vs PMAD logit-only w0.15 mean is `-0.008643`.
- Delta vs original TGGA aux run01 is `-0.010054`; delta vs original TGGA aux run02 is `-0.005285`.
- No epoch exceeds the clean baseline mean.
- Late-curve caveat remains: epoch 48 best `0.512152`, then epoch 49 drops to `0.469468`, final epoch `0.492633`.
- Gate diagnostics: final c3 gate is very open (`gate_c3_mean=0.474472`, `gate_c3_std=0.311781`), and c4 opens much more than in original TGGA (`gate_c4_mean=0.230513`, `gate_c4_std=0.135297`).
- Interpretation: **negative diagnostic.** Removing auxiliary CE removes the high peak but does not remove late instability. Aux CE is not the only problem; TGGA gate/residual dynamics are not safe enough in this form.
- Paper boundary: do not cite no-aux TGGA as an improvement. Use it only as diagnostic evidence that the TGGA line is unstable and that the original TGGA peak depends partly on auxiliary CE.

## 2026-05-12 TGGA No-Aux Diagnostic Boundary

- Implemented `dformerv2_tgga_c34_noaux_semgrad_beta002_simplefpn_v1` as the next diagnostic after TGGA run01-run02.
- Purpose: test whether late collapse comes from auxiliary semantic CE or from TGGA gate/residual dynamics.
- Run01 is negative: best val/mIoU `0.512152`, below the clean baseline mean by `0.005245`.
- Boundary nuance: because the original `detachsem` semantic cue would become untrained if aux CE were simply removed, this diagnostic uses semantic-gradient gating and final CE only.
- Paper boundary: no-aux TGGA is diagnostic-only and cannot be cited as effective.

## 2026-05-12 TGGA Run02 and Two-Run Boundary

- `dformerv2_tgga_c34_beta002_aux003_detachsem_simplefpn_v2_run02` completed 50 validation epochs.
- Run02 best val/mIoU is `0.517437` at epoch 49; last val/mIoU is `0.486566`.
- Run02 is only `+0.000040` above the clean 10-run baseline mean `0.517397`, below baseline mean + 1 std `0.522298`, below baseline best single `0.524425`, and below PMAD logit-only w0.15 5-run mean `0.520795`.
- Run01-run02 mean best val/mIoU is `0.519822`, which is `+0.002425` over the clean baseline mean but only `+0.495` baseline std units and still `-0.000973` below PMAD w0.15.
- Both runs collapse late: run01 final `0.489865`; run02 final `0.486566`.
- Run02 c3 TGGA opens more than run01: final `gate_c3_mean=0.409689`, `gate_c3_std=0.346383`; c4 remains weak and low variance with `gate_c4_mean=0.126628`, `gate_c4_std=0.010253`.
- Interpretation: **weak positive but unstable.** TGGA c3/c4 has a real late-epoch signal, but the repeat does not support a stable paper improvement claim.
- Paper boundary: do not report TGGA c3/c4 as an improved method. It can be discussed only as an unstable structure-side candidate pending diagnostics.
- Next paper-relevant diagnostic: `dformerv2_tgga_c34_noaux_semgrad_beta002_simplefpn_v1`.

## 2026-05-12 TGGA Diagnostic Variant Boundary

- Original TGGA c3/c4 run01 remains promising but unstable: best `0.522206` at epoch 48, final `0.489865`.
- Run02 weakens the claim: best `0.517437` at epoch 49, final `0.486566`, with the same late-collapse pattern.
- Implemented `dformerv2_tgga_c4only_beta002_aux003_detachsem_simplefpn_v1` as a diagnostic variant to isolate whether c3 high-resolution residual/gate noise contributes to late collapse.
- Implemented `dformerv2_tgga_c34_weakc3_beta001_c4beta002_aux003_detachsem_v1` as a diagnostic variant to preserve c3 capacity with weaker c3 residual strength and a more conservative c3 gate.
- Neither variant has an mIoU result yet, so neither can be cited as effective.
- TGGA+PMAD remains a conditional future experiment only; it is not implemented as an active mainline here.

## 2026-05-12 Active/Archived Boundary After Cleanup

- Active main baseline: `dformerv2_mid_fusion`.
- Active unstable structure candidate: `dformerv2_tgga_c34_beta002_aux003_detachsem_simplefpn_v2`.
- Active marginal-positive KD candidate: `dformerv2_primkd_logit_only`.
- Active PMAD dependency: `dformerv2_geometry_primary_teacher`.
- Archived / deprecated as positive claims: DGBF, CGPC, SGBR-Lite, CGCD/ClassContext, context decoder/PPM, FFT freq enhance, FFT HiLo, depth FFT select, CE+Dice, and FreqCov-style auxiliary losses.
- TGGA run01-run02 is weak positive but unstable: mean best `0.519822`, mean final `0.488215`; do not claim stable improvement.
- PMAD logit-only remains marginal positive, not a strong main result.
- CGCD remains seed-sensitive; DGBF/CGPC/SGBR/FFT-style branches are negative or unstable ablations only.

## 2026-05-12 TGGA C3/C4 Run01 Boundary

- `dformerv2_tgga_c34_beta002_aux003_detachsem_simplefpn_v2_run01` completed 50 validation epochs.
- Setting: `DFormerv2_S + TGGA(c3,c4) + ResNet-18 DepthEncoder + GatedFusion + SimpleFPNDecoder`, with `0.03` auxiliary CE on each TGGA semantic head.
- Best val/mIoU is `0.522206` at epoch 48; last val/mIoU is `0.489865`.
- Clean ten-run RGB-D baseline mean best is `0.517397`, std `0.004901`, mean + 1 std `0.522298`, best single `0.524425`.
- Delta vs clean baseline mean is `+0.004809`, equal to `+0.981` baseline std units.
- Delta vs baseline mean + 1 std is `-0.000092`; delta vs baseline best single is `-0.002219`.
- Compared with PMAD logit-only w0.15 5-run mean `0.520795`, TGGA run01 is higher by `0.001411`.
- Compared with bounded CGCD 5-run mean `0.515986`, TGGA run01 is higher by `0.006220`.
- Late-curve caveat: epoch 48 is a late high point, followed by `0.516722` at epoch 49 and `0.489865` at epoch 50. The final-epoch value is poor.
- TGGA diagnostics: final `tgga_beta_c3=0.035080`, `tgga_beta_c4=0.023389`; c3 gate opens substantially (`gate_c3_mean=0.351956`, `gate_c3_std=0.305273`) while c4 remains conservative (`gate_c4_mean=0.131228`).
- Interpretation: **promising single-run candidate, not stable yet.** TGGA is the first recent structure-side experiment to nearly reach the baseline mean + 1 std threshold and outperform PMAD's five-run mean in a single run. However, the gain is late and unstable, so it must be repeated before being treated as a main result.
- Paper boundary: this run01-only boundary is superseded by the run01-run02 boundary above; do not claim TGGA c3/c4 as a stable improvement.

## 2026-05-12 Bounded Class Context Decoder 5-Run Boundary

- `dformerv2_class_context_decoder_bounded_a02_run01` through `run05` completed 50 validation epochs each.
- Setting: `DFormerv2_S + ResNet-18 DepthEncoder + GatedFusion + ClassContextFPNDecoder`, with `class_context_channels=64`, `class_context_aux_weight=0.2`, `class_context_alpha_init=0.1`, and bounded `class_context_alpha_max=0.2`.
- Five-run mean best val/mIoU is `0.515986`, population std `0.005208`.
- Clean ten-run RGB-D baseline mean best is `0.517397`, std `0.004901`, mean + 1 std `0.522298`, best single `0.524425`.
- Mean delta vs clean baseline mean is `-0.001411`, equal to `-0.288` baseline std units.
- Compared with PMAD logit-only w0.15 5-run mean `0.520795`, bounded class context is lower by `0.004809`.
- Runs above baseline mean: `2/5`; runs above baseline mean + 1 std: `1/5`; runs above baseline best single: `1/5`.
- Best single run is `0.525156` (`run01`), but the other four runs are `0.511353`, `0.511318`, `0.514017`, and `0.518087`.
- Bounded alpha keeps the final context alpha stable around `0.134-0.135`, so it fixes the runaway-alpha symptom observed in the unbounded run.
- Interpretation: **not a stable improvement.** The OCR-style class-context decoder can produce a high single run, but repeated-run statistics fall below the clean baseline mean. The main problem is no longer alpha explosion; it is seed sensitivity / inconsistent benefit from the class-context decoder.
- Paper boundary: do not cite bounded CGCD as an improved method. It can be used as an ablation showing that class-context decoder refinement has occasional positive signal but does not reliably outperform the DFormerv2 mid-fusion baseline.

## 2026-05-12 Class Context Decoder Run01 Boundary

- `dformerv2_class_context_decoder_run01` completed 50 validation epochs.
- Setting: `DFormerv2_S + ResNet-18 DepthEncoder + GatedFusion + ClassContextFPNDecoder`, with `class_context_channels=64`, `class_context_aux_weight=0.2`, and `class_context_alpha_init=0.1`.
- Best val/mIoU is `0.519807` at epoch 46; last val/mIoU is `0.503151`.
- Clean ten-run RGB-D baseline mean best is `0.517397`, std `0.004901`, mean + 1 std `0.522298`, best single `0.524425`.
- Delta vs clean baseline mean is `+0.002410`, equal to `+0.492` baseline std units.
- Compared with PMAD logit-only w0.15 5-run mean `0.520795`, this run is lower by `0.000988`.
- Compared with CGPC c3 best `0.515838` and CGPC c4 best `0.512659`, this run is higher by `0.003969` and `0.007148`.
- The late curve is unstable: mean val/mIoU over the last 10 epochs is `0.501142`, and epoch 47 drops sharply to `0.433597`.
- Interpretation: **marginal positive but unstable decoder result.** The OCR-style class-context decoder is healthier than recent class-guided contrastive losses and beats the clean baseline mean in one run, but it does not cross the baseline mean + 1 std threshold.
- Paper boundary: do not cite as a stable improvement yet. It can be kept as a promising single-run decoder ablation pending repeat or a stability-focused follow-up.

## 2026-05-11 CGPC C4 Run01 Boundary

- `dformerv2_mid_fusion_cgpc_w001_t01_c4_detach_run01` completed 50 validation epochs.
- Setting: unchanged `dformerv2_mid_fusion` architecture with class-guided prototype contrastive loss on fused c4, `cgpc_weight=0.01`, `temperature=0.1`, detached prototypes.
- Best val/mIoU is `0.512659` at epoch 49; last val/mIoU is `0.507206`.
- Clean ten-run RGB-D baseline mean best is `0.517397`, std `0.004901`, mean + 1 std `0.522298`, best single `0.524425`.
- Delta vs clean baseline mean is `-0.004738`, equal to `-0.967` baseline std units.
- Compared with CGPC c3 best `0.515838`, c4 is lower by `0.003179`.
- CGPC diagnostics are active but lower-coverage than c3: final `cgpc_num_classes=9.790932`, `cgpc_num_queries=508.498749`, and `cgpc_loss` decreases from `0.884067` to `0.364393`.
- Interpretation: **negative stage ablation.** Moving prototype contrast from c3 to the more semantic c4 stage did not improve segmentation quality and performed worse than c3.
- Paper boundary: do not cite CGPC c4 as an improvement. It can be used as a negative stage ablation showing that simple batch-local class prototype contrast is not sufficient for this DFormerV2 mid-fusion baseline.

## 2026-05-11 CGPC Run01 Boundary

- `dformerv2_mid_fusion_cgpc_w001_t01_c3_detach_run01` completed 50 validation epochs.
- Setting: unchanged `dformerv2_mid_fusion` architecture with class-guided prototype contrastive loss on fused c3, `cgpc_weight=0.01`, `temperature=0.1`, detached prototypes.
- Best val/mIoU is `0.515838` at epoch 50; last val/mIoU is also `0.515838`.
- Clean ten-run RGB-D baseline mean best is `0.517397`, std `0.004901`, mean + 1 std `0.522298`, best single `0.524425`.
- Delta vs clean baseline mean is `-0.001559`, equal to `-0.318` baseline std units.
- CGPC diagnostics are healthy: final `cgpc_num_classes=13.440806`, `cgpc_num_queries=1120.790894`, and `cgpc_loss` decreases from `0.968080` to `0.372201`.
- Interpretation: **neutral-to-negative first CGPC result.** Unlike DGBF, the auxiliary signal is active and well-sampled, but it still does not improve the final segmentation metric over CE. This suggests the simple batch-local c3 prototype contrast is not enough to add class-discriminative value to the already strong fused representation.
- Paper boundary: do not cite as improvement. It can be discussed as a label-guided auxiliary loss attempt that is healthier than generic alignment but still below baseline.

## 2026-05-11 DGBF Run01 Boundary

- `dformerv2_mid_fusion_dgbf_a1_g2_depthsem_run01` completed 50 validation epochs.
- Setting: unchanged `dformerv2_mid_fusion` architecture with output-level `DGBFLoss`, `alpha=1.0`, `gamma=2.0`, `mode=depth_semantic`.
- Best val/mIoU is `0.513194` at epoch 49; last val/mIoU is `0.443671`.
- Clean ten-run RGB-D baseline mean best is `0.517397`, std `0.004901`, mean + 1 std `0.522298`, best single `0.524425`.
- Delta vs clean baseline mean is `-0.004203`, equal to `-0.857` baseline std units.
- DGBF diagnostics show that the effective weighting is tiny on average: final `dgbf_boundary_mean=0.002385` and `dgbf_weight_mean=1.000793`.
- Interpretation: **negative first DGBF result.** The depth-semantic boundary weighting is too sparse/weak in this configuration and mostly behaves like CE, while still adding instability near the end of training.
- Paper boundary: do not cite this as an improvement. It can be used as a negative ablation showing that naive output-level depth-boundary weighting is insufficient for this architecture.

## 2026-05-11 PMAD Logit-Only KD 5-Run Summary Boundary

- `dformerv2_primkd_logit_only` at `kd_weight=0.15`, `kd_temperature=4.0` has now been run 5 times.
- run01: best `0.522998` at epoch 48, delta vs baseline mean `+0.005601` (+1.14 std)
- run02: best `0.520144` at epoch 47, delta vs baseline mean `+0.002747` (+0.56 std)
- run03: best `0.522600` at epoch 41, delta vs baseline mean `+0.005203` (+1.06 std)
- run04: best `0.524028` at epoch 45, delta vs baseline mean `+0.006631` (+1.35 std)
- run05: best `0.514204` at epoch 41, delta vs baseline mean `-0.003193` (-0.65 std)
- 5-run mean best: `0.520795`, 5-run std: `0.003799`
- 5-run mean delta vs clean baseline mean: `+0.003398` (+0.69 baseline std units)
- 5-run mean delta vs baseline mean + 1 std: `-0.001503`
- 5-run best single: `0.524028` (run04), nearly matches baseline best single `0.524425`
- Runs above baseline mean: 4/5; runs above baseline mean + 1 std: 2/5
- KD weight ablation: w=0.10 (0.5101, -1.50 std), w=0.15 (0.5208, +0.69 std), w=0.175 (0.5182, +0.16 std), w=0.20 (0.5145, -0.60 std)
- Interpretation: **marginal positive candidate.** The 5-run mean beats the baseline mean by +0.69 std but does not cross the mean + 1 std threshold. 4/5 runs are positive, with one outlier (run05) pulling the mean down. The signal is directionally positive but not strong enough for a definitive claim. w=0.15 is clearly the best KD weight; higher weights show diminishing returns.
- Paper boundary: PMAD logit-only KD at w=0.15 can be reported as a marginal positive ablation study. It demonstrates that geometry-primary teacher distillation with conservative logit KD can modestly improve RGB-D segmentation. Do not claim as a strong main result. The contribution is the KD framework and the ablation finding, not a state-of-the-art number.
- Strategic implication: PMAD logit-only is sufficient. Do not add feature KD. The paper can include this as an ablation showing the effect of KD weight on segmentation quality, with w=0.15 as the recommended setting.

## 2026-05-10 PMAD Logit-Only KD Weight 0.15 Run02 Boundary

- `dformerv2_primkd_logit_only_w015_t4_run02` completed 50 validation epochs.
- Setting: student = full `dformerv2_mid_fusion`; teacher = frozen geometry-primary teacher; logit-only KD with `kd_weight=0.15`, `kd_temperature=4.0`.
- Best val/mIoU is `0.520144` at epoch 47; last val/mIoU is `0.498182`.
- Clean ten-run RGB-D baseline mean best is `0.517397`, std `0.004901`, mean + 1 std `0.522298`, best single `0.524425`.
- Delta vs clean baseline mean is `+0.002747`, equal to `+0.560` baseline std units.
- Delta vs the previous `kd_weight=0.15` PMAD run is `-0.002854`; delta vs `kd_weight=0.10` run is `+0.010075`.
- Current `kd_weight=0.15` two-run mean best is `0.521571`, with population std `0.001427`.
- Interpretation: **positive but not strong repeat.** This supports that `kd_weight=0.15` is healthier than `kd_weight=0.10`, but it does not yet prove stable improvement over the clean baseline because the two-run mean remains slightly below the mean + 1 std threshold.
- Paper boundary: can continue treating PMAD logit-only as a candidate direction. Do not claim stable superiority yet; wait for `run03`.
- Strategic implication: run `dformerv2_primkd_logit_only_w015_t4_run03` before feature KD. Feature KD should only be considered if the 3-run mean remains positive.

## 2026-05-10 PMAD Logit-Only KD Weight 0.10 Boundary

- `dformerv2_primkd_logit_only_w010_t4_run01` completed 50 validation epochs.
- Setting: student = full `dformerv2_mid_fusion`; teacher = frozen geometry-primary teacher; logit-only KD with `kd_weight=0.10`, `kd_temperature=4.0`.
- Best val/mIoU is `0.510068` at epoch 50; last val/mIoU is also `0.510068`.
- Clean ten-run RGB-D baseline mean best is `0.517397`, std `0.004901`, mean + 1 std `0.522298`, best single `0.524425`.
- Delta vs clean baseline mean is `-0.007329`, equal to `-1.495` baseline std units.
- Delta vs the previous `kd_weight=0.15` PMAD run is `-0.012930`.
- Interpretation: **negative KD-weight ablation.** The `0.10` KD weight is not a safer version of PMAD; it substantially underperforms both the baseline and the `0.15` PMAD single run.
- Paper boundary: do not cite `kd_weight=0.10` as a promising setting. It can be used only as an ablation showing that PMAD is sensitive to KD strength and that the initial `0.15` result is not automatically robust across nearby weights.
- Strategic implication: do not repeat `0.10` and do not add feature KD on top of it. The remaining decision-value choices are `kd_weight=0.20` or repeating the best `0.15` setting before any feature-KD work.

## 2026-05-10 PMAD Logit-Only Run01 Boundary

- `dformerv2_primkd_logit_only_w015_t4_run01` completed 50 validation epochs.
- Setting: student = full `dformerv2_mid_fusion`; teacher = frozen geometry-primary teacher `DFormerv2_S(rgb, real_depth) + SimpleFPNDecoder`; logit-only KD with `kd_weight=0.15`, `kd_temperature=4.0`.
- Best val/mIoU is `0.522998` at epoch 48; last val/mIoU is `0.513176`.
- Clean ten-run RGB-D baseline mean best is `0.517397`, std `0.004901`, mean + 1 std `0.522298`, best single `0.524425`.
- Delta vs clean baseline mean is `+0.005601`, equal to `+1.143` baseline std units.
- Delta vs clean baseline mean + 1 std is `+0.000700`; delta vs clean baseline best single is `-0.001427`.
- Interpretation: **positive single-run signal.** This is the first PMAD logit-only run and it crosses the pre-defined strong-signal threshold, but it is not yet a stable improvement because there is no repeat evidence.
- Paper boundary: can be described as a promising candidate result, not as a final method claim. Do not claim stable improvement until at least 3-run mean confirms it.
- Strategic implication: keep PMAD logit-only as the active main direction. Next test only small KD-weight ablations (`0.10`, `0.20`) or repeat the best setting; do not add feature KD yet.

## 2026-05-10 Geometry-Primary Teacher Sanity Boundary

- `dformerv2_geometry_primary_teacher_run01` completed 50 validation epochs as the revised Phase 0 teacher sanity check for PMAD / PrimKD.
- The model uses `Dformerv2_S(rgb, real_depth) + SimpleFPNDecoder`, preserving DFormerV2's internal depth geometry prior while removing the extra `DepthEncoder + GatedFusion` branch.
- Best val/mIoU is `0.516824` at epoch 38; last val/mIoU is `0.509223`.
- Clean ten-run RGB-D baseline mean best remains `0.517397`, std `0.004901`; the minimum teacher usability gate is `0.515000`.
- Comparison to failed constant-zero teacher: `0.516824` vs `0.488489`, delta `+0.028335`.
- Interpretation: **usable teacher, not strong teacher.** It passes the `0.515` gate but does not exceed the strong-teacher threshold `0.522298`.
- Paper boundary: this can support entering PMAD Phase 1, but it is not a main result. The more important claim is that retaining real DFormerV2 geometry prior is necessary for a usable teacher.
- Strategic implication: proceed with logit-only PMAD using conservative `kd_weight=0.15`, `kd_temperature=4.0`. Do not repeat teacher unless PMAD later becomes promising.

## 2026-05-10 RGB-Only Teacher Sanity Boundary

- `dformerv2_rgb_teacher_constdepth_run01` completed 50 validation epochs as the Phase 0 teacher sanity check for PMAD / PrimKD.
- The model uses `Dformerv2_S + SimpleFPNDecoder` with constant zero depth inside the model, so it is an RGB-only teacher candidate with position-only DFormerV2 geometry prior.
- Best val/mIoU is `0.488489` at epoch 43; last val/mIoU is `0.456266`.
- Clean ten-run RGB-D baseline mean best remains `0.517397`, std `0.004901`; the minimum teacher usability gate is `0.515000`.
- Interpretation: **teacher sanity failed.** The teacher is `0.026511` below the minimum gate and `0.028908` below the RGB-D baseline mean.
- Paper boundary: do not cite PMAD as an active improvement based on this teacher. This run should be described as a failed teacher-sanity check if discussed at all.
- Strategic implication: PMAD is blocked by teacher quality, not yet by KD loss design. Do not run formal KD until a teacher reaches at least `0.515`.

## 2026-05-10 C4 PPM Context Decoder Run01 Boundary

- `dformerv2_context_decoder_c4ppm_run01` completed 50 validation epochs.
- Setting: C4 PPM context refinement before FPN lateral4, `pool_scales=(1,2,3,6)`, `alpha_init=0.1`, `loss_type=ce`, model = `DFormerV2 + DepthEncoder + GatedFusion + ContextFPNDecoder`.
- Best val/mIoU is `0.507293` at epoch 49; last val/mIoU is `0.507293`.
- Clean ten-run `dformerv2_mid_fusion` GatedFusion baseline mean best is `0.517397`, with population std `0.004901` and best single run `0.524425`.
- Delta vs the clean ten-run baseline mean is `-0.010104`; delta in baseline std units is `-2.062`.
- Interpretation: **clear negative result.** C4 PPM context decoder significantly underperforms the clean baseline. The PPM block does not help this architecture on NYUDepthV2. val/loss rises from 1.04 (epoch 11) to 1.27 (epoch 49) while train/loss decreases to 0.14, showing moderate overfitting.
- Paper boundary: **do not cite as improvement.** The decoder-side PPM context refinement is a negative ablation. This completes the decoder-side modification category — the SimpleFPNDecoder baseline is stronger than the PPM-enhanced variant.
- Strategic implication: **all tested modification categories have now failed to beat the clean baseline:**
  - Fusion replacement (DGC-AF, SA-Gate, PG-SparseComp, etc.): negative
  - FFT enhancement (freq_enhance, HiLo, depth_fft_select): not stable
  - Auxiliary loss (freqcov, maskrec, InfoNCE): negative
  - Loss recipe (CE + Dice): negative
  - Decoder context (C4 PPM): negative
- The GatedFusion + SimpleFPNDecoder baseline appears to be near-optimal for this architecture on NYUDepthV2. Further improvements likely require either a fundamentally different architectural approach or a different experimental setup (larger dataset, different backbone, different training recipe).
- Interpretation: **clear negative result.** CE + Dice at weight 0.5 significantly underperforms the pure CE baseline. The Dice loss causes train-val divergence and reduces final segmentation quality.
- Paper boundary: **do not cite as improvement.** CE + Dice at w=0.5 is harmful for this architecture. If pursuing loss experiments, test weaker Dice weight (0.1 or 0.2) or pure FocalLoss for class imbalance. Otherwise, the pure CE baseline is the stronger choice.
- Strategic implication: the GatedFusion baseline with pure CE loss remains the best-performing configuration. All tested modifications (fusion replacements, FFT variants, auxiliary losses, CE+Dice) have failed to produce stable improvements. The baseline appears to be near-optimal for this architecture on NYUDepthV2.

## 2026-05-09 FFT Freq Enhance 3-Run Summary Boundary

- `dformerv2_fft_freq_enhance` (cutoff=0.25, gamma=0.1) has now been run 3 times.
- run01: best val/mIoU `0.522688` at epoch 41, delta vs baseline mean `+0.005291`
- run02: best val/mIoU `0.5159` at epoch 38, delta vs baseline mean `-0.001497`
- run03: best val/mIoU `0.5145` at epoch 42, delta vs baseline mean `-0.002897`
- 3-run mean best val/mIoU: `0.517696`, 3-run std: `0.003664`
- 3-run mean delta vs clean 10-run baseline mean: `+0.000299` (0.061 baseline std)
- Interpretation: **not a stable improvement**. The 3-run mean is essentially identical to the baseline. run01 was a high-variance outlier; run02 and run03 both fell below baseline mean. Both run02 and run03 showed severe late collapse (run02: 0.5159→0.443, run03: 0.5145→0.489), while run01 was more stable (0.5227→0.5147).
- Paper boundary: **FFT freq_enhance is not a valid paper improvement.** The initial positive signal was statistical noise. Do not cite as improvement. Can be documented as a negative ablation showing that post-encoder FFT high-frequency enhancement does not consistently beat the GatedFusion baseline.
- Strategic implication: the FFT inference-path direction is now deprioritized. All inference-path modifications tested (FFT freq_enhance, FFT HiLo, depth FFT select) and all auxiliary losses (freqcov, maskrec, InfoNCE) have failed to produce stable gains. The GatedFusion baseline appears to be near-optimal for this architecture.

## 2026-05-09 FFT HiLo Enhancement Run01 Boundary

- `dformerv2_fft_hilo_enhance_w1111_c025_ah01_al003_am05_run01` completed 50 validation epochs.
- Setting: `cutoff_ratio=0.25`, `alpha_high_init=0.10`, `alpha_low_init=0.03`, `alpha_max=0.5`, `hilo_stage_weights=1,1,1,1`, four-stage dual-band enhancement on both primary/DFormerV2 and aligned depth features before GatedFusion.
- Best val/mIoU is `0.519128` at recorded epoch `41`; last val/mIoU is `0.518313`.
- Clean ten-run `dformerv2_mid_fusion` GatedFusion baseline mean best is `0.517397`, with population std `0.004901` and best single run `0.524425`.
- Delta vs the clean ten-run baseline mean is `+0.001731`; delta vs the clean baseline best single run is `-0.005297`.
- Comparison: `dformerv2_fft_freq_enhance` (gamma=0.1) single run best is `0.522688`; HiLo is `-0.003560` below.
- Interpretation: positive single-run signal but marginal, and weaker than the simpler freq_enhance design. The HiLo dual-band approach adds low-frequency enhancement and dual-band gating but does not outperform high-frequency-only enhancement. Training shows significant oscillation (drops at epochs 21, 29-30, 36, 42-43), suggesting the dual-band enhancement introduces more instability.
- Paper boundary: do not cite as improvement over freq_enhance. The HiLo design can be documented as a negative/neutral ablation showing that adding low-frequency enhancement does not help beyond high-frequency-only enhancement. The simpler `dformerv2_fft_freq_enhance` remains the stronger candidate.

## 2026-05-09 Cross-Modal InfoNCE Run01 Boundary

- `dformerv2_cm_infonce_c34_lam005_t01_s256_run01` completed 50 validation epochs.
- Setting: one-way depth-to-primary InfoNCE on c3/c4, `lambda_contrast=0.005`, `temperature=0.1`, `proj_dim=64`, `sample_points=256`, primary encoder gradient detached.
- Best val/mIoU is `0.514461` at recorded epoch `46`; last val/mIoU is `0.498469`.
- Clean ten-run `dformerv2_mid_fusion` GatedFusion baseline mean best is `0.517397`, with population std `0.004901` and best single run `0.524425`.
- Delta vs the clean ten-run baseline mean is `-0.002936`; delta vs the clean baseline best single run is `-0.009964`.
- Contrast loss converged from `5.575` to `3.089` (45% drop), confirming the InfoNCE signal is active and learning cross-modal alignment.
- Interpretation: negative single-run result. The contrastive alignment loss is technically correct (loss decreases) but does not translate to improved segmentation. The late collapse pattern (epoch 46→49: `0.5145→0.4985`) is consistent with other auxiliary loss experiments.
- Paper boundary: do not cite as improvement. Cross-modal InfoNCE at `lambda=0.005` is weaker than the clean baseline mean. Can be documented as a negative ablation showing that cross-modal contrastive alignment at this weight does not help.

## 2026-05-08 Cross-Modal InfoNCE Candidate Boundary

- New candidate entry is `dformerv2_cm_infonce`.
- Motivation: prior reconstruction/covariance auxiliary losses did not provide stable gains, so this branch tests a more direct cross-modal alignment signal without changing inference.
- The inference architecture is unchanged: DFormerV2_S primary encoder, DepthEncoder, GatedFusion, and SimpleFPNDecoder.
- Contrastive design is one-way depth-to-primary: depth features are queries and DFormerV2 primary features are keys.
- Key/query formula: `k = primary_proj(P.detach())`, `q = depth_proj(D)`.
- The `P.detach()` boundary protects the DFormerV2 primary encoder, while `primary_proj` remains trainable under the contrastive loss.
- First planned configuration uses c3+c4 only, `lambda_contrast=0.005`, `temperature=0.1`, `proj_dim=64`, and `sample_points=256`.
- This is not symmetric contrast, not KD, not memory bank contrast, not label-aware contrast, not mask reconstruction, and not a change to validation/inference.
- Paper status: implementation-only candidate; no mIoU claim until formal training logs and `miou_list` evidence exist.

## 2026-05-08 Depth FFT Frequency Selection Run01 Boundary

- `dformerv2_depth_fft_select_c030_run01` completed 50 validation epochs.
- Setting: `cutoff_ratio=0.30`, internal depth FFT low/high frequency selection after c2/c3/c4, c1 skipped, segmentation loss only.
- Best val/mIoU is `0.513871` at recorded epoch `43`; last val/mIoU is `0.482797`.
- Clean ten-run `dformerv2_mid_fusion` GatedFusion baseline mean best is `0.517397`, with population std `0.004901` and best single run `0.524425`.
- Delta vs the clean ten-run baseline mean is `-0.003526`; delta vs the clean baseline best single run is `-0.010554`.
- Checkpoint diagnostics show the selection modules stayed near identity: c2 low/high `0.997994/1.007397`, c3 low/high `0.993755/1.011458`, c4 low/high `0.993354/1.007414` using `sigmoid(bias) * 2`.
- Interpretation: negative result. The depth encoder internal FFT selection branch did not learn a meaningful selection away from identity, and the segmentation result is below the repeated baseline mean.
- Paper boundary: do not use this as a main improvement. It can be documented as a negative ablation showing that internal FFT selection on abstract DepthEncoder features is weaker than post-encoder/pre-fusion frequency enhancement.

## 2026-05-08 Depth FFT Frequency Selection Candidate Boundary

- New candidate entry is `dformerv2_depth_fft_select`.
- Motivation: previous `dformerv2_fft_freq_enhance` applied FFT high-frequency enhancement after encoder outputs and showed a positive single-run signal, but it was gamma-sensitive. This branch moves frequency selection inside the depth encoder and removes gamma.
- The DFormerV2 primary branch, GatedFusion, SimpleFPNDecoder, BaseLitSeg, loss, dataset, and dataloader remain unchanged.
- The depth branch keeps the original ResNet-18 structure but inserts FFT low/high selection after c2, c3, and c4; c1 is skipped to avoid shallow depth noise.
- The module uses true spatial FFT decomposition, not AvgPool high-pass or channel-only attention.
- Selection is initialized as exact identity through zero-initialized depthwise convolutions and `sigmoid(0) * 2 = 1`.
- Formula: `D_out = D + (w_low - 1) * D_low + (w_high - 1) * D_high`.
- First planned run: `cutoff_ratio=0.30`, segmentation loss only.
- Paper status: implementation-only candidate; no mIoU claim until formal training logs and `miou_list` evidence exist.

## 2026-05-08 FFT Frequency Enhancement Run01 Boundary

- `dformerv2_fft_freq_enhance_hh_w1111_c025_g01_run01` completed 50 validation epochs.
- Setting: `cutoff_ratio=0.25`, `gamma_init=0.1`, four-stage FFT enhancement on both primary/DFormerV2 and aligned depth features before GatedFusion.
- Best val/mIoU is `0.522688` at recorded epoch `41`; last val/mIoU is `0.514651`.
- Clean ten-run `dformerv2_mid_fusion` GatedFusion baseline mean best is `0.517397`, with population std `0.004901` and best single run `0.524425`.
- Delta vs the clean ten-run baseline mean is `+0.005291`; delta vs the clean baseline best single run is `-0.001737`.
- Interpretation: positive single-run signal and more promising than prior auxiliary-loss attempts, but not yet a stable paper improvement because repeated-run evidence is missing and the strongest baseline single run is still higher.
- Paper boundary: can be cited as a promising candidate ablation only after repeat runs; currently it should not be reported as the final main result.

## 2026-05-08 FFT Frequency Enhancement Run02 (gamma=0.2) Boundary

- `dformerv2_fft_freq_enhance_hh_w1111_c025_g02_run01` completed 50 validation epochs.
- Setting: `cutoff_ratio=0.25`, `gamma_init=0.20`, four-stage FFT enhancement on both primary/DFormerV2 and aligned depth features before GatedFusion.
- Best val/mIoU is `0.515696` at recorded epoch `47`; last val/mIoU is `0.476463`.
- g01 best (gamma=0.1) is `0.522688`; delta vs g01 is `-0.006992`.
- Clean ten-run `dformerv2_mid_fusion` GatedFusion baseline mean best is `0.517397`; delta vs baseline mean is `-0.001701`.
- The run peaks exceptionally late (epoch 47) and collapses sharply in the last 3 epochs (0.5157 -> 0.5110 -> 0.4964 -> 0.4765).
- Interpretation: negative result. g01 extracted gamma values (all learned to 0.30-0.47 from init 0.10) already indicated that the model wants higher gamma, but giving it at initialization does not help — the gradual growth path from 0.10 is more stable. The late peak + sharp collapse pattern suggests gamma=0.2 causes over-enhancement that the model cannot recover from.
- Paper boundary: gamma=0.2 is worse than gamma=0.1. Future sweeps should focus on gamma <= 0.1 or adjust cutoff_ratio instead. Do not cite gamma=0.2 as an improvement.

## 2026-05-08 FFT Frequency Enhancement Candidate Boundary

- New candidate entry is `dformerv2_fft_freq_enhance`.
- Motivation: prior frequency covariance and mask reconstruction losses did not provide stable gains, so this branch tests a direct inference-path frequency enhancement before GatedFusion.
- The module is true spatial FFT frequency enhancement, not channel-only attention or AvgPool high-pass approximation.
- The inference path is changed only by inserting `FFTFrequencyEnhance` before GatedFusion for both primary/DFormerV2 features and aligned depth features.
- Frequency decomposition uses `torch.fft.fft2` over H/W, a hard circular low-pass mask in the shifted Fourier plane, and `torch.fft.ifft2(...).real`.
- Formula: `F_high = F - IFFT2(M_low * FFT2(F)).real`; output is `F + gamma * gate * Clean(F_high)`.
- First planned run: `cutoff_ratio=0.25`, `gamma_init=0.05`, four stages enabled, segmentation loss only.
- This is not an auxiliary loss, FreqFusion module import, FADC AdaptiveDilatedConv, CARAFE, new decoder, or new backbone.
- Paper status: implementation-only candidate; no mIoU claim until formal training logs and `miou_list` evidence exist.

## 2026-05-08 Feature Mask Reconstruction Run01 Boundary

- `dformerv2_feat_maskrec_c34_w0011_lam01_run01` completed 50 validation epochs.
- Setting: `lambda_mask=0.1`, `maskrec_stage_weights=0,0,1,1`, `mask_ratio_depth=0.30`, `mask_ratio_primary=0.15`, `maskrec_alpha=0.5`.
- Best val/mIoU is `0.515327` at recorded epoch `33`; last val/mIoU is `0.501496`.
- Clean ten-run `dformerv2_mid_fusion` GatedFusion baseline mean best is `0.517397`, with population std `0.004901`.
- Delta vs the clean ten-run baseline mean is `-0.002070`.
- Interpretation: negative single-run result. The auxiliary reconstruction loss decreases and is active, but it has not improved the segmentation objective.
- Paper boundary: this run can only be cited as a negative/neutral auxiliary-loss ablation, not as an improvement.

## 2026-05-08 Feature Mask Reconstruction Candidate Boundary

- New candidate entry is `dformerv2_feat_maskrec_c34`.
- Motivation: after freqcov showed only limited single-run gains, test a more direct training-only cross-modal auxiliary signal without changing the segmentation inference architecture.
- The inference architecture is unchanged: DFormerV2_S primary encoder, DepthEncoder, GatedFusion, and SimpleFPNDecoder.
- The auxiliary loss is feature-level, not image-level reconstruction: it supports c1-c4 primary/DFormerv2 and aligned depth features.
- Stage participation is explicit through `--maskrec_stage_weights`; the first planned run uses `0,0,1,1` for c3+c4.
- Primary-to-depth reconstructs masked depth features from primary features plus visible depth context, with default depth mask ratio `0.30`.
- Depth-to-primary reconstructs masked primary/DFormerv2 features from depth features plus visible primary context, with default primary mask ratio `0.15`.
- Targets are detached for stable supervision; source features and masked target inputs remain attached for gradient flow.
- This is not a new fusion block, not KD, not MultiMAE pretraining, not M3AE missing-modality inference, and not a replacement for the baseline.
- Paper status: implementation-only candidate; no mIoU claim until formal training logs and `miou_list` evidence exist.

## 2026-05-08 MS-FreqCov Sweep Boundary

- Active baseline for comparisons remains clean 10-run `dformerv2_mid_fusion` GatedFusion: mean best val/mIoU `0.517397`, population std `0.004901`, best single run `0.524425`.
- `dformerv2_ms_freqcov` keeps the inference architecture unchanged and adds only a training-time c1-c4 frequency covariance auxiliary loss.
- Seven freqcov settings have completed 50 epochs: default `lambda=0.01`, `lambda=0.1`, `lambda=1.0`, high-stage weighted `0.5,1,1,2` at `lambda=0.1/1.0/2.0`, and high-stage weighted with stage1 disabled `0,0.5,1,2` at `lambda=1.0`.
- Best freqcov single run is default `lambda=0.01`, best val/mIoU `0.520539`; second useful signal is `lambda=1.0`, weights `0,0.5,1,2`, best val/mIoU `0.520060`.
- Sweep mean best val/mIoU is `0.515697`, below the clean baseline mean by `0.001700`.
- `lambda=2.0` with weights `0.5,1,1,2` is a clear negative result: best val/mIoU `0.504229`.
- Paper boundary: freqcov can be described as a training-only auxiliary ablation with limited single-run positive signals, but it is not yet a stable paper improvement and must not be reported as the main result unless repeated runs beat the clean baseline mean.

## 2026-05-07 MS-FreqCov Run01 Result Boundary

- `dformerv2_ms_freqcov_run01` completed 50 validation epochs.
- Best val/mIoU is `0.520539` at recorded epoch `50`.
- Clean ten-run `dformerv2_mid_fusion` GatedFusion baseline mean best is `0.517397`, with population std `0.004901`.
- Run01 delta vs the clean ten-run baseline mean is `+0.003142`.
- Interpretation: promising but not stable yet. The gain is smaller than one baseline standard deviation, so repeated runs are required before this can be written as a paper improvement.

## 2026-05-07 MS-FreqCov Candidate Boundary

- New candidate entry is `dformerv2_ms_freqcov`.
- Motivation: test whether c1-c4 RGB/DFormer and depth features benefit from a training-only second-order frequency covariance auxiliary loss.
- The inference architecture is unchanged: DFormerV2_S primary encoder, DepthEncoder, GatedFusion, and SimpleFPNDecoder.
- This is not a new fusion block, not a reconstruction head, not KD, and not full VICReg or Barlow Twins SSL.
- Paper status: implementation-only candidate; no mIoU claim until formal training logs and `miou_list` evidence exist.

## 2026-05-07 GatedFusion Baseline Boundary

- New repeated baseline evidence is `dformerv2_mid_fusion_gate_baseline`.
- Run01-run09 completed all 50 validation epochs and give mean best val/mIoU `0.516789`.
- Run10 is partial with 43 recorded validation epochs, best recorded val/mIoU `0.514412`; it is kept as partial evidence only and is not counted in the complete-run mean.
- This raises the practical comparison bar above the earlier baseline mean `0.513406`.
- Candidate fusion or guided-depth modules should now be compared against the nine-complete-run GatedFusion baseline unless a later clean ten-run baseline is completed.

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
