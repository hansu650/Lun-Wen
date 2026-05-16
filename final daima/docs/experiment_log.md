# Experiment Log

## 2026-05-16 R047 result: GatedFusion local GroupNorm negative

- branch: `exp/R047-gatedfusion-local-gn-v1`
- model: `dformerv2_gatedfusion_gn`
- run: `R047_gatedfusion_local_gn_run01`
- hypothesis: replace batch-dependent normalization inside the local `GatedFusion` blocks with `GroupNorm(32, C)` to test whether batch_size=2 fusion-statistics drift explains R016 late instability.
- implementation: added independent model entry; original `GatedFusion` and baseline remain unchanged. `GatedFusionGN` keeps the original depth projection, gate equation, fused mixture, and refine shape, but uses GroupNorm in gate/refine.
- literature/code evidence: Group Normalization ECCV 2018 / arXiv `1803.08494`; direct PyTorch `torch.nn.GroupNorm`; MoBaNet 2026 multimodal gated fusion as recent supporting pattern.
- smoke status: `py_compile`, `train.py --help`, random tensor forward/backward, and static code review passed.
- full train status: completed; `Trainer.fit` reached `max_epochs=50`.
- recorded validation epochs: `50`
- best val/mIoU: `0.528301` at validation epoch `25`
- last val/mIoU: `0.472746`
- last-5 mean val/mIoU: `0.509970`
- last-10 mean val/mIoU: `0.513930`
- best-to-last drop: `0.055555`
- best val/loss: `0.971623` at validation epoch `15`
- last val/loss: `1.294296`
- final train/loss_epoch: `0.103899`
- checkpoint: `checkpoints\R047_gatedfusion_local_gn_run01\dformerv2_gatedfusion_gn-epoch=24-val_mIoU=0.5283.pt`
- TensorBoard event: `checkpoints\R047_gatedfusion_local_gn_run01\lightning_logs\version_0\events.out.tfevents.1778920890.Administrator.5584.0`
- evidence: `miou_list/R047_gatedfusion_local_gn_run01.md`
- comparison: below R016 `0.541121` by `-0.012820`, below R036 `0.539790` by `-0.011489`, and below R041 `0.537098` by `-0.008797`.
- conclusion: **negative.** Full BN-to-GN replacement in `GatedFusion` lowers the peak and worsens late collapse; do not continue full GN replacement.
- next step: archive code under `feiqi/failed_experiments_r047_20260516/` and pivot to a distinct R048 hypothesis, preferably outside another local fusion normalization tweak.

## 2026-05-16 R046 result: DGFusion c4 depth-token negative below R016

- branch: `exp/R046-dgfusion-c4-depth-token-v1`
- model: `dformerv2_dgfusion_c4_depth_token`
- run: `R046_dgfusion_c4_depth_token_run01`
- hypothesis: add a DGFusion-style c4 local depth-token feature residual on top of the original c4 GatedFusion output to test local learned geometry-conditioned interaction without replacing GatedFusion or changing decoder/loss/training recipe.
- implementation: c1-c3 stay on original GatedFusion. c4 uses original GatedFusion as the base, then a local pooled c4 depth token conditions a zero-initialized feature residual through normalized query/key affinity.
- literature/code evidence: DGFusion arXiv `2509.09828`, official repo `https://github.com/timbroed/DGFusion`, core `dgfusion/modeling/modality_fusion/depth_token_guided_pca.py`. Only the minimal local depth-token idea was ported; no full framework, global condition token, auxiliary depth head, sampling offset, raw-depth cue, diff cue, or self-adapter was added.
- smoke status: `py_compile`, `train.py --help`, and random tensor forward/backward passed. Initial `c4_token_delta_abs` was exactly `0.0`, and gradients reached the zero-initialized out projection.
- full train status: completed with exit code `0`; `Trainer.fit` reached `max_epochs=50`.
- recorded validation epochs: `50`
- best val/mIoU: `0.531838` at validation epoch `44`
- last val/mIoU: `0.527239`
- last-5 mean val/mIoU: `0.510100`
- last-10 mean val/mIoU: `0.514172`
- best-to-last drop: `0.004599`
- best val/loss: `0.961911` at validation epoch `10`
- last val/loss: `1.208670`
- final train/loss_epoch: `0.054207`
- c4_token_delta_abs first/last/min/max: `0.009067` / `0.273799` / `0.009067` / `0.289744`
- c4_token_affinity_mean first/last/min/max: `0.623793` / `0.287895` / `0.287302` / `0.623793`
- c4_token_affinity_std first/last/min/max: `0.026211` / `0.028371` / `0.026211` / `0.048227`
- checkpoint: `checkpoints/R046_dgfusion_c4_depth_token_run01/dformerv2_dgfusion_c4_depth_token-epoch=43-val_mIoU=0.5318.pt`
- TensorBoard event: `checkpoints/R046_dgfusion_c4_depth_token_run01/lightning_logs/version_0/events.out.tfevents.1778913804.Administrator.33260.0`
- saved command: `checkpoints/R046_dgfusion_c4_depth_token_run01/run_r046.ps1`
- evidence: `miou_list/R046_dgfusion_c4_depth_token_run01.md`
- comparison: R046 is below R016 `0.541121` by `-0.009283`, below R036 `0.539790` by `-0.007952`, and below R041 `0.537098` by `-0.005260`.
- conclusion: **negative below corrected baseline.** It is more stable than R045 but the fixed-recipe peak remains too low; local c4 depth-token interaction is insufficient in this minimal form.
- next step: do not tune DGFusion-lite window/token/scale. Archive the implementation under `feiqi/failed_experiments_r046_20260516/` and pivot to a distinct hypothesis.

## 2026-05-16 R045 result: c3/c4 zero-init modality adapter negative below R016

- branch: `exp/R045-c34-zero-init-modality-adapter-v1`
- model: `dformerv2_c34_zero_init_modality_adapter`
- run: `R045_c34_zero_init_modality_adapter_run01`
- hypothesis: add c3/c4-only zero-initialized modality adapters before the original GatedFusion modules to test whether high-stage feature-stream adaptation improves over R016 without depth input conversion or output residual correction.
- implementation: c1/c2 stay on the original path; c3/c4 DFormerv2 and aligned DepthEncoder features pass through separate zero-initialized bottleneck adapters before the unchanged `GatedFusion` modules and unchanged `SimpleFPNDecoder`.
- literature/code evidence: selected from a StitchFusion/KTB-style adapter gate after subagent paper-code reading; only the minimal zero-init bottleneck adapter idea was ported. No full framework, cross-attention stack, auxiliary loss, decoder change, or training recipe change was added.
- smoke status: `py_compile`, `train.py --help`, and random tensor forward/backward passed. Initial adapter deltas were exactly `0.0`, and gradients reached the zero-initialized final adapter conv.
- full train status: completed with exit code `0`; `Trainer.fit` reached `max_epochs=50`.
- recorded validation epochs: `50`
- best val/mIoU: `0.531454` at validation epoch `48`
- last val/mIoU: `0.505130`
- last-5 mean val/mIoU: `0.511191`
- last-10 mean val/mIoU: `0.509930`
- best-to-last drop: `0.026324`
- best val/loss: `0.981750` at validation epoch `8`
- last val/loss: `1.161006`
- final train/loss_epoch: `0.151835`
- rgb_c3_adapter_delta_abs first/last/max: `0.008236` / `0.043839` / `0.043839`
- rgb_c4_adapter_delta_abs first/last/max: `0.010011` / `0.040954` / `0.040954`
- depth_c3_adapter_delta_abs first/last/max: `0.005683` / `0.101323` / `0.108361`
- depth_c4_adapter_delta_abs first/last/max: `0.018327` / `0.098307` / `0.098307`
- checkpoint: `checkpoints/R045_c34_zero_init_modality_adapter_run01/dformerv2_c34_zero_init_modality_adapter-epoch=47-val_mIoU=0.5315.pt`
- TensorBoard event: `checkpoints/R045_c34_zero_init_modality_adapter_run01/lightning_logs/version_0/events.out.tfevents.1778907691.Administrator.38920.0`
- saved command: `checkpoints/R045_c34_zero_init_modality_adapter_run01/run_r045.ps1`
- evidence: `miou_list/R045_c34_zero_init_modality_adapter_run01.md`
- comparison: R045 is below R016 `0.541121` by `-0.009667`, below R036 `0.539790` by `-0.008336`, and below R041 `0.537098` by `-0.005644`.
- conclusion: **negative below corrected baseline.** The zero-init high-stage adapter opens but does not improve the fixed-recipe peak and has a severe late drop.
- next step: do not tune adapter scale/reduction/stage. Archive the implementation under `feiqi/failed_experiments_r045_20260516/` and pivot to a distinct hypothesis, with DGFusion-style c4 local depth-token interaction as the current R046 lead if R045 remains the final result.

# Experiment Log

## 2026-05-16 R044 result: conditioned c3/c4 residual negative below R016

- branch: `exp/R044-conditioned-c34-bounded-residual-v1`
- model: `dformerv2_conditioned_c34_bounded_residual`
- run: `R044_conditioned_c34_bounded_residual_run01`
- hypothesis: condition the c3/c4 bounded depth residual amplitude on DFormerv2 c4 global pooled features to test whether image/channel-specific residual opening improves over R036 static alpha while preserving original GatedFusion as the base.
- implementation: added a separate model entry. c1/c2 used original `GatedFusion`; c3/c4 used original `GatedFusion` as `base` plus `alpha * residual`, where `alpha <= 0.05` is generated from c4 global pooled DFormerv2 features by a zero-initialized channel head. The residual final projection was zero-initialized.
- literature/code evidence: motivated by CAFuser condition-aware fusion and DGFusion depth-guided fusion, but only a minimal conditioned residual amplitude unit was ported. No full framework, condition classifier, cross-attention stack, auxiliary depth loss, decoder change, or training recipe change was added.
- smoke status: syntax compile, `train.py --help`, and real NYU batch CUDA forward/backward passed. Smoke confirmed c1/c2 original `GatedFusion`, c3/c4 `ImageConditionedC34BoundedDepthResidual`, finite CE, unchanged DFormerv2 pretrained load stats, and nonzero gradients through residual/depth paths after one optimizer step.
- full train status: completed with exit code `0`; `Trainer.fit` reached `max_epochs=50`.
- recorded validation epochs: `50`
- best val/mIoU: `0.535663` at validation epoch `49`
- last val/mIoU: `0.520020`
- last-5 mean val/mIoU: `0.521228`
- last-10 mean val/mIoU: `0.519138`
- best-to-last drop: `0.015643`
- best val/loss: `0.955417` at validation epoch `10`
- last val/loss: `1.279686`
- final train/loss_epoch: `0.051134`
- c3 conditioned alpha mean first/last/max: `0.027802` / `0.049420` / `0.049420`
- c3 conditioned alpha max first/last/max: `0.030972` / `0.049999` / `0.049999`
- c3 residual_abs first/last/max: `0.066489` / `0.737136` / `0.737921`
- c4 conditioned alpha mean first/last/max: `0.025361` / `0.027943` / `0.028668`
- c4 conditioned alpha max first/last/max: `0.026435` / `0.044898` / `0.045183`
- c4 residual_abs first/last/max: `0.088054` / `0.708959` / `0.708959`
- checkpoint: `checkpoints/R044_conditioned_c34_bounded_residual_run01/dformerv2_conditioned_c34_bounded_residual-epoch=48-val_mIoU=0.5357.pt`
- TensorBoard event: `checkpoints/R044_conditioned_c34_bounded_residual_run01/lightning_logs/version_0/events.out.tfevents.1778900799.Administrator.11996.0`
- saved command: `checkpoints/R044_conditioned_c34_bounded_residual_run01/run_r044.cmd`
- evidence: `miou_list/R044_conditioned_c34_bounded_residual_run01.md`
- comparison: R044 is below R016 `0.541121` by `-0.005458`, below R036 `0.539790` by `-0.004127`, and below R041 `0.537098` by `-0.001435`. It is only `+0.000071` over R043, which is not meaningful.
- conclusion: **negative/diagnostic below corrected baseline.** The conditioned residual path opens, especially c3, but it saturates near the alpha cap and residual magnitude grows without improving the fixed-recipe peak. The final drop also crosses the `0.015` instability tripwire.
- next step: do not promote this exact module and do not tune alpha bound or hidden size. Archive the implementation under `feiqi/failed_experiments_r044_20260516/` and pivot to a distinct hypothesis rather than another conditioned c3/c4 residual micro-variant.

## 2026-05-16 R043 result: raw depth geometry c4 cue partial signal below R016

- branch: `exp/R043-depthgeo-c4-cue-v1`
- model: `dformerv2_depthgeo_c4_cue`
- run: `R043_depthgeo_c4_cue_run01`
- hypothesis: a c4-only raw-depth Sobel/normal-like geometry cue can improve high-level gate decisions more safely than RGB-depth feature disagreement, aligning with DFormerv2's geometry-prior motivation.
- implementation: added a separate model entry. c1-c3 outputs use the original `GatedFusion`; inherited c4 fusion is disabled with `Identity`; c4 uses the original gate/refine form plus a zero-initialized gate-logit correction from on-the-fly depth Sobel `dx/dy`, magnitude, and normal-like `nx/ny/nz`.
- literature/code evidence: R043 builds on DFormerv2 geometry-prior motivation plus geometry-aware depth/normal cue patterns from 2025 RGB-D literature. It ports only a minimal c4 gate-logit cue, not a new backbone, dataloader preprocessing, or large framework.
- smoke status: syntax compile, `train.py --help`, and real NYU batch CUDA forward/backward passed. Smoke confirmed c1-c3 original `GatedFusion`, c4 `DepthGeometryC4CueFusion`, logits `(2, 40, 480, 640)`, finite CE `3.770108`, unchanged DFormerv2 pretrained load stats `loaded_keys=774`, `missing_keys=6`, `unexpected_keys=11`, finite raw-depth geometry cue tensors, and nonzero gradients through the geo gate, c4 depth projection, base gate, and refine path.
- full train status: completed with exit code `0`; `Trainer.fit` reached `max_epochs=50`.
- recorded validation epochs: `50`
- best val/mIoU: `0.535592` at validation epoch `42`
- last val/mIoU: `0.522214`
- last-5 mean val/mIoU: `0.518946`
- last-10 mean val/mIoU: `0.522097`
- best-to-last drop: `0.013378`
- best val/loss: `0.961905` at validation epoch `11`
- last val/loss: `1.208118`
- final train/loss_epoch: `0.052872`
- depthgeo c4 geo_logit_abs first/last/max: `0.015203` / `0.153483` / `0.153980`
- depthgeo c4 gate_mean first/last/max: `0.499898` / `0.512162` / `0.512432`
- depthgeo c4 gate_std first/last/max: `0.207853` / `0.212319` / `0.213092`
- depth edge mean first/last: `0.145911` / `0.145908`
- checkpoint: `checkpoints/R043_depthgeo_c4_cue_run01/dformerv2_depthgeo_c4_cue-epoch=41-val_mIoU=0.5356.pt`
- TensorBoard event: `checkpoints/R043_depthgeo_c4_cue_run01/lightning_logs/version_0/events.out.tfevents.1778893712.Administrator.9528.0`
- saved command: `checkpoints/R043_depthgeo_c4_cue_run01/run_r043.cmd`
- evidence: `miou_list/R043_depthgeo_c4_cue_run01.md`
- comparison: R043 is above R042 by `+0.004863` and far less unstable, but below R041 `0.537098` by `-0.001506`, below R036 `0.539790` by `-0.004198`, and below R016 `0.541121` by `-0.005529`.
- conclusion: **partial geometry-cue signal below corrected baseline.** The explicit depth geometry cue opens conservatively and is safer than c3-propagated feature disagreement, but it does not beat the strongest fixed-recipe results and late-window dips remain.
- next step: do not promote this exact module and do not tune Sobel/normal hidden size or scale. Archive the code under `feiqi/failed_experiments_r043_20260516/` and pivot to a distinct R044 hypothesis.

## 2026-05-16 R042 result: c3-to-c4 DiffPixel cue negative

- branch: `exp/R042-diffpixel-c3toc4-cue-v1`
- model: `dformerv2_diffpixel_c3toc4_cue`
- run: `R042_diffpixel_c3toc4_cue_run01`
- hypothesis: a c3 RGB-depth differential cue propagated to c4 can condition high-level `GatedFusion` better than R041's c4-only local difference, without changing c3 outputs or adding output residuals.
- implementation: added a separate model entry. c1-c3 outputs use the original `GatedFusion`; inherited c4 fusion is disabled with `Identity`; c4 uses a new fusion block whose gate logit adds a zero-initialized, downsampled cue from c3 `[rgb-depth_proj, abs(rgb-depth_proj)]`.
- literature/code evidence: R042 builds on DiffPixelFormer (`arXiv:2511.13047`, official repo `github.com/gongyan1/DiffPixelFormer`) and the R041 partial-positive c4 differential cue result. It ports only a minimal cross-stage differential cue, not the full paper framework.
- smoke status: syntax compile, `train.py --help`, and real NYU batch CUDA forward/backward passed. Smoke confirmed c1-c3 original `GatedFusion`, c4 extra `C3ToC4DiffPixelCueFusion`, logits `(2, 40, 480, 640)`, finite CE `3.726418`, unchanged DFormerv2 pretrained load stats `loaded_keys=774`, `missing_keys=6`, `unexpected_keys=11`, and nonzero gradients through the final c3 cue projection, c4 depth projection, gate, and refine path.
- full train status: completed with exit code `0`; `Trainer.fit` reached `max_epochs=50`.
- recorded validation epochs: `50`
- best val/mIoU: `0.530729` at validation epoch `43`
- last val/mIoU: `0.458179`
- last-5 mean val/mIoU: `0.505197`
- last-10 mean val/mIoU: `0.510157`
- best-to-last drop: `0.072551`
- best val/loss: `0.957006` at validation epoch `10`
- last val/loss: `1.417108`
- final train/loss_epoch: `0.065262`
- c3-to-c4 cue_abs first/last: `0.018134` / `0.246822`
- c3-to-c4 gate_mean first/last: `0.500000` / `0.545177`
- checkpoint: `checkpoints/R042_diffpixel_c3toc4_cue_run01/dformerv2_diffpixel_c3toc4_cue-epoch=42-val_mIoU=0.5307.pt`
- TensorBoard event: `checkpoints/R042_diffpixel_c3toc4_cue_run01/lightning_logs/version_0/events.out.tfevents.1778887449.Administrator.10152.0`
- saved command: `checkpoints/R042_diffpixel_c3toc4_cue_run01/run_r042.cmd`
- evidence: `miou_list/R042_diffpixel_c3toc4_cue_run01.md`
- comparison: R042 is below R041 `0.537098` by `-0.006369`, below R036 `0.539790` by `-0.009061`, and below R016 `0.541121` by `-0.010392`.
- conclusion: **negative and unstable.** The c3 cue branch opens, but propagating c3 disagreement into c4 appears to amplify mid-level noise and worsens late collapse.
- next step: do not continue c3-propagated differential cue or hidden-size/scale variants. Pivot to a distinct hypothesis such as explicit geometry derivative/normal-like depth cue or a higher-capacity stability module.

## 2026-05-16 R041 result: DiffPixel c4 cue partial positive below R016

- branch: `exp/R041-diffpixel-c4-cue-v1`
- model: `dformerv2_diffpixel_c4_cue`
- run: `R041_diffpixel_c4_cue_run01`
- hypothesis: DiffPixelFormer-style RGB-depth differential cue can improve c4 gate decisions by explicitly modeling local modality disagreement before fusion, without adding an output residual.
- implementation: added a separate model entry with original c1-c3 `GatedFusion`; c4 computes the normal gate logit from `[rgb, depth_proj]`, then adds a zero-initialized differential gate logit from `[rgb-depth_proj, abs(rgb-depth_proj)]` before the original sigmoid and refine path.
- literature/code evidence: read DiffPixelFormer (`arXiv:2511.13047`, official repo `github.com/gongyan1/DiffPixelFormer`) and the official differential cue path using `x0 - x1`; ported only the minimal c4 gate-logit correction idea, not the full Transformer/cross-attention framework.
- smoke status: syntax compile, `train.py --help`, and real NYU batch CUDA forward/backward passed. Smoke confirmed c1-c3 original `GatedFusion`, c4 `DiffPixelC4CueFusion`, logits `(2, 40, 480, 640)`, finite CE `3.609063`, nonzero diff-gate/depth/gate/refine gradients, and unchanged DFormerv2 pretrained load stats `loaded_keys=774`, `missing_keys=6`, `unexpected_keys=11`.
- full train status: completed with exit code `0`; `Trainer.fit` reached `max_epochs=50`.
- recorded validation epochs: `50`
- best val/mIoU: `0.537098` at validation epoch `44`
- last val/mIoU: `0.529552`
- last-5 mean val/mIoU: `0.507803`
- last-10 mean val/mIoU: `0.516635`
- best-to-last drop: `0.007546`
- best val/loss: `0.974793` at validation epoch `12`
- last val/loss: `1.197305`
- final train/loss_epoch: `0.071763`
- DiffPixel c4 gate_mean first/last: `0.500861` / `0.544060`
- DiffPixel c4 gate_std first/last: `0.208216` / `0.221571`
- DiffPixel c4 diff_gate_abs first/last: `0.015047` / `0.253500`
- DiffPixel c4 diff_context_abs first/last: `1.039935` / `0.954306`
- checkpoint: `checkpoints/R041_diffpixel_c4_cue_run01/dformerv2_diffpixel_c4_cue-epoch=43-val_mIoU=0.5371.pt`
- TensorBoard event: `checkpoints/R041_diffpixel_c4_cue_run01/lightning_logs/version_0/events.out.tfevents.1778881132.Administrator.43760.0`
- saved command: `checkpoints/R041_diffpixel_c4_cue_run01/run_r041.cmd`
- evidence: `miou_list/R041_diffpixel_c4_cue_run01.md`
- comparison: R041 is above R040 `0.527946`, R038 `0.530810`, and R037 `0.534656`, but below R036 `0.539790` by `-0.002692` and below R016 `0.541121` by `-0.004023`.
- conclusion: **partial positive below corrected baseline.** The differential cue is the strongest recent c4-only fusion cue and crosses `0.53`, but it does not beat R016 and still has late-window dips.
- next step: do not promote this exact module as active mainline. Use it as evidence that differential cues are more useful than prompt/sparse c4 variants, then either test a distinct differential-cue extension or pivot to a higher-capacity/stability design.

## 2026-05-16 R040 result: c4 low-rank depth prompt negative below R016

- branch: `exp/R040-c4-lowrank-depth-prompt-v1`
- model: `dformerv2_c4_lowrank_depth_prompt`
- run: `R040_c4_lowrank_depth_prompt_run01`
- hypothesis: MixPrompt-style low-rank c4 depth prompt can condition the DFormerv2 c4 primary feature before the original `GatedFusion`, testing whether depth is more useful as fusion-before prompt conditioning than as output-side residual correction.
- implementation: added a separate model entry with original c1-c3 `GatedFusion`; c4 projects depth, builds a low-rank prompt from `[depth_proj, abs(rgb-depth_proj)]`, adds the zero-init projected prompt to c4 RGB/DFormerv2 features, then runs the original gate/refine fusion logic.
- literature/code evidence: read MixPrompt (`NeurIPS 2025`, official repo `github.com/xiaoshideta/MixPrompt`) and the official `ADA` low-rank prompt code; ported only the minimal c4 prompt-conditioning idea, not the full MixPrompt framework.
- smoke status: syntax compile, `train.py --help`, and real NYU batch CUDA forward/backward passed. Smoke confirmed c1-c3 original `GatedFusion`, c4 `C4LowRankDepthPromptFusion`, logits `(2, 40, 480, 640)`, finite CE `3.753238`, nonzero prompt/depth/gate/refine gradients, and unchanged DFormerv2 pretrained load stats `loaded_keys=774`, `missing_keys=6`, `unexpected_keys=11`.
- full train status: completed with exit code `0`; `Trainer.fit` reached `max_epochs=50`.
- recorded validation epochs: `50`
- best val/mIoU: `0.527946` at validation epoch `37`
- last val/mIoU: `0.524679`
- last-5 mean val/mIoU: `0.509687`
- last-10 mean val/mIoU: `0.508256`
- best-to-last drop: `0.003267`
- best val/loss: `0.950243` at validation epoch `9`
- last val/loss: `1.172903`
- final train/loss_epoch: `0.054379`
- C4 prompt_abs first/last: `0.030488` / `0.014483`
- C4 prompt_raw_abs first/last: `0.156883` / `0.013475`
- C4 prompt_gate_mean first/last: `0.499766` / `0.501595`
- C4 prompt_gate_std first/last: `0.207389` / `0.206496`
- checkpoint: `checkpoints/R040_c4_lowrank_depth_prompt_run01/dformerv2_c4_lowrank_depth_prompt-epoch=36-val_mIoU=0.5279.pt`
- TensorBoard event: `checkpoints/R040_c4_lowrank_depth_prompt_run01/lightning_logs/version_0/events.out.tfevents.1778875233.Administrator.19400.0`
- saved command: `checkpoints/R040_c4_lowrank_depth_prompt_run01/run_r040.cmd`
- evidence: `miou_list/R040_c4_lowrank_depth_prompt_run01.md`
- comparison: R040 is below the stage threshold `0.53` by `-0.002054`, below R016 `0.541121` by `-0.013175`, below R036 `0.539790`, and only slightly above R010 `0.527469`.
- conclusion: **negative below corrected baseline.** The low-rank c4 prompt avoids R039-style gate explosion and recovers at the final epoch, but late-window dips remain severe and the low ceiling does not improve the fixed-recipe pipeline.
- next step: do not tune prompt rank/down-ratio/c4 scale. Pivot to a distinct DiffPixelFormer-style c4 differential cue that explicitly models RGB-depth differences before fusion.

## 2026-05-16 R039 result: MIIM-lite c4 global-local residual negative below R016

- branch: `exp/R039-miim-c4-lite-v1`
- model: `dformerv2_miim_c4_lite`
- run: `R039_miim_c4_lite_run01`
- hypothesis: HDBFormer/MIIM-style global-local interaction at c4 may provide a better high-level RGB-depth update than R038 sparse sampling, while preserving c1-c3 original `GatedFusion`, `SimpleFPNDecoder`, CE loss, and the fixed training recipe.
- implementation: added a separate model entry with original c1-c3 `GatedFusion`; c4 first computes the normal R016 `GatedFusion` base and then adds a tiny bounded MIIM-lite residual `base + alpha * global_gate * local_update`.
- literature/code evidence: read HDBFormer paper/repo (`arXiv:2504.13579`, `github.com/Weishuobin/HDBFormer`) and the official `models/MIIM.py`; ported only the minimal global pooling + local depthwise-conv interaction idea, not the full HDBFormer framework.
- smoke status: syntax compile, `train.py --help`, registry lookup, and real NYU batch CUDA forward/backward passed. Smoke confirmed c1-c3 original `GatedFusion`, c4 `MIIMC4LiteFusion`, logits `(1, 40, 480, 640)`, finite CE `3.816422`, nonzero c4 MIIM/global/local/alpha gradients, and unchanged DFormerv2 pretrained load stats `loaded_keys=774`, `missing_keys=6`, `unexpected_keys=11`.
- full train status: completed with exit code `0`; `Trainer.fit` reached `max_epochs=50`.
- recorded validation epochs: `50`
- best val/mIoU: `0.534131` at validation epoch `41`
- last val/mIoU: `0.509767`
- last-5 mean val/mIoU: `0.511286`
- last-10 mean val/mIoU: `0.513125`
- best-to-last drop: `0.024364`
- best val/loss: `0.960918` at validation epoch `9`
- last val/loss: `1.160239`
- final train/loss_epoch: `0.126658`
- MIIM c4 alpha first/last: `0.002511` / `0.003592`
- MIIM c4 gate_mean first/last: `0.507850` / `0.895789`
- MIIM c4 gate_std first/last: `0.094741` / `0.298645`
- MIIM c4 update_abs first/last: `0.303729` / `0.392390`
- checkpoint: `checkpoints/R039_miim_c4_lite_run01/dformerv2_miim_c4_lite-epoch=40-val_mIoU=0.5341.pt`
- TensorBoard event: `checkpoints/R039_miim_c4_lite_run01/lightning_logs/version_0/events.out.tfevents.1778869284.Administrator.3304.0`
- saved command: `checkpoints/R039_miim_c4_lite_run01/run_r039.cmd`
- evidence: `miou_list/R039_miim_c4_lite_run01.md`
- comparison: R039 is below R016 `0.541121` by `-0.006990`, below R036 `0.539790`, and slightly below R037 `0.534656`.
- conclusion: **negative below corrected baseline with late collapse.** MIIM-lite c4 opens and produces a late peak above `0.53`, but the high gate mean and large best-to-last drop show that this minimal global-local residual does not stabilize or improve the current pipeline.
- next step: do not tune MIIM alpha/channel. Pivot to a distinct 2025/2026 hypothesis such as low-rank prompt-style depth guidance or a differential cue module, with the same fixed recipe.

## 2026-05-16 R038 result: DSCF-lite c4-only negative below R016

- branch: `exp/R038-dscf-c4-lite-v1`
- model: `dformerv2_dscf_c4_lite`
- run: `R038_dscf_c4_lite_run01`
- hypothesis: KTB/CVPR 2025 DSCF-style dynamic sparse cross-modal sampling at c4 may suppress depth noise better than dense c4 `GatedFusion`, while leaving c1-c3, decoder, loss, and training recipe unchanged.
- implementation: added a separate model entry that replaces only c4 fusion with `DSCFC4LiteFusion`; c1-c3 remain original `GatedFusion`.
- smoke status: syntax compile, `train.py --help`, registry lookup, and real NYU batch CUDA forward/backward passed. Smoke confirmed c1-c3 original `GatedFusion`, c4 `DSCFC4LiteFusion`, logits `(1, 40, 480, 640)`, finite CE `3.778212`, offset/sample-weight/gate/refine gradients nonzero, non-identical K-branch gradients, and unchanged DFormerv2 pretrained load stats.
- full train status: completed with exit code `0`; `Trainer.fit` reached `max_epochs=50`.
- recorded validation epochs: `50`
- best val/mIoU: `0.530810` at validation epoch `38`
- last val/mIoU: `0.530308`
- last-5 mean val/mIoU: `0.526104`
- last-10 mean val/mIoU: `0.522189`
- best-to-last drop: `0.000502`
- best val/loss: `0.936448` at validation epoch `10`
- last val/loss: `1.218423`
- final train/loss_epoch: `0.056458`
- DSCF c4 offset_abs first/last: `0.961821` / `1.675011`
- DSCF c4 weight_entropy first/last: `1.376656` / `1.336370`
- checkpoint: `checkpoints/R038_dscf_c4_lite_run01/dformerv2_dscf_c4_lite-epoch=37-val_mIoU=0.5308.pt`
- TensorBoard event: `checkpoints/R038_dscf_c4_lite_run01/lightning_logs/version_0/events.out.tfevents.1778863547.Administrator.8372.0`
- saved command: `checkpoints/R038_dscf_c4_lite_run01/run_r038.cmd`
- evidence: `miou_list/R038_dscf_c4_lite_run01.md`
- comparison: R038 is below R016 `0.541121` by `-0.010311`, below R037 `0.534656`, and below R036 `0.539790`.
- conclusion: **negative below corrected baseline.** DSCF-lite c4 sampling opens and is stable, but suppresses peak performance; do not tune K or offset scale.
- next step: pivot to a distinct global-local interaction hypothesis such as HDBFormer MIIM-lite c4-only, or reassess the R016 contract gap before more fusion swaps.

## 2026-05-15 R037 result: DGL minimal stable but below R016

- branch: `exp/R037-dgl-minimal-v1`
- model: `dformerv2_dgl_minimal`
- run: `R037_dgl_minimal_run01`
- hypothesis: DGL-style gradient disentanglement may reduce multimodal optimization conflict by routing fused CE gradients only through fusion/decoder and routing encoder gradients through primary/depth auxiliary CE heads.
- implementation: added a separate model entry where the fused logits are computed from detached primary/depth features; training uses `fusion CE + 0.03 * (primary aux CE + depth aux CE)`. Validation/inference returns only fused logits and does not use the auxiliary heads.
- smoke status: syntax compile, `train.py --help`, registry lookup, and real NYU batch CUDA forward/backward passed. Smoke confirmed fused CE gradients are zero for DFormerv2/DepthEncoder and nonzero for fusion/decoder; aux CE gradients are nonzero for encoders/aux heads and zero for fusion/decoder.
- full train status: completed with exit code `0`; `Trainer.fit` reached `max_epochs=50`.
- recorded validation epochs: `50`
- best val/mIoU: `0.534656` at validation epoch `42`
- last val/mIoU: `0.530153`
- last-5 mean val/mIoU: `0.526926`
- last-10 mean val/mIoU: `0.526304`
- best-to-last drop: `0.004503`
- best val/loss: `0.949518` at validation epoch `12`
- last val/loss: `1.118480`
- final train/loss_epoch: `0.056264`
- DGL aux weight: `0.03`
- primary aux CE first/last: `2.388133` / `0.066309`
- depth aux CE first/last: `2.517628` / `0.131319`
- checkpoint: `checkpoints/R037_dgl_minimal_run01/dformerv2_dgl_minimal-epoch=41-val_mIoU=0.5347.pt`
- TensorBoard event: `checkpoints/R037_dgl_minimal_run01/lightning_logs/version_0/events.out.tfevents.1778857094.Administrator.1736.0`
- evidence: `miou_list/R037_dgl_minimal_run01.md`
- comparison: R037 is below R016 `0.541121` by `-0.006465`; it is stable but not competitive with the current best.
- conclusion: **stable but below corrected baseline.** DGL-style gradient disentanglement reduces late drop, but the peak is suppressed; do not promote or tune aux weight as a micro-search.
- next step: pivot to a distinct fusion-operator hypothesis, with KTB/CVPR 2025 DSCF-lite c4-only as the next highest-value R038 candidate.

## 2026-05-15 R036 result: c3/c4 bounded depth residual partial positive below R016

- branch: `exp/R036-c34-bounded-depth-residual-v1`
- model: `dformerv2_c34_bounded_depth_residual`
- run: `R036_c34_bounded_depth_residual_run01`
- hypothesis: c3/c4-only, zero-initialized, low-amplitude bounded depth residual after the proven R016 `GatedFusion` output may recover residual-depth signal without all-stage instability.
- implementation: added a separate model entry with c1/c2 original `GatedFusion`, c3/c4 `GatedFusionC34BoundedDepthResidual`, and `alpha_max=0.05`; the corrected baseline `dformerv2_mid_fusion`, dataset, metric, val loader, loss type, optimizer, scheduler, batch size, epoch count, learning rate, workers, early stopping, DFormerv2-S level, pretrained loading, DepthEncoder, and SimpleFPNDecoder remain unchanged.
- smoke status: syntax compile, `train.py --help`, registry lookup, and real NYU batch CUDA forward/backward passed. Smoke confirmed exact initial parity with the base GatedFusion path, finite CE loss `3.837778`, nonzero c3/c4 residual gradients, and DFormerv2 pretrained load stats `loaded_keys=774`, `missing_keys=6`, `unexpected_keys=11`.
- full train status: completed with exit code `0`; `Trainer.fit` reached `max_epochs=50`.
- recorded validation epochs: `50`
- best val/mIoU: `0.539790` at validation epoch `44`
- last val/mIoU: `0.528882`
- last-5 mean val/mIoU: `0.516304`
- last-10 mean val/mIoU: `0.521443`
- best-to-last drop: `0.010908`
- best val/loss: `0.950258` at validation epoch `8`
- last val/loss: `1.208208`
- final train/loss_epoch: `0.052817`
- c3 residual alpha first/last: `0.025097` / `0.026970`
- c4 residual alpha first/last: `0.025034` / `0.025553`
- checkpoint: `checkpoints/R036_c34_bounded_depth_residual_run01/dformerv2_c34_bounded_depth_residual-epoch=43-val_mIoU=0.5398.pt`
- TensorBoard event: `checkpoints/R036_c34_bounded_depth_residual_run01/lightning_logs/version_0/events.out.tfevents.1778851092.Administrator.20952.0`
- evidence: `miou_list/R036_c34_bounded_depth_residual_run01.md`
- comparison: R036 is below R016 `0.541121` by `-0.001331`, but stronger than R034 `0.539322` and R035 `0.529498`.
- conclusion: **partial-positive below corrected baseline.** The c3/c4 bounded residual path opens slightly, but does not beat R016 and still has a `0.010908` best-to-last drop.
- next step: do not continue bounded-residual micro-search; choose a distinct next hypothesis, likely geometry-prior consistency or a better-supported 2025/2026 modality-balance mechanism.

## 2026-05-15 R035 result: Gate balance regularizer negative

- branch: `exp/R035-gate-balance-reg-v1`
- model: `dformerv2_gate_balance_reg`
- run: `R035_gate_balance_reg_run01`
- hypothesis: a tiny training-only regularizer on global GatedFusion gate means may reduce modality bias and late instability without changing inference architecture.
- implementation: added a separate gate-stat model entry with `0.01 * mean((gate_mean - 0.5)^2)` during training only; the baseline `dformerv2_mid_fusion`, dataset, metric, val loader, loss type, optimizer, scheduler, batch size, epoch count, learning rate, workers, early stopping, DFormerv2-S level, pretrained loading, DepthEncoder, and SimpleFPNDecoder remain unchanged.
- smoke status: syntax compile, `train.py --help`, registry lookup, and real NYU batch CUDA forward/backward passed. Smoke confirmed finite loss `3.750904`, gate means `[0.500235, 0.499949, 0.499781, 0.499728]`, nonzero `depth_proj` gradient, and unchanged DFormerv2 pretrained load stats.
- full train status: completed with exit code `0`; `Trainer.fit` reached `max_epochs=50`.
- recorded validation epochs: `50`
- best val/mIoU: `0.529498` at validation epoch `38`
- last val/mIoU: `0.521308`
- last-5 mean val/mIoU: `0.506682`
- last-10 mean val/mIoU: `0.510080`
- best-to-last drop: `0.008190`
- best val/loss: `0.966273` at validation epoch `11`
- last val/loss: `1.177837`
- final train/loss_epoch: `0.068845`
- checkpoint: `checkpoints/R035_gate_balance_reg_run01/dformerv2_gate_balance_reg-epoch=37-val_mIoU=0.5295.pt`
- TensorBoard event: `checkpoints/R035_gate_balance_reg_run01/lightning_logs/version_0/events.out.tfevents.1778845626.Administrator.38684.0`
- evidence: `miou_list/R035_gate_balance_reg_run01.md`
- comparison: R035 is below the `0.53` stage threshold by `-0.000502` and below R016 `0.541121` by `-0.011623`.
- conclusion: **negative.** Gate balance regularization slightly constrains late drop but suppresses peak performance below `0.53`; do not tune its lambda.
- next step: pivot to the distinct R036 candidate: c3/c4 bounded low-amplitude depth residual on top of the R016 GatedFusion base.

## 2026-05-15 R034 result: MASG gate-only depth stop-gradient negative unstable

- branch: `exp/R034-masg-gated-fusion-v1`
- model: `dformerv2_masg_fusion`
- run: `R034_masg_gated_fusion_run01`
- hypothesis: detach only the depth projection used by the `GatedFusion` gate to test whether depth-to-gate gradient coupling causes R016 late instability, while preserving the RGB gate path and depth value-path gradients.
- implementation: added a separate `GatedFusionGateStopGrad` module and `DFormerV2MASGFusionSegmentor`; the baseline `dformerv2_mid_fusion`, `DepthEncoder`, `SimpleFPNDecoder`, data pipeline, metric, loss, optimizer, scheduler, batch size, epoch count, learning rate, workers, early stopping, DFormerv2-S level, and pretrained loading remain unchanged.
- smoke status: syntax compile, `train.py --help`, registry lookup, small CUDA forward/backward, and real NYU batch CUDA forward/backward passed. Smoke confirmed logits `(1, 40, 480, 640)`, CE loss `3.733857`, nonzero `depth_proj` gradient, nonzero gate gradient, and DFormerv2 pretrained load stats `loaded_keys=774`, `missing_keys=6`, `unexpected_keys=11`.
- full train status: completed with exit code `0`; `Trainer.fit` reached `max_epochs=50`.
- recorded validation epochs: `50`
- best val/mIoU: `0.539322` at validation epoch `40`
- last val/mIoU: `0.518738`
- last-5 mean val/mIoU: `0.504633`
- last-10 mean val/mIoU: `0.512033`
- best-to-last drop: `0.020584`
- best val/loss: `0.966439` at validation epoch `8`
- last val/loss: `1.230551`
- final train/loss_epoch: `0.059559`
- checkpoint: `checkpoints/R034_masg_gated_fusion_run01/dformerv2_masg_fusion-epoch=39-val_mIoU=0.5393.pt`
- TensorBoard event: `checkpoints/R034_masg_gated_fusion_run01/lightning_logs/version_0/events.out.tfevents.1778839940.Administrator.26056.0`
- evidence: `miou_list/R034_masg_gated_fusion_run01.md`
- comparison: R034 is below R016 `0.541121` by `-0.001799` and has a larger best-to-last drop than R016 (`0.020584` vs `0.013702`).
- conclusion: **negative/unstable relative to corrected baseline.** Depth-only gate stop-gradient does not solve the R016 late-instability problem and should not be promoted to active mainline.
- next step: stop MASG detach micro-search; choose a distinct next hypothesis such as modality-balance regularization or bounded high-stage depth residual.

## 2026-05-15 Post-R033 cleanup checkpoint

- purpose: pause the experiment loop after R033, summarize all `0.53+` runs, and clean active code before the next discussion.
- active best remains R016 `dformerv2_mid_fusion`, best val/mIoU `0.541121`.
- active registry now keeps only the corrected baseline and two stable reference diagnostics: `dformerv2_mid_fusion`, `dformerv2_ham_decoder`, and `dformerv2_geometry_primary_ham_decoder`.
- archived code path: `feiqi/failed_experiments_r019_r033_20260515/`.
- archived directions: R019/R020 depth adapter/blend, R025 DepthEncoder BN eval, R026 official init, R027 primary residual depth injection, R030 residual-top, R031 SimpleFPN classifier dropout, R032 c1 detail gate, and R033 SimpleFPN+Ham logit fusion.
- evidence status: all reports, mIoU lists, metrics, leaderboard entries, and experiment ledgers are retained.
- next status: pause for user discussion; do not launch R034 until explicitly resumed.

## 2026-05-15 R033 result: SimpleFPN Ham logit fusion below corrected baseline

- branch: `exp/R033-simplefpn-ham-logit-fusion-v1`
- model: `dformerv2_simplefpn_ham_logit_fusion`
- run: `R033_simplefpn_ham_logit_fusion_run01`
- hypothesis: SimpleFPN has the strongest corrected peak while Ham variants provide contextual logits, so a small learnable Ham logit residual may complement the R016 SimpleFPN path.
- implementation: added a separate `SimpleFPNHamLogitFusionDecoder` that computes `simple_fpn_logits + alpha * ham_logits`, with `alpha = sigmoid(ham_logit_logit)` initialized near `0.05`; the baseline `dformerv2_mid_fusion` entry remains unchanged.
- smoke status: syntax compile, `train.py --help`, and real-batch CUDA forward/backward smoke passed. Smoke confirmed decoder type `SimpleFPNHamLogitFusionDecoder`, initial alpha `0.050000`, logits `(2, 40, 480, 640)`, CE loss `3.807222`, nonzero alpha gradient, nonzero SimpleFPN/Ham classifier gradients, and peak memory about `5775.9 MB`.
- full train status: completed with exit code `0`; `Trainer.fit` reached `max_epochs=50`.
- recorded validation epochs: `50`
- best val/mIoU: `0.533020` at validation epoch `49`
- last val/mIoU: `0.528883`
- last-5 mean val/mIoU: `0.527628`
- last-10 mean val/mIoU: `0.519951`
- best-to-last drop: `0.004137`
- best val/loss: `0.977882` at validation epoch `9`
- final train/loss_epoch: `0.052197`
- ham logit alpha first/last: `0.050669` / `0.090593`
- checkpoint: `checkpoints/R033_simplefpn_ham_logit_fusion_run01/dformerv2_simplefpn_ham_logit_fusion-epoch=48-val_mIoU=0.5330.pt`
- TensorBoard event: `checkpoints/R033_simplefpn_ham_logit_fusion_run01/lightning_logs/version_0/events.out.tfevents.1778829755.Administrator.10268.0`
- evidence: `miou_list/R033_simplefpn_ham_logit_fusion_run01.md`
- comparison: R033 crosses `0.53`, but is below R016 `0.541121` by `-0.008101`, below R015 `0.537398`, below R027 `0.536739`, below R032 `0.536603`, below R030 `0.536454`, and below the final `0.56` goal.
- conclusion: **partial-positive but not mainline.** The Ham residual branch was learned open, but logits-level Ham complementarity is weaker than the corrected SimpleFPN baseline.
- next step: pause for discussion as requested. If continuing fixed-recipe search, avoid more Ham-logit scalar tuning and consider a cleaner high-stage-only residual-top diagnostic or a broader stability/contract review.

## 2026-05-15 R032 result: SimpleFPN c1 detail gate partial positive below R016

- branch: `exp/R032-simplefpn-c1-detail-gate-v1`
- model: `dformerv2_simplefpn_c1_detail_gate`
- run: `R032_simplefpn_c1_detail_gate_run01`
- hypothesis: the strongest corrected SimpleFPN path may be limited by noisy high-resolution `c1` detail, so a learnable c1 detail strength initialized near baseline-equivalent 1.0 may improve stability or peak mIoU.
- implementation: added a separate `SimpleFPNDecoderC1DetailGate` entry with `p1 = alpha * lateral1(c1) + upsample(p2)` and logged `train/c1_detail_alpha`; the baseline `dformerv2_mid_fusion` entry remains unchanged.
- full train status: completed with exit code `0`; `Trainer.fit` reached `max_epochs=50`.
- recorded validation epochs: `50`
- best val/mIoU: `0.536603` at validation epoch `50`
- last val/mIoU: `0.536603`
- last-5 mean val/mIoU: `0.505390`
- last-10 mean val/mIoU: `0.509657`
- best-to-last drop: `0.000000`
- best val/loss: `0.965559` at validation epoch `9`
- final train/loss_epoch: `0.061732`
- c1 detail alpha first/last: `0.998994` / `0.998770`
- checkpoint: `checkpoints/R032_simplefpn_c1_detail_gate_run01/dformerv2_simplefpn_c1_detail_gate-epoch=49-val_mIoU=0.5366.pt`
- TensorBoard event: `checkpoints/R032_simplefpn_c1_detail_gate_run01/lightning_logs/version_0/events.out.tfevents.1778824352.Administrator.34648.0`
- evidence: `miou_list/R032_simplefpn_c1_detail_gate_run01.md`
- comparison: R032 crosses `0.53`, but is below R016 `0.541121` by `-0.004518`, below R027 `0.536739` by `-0.000136`, and far below the final `0.56` goal.
- conclusion: **partial-positive below corrected baseline.** The last-epoch spike is useful evidence, but alpha barely moved, so this exact c1 detail gate is not enough.
- next step: do not tune the c1 gate. Run R033, a distinct SimpleFPN + Ham logit fusion experiment, to test whether Ham semantic context complements the SimpleFPN head.

## 2026-05-15 R031 result: SimpleFPN classifier dropout negative

- branch: `exp/R031-simplefpn-classifier-dropout-v1`
- model: `dformerv2_simplefpn_classifier_dropout`
- run: `R031_simplefpn_classifier_dropout_run01`
- hypothesis: the strongest corrected R016 SimpleFPN path may benefit from the same small classifier regularizer that helped R022 Ham dropout parity.
- implementation: added a separate `SimpleFPNDecoderWithClassifierDropout` entry that applies `Dropout2d(0.1)` immediately before the SimpleFPN classifier; the baseline `dformerv2_mid_fusion` entry remains unchanged.
- full train status: completed with exit code `0`; `Trainer.fit` reached `max_epochs=50`.
- recorded validation epochs: `50`
- best val/mIoU: `0.531544` at validation epoch `40`
- last val/mIoU: `0.525760`
- last-5 mean val/mIoU: `0.508009`
- last-10 mean val/mIoU: `0.507366`
- best-to-last drop: `0.005784`
- best val/loss: `0.971063` at validation epoch `11`
- final train/loss_epoch: `0.064540`
- checkpoint: `checkpoints/R031_simplefpn_classifier_dropout_run01/dformerv2_simplefpn_classifier_dropout-epoch=39-val_mIoU=0.5315.pt`
- TensorBoard event: `checkpoints/R031_simplefpn_classifier_dropout_run01/lightning_logs/version_0/events.out.tfevents.1778819244.Administrator.21360.0`
- evidence: `miou_list/R031_simplefpn_classifier_dropout_run01.md`
- comparison: R031 crosses `0.53`, but is below R016 `0.541121` by `-0.009577`, below R027 `0.536739`, below R030 `0.536454`, and below R022 `0.534332`.
- conclusion: **negative relative to corrected baseline.** Classifier dropout helps the Ham decoder path, but does not transfer to the SimpleFPN baseline.
- next step: stop SimpleFPN dropout variants. Run R032, a separate SimpleFPN c1 detail-gate entry, to test whether high-resolution c1 detail noise is limiting the strongest path.

## 2026-05-15 R030 result: GatedFusion residual-top partial positive below R016

- branch: `exp/R030-gated-fusion-residual-top-v1`
- model: `dformerv2_gated_fusion_residual_top`
- run: `R030_gated_fusion_residual_top_run01`
- hypothesis: R027's residual-depth signal may help if it is added on top of the proven R016 `GatedFusion` path instead of replacing it.
- implementation: added `GatedFusionResidualTop`, which computes the original `GatedFusion` base and then adds a zero-initialized residual from `rgb_feat`, `depth_proj`, `base`, and `abs(rgb_feat - depth_proj)`.
- full train status: completed with exit code `0`; `Trainer.fit` reached `max_epochs=50`.
- recorded validation epochs: `50`
- best val/mIoU: `0.536454` at validation epoch `42`
- last val/mIoU: `0.529803`
- last-5 mean val/mIoU: `0.506209`
- last-10 mean val/mIoU: `0.511101`
- best-to-last drop: `0.006651`
- best val/loss: `0.975530` at validation epoch `13`
- final train/loss_epoch: `0.057927`
- checkpoint: `checkpoints/R030_gated_fusion_residual_top_run01/dformerv2_gated_fusion_residual_top-epoch=41-val_mIoU=0.5365.pt`
- TensorBoard event: `checkpoints/R030_gated_fusion_residual_top_run01/lightning_logs/version_0/events.out.tfevents.1778813976.Administrator.33508.0`
- evidence: `miou_list/R030_gated_fusion_residual_top_run01.md`
- comparison: R030 crosses `0.53`, but is below R016 `0.541121` by `-0.004667`, below R027 peak `0.536739` by `-0.000285`, and below the final `0.56` goal by `-0.023546`.
- conclusion: **partial-positive below corrected baseline.** Preserving R016 `GatedFusion` improves late recovery versus R027, but all-stage residual-top correction still does not beat the corrected baseline.
- next step: stop residual-family variants for now. Pivot to a distinct low-risk R031 hypothesis: add `Dropout2d(0.1)` only before the SimpleFPN classifier in a separate model entry, inspired by the R022 Ham dropout parity gain.

## 2026-05-15 R027 result: primary residual depth injection partial positive but unstable

- branch: `exp/R027-primary-residual-depth-injection-v1`
- model: `dformerv2_primary_residual_depth`
- run: `R027_primary_residual_depth_run01`
- hypothesis: the main fusion issue may be feature replacement; preserve DFormerv2 primary features and inject external DepthEncoder information only through a zero-initialized residual branch.
- implementation: replaced the four `GatedFusion` modules with `PrimaryResidualDepthInjection`, where `out = rgb_feat + residual(depth_proj, abs(rgb_feat - depth_proj))` and the last residual conv is zero-initialized.
- full train status: completed with exit code `0`; `Trainer.fit` reached `max_epochs=50`.
- recorded validation epochs: `50`
- best val/mIoU: `0.536739` at validation epoch `41`
- last val/mIoU: `0.505286`
- last-5 mean val/mIoU: `0.519799`
- last-10 mean val/mIoU: `0.522758`
- best-to-last drop: `0.031453`
- best val/loss: `0.964226` at validation epoch `9`
- final train/loss_epoch: `0.122960`
- checkpoint: `checkpoints/R027_primary_residual_depth_run01/dformerv2_primary_residual_depth-epoch=40-val_mIoU=0.5367.pt`
- TensorBoard event: `checkpoints/R027_primary_residual_depth_run01/lightning_logs/version_0/events.out.tfevents.1778808666.Administrator.14000.0`
- evidence: `miou_list/R027_primary_residual_depth_run01.md`
- comparison: R027 crosses the fixed-recipe `0.53` threshold, but is below R016 `0.541121` by `-0.004382` and below the final `0.56` goal by `-0.023261`.
- conclusion: **partial-positive peak, stability negative.** Primary residual depth injection has signal, but replacing the proven R016 `GatedFusion` path is too unstable to promote.
- next step: do not tune R027 scales. Run a narrower residual-on-top experiment that preserves R016 `GatedFusion` output at initialization and adds only a zero-initialized correction.

## 2026-05-15 R026 result: official-style local init negative

- branch: `exp/R026-official-init-local-modules-v1`
- model: `dformerv2_official_init_local_modules`
- run: `R026_official_init_local_modules_run01`
- hypothesis: local random fusion/decoder modules may be under-initialized relative to the official DFormer decode head contract.
- implementation: added a model entry that applies official-style initialization only to `GatedFusion` and `SimpleFPNDecoder`: Conv2d Kaiming fan-in/relu, BatchNorm2d `eps=1e-3`, `momentum=0.1`, weight `1`, bias `0`.
- full train status: completed with exit code `0`; `Trainer.fit` reached `max_epochs=50`.
- recorded validation epochs: `50`
- best val/mIoU: `0.507906` at validation epoch `33`
- last val/mIoU: `0.499770`
- last-5 mean val/mIoU: `0.496476`
- last-10 mean val/mIoU: `0.495483`
- best-to-last drop: `0.008136`
- best val/loss: `1.073346` at validation epoch `5`
- final train/loss_epoch: `0.054818`
- checkpoint: `checkpoints/R026_official_init_local_modules_run01/dformerv2_official_init_local_modules-epoch=32-val_mIoU=0.5079.pt`
- TensorBoard event: `checkpoints/R026_official_init_local_modules_run01/lightning_logs/version_0/events.out.tfevents.1778803189.Administrator.35684.0`
- evidence: `miou_list/R026_official_init_local_modules_run01.md`
- comparison: R026 is below R016 `0.541121` by `-0.033215`, below R025 `0.532572` by `-0.024666`, and below the `0.53` stage threshold.
- conclusion: **negative.** Official-style initialization of only the local random modules hurts this local pipeline and should not be continued.
- next step: switch to a fusion-form hypothesis: primary-preserving residual depth injection initialized as DFormerv2 identity.

## 2026-05-15 R025 result: DepthEncoder BN eval peak positive but unstable

- branch: `exp/R025-depth-encoder-bn-eval-v1`
- model: `dformerv2_depth_encoder_bn_eval`
- run: `R025_depth_encoder_bn_eval_run01`
- hypothesis: small-batch BatchNorm drift inside the external ResNet-18 DepthEncoder may destabilize the corrected mid-fusion path.
- implementation: added a model entry that keeps only DepthEncoder `BatchNorm2d` modules in eval mode during training; BN affine parameters remain trainable.
- full train status: completed with exit code `0`; `Trainer.fit` reached `max_epochs=50`.
- recorded validation epochs: `50`
- best val/mIoU: `0.532572` at validation epoch `47`
- last val/mIoU: `0.496030`
- last-5 mean val/mIoU: `0.520333`
- last-10 mean val/mIoU: `0.517969`
- best-to-last drop: `0.036541`
- best val/loss: `0.961156` at validation epoch `9`
- final train/loss_epoch: `0.095029`
- checkpoint: `checkpoints/R025_depth_encoder_bn_eval_run01/dformerv2_depth_encoder_bn_eval-epoch=46-val_mIoU=0.5326.pt`
- TensorBoard event: `checkpoints/R025_depth_encoder_bn_eval_run01/lightning_logs/version_0/events.out.tfevents.1778798018.Administrator.26772.0`
- evidence: `miou_list/R025_depth_encoder_bn_eval_run01.md`
- comparison: R025 is above R024 `0.530186` by `+0.002386`, but below R022 `0.534332` by `-0.001760` and below R016 `0.541121` by `-0.008549`.
- conclusion: **partial-positive peak, stability negative.** DepthEncoder BN eval did not fix late collapse and should not become the next base.
- next step: do not build on BN eval. Test official-style initialization of only local random modules (`GatedFusion` + `SimpleFPNDecoder`) while leaving pretrained DFormerv2 and DepthEncoder untouched.

## 2026-05-15 R024 result: raw DFormerv2-S + Ham stable positive below corrected baseline

- branch: `exp/R024-geometry-primary-ham-decoder-v1`
- model: `dformerv2_geometry_primary_ham_decoder`
- run: `R024_geometry_primary_ham_decoder_run01`
- hypothesis: raw `DFormerv2_S(rgb, depth)` features may match the official Ham decoder contract better than the local post-backbone external DepthEncoder/GatedFusion stack.
- implementation: added a model entry for `DFormerv2_S(rgb, depth) -> OfficialHamDecoder`, with no external DepthEncoder or GatedFusion.
- full train status: completed with exit code `0`; `Trainer.fit` reached `max_epochs=50`.
- recorded validation epochs: `50`
- best val/mIoU: `0.530186` at validation epoch `45`
- last val/mIoU: `0.529383`
- last-5 mean val/mIoU: `0.521843`
- last-10 mean val/mIoU: `0.522327`
- best-to-last drop: `0.000803`
- best val/loss: `1.062941` at validation epoch `8`
- final train/loss_epoch: `0.071329`
- checkpoint: `checkpoints/R024_geometry_primary_ham_decoder_run01/dformerv2_geometry_primary_ham_decoder-epoch=44-val_mIoU=0.5302.pt`
- TensorBoard event: `checkpoints/R024_geometry_primary_ham_decoder_run01/lightning_logs/version_0/events.out.tfevents.1778793121.Administrator.20368.0`
- evidence: `miou_list/R024_geometry_primary_ham_decoder_run01.md`
- comparison: R024 is above the fixed-recipe `0.53` stage threshold, but below R022 `0.534332` by `-0.004146` and below R016 `0.541121` by `-0.010935`.
- conclusion: **stable positive structure diagnostic, not a new best.** Raw DFormerv2-S + Ham is stable and crosses `0.53`, but it does not beat the local external-fusion baseline, so the external DepthEncoder/GatedFusion path is not simply harmful.
- next step: do not continue Ham micro-fixes. Test a targeted stability hypothesis on the stronger corrected mid-fusion path, such as external DepthEncoder BatchNorm eval/freeze during training.

## 2026-05-15 R023 result: corrected-contract geometry-primary teacher negative gate

- branch: `exp/R023-corrected-contract-teacher-refresh-v1`
- model: `dformerv2_geometry_primary_teacher`
- run: `R023_geometry_primary_teacher_corrected_contract_run01`
- hypothesis: refreshing the geometry-primary teacher under the corrected R015/R016 label/depth contract can decide whether corrected-contract PMAD/KD deserves another student experiment.
- implementation: no model-code change; existing `DFormerV2GeometryPrimaryTeacherSegmentor` was trained under the current corrected data contract.
- full train status: completed with exit code `0`; `Trainer.fit` reached `max_epochs=50`.
- recorded validation epochs: `50`
- best val/mIoU: `0.524498` at validation epoch `43`
- last val/mIoU: `0.507023`
- last-5 mean val/mIoU: `0.510467`
- last-10 mean val/mIoU: `0.512531`
- best-to-last drop: `0.017475`
- best val/loss: `0.988437` at validation epoch `10`
- final train/loss_epoch: `0.092329`
- checkpoint: `checkpoints/R023_geometry_primary_teacher_corrected_contract_run01/dformerv2_geometry_primary_teacher-epoch=42-val_mIoU=0.5245.pt`
- TensorBoard event: `checkpoints/R023_geometry_primary_teacher_corrected_contract_run01/lightning_logs/version_0/events.out.tfevents.1778787920.Administrator.37368.0`
- evidence: `miou_list/R023_geometry_primary_teacher_corrected_contract_run01.md`
- comparison: R023 is below R016 `0.541121` by `-0.016623`, below R022 `0.534332` by `-0.009834`, and below the `0.53` teacher gate.
- conclusion: **negative teacher gate.** The corrected-contract geometry-primary teacher is still too weak and unstable to justify a corrected PMAD student run from this checkpoint.
- next step: skip corrected PMAD for now and run a non-KD structure isolation experiment: raw `DFormerv2_S(rgb, depth) -> OfficialHamDecoder`, with no external DepthEncoder or GatedFusion.

## 2026-05-15 R022 result: Ham dropout parity fix partial positive, still below R016

- branch: `exp/R022-ham-dropout-parity-v1`
- model: `dformerv2_ham_decoder`
- run: `R022_ham_dropout_parity_run01`
- hypothesis: R021 may underperform because it omitted official `BaseDecodeHead.cls_seg()` `Dropout2d(0.1)`; add only that dropout before the Ham classifier.
- implementation: `OfficialHamDecoder` now applies `Dropout2d(0.1)` before `classifier`.
- full train status: completed with exit code `0`; `Trainer.fit` reached `max_epochs=50`.
- recorded validation epochs: `50`
- best val/mIoU: `0.534332` at validation epoch `50`
- last val/mIoU: `0.534332`
- last-5 mean val/mIoU: `0.527687`
- last-10 mean val/mIoU: `0.512629`
- best-to-last drop: `0.000000`
- best val/loss: `1.106345` at validation epoch `21`
- final train/loss_epoch: `0.059158`
- checkpoint: `checkpoints/R022_ham_dropout_parity_run01/dformerv2_ham_decoder-epoch=49-val_mIoU=0.5343.pt`
- TensorBoard event: `checkpoints/R022_ham_dropout_parity_run01/lightning_logs/version_0/events.out.tfevents.1778782418.Administrator.36360.0`
- evidence: `miou_list/R022_ham_dropout_parity_run01.md`
- comparison: R022 improves R021 `0.527353` by `+0.006979`, improves R020 `0.532924` by `+0.001408`, but remains below R016 `0.541121` by `-0.006790`.
- conclusion: **partial-positive parity fix, not a new corrected baseline.** Official classifier dropout matters, and the Ham decoder path is now the strongest retained method variant, but it still does not beat the corrected baseline.
- next step: stop Ham decoder micro-fixes for now. Run corrected-contract geometry-primary teacher refresh, then decide whether corrected PMAD student is worth running.

## 2026-05-15 R021 result: LightHam-like decoder negative below R016

- branch: `exp/R021-official-ham-decoder-parity-v1`
- model: `dformerv2_ham_decoder`
- run: `R021_official_ham_decoder_parity_run01`
- hypothesis: after R015/R016 align label and depth contracts, the remaining gap to DFormerv2-S reference performance may come from the decoder/head contract; test a self-contained c2-c4 LightHam-like decoder instead of `SimpleFPNDecoder`.
- implementation: c2/c3/c4 fused features are upsampled to c2, concatenated, passed through `squeeze -> NMF Hamburger -> align -> classifier`, and upsampled to input resolution.
- audit caveat: this implementation omits official `BaseDecodeHead.cls_seg()` `Dropout2d(0.1)`, so R021 should be described as LightHam-like rather than strict official Ham parity.
- full train status: completed with exit code `0`; `Trainer.fit` reached `max_epochs=50`.
- recorded validation epochs: `50`
- best val/mIoU: `0.527353` at validation epoch `39`
- last val/mIoU: `0.501377`
- last-5 mean val/mIoU: `0.503158`
- last-10 mean val/mIoU: `0.506140`
- best-to-last drop: `0.025976`
- best val/loss: `1.121119` at validation epoch `7`
- final train/loss_epoch: `0.177592`
- checkpoint: `checkpoints/R021_official_ham_decoder_parity_run01/dformerv2_ham_decoder-epoch=38-val_mIoU=0.5274.pt`
- TensorBoard event: `checkpoints/R021_official_ham_decoder_parity_run01/lightning_logs/version_0/events.out.tfevents.1778777116.Administrator.12456.0`
- evidence: `miou_list/R021_official_ham_decoder_parity_run01.md`
- comparison: R021 is below R016 `0.541121` by `-0.013768`, below R020 `0.532924` by `-0.005571`, and barely below R010 `0.527469` by `-0.000116`.
- conclusion: **negative decoder result.** The current LightHam-like decoder does not improve the corrected pipeline and shows late instability.
- next step: run one minimal R022 Ham parity fix by adding official `Dropout2d(0.1)` before the classifier. If it remains below R016/R020, stop Ham decoder work and move to corrected-contract PMAD teacher refresh.

## 2026-05-15 R020 result: branch-specific depth blend adapter improves stability but remains below R016

- branch: `exp/R020-depth-blend-adapter-v1`
- model: `dformerv2_branch_depth_blend_adapter`
- run: `R020_branch_depth_blend_adapter_run01`
- hypothesis: R019's hard DepthEncoder input switch can be stabilized by a learnable global convex blend between R016 official-normalized depth and reconstructed `[0,1]` depth.
- implementation: DFormerv2 receives the original R016 normalized depth; DepthEncoder receives `(1-alpha)*depth + alpha*depth01`, where `depth01 = clamp(depth * 0.28 + 0.48, 0, 1)` and `alpha` is initialized near `0.05`.
- full train status: completed with exit code `0`; `Trainer.fit` reached `max_epochs=50`.
- recorded validation epochs: `50`
- best val/mIoU: `0.532924` at validation epoch `41`
- last val/mIoU: `0.503238`
- last-5 mean val/mIoU: `0.520456`
- last-10 mean val/mIoU: `0.516804`
- best-to-last drop: `0.029686`
- best val/loss: `0.979484` at validation epoch `8`
- final train/loss_epoch: `0.089993`
- alpha first/last: `0.050022` / `0.051455`
- checkpoint: `checkpoints/R020_branch_depth_blend_adapter_run01/dformerv2_branch_depth_blend_adapter-epoch=40-val_mIoU=0.5329.pt`
- TensorBoard event: `checkpoints/R020_branch_depth_blend_adapter_run01/lightning_logs/version_0/events.out.tfevents.1778771221.Administrator.41764.0`
- evidence: `miou_list/R020_branch_depth_blend_adapter_run01.md`
- comparison: R020 is slightly above R019 `0.532539` by `+0.000385`, but below R016 `0.541121` by `-0.008197`.
- conclusion: **partial-positive stabilization signal, not a new best.** R020 improves R019 stability through late epochs but still drops at the final validation epoch and does not beat the corrected baseline.
- next step: do not blindly repeat global alpha blend. Either target late stability directly or test a richer branch-specific adapter; if the goal is reference-gap diagnosis, run official Ham parity audit.

## 2026-05-14 R019 result: branch-specific depth adapter partial positive, unstable

- branch: `exp/R019-branch-depth-adapter-v1`
- model: `dformerv2_branch_depth_adapter`
- run: `R019_branch_depth_adapter_run01`
- hypothesis: R016 official depth normalization helps the DFormerv2 geometry branch, but the external ResNet-18 DepthEncoder may need a branch-specific `[0,1]` depth representation reconstructed inside the model.
- implementation: DFormerv2 receives the original R016 normalized depth; DepthEncoder receives `torch.clamp(depth * 0.28 + 0.48, 0, 1)`.
- full train status: completed with exit code `0`; `Trainer.fit` reached `max_epochs=50`.
- recorded validation epochs: `50`
- best val/mIoU: `0.532539` at validation epoch `46`
- last val/mIoU: `0.495229`
- last-5 mean val/mIoU: `0.509575`
- last-10 mean val/mIoU: `0.518038`
- best-to-last drop: `0.037311`
- best val/loss: `0.958302` at validation epoch `8`
- final train/loss_epoch: `0.067030`
- checkpoint: `checkpoints/R019_branch_depth_adapter_run01/dformerv2_branch_depth_adapter-epoch=45-val_mIoU=0.5325.pt`
- TensorBoard event: `checkpoints/R019_branch_depth_adapter_run01/lightning_logs/version_0/events.out.tfevents.1778765914.Administrator.27112.0`
- evidence: `miou_list/R019_branch_depth_adapter_run01.md`
- comparison: R019 is above the fixed-recipe 0.53 threshold, but below the current R016 corrected baseline `0.541121` by `-0.008582`.
- comparison: R019 is above R018 `0.526282` by `+0.006257`, but the late collapse is much worse.
- conclusion: **partial-positive original-method signal, not a new best.** Branch-specific depth representation can create a high peak, but the simple `[0,1]` reconstruction is unstable.
- next step: do not claim R019 as the main result. If continuing this line, stabilize the adapter instead of repeating blindly; otherwise run official Ham parity audit to quantify the remaining DFormer reference-structure gap.

## 2026-05-14 R018 result: official drop_path_rate 0.25 negative

- branch: `exp/R018-droppath025-contract-v1`
- model tested: `dformerv2_mid_fusion_dpr025`
- official result run: `R018_dformerv2_mid_fusion_dpr025_retry1`
- hypothesis: official DFormerv2-S NYUDepthV2 uses `drop_path_rate=0.25`; testing that value after R015/R016 corrected label and depth contracts might close part of the gap to the DFormerv2-S reference result.
- full train status: retry1 completed with exit code `0`; `Trainer.fit` reached `max_epochs=50`.
- recorded validation epochs: `50`
- best val/mIoU: `0.526282` at validation epoch `46`
- last val/mIoU: `0.522893`
- last-5 mean val/mIoU: `0.512694`
- last-10 mean val/mIoU: `0.513363`
- best-to-last drop: `0.003389`
- best val/loss: `0.948631` at validation epoch `10`
- final train/loss_epoch: `0.078764`
- checkpoint: `checkpoints/R018_dformerv2_mid_fusion_dpr025_retry1/dformerv2_mid_fusion_dpr025-epoch=45-val_mIoU=0.5263.pt`
- TensorBoard event: `checkpoints/R018_dformerv2_mid_fusion_dpr025_retry1/lightning_logs/version_0/events.out.tfevents.1778760450.Administrator.7836.0`
- evidence: `miou_list/R018_dformerv2_mid_fusion_dpr025_retry1.md`
- comparison: R016 corrected baseline best was `0.541121`; R018 is lower by `-0.014839`.
- comparison: R010 PMAD logit-only best was `0.527469`; R018 is lower by `-0.001187`.
- conclusion: **negative official-contract gate.** `drop_path_rate=0.25` should not be promoted for this local DFormerv2 + DepthEncoder + GatedFusion + SimpleFPN pipeline.
- process note: the first foreground launch reached 42 validation epochs and then stopped progressing after the command wrapper timed out and the stdout/progress pipe became unhealthy. It is partial process evidence only and is excluded from the result.
- archive: failed active-code diff recorded in `feiqi/failed_experiments_r014_plus_20260514/R018_droppath025_contract.md`; `src/models/mid_fusion.py` and `train.py` are restored to the R017/R016 mainline state.
- next step: keep R016 as the corrected baseline. If continuing below `0.56`, stop baseline-contract micro-gates unless a new mismatch is found; next highest-value candidates are official Ham parity audit, corrected-contract PMAD teacher refresh, or a branch-specific depth input adapter.

## 2026-05-14 R017 result: official RGB/BGR channel contract negative

- branch: `exp/R017-rgb-bgr-contract-v1`
- model: `dformerv2_mid_fusion`
- run: `R017_rgb_bgr_official_contract`
- hypothesis: after R015/R016 align official NYU label and depth contracts, RGB channel order might also need to match official DFormer NYUDepthV2 BGR input behavior.
- full train status: completed with exit code `0`; `Trainer.fit` reached `max_epochs=50`.
- recorded validation epochs: `50`
- best val/mIoU: `0.529090` at validation epoch `38`
- last val/mIoU: `0.523078`
- last-5 mean val/mIoU: `0.494251`
- last-10 mean val/mIoU: `0.506949`
- best-to-last drop: `0.006011`
- best val/loss: `0.973518` at validation epoch `9`
- last val/loss: `1.228286`
- final train/loss_epoch: `0.063107`
- checkpoint: `checkpoints/R017_rgb_bgr_official_contract/dformerv2_mid_fusion-epoch=37-val_mIoU=0.5291.pt`
- TensorBoard event: `checkpoints/R017_rgb_bgr_official_contract/lightning_logs/version_0/events.out.tfevents.1778750750.Administrator.38244.0`
- evidence: `miou_list/R017_rgb_bgr_official_contract.md`
- comparison: R016 official-label-and-depth baseline best was `0.541121`; R017 is lower by `-0.012031`.
- conclusion: **negative official-contract gate.** Although DFormer official NYUDepthV2 keeps BGR input, this local adaptation performs better with the R016 RGB input path. Do not merge active BGR code into main.
- archive: failed active-code diff recorded in `feiqi/failed_experiments_r014_plus_20260514/R017_rgb_bgr_contract.md`; `src/data_module.py` is restored to the R016 RGB path.
- next step: keep R016 as corrected baseline and test the next official baseline-contract gate, DFormerv2-S `drop_path_rate=0.25`.

## 2026-05-14 R017 dry-check: official RGB/BGR channel contract

- branch: `exp/R017-rgb-bgr-contract-v1`
- model: `dformerv2_mid_fusion`
- planned run: `R017_rgb_bgr_official_contract`
- hypothesis: after R015/R016 align the official NYU label and depth contracts, RGB channel order should match the official DFormer NYUDepthV2 input contract. Official DFormer keeps OpenCV BGR for non-SUNRGBD datasets, while the local data module converted BGR to RGB.
- official code evidence: `ref_codes/DFormer/utils/dataloader/RGBXDataset.py` sets `rgb_mode = "BGR"` for non-SUNRGBD datasets and `_open_image(..., "BGR")` returns `cv2.imread(...)` without `BGR2RGB`.
- implementation scope: `src/data_module.py` only. Remove `cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)` and keep R015 label mapping plus R016 depth normalization unchanged.
- forbidden-change check: no model, decoder, backbone, split file, label mapping, depth normalization, metric, mIoU calculation, optimizer, scheduler, batch size, epoch count, learning rate, worker count, early stopping, pretrained-loading logic, checkpoint artifact, dataset, or TensorBoard event change is approved.
- result status: smoke and full train pending; no mIoU claim yet.

## 2026-05-14 R016 result: official depth normalization improves official-contract baseline

- branch: `exp/R016-depth-norm-contract-v1`
- model: `dformerv2_mid_fusion`
- run: `R016_depth_norm_official_baseline_retry1`
- hypothesis: after R015 aligns the official NYU label/ignore contract, depth input should also follow the official DFormer modal_x normalization contract: `raw / 255.0`, then `(x - 0.48) / 0.28`.
- full train status: retry1 completed with exit code `0`; `Trainer.fit` reached `max_epochs=50`.
- recorded validation epochs: `50`
- best val/mIoU: `0.541121` at validation epoch `49`
- last val/mIoU: `0.527420`
- last-5 mean val/mIoU: `0.535500`
- last-10 mean val/mIoU: `0.524063`
- best-to-last drop: `0.013702`
- best val/loss: `0.978448` at validation epoch `14`
- last val/loss: `1.211359`
- final train/loss_epoch: `0.053537`
- checkpoint: `checkpoints/R016_depth_norm_official_baseline_retry1/dformerv2_mid_fusion-epoch=48-val_mIoU=0.5411.pt`
- TensorBoard event: `checkpoints/R016_depth_norm_official_baseline_retry1/lightning_logs/version_0/events.out.tfevents.1778745208.Administrator.36016.0`
- evidence: `miou_list/R016_depth_norm_official_baseline_retry1.md`
- comparison: R015 official-label baseline best was `0.537398`; R016 retry1 improves by `+0.003723`.
- conclusion: **positive official-contract alignment.** R016 is the strongest current full-train run and confirms that preprocessing-contract alignment is the highest-value direction so far.
- paper boundary: this is not a novel method contribution. It should be reported as DFormer official modal_x/depth preprocessing alignment, with DFormer cited, and used as the corrected baseline for later original method work.
- process note: the first `R016_depth_norm_official_baseline` launch was interrupted after 47 validation epochs by `forrtl error (200): program aborting due to window-CLOSE event` when the command window was closed. It is recorded only as partial process evidence; the official result is retry1.
- next step: continue toward `0.56` from this official-label-and-depth baseline, likely with the next contract mismatch gate (`RGB/BGR` input contract) before returning to original lightweight method design.

## 2026-05-14 R016 dry-check: official depth normalization contract

- branch: `exp/R016-depth-norm-contract-v1`
- model: `dformerv2_mid_fusion`
- planned run: `R016_depth_norm_official_baseline`
- hypothesis: after R015 aligns the official NYU label/ignore contract, depth input should also follow the official DFormer modal_x normalization contract: `raw / 255.0`, then `(x - 0.48) / 0.28`.
- official code evidence: `ref_codes/DFormer/utils/dataloader/dataloader.py` normalizes `modal_x` with `[0.48, 0.48, 0.48]` / `[0.28, 0.28, 0.28]`; `ref_codes/DFormer/utils/transforms.py` implements normalize as `/255.0`, subtract mean, divide std.
- implementation scope: `src/data_module.py` only. `depth` is now an Albumentations `mask` target so RGB ImageNet Normalize does not touch it, then depth is manually normalized with DFormer stats.
- forbidden-change check: no model, decoder, backbone, split file, label mapping, metric, mIoU calculation, optimizer, scheduler, batch size, epoch count, learning rate, worker count, early stopping, pretrained-loading logic, checkpoint artifact, dataset, or TensorBoard event change is approved.
- smoke status: `py_compile`, `train.py --help`, real-batch stats, and CUDA forward sanity passed. Real-batch depth range was `[-1.714286, 1.857143]`; logits were `(2, 40, 480, 640)` with CE loss `3.711953`.
- result status: full train running; no mIoU claim yet.

## 2026-05-14 R015 result: official-label baseline success for 0.53 stage

- branch: `exp/R015-label-ignore-contract-v1`
- model: `dformerv2_mid_fusion`
- run: `R015_label_ignore_official_baseline`
- hypothesis: official DFormer NYU label mapping `0 -> 255 ignore`, `1..40 -> 0..39` is required before judging the gap to official DFormerv2-S results.
- full train status: completed with exit code `0`; `Trainer.fit` reached `max_epochs=50`.
- recorded validation epochs: `50`
- best val/mIoU: `0.537398` at validation epoch `45`
- last val/mIoU: `0.499418`
- last-5 mean val/mIoU: `0.520010`
- last-10 mean val/mIoU: `0.520691`
- best-to-last drop: `0.037981`
- best val/loss: `0.969897` at validation epoch `10`
- last val/loss: `1.291720`
- final train/loss_epoch: `0.093611`
- checkpoint: `checkpoints/R015_label_ignore_official_baseline/dformerv2_mid_fusion-epoch=44-val_mIoU=0.5374.pt`
- TensorBoard event: `checkpoints/R015_label_ignore_official_baseline/lightning_logs/version_0/events.out.tfevents.1778734783.Administrator.15996.0`
- evidence: `miou_list/R015_label_ignore_official_baseline.md`
- conclusion: **fixed-recipe 0.53 stage success, new official-label baseline.** R015 crosses `0.53` without changing model structure, split, loader, augmentation, optimizer, scheduler, epoch, batch, lr, worker, early stopping, DFormerv2-S level, or pretrained loading.
- contract boundary: because R015 changes label/ignore semantics, it is not a direct old-contract gain claim over the previous `0.517397` baseline. Later runs should compare against R015 under the official-label contract.
- caveat: late instability remains; the best-to-last drop is `0.037981`, so this is not yet a stable `0.56` solution.
- next step: continue from this official-label baseline and test official depth normalization contract alignment as a single fixed-recipe hypothesis.

## 2026-05-14 R015 implementation dry-check

- branch: `exp/R015-label-ignore-contract-v1`
- planned model: `dformerv2_mid_fusion`
- planned run: `R015_label_ignore_official_baseline`
- hypothesis: official DFormer NYU label mapping `0 -> 255 ignore`, `1..40 -> 0..39` is required before judging the gap to official DFormerv2-S results.
- implementation scope: map labels in `src/data_module.py`; simplify `src/utils/metrics.py` to expect train-id labels plus `255` ignore.
- contract boundary: this changes label/ignore semantics and must be treated as a new official-label baseline. It is not directly comparable to old runs as a simple improvement claim.
- forbidden-change check: no dataset split file, sample list, validation/test loader behavior, data augmentation, optimizer, scheduler, epoch, batch, lr, worker, checkpoint artifact, dataset, pretrained weight, TensorBoard log, backbone, encoder, decoder, or model structure change is approved.
- smoke status: `py_compile`, `train.py --help`, label unit mapping, and real-batch forward smoke passed. Real-batch labels were canonical `0..39/255`, class `39` was present, and the model produced logits `(2, 40, 480, 640)` with CE loss `3.827293`.
- result status: full train pending; no mIoU claim yet.

## 2026-05-14 cleanup/nyu056-mainline

- purpose: clean active code and branch state before R014 and the longer `0.56` goal loop.
- action: merged R010/R012/R013 evidence ledgers into the cleanup branch while keeping failed model code out of the active training path.
- active models after cleanup: `dformerv2_mid_fusion`, `dformerv2_tgga_c4only_beta002_aux003_detachsem_simplefpn_v1`, `dformerv2_geometry_primary_teacher`, `dformerv2_primkd_logit_only`, plus legacy `early` and `mid_fusion`.
- archived code: TGGA c3/c4 and weak-c3 snapshot, pre-cleanup registry snapshot, depth FFT select, FFT frequency enhance, and FFT HiLo under `feiqi/failed_experiments_r001_r013_20260514/`.
- kept evidence: no `docs`, `miou_list`, `reports`, `metrics`, or experiment ledger evidence was deleted.
- training entrypoint: fixed the callback wiring so `TQDMProgressBar` remains active and added UTF-8 stdout/stderr reconfigure for Windows/Rich output.
- forbidden-change check: no dataset split, dataloader, augmentation, validation, metric, mIoU, optimizer, scheduler, epoch, batch, lr, worker, checkpoint artifact, dataset, pretrained-weight, or TensorBoard-log change was made.
- next step: create `exp/R014-pmad-tgga-c4-v1` from the cleaned mainline and run one fixed-recipe PMAD logit-only + TGGA c4-only full train.

## 2026-05-13 dformerv2_lmlp_decoder_run01

- model: `dformerv2_lmlp_decoder`
- method: DFormer/SegFormer-style c2-c4 LMLP decoder head.
- purpose: test whether SimpleFPN top-down additive fusion is limiting the fused DFormerv2 RGB-D features.
- architecture: `DFormerv2_S + ResNet-18 DepthEncoder + GatedFusion + LMLPDecoder`.
- decoder: c2/c3/c4 MLP projections to `embed_dim=768`, upsample c3/c4 to c2, concatenate, `1x1` fuse + BN + ReLU + dropout + classifier, then upsample logits to input size.
- settings: `batch_size=2`, `max_epochs=50`, `lr=6e-5`, `num_workers=4`, `early_stop_patience=30`, `loss_type=ce`
- recorded validation epochs: `50`
- best val/mIoU: `0.517981` at epoch `41`
- last val/mIoU: `0.490231`
- last-5 mean val/mIoU: `0.505172`
- last-10 mean val/mIoU: `0.508065`
- best val/loss: `1.018381` at epoch `4`
- final train/loss: `0.208969`
- comparison clean 10-run RGB-D baseline mean best: `0.517397`
- comparison clean 10-run RGB-D baseline std: `0.004901`
- comparison clean 10-run RGB-D baseline mean + 1 std: `0.522298`
- comparison clean 10-run RGB-D baseline best single: `0.524425`
- comparison R010 PMAD run06_retry1 best: `0.527469`
- comparison R004 TGGA c4-only best: `0.522849`
- delta vs clean baseline mean: `+0.000584` (`+0.119` baseline std units)
- delta vs clean baseline mean + 1 std: `-0.004317`
- delta vs clean baseline best single: `-0.006444`
- delta vs R010 run06_retry1: `-0.009488`
- delta vs R004 TGGA c4-only: `-0.004868`
- best-to-last delta: `-0.027750`
- gap to `0.53` goal: `-0.012019`
- checkpoint: `checkpoints/dformerv2_lmlp_decoder_run01/dformerv2_lmlp_decoder-epoch=40-val_mIoU=0.5180.pt`
- TensorBoard event: `checkpoints/dformerv2_lmlp_decoder_run01/lightning_logs/version_0/events.out.tfevents.1778682038.Administrator.26812.0`
- evidence: `miou_list/dformerv2_lmlp_decoder_run01.md`
- conclusion: **weak near-baseline result, negative for the goal.** LMLP slightly exceeds the baseline mean at its best epoch but remains below baseline mean + 1 std, below stronger partial-positive branches, and far below `0.53`.
- next step: pause per user request. Decoder replacement alone is not the obvious path; discuss whether to pursue safer c4-only reliability calibration, stronger-but-careful KD, or a broader model/recipe change that would require changing the experiment contract.

## 2026-05-13 dformerv2_primkd_logit_only_w015_t4_run07

- model: `dformerv2_primkd_logit_only`
- method: PMAD/PrimKD-style logit-only KD repeat at the strongest prior setting.
- purpose: test whether R010's high-tail `kd_weight=0.15`, `kd_temperature=4.0` result can reproduce or reach the `0.53` goal as run07.
- architecture: `DFormerv2_S + ResNet-18 DepthEncoder + GatedFusion + SimpleFPNDecoder` student, frozen geometry-primary DFormerv2 teacher.
- teacher checkpoint: `checkpoints/dformerv2_geometry_primary_teacher_run01/dformerv2_geometry_primary_teacher-epoch=37-val_mIoU=0.5168.pt`
- settings: `batch_size=2`, `max_epochs=50`, `lr=6e-5`, `num_workers=4`, `early_stop_patience=30`, `loss_type=ce`
- KD settings: `kd_weight=0.15`, `kd_temperature=4.0`, logit-only, no feature KD, `--save_student_only`
- recorded validation epochs: `50`
- best val/mIoU: `0.516967` at epoch `43`
- last val/mIoU: `0.508205`
- last-5 mean val/mIoU: `0.496441`
- last-10 mean val/mIoU: `0.503120`
- best val/loss: `1.039913` at epoch `8`
- final train/loss: `0.216768`
- final train/ce_loss: `0.153825`
- final train/kd_loss: `0.419615`
- comparison clean 10-run RGB-D baseline mean best: `0.517397`
- comparison clean 10-run RGB-D baseline std: `0.004901`
- comparison clean 10-run RGB-D baseline mean + 1 std: `0.522298`
- comparison clean 10-run RGB-D baseline best single: `0.524425`
- comparison prior PMAD w0.15/T4 five-run mean: `0.520795`
- comparison R010 PMAD run06_retry1 best: `0.527469`
- delta vs clean baseline mean: `-0.000430` (`-0.088` baseline std units)
- delta vs clean baseline mean + 1 std: `-0.005331`
- delta vs clean baseline best single: `-0.007458`
- delta vs prior PMAD five-run mean: `-0.003828`
- delta vs R010 run06_retry1: `-0.010502`
- gap to `0.53` goal: `-0.013033`
- updated PMAD w0.15/T4 seven-run mean best: `0.521201`
- updated PMAD w0.15/T4 seven-run population std: `0.004148`
- checkpoint: `checkpoints/dformerv2_primkd_logit_only_w015_t4_run07/dformerv2_primkd_logit_only-epoch=42-val_mIoU=0.5170.pt`
- TensorBoard event: `checkpoints/dformerv2_primkd_logit_only_w015_t4_run07/lightning_logs/version_0/events.out.tfevents.1778675198.Administrator.22916.0`
- evidence: `miou_list/dformerv2_primkd_logit_only_w015_t4_run07.md`
- process note: launched through a hidden `cmd.exe /c` script with UTF-8 Python environment variables. Training completed with exit code `0` and no Windows/Rich teardown crash.
- conclusion: **negative repeat.** R012 is below the clean baseline mean, below the prior PMAD mean, far below R010's high-tail result, and below the `0.53` success criterion.
- next step: stop blind PMAD w0.15/T4 repeats. If the loop continues, choose a distinct hypothesis with higher decision value, currently gated around an LMLP/SegFormer-style decoder-head test rather than another PMAD threshold or feature-hint variant.

## 2026-05-13 dformerv2_primkd_logit_only_w015_t4_run06_retry1

- model: `dformerv2_primkd_logit_only`
- method: PMAD/PrimKD-style logit-only KD repeat at the strongest prior setting.
- purpose: test whether the repeat-backed `kd_weight=0.15`, `kd_temperature=4.0` setting can produce a high-tail run near the `0.53` goal while adding a sixth stability sample.
- architecture: `DFormerv2_S + ResNet-18 DepthEncoder + GatedFusion + SimpleFPNDecoder` student, frozen geometry-primary DFormerv2 teacher.
- teacher checkpoint: `checkpoints/dformerv2_geometry_primary_teacher_run01/dformerv2_geometry_primary_teacher-epoch=37-val_mIoU=0.5168.pt`
- settings: `batch_size=2`, `max_epochs=50`, `lr=6e-5`, `num_workers=4`, `early_stop_patience=30`, `loss_type=ce`
- KD settings: `kd_weight=0.15`, `kd_temperature=4.0`, logit-only, no feature KD, `--save_student_only`
- recorded validation epochs: `50`
- best val/mIoU: `0.527469` at epoch `49`
- last val/mIoU: `0.526316`
- last-5 mean val/mIoU: `0.519330`
- last-10 mean val/mIoU: `0.516229`
- best val/loss: `1.064507` at epoch `7`
- final train/loss: `0.209408`
- final train/ce_loss: `0.149212`
- final train/kd_loss: `0.401306`
- comparison clean 10-run RGB-D baseline mean best: `0.517397`
- comparison clean 10-run RGB-D baseline std: `0.004901`
- comparison clean 10-run RGB-D baseline mean + 1 std: `0.522298`
- comparison clean 10-run RGB-D baseline best single: `0.524425`
- comparison prior PMAD w0.15/T4 five-run mean: `0.520795`
- comparison prior PMAD w0.15/T4 best single: `0.524028`
- delta vs clean baseline mean: `+0.010072` (`+2.055` baseline std units)
- delta vs clean baseline mean + 1 std: `+0.005171`
- delta vs clean baseline best single: `+0.003044`
- delta vs prior PMAD five-run mean: `+0.006674`
- delta vs prior PMAD best single: `+0.003441`
- gap to `0.53` goal: `-0.002531`
- updated PMAD w0.15/T4 six-run mean best: `0.521907`
- updated PMAD w0.15/T4 six-run population std: `0.004073`
- checkpoint: `checkpoints/dformerv2_primkd_logit_only_w015_t4_run06_retry1/dformerv2_primkd_logit_only-epoch=48-val_mIoU=0.5275.pt`
- TensorBoard event: `checkpoints/dformerv2_primkd_logit_only_w015_t4_run06_retry1/lightning_logs/version_0/events.out.tfevents.1778662831.Administrator.24204.0`
- evidence: `miou_list/dformerv2_primkd_logit_only_w015_t4_run06_retry1.md`
- process note: first launch `dformerv2_primkd_logit_only_w015_t4_run06` stopped during epoch 0 with Windows `forrtl error (200): program aborting due to window-CLOSE event` and no `val/mIoU`; it is excluded from results.
- conclusion: **partial positive, below goal.** R010 is the best single PMAD repeat so far and raises the six-run PMAD mean, but it still misses the `0.53` success criterion.
- next step: do not claim success. Continue with a new single-hypothesis candidate; based on subagent review, the next highest decision-value direction is a training-only IoU-surrogate loss such as a small-weight Lovasz-Softmax term.

## 2026-05-13 dformerv2_tgga_c34_weakc3_beta001_c4beta002_aux003_detachsem_v1_run01

- model: `dformerv2_tgga_c34_weakc3_beta001_c4beta002_aux003_detachsem_v1`
- method: TGGA diagnostic with weak c3 plus c4 semantic/geometry gates.
- purpose: test whether R004's c4-only calibration signal can be improved by reintroducing a conservative c3 detail path without the original TGGA c3/c4 instability.
- architecture: `DFormerv2_S + TGGA(weak c3, c4) + ResNet-18 DepthEncoder + GatedFusion + SimpleFPNDecoder`.
- TGGA settings: c3 `beta_init=0.01`, `beta_max=0.05`, `gate_bias_init=-3.0`; c4 `beta_init=0.02`, `beta_max=0.1`, `gate_bias_init=-2.0`; aux CE weight `0.03` on c3 and c4; detached semantic cue.
- settings: `batch_size=2`, `max_epochs=50`, `lr=6e-5`, `num_workers=4`, `early_stop_patience=30`, `loss_type=ce`
- pretrained: `C:/Users/qintian/Desktop/qintian/dformer_work/checkpoints/pretrained/DFormerv2_Small_pretrained.pth`
- recorded validation epochs: `50`
- best val/mIoU: `0.518253` at epoch `43`
- last val/mIoU: `0.514908`
- last-5 mean val/mIoU: `0.508991`
- last-10 mean val/mIoU: `0.507070`
- best val/loss: `1.016042` at epoch `13`
- final train/loss: `0.150705`
- final train/main_loss: `0.130934`
- final train/tgga_aux_loss_c3: `0.297266`
- final train/tgga_aux_loss_c4: `0.361774`
- final train/tgga_beta_c3: `0.020015`
- final train/tgga_beta_c4: `0.022356`
- final train/tgga_gate_c3_mean: `0.293138`
- final train/tgga_gate_c3_std: `0.331622`
- final train/tgga_gate_c4_mean: `0.131140`
- final train/tgga_gate_c4_std: `0.012837`
- comparison clean 10-run RGB-D baseline mean best: `0.517397`
- comparison clean 10-run RGB-D baseline std: `0.004901`
- comparison clean 10-run RGB-D baseline mean + 1 std: `0.522298`
- comparison clean 10-run RGB-D baseline best single: `0.524425`
- comparison PMAD w0.15/T4 five-run mean: `0.520795`
- comparison R004 TGGA c4-only best: `0.522849`
- delta vs clean baseline mean: `+0.000856` (`+0.175` baseline std units)
- delta vs clean baseline mean + 1 std: `-0.004045`
- delta vs clean baseline best single: `-0.006172`
- delta vs PMAD w0.15/T4 five-run mean: `-0.002542`
- delta vs R004 TGGA c4-only: `-0.004596`
- best-to-last delta: `-0.003345`
- checkpoint: `checkpoints/dformerv2_tgga_c34_weakc3_beta001_c4beta002_aux003_detachsem_v1_run01/dformerv2_tgga_c34_weakc3_beta001_c4beta002_aux003_detachsem_v1-epoch=42-val_mIoU=0.5183.pt`
- TensorBoard event: `checkpoints/dformerv2_tgga_c34_weakc3_beta001_c4beta002_aux003_detachsem_v1_run01/lightning_logs/version_0/events.out.tfevents.1778626367.Administrator.41400.0`
- evidence: `miou_list/dformerv2_tgga_c34_weakc3_beta001_c4beta002_aux003_detachsem_v1_run01.md`
- process note: `Trainer.fit` reached `max_epochs=50`; after metric/checkpoint writing, Rich progress teardown raised a Windows GBK `UnicodeEncodeError`.
- conclusion: **weak positive versus baseline mean, negative versus the active TGGA decision.** Weak-c3 does not improve on R004 c4-only and does not reach baseline mean + 1 std, PMAD mean, or the `0.53` goal.
- next step: do not continue this weak-c3 variant unchanged. Treat R004 c4-only as the better TGGA diagnostic and pause for external review before selecting another branch.

## 2026-05-13 dformerv2_tgga_c4only_beta002_aux003_detachsem_simplefpn_v1_run01

- model: `dformerv2_tgga_c4only_beta002_aux003_detachsem_simplefpn_v1`
- method: TGGA diagnostic with only the c4 semantic/geometry gate active.
- purpose: test whether removing the c3 high-resolution TGGA gate/residual path preserves the original TGGA c3/c4 high-epoch signal while reducing late collapse risk.
- architecture: `DFormerv2_S + TGGA(c4 only) + ResNet-18 DepthEncoder + GatedFusion + SimpleFPNDecoder`.
- TGGA settings: `beta_c4_init=0.02`, `aux_weight=0.03`, detached semantic cue, no c3 TGGA.
- settings: `batch_size=2`, `max_epochs=50`, `lr=6e-5`, `num_workers=4`, `early_stop_patience=30`, `loss_type=ce`
- pretrained: `C:/Users/qintian/Desktop/qintian/dformer_work/checkpoints/pretrained/DFormerv2_Small_pretrained.pth`
- recorded validation epochs: `50`
- best val/mIoU: `0.522849` at epoch `42`
- last val/mIoU: `0.509320`
- last-5 mean val/mIoU: `0.505936`
- last-10 mean val/mIoU: `0.510887`
- best val/loss: `1.003732` at epoch `12`
- final train/loss: `0.146967`
- final train/main_loss: `0.135671`
- final train/tgga_aux_loss_c4: `0.376528`
- final train/tgga_beta_c4: `0.022874`
- final train/tgga_gate_c4_mean: `0.130742`
- final train/tgga_gate_c4_std: `0.014394`
- comparison clean 10-run RGB-D baseline mean best: `0.517397`
- comparison clean 10-run RGB-D baseline std: `0.004901`
- comparison clean 10-run RGB-D baseline mean + 1 std: `0.522298`
- comparison clean 10-run RGB-D baseline best single: `0.524425`
- comparison PMAD w0.15/T4 five-run mean: `0.520795`
- comparison original TGGA c3/c4 run01 best: `0.522206`
- comparison original TGGA c3/c4 run01-run02 mean best: `0.519822`
- delta vs clean baseline mean: `+0.005452` (`+1.112` baseline std units)
- delta vs clean baseline mean + 1 std: `+0.000551`
- delta vs clean baseline best single: `-0.001576`
- delta vs PMAD w0.15/T4 five-run mean: `+0.002054`
- delta vs original TGGA c3/c4 run01: `+0.000643`
- best-to-last delta: `-0.013529`
- checkpoint: `checkpoints/dformerv2_tgga_c4only_beta002_aux003_detachsem_simplefpn_v1_run01/dformerv2_tgga_c4only_beta002_aux003_detachsem_simplefpn_v1-epoch=41-val_mIoU=0.5228.pt`
- TensorBoard event: `checkpoints/dformerv2_tgga_c4only_beta002_aux003_detachsem_simplefpn_v1_run01/lightning_logs/version_0/events.out.tfevents.1778620555.Administrator.23676.0`
- evidence: `miou_list/dformerv2_tgga_c4only_beta002_aux003_detachsem_simplefpn_v1_run01.md`
- process note: `Trainer.fit` reached `max_epochs=50`.
- conclusion: **strong diagnostic signal but not goal success.** C4-only TGGA is the strongest orchestration-loop run so far, crosses the clean baseline mean + 1 std threshold, and beats PMAD w0.15/T4 mean in this single run, but it remains below the `0.53` target and below the best single clean baseline run.
- next step: do not claim success. Because c4-only TGGA is safer than the original c3/c4 gate but still drops after its best epoch, the next round should test one narrow follow-up that uses this c4 signal without repeating failed PMAD filtering or decoder frequency fusion.

## 2026-05-13 dformerv2_primkd_correct_entropy_w015_t4_h025_run01

- model: `dformerv2_primkd_correct_entropy`
- method: PMAD / PrimKD logit-only KD with a correct-and-low-entropy teacher trust gate.
- purpose: test whether filtering KD to teacher-correct, low-entropy training pixels can preserve PMAD w0.15/T4 gains while avoiding harmful teacher transfer.
- architecture: `DFormerv2_S + ResNet-18 DepthEncoder + GatedFusion + SimpleFPNDecoder`; frozen geometry-primary teacher `DFormerv2_S(rgb, real_depth) + SimpleFPNDecoder`.
- loss: `CE(student, label) + 0.15 * gated_KL(student, teacher, T=4.0)`.
- selector settings: sanitized-label correctness `teacher_argmax == label`, normalized teacher entropy `<=0.25`, selected-pixel KL normalized by valid-pixel count.
- settings: `batch_size=2`, `max_epochs=50`, `lr=6e-5`, `num_workers=4`, `early_stop_patience=30`, `loss_type=ce`
- pretrained: `C:/Users/qintian/Desktop/qintian/dformer_work/checkpoints/pretrained/DFormerv2_Small_pretrained.pth`
- teacher checkpoint: `checkpoints/dformerv2_geometry_primary_teacher_run01/dformerv2_geometry_primary_teacher-epoch=37-val_mIoU=0.5168.pt`
- recorded validation epochs: `50`
- best val/mIoU: `0.516597` at epoch `50`
- last val/mIoU: `0.516597`
- last-5 mean val/mIoU: `0.505977`
- last-10 mean val/mIoU: `0.500502`
- best val/loss: `1.079330` at epoch `8`
- final train/loss: `0.207509`
- final train/ce_loss: `0.150859`
- final train/kd_loss: `0.377668`
- final train/kd_mask_ratio: `0.910636`
- final train/kd_entropy_mean: `0.047532`
- final train/kd_entropy_selected_mean: `0.030076`
- final train/kd_teacher_valid_acc: `0.932230`
- final train/kd_teacher_selected_acc: `1.000000`
- final train/kd_selected_kl: `0.414943`
- comparison clean 10-run RGB-D baseline mean best: `0.517397`
- comparison clean 10-run RGB-D baseline std: `0.004901`
- comparison clean 10-run RGB-D baseline mean + 1 std: `0.522298`
- comparison clean 10-run RGB-D baseline best single: `0.524425`
- comparison PMAD w0.15/T4 five-run mean: `0.520795`
- delta vs clean baseline mean: `-0.000800` (`-0.163` baseline std units)
- delta vs clean baseline mean + 1 std: `-0.005701`
- delta vs PMAD w0.15/T4 five-run mean: `-0.004198`
- checkpoint: `checkpoints/dformerv2_primkd_correct_entropy_w015_t4_h025_run01/dformerv2_primkd_correct_entropy-epoch=49-val_mIoU=0.5166.pt`
- TensorBoard event: `checkpoints/dformerv2_primkd_correct_entropy_w015_t4_h025_run01/lightning_logs/version_0/events.out.tfevents.1778613991.Administrator.34044.0`
- evidence: `miou_list/dformerv2_primkd_correct_entropy_w015_t4_h025_run01.md`
- process note: `Trainer.fit` reached `max_epochs=50`.
- conclusion: **near-baseline but negative result.** The selector is meaningfully selective compared with R001 and selected teacher pixels are label-correct, but the run still peaks below the clean baseline mean and below PMAD w0.15/T4 mean.
- next step: do not repeat this exact correct-and-entropy gate. This weakens the idea that PMAD can be improved simply by removing teacher-wrong pixels; the next round should move away from stricter PMAD filtering unless a clearly different KD mechanism is proposed.

## 2026-05-13 dformerv2_freqfpn_decoder_run01

- model: `dformerv2_freqfpn_decoder`
- method: decoder-side frequency-aware FPN top-down fusion.
- purpose: test whether low/high-frequency correction inside the decoder top-down path can reduce boundary displacement and frequency mismatch without changing encoder, GatedFusion, loss, data, or training recipe.
- architecture: `DFormerv2_S + ResNet-18 DepthEncoder + GatedFusion + FrequencyAwareFPNDecoder`.
- decoder change: `SimpleFPNDecoder` baseline remains unchanged; this model replaces only top-down FPN additions with `FrequencyAwareTopDownFuse`.
- settings: `batch_size=2`, `max_epochs=50`, `lr=6e-5`, `num_workers=4`, `early_stop_patience=30`, `loss_type=ce`
- pretrained: `C:/Users/qintian/Desktop/qintian/dformer_work/checkpoints/pretrained/DFormerv2_Small_pretrained.pth`
- recorded validation epochs: `50`
- best val/mIoU: `0.516915` at epoch `44`
- last val/mIoU: `0.486524`
- last-5 mean val/mIoU: `0.498475`
- last-10 mean val/mIoU: `0.504222`
- best val/loss: `1.022916` at epoch `9`
- final train/loss: `0.174152`
- comparison clean 10-run RGB-D baseline mean best: `0.517397`
- comparison clean 10-run RGB-D baseline std: `0.004901`
- comparison clean 10-run RGB-D baseline mean + 1 std: `0.522298`
- comparison clean 10-run RGB-D baseline best single: `0.524425`
- comparison PMAD w0.15/T4 five-run mean: `0.520795`
- delta vs clean baseline mean: `-0.000482` (`-0.098` baseline std units)
- delta vs clean baseline mean + 1 std: `-0.005383`
- delta vs PMAD w0.15/T4 five-run mean: `-0.003880`
- checkpoint: `checkpoints/dformerv2_freqfpn_decoder_run01/dformerv2_freqfpn_decoder-epoch=43-val_mIoU=0.5169.pt`
- TensorBoard event: `checkpoints/dformerv2_freqfpn_decoder_run01/lightning_logs/version_0/events.out.tfevents.1778607762.Administrator.22656.0`
- evidence: `miou_list/dformerv2_freqfpn_decoder_run01.md`
- process note: `Trainer.fit` reached `max_epochs=50`.
- conclusion: **neutral/negative result.** Decoder-side frequency-aware top-down fusion peaks very close to the clean baseline mean but does not exceed it, does not approach the `0.53` target, and shows late instability after epoch 47.
- next step: do not repeat this exact decoder unchanged. Prefer the next round to target a different high-value hypothesis with clearer room above the baseline mean.

## 2026-05-13 dformerv2_primkd_boundary_conf_w015_t4_run01

- model: `dformerv2_primkd_boundary_conf`
- method: PMAD / PrimKD logit-only KD with deterministic confidence weighting and semantic-boundary boosting.
- purpose: test whether boundary/confidence-selective KD can preserve the positive `dformerv2_primkd_logit_only` w0.15/T4 signal while reducing harmful teacher transfer.
- architecture: `DFormerv2_S + ResNet-18 DepthEncoder + GatedFusion + SimpleFPNDecoder`; frozen geometry-primary teacher `DFormerv2_S(rgb, real_depth) + SimpleFPNDecoder`.
- loss: `CE(student, label) + 0.15 * selective_KL(student, teacher, T=4.0)`.
- selective KD settings: confidence threshold `0.40`, confidence power `1.5`, boundary boost `1.0`.
- settings: `batch_size=2`, `max_epochs=50`, `lr=6e-5`, `num_workers=4`, `early_stop_patience=30`, `loss_type=ce`
- pretrained: `C:/Users/qintian/Desktop/qintian/dformer_work/checkpoints/pretrained/DFormerv2_Small_pretrained.pth`
- teacher checkpoint: `checkpoints/dformerv2_geometry_primary_teacher_run01/dformerv2_geometry_primary_teacher-epoch=37-val_mIoU=0.5168.pt`
- recorded validation epochs: `50`
- best val/mIoU: `0.511646` at epoch `50`
- last val/mIoU: `0.511646`
- last-5 mean val/mIoU: `0.507105`
- last-10 mean val/mIoU: `0.502303`
- best val/loss: `1.059111`
- final train/loss: `0.210573`
- final train/ce_loss: `0.151268`
- final train/kd_loss: `0.395367`
- final train/kd_mask_ratio: `0.998182`
- final train/kd_boundary_ratio: `0.061130`
- final train/kd_conf_mean: `0.938347`
- comparison clean 10-run RGB-D baseline mean best: `0.517397`
- comparison clean 10-run RGB-D baseline std: `0.004901`
- comparison clean 10-run RGB-D baseline mean + 1 std: `0.522298`
- comparison PMAD logit-only w0.15/T4 five-run mean best: `0.520795`
- delta vs clean baseline mean: `-0.005751` (`-1.173` baseline std units)
- delta vs clean baseline mean + 1 std: `-0.010652`
- delta vs PMAD w0.15/T4 five-run mean: `-0.009149`
- checkpoint: `checkpoints/dformerv2_primkd_boundary_conf_w015_t4_run01/dformerv2_primkd_boundary_conf-epoch=49-val_mIoU=0.5116.pt`
- TensorBoard event: `checkpoints/dformerv2_primkd_boundary_conf_w015_t4_run01/lightning_logs/version_0/events.out.tfevents.1778600707.Administrator.4516.0`
- evidence: `miou_list/dformerv2_primkd_boundary_conf_w015_t4_run01.md`
- process note: `Trainer.fit` reached `max_epochs=50`; after metric/checkpoint writing, Rich progress teardown raised a Windows GBK `UnicodeEncodeError`.
- conclusion: **negative result.** The run is below the clean baseline mean and below the existing PMAD w0.15/T4 repeated mean. The confidence threshold `0.40` was effectively non-selective because `kd_mask_ratio` stayed near `0.998`, so this setting mainly tested confidence-weighted and boundary-boosted KD rather than true uncertain-pixel filtering.
- next step: do not repeat this exact setting. If PMAD is revisited, use a genuinely selective confidence threshold or entropy gate; otherwise prioritize a different high-decision-value direction such as decoder-side frequency fusion.

## 2026-05-12 dformerv2_tgga_c34_noaux_semgrad_beta002_simplefpn_v1_run01

- model: `dformerv2_tgga_c34_noaux_semgrad_beta002_simplefpn_v1`
- method: TGGA on DFormerV2 c3/c4, no auxiliary CE, semantic cue trained only through final CE via gate path.
- loss: `CE(final_logits, label)` only.
- purpose: diagnose whether original TGGA late collapse is mainly caused by auxiliary CE conflict or by TGGA gate/residual dynamics.
- architecture: `DFormerv2_S + TGGA(c3,c4 no-aux semgrad) + ResNet-18 DepthEncoder + GatedFusion + SimpleFPNDecoder`
- settings: `batch_size=2`, `max_epochs=50`, `lr=6e-5`, `num_workers=4`, `early_stop_patience=30`
- pretrained: `C:/Users/qintian/Desktop/qintian/dformer_work/checkpoints/pretrained/DFormerv2_Small_pretrained.pth`
- recorded validation epochs: `50`
- best val/mIoU: `0.512152` at epoch `48`
- last val/mIoU: `0.492633`
- best val/loss: `1.032364` at epoch `15`
- last val/loss: `1.275699`
- mean val/mIoU over last 10 epochs: `0.501366`
- mean val/mIoU over last 5 epochs: `0.498370`
- post-best mean val/mIoU: `0.481050`
- final train/loss: `0.187033`
- final train/main_loss: `0.187033`
- final train/tgga_aux_ce_c3_diag: `3.880328`
- final train/tgga_aux_ce_c4_diag: `3.911476`
- final train/tgga_beta_c3: `0.035324`
- final train/tgga_beta_c4: `0.025326`
- final train/tgga_gate_c3_mean: `0.474472`
- final train/tgga_gate_c4_mean: `0.230513`
- final train/tgga_gate_c3_std: `0.311781`
- final train/tgga_gate_c4_std: `0.135297`
- comparison clean 10-run RGB-D baseline mean best: `0.517397`
- comparison clean 10-run RGB-D baseline std: `0.004901`
- comparison clean 10-run RGB-D baseline mean + 1 std: `0.522298`
- comparison PMAD logit-only w0.15 5-run mean best: `0.520795`
- comparison original TGGA aux run01 best: `0.522206`
- comparison original TGGA aux run02 best: `0.517437`
- delta vs clean baseline mean: `-0.005245` (`-1.070` baseline std units)
- delta vs PMAD w0.15 mean: `-0.008643`
- delta vs original TGGA aux run01: `-0.010054`
- delta vs original TGGA aux run02: `-0.005285`
- epochs above clean baseline mean: `0/50`
- late-curve check: epoch 41-50 val/mIoU = `0.495870, 0.503853, 0.504099, 0.508151, 0.509840, 0.510360, 0.507238, 0.512152, 0.469468, 0.492633`. The best epoch is epoch 48, then epoch 49 drops sharply.
- checkpoint: `checkpoints/dformerv2_tgga_c34_noaux_semgrad_beta002_simplefpn_v1_run01/dformerv2_tgga_c34_noaux_semgrad_beta002_simplefpn_v1-epoch=47-val_mIoU=0.5122.pt`
- evidence: `miou_list/dformerv2_tgga_c34_noaux_semgrad_beta002_simplefpn_v1_run01.md`
- Pro discussion prompt: provided in chat only; no standalone prompt file is kept.
- conclusion: **negative diagnostic result.** Removing auxiliary CE does not improve stability or performance. It lowers the peak relative to original TGGA and still shows late collapse, so auxiliary CE is not the only instability source; TGGA gate/residual dynamics are likely unsafe in this form.
- next step: do not claim TGGA no-aux as improvement. Ask Pro to decide whether the only remaining worthwhile diagnostic is weak-c3/c4-only or whether to stop TGGA and return to PMAD/clean-baseline-safe directions.

## 2026-05-12 TGGA c3/c4 no-aux semantic-gradient diagnostic implementation

- type: code implementation note, not a training run.
- implemented model: `dformerv2_tgga_c34_noaux_semgrad_beta002_simplefpn_v1`
- purpose: test whether TGGA late collapse is caused mainly by the auxiliary semantic CE loss or by TGGA gate/residual dynamics.
- structure: same c3/c4 TGGA insertion as `dformerv2_tgga_c34_beta002_aux003_detachsem_simplefpn_v2`.
- loss: `CE(final_logits, label)` only.
- semantic cue note: the original `detachsem` path would leave the aux head untrained if aux CE were removed, so this diagnostic uses semantic-gradient gating (`detach_semantic=False`) and lets final CE train the semantic cue through the gate.
- diagnostic logs: `train/tgga_aux_ce_c3_diag`, `train/tgga_aux_ce_c4_diag`, `train/tgga_beta_c3`, `train/tgga_beta_c4`, `train/tgga_gate_c3_mean`, `train/tgga_gate_c4_mean`, `train/tgga_gate_c3_std`, and `train/tgga_gate_c4_std`.
- result note: run01 best val/mIoU is `0.512152`, below the clean baseline mean by `0.005245`; see `miou_list/dformerv2_tgga_c34_noaux_semgrad_beta002_simplefpn_v1_run01.md`.

## 2026-05-12 dformerv2_tgga_c34_beta002_aux003_detachsem_simplefpn_v2 run01-run02 summary

- model: `dformerv2_tgga_c34_beta002_aux003_detachsem_simplefpn_v2`
- completed runs: `2`
- run01 best val/mIoU: `0.522206` at epoch `48`; last val/mIoU `0.489865`; last-10 mean `0.510627`
- run02 best val/mIoU: `0.517437` at epoch `49`; last val/mIoU `0.486566`; last-10 mean `0.501329`
- two-run mean best val/mIoU: `0.519822`
- two-run population std best val/mIoU: `0.002384`
- two-run mean last val/mIoU: `0.488215`
- clean 10-run baseline mean best: `0.517397`, std `0.004901`, mean + 1 std `0.522298`, best single `0.524425`
- delta vs clean baseline mean: `+0.002425` (`+0.495` baseline std units)
- delta vs clean baseline mean + 1 std: `-0.002476`
- delta vs PMAD logit-only w0.15 5-run mean `0.520795`: `-0.000973`
- evidence: `miou_list/dformerv2_tgga_c34_beta002_aux003_detachsem_simplefpn_v2_run01.md`, `miou_list/dformerv2_tgga_c34_beta002_aux003_detachsem_simplefpn_v2_run02.md`, and `miou_list/dformerv2_tgga_c34_beta002_aux003_detachsem_simplefpn_v2_summary_run01_02.md`
- GPT discussion record: conclusion summarized here; no standalone discussion file is kept.
- conclusion: **weak positive but unstable, not a stable improvement.** The two-run best-mIoU mean is above the clean baseline mean but below PMAD w0.15 and below baseline mean + 1 std. Both runs collapse late to final mIoU around `0.49`, so TGGA should not be claimed as a paper improvement.
- next step: do not blindly prioritize run03. The highest-value diagnostic is `dformerv2_tgga_c34_noaux_semgrad_beta002_simplefpn_v1`.

## 2026-05-12 dformerv2_tgga_c34_beta002_aux003_detachsem_simplefpn_v2_run02

- model: `dformerv2_tgga_c34_beta002_aux003_detachsem_simplefpn_v2`
- method: Task-Guided Geometry Calibration Adapter on DFormerV2 c3/c4 before external `DepthEncoder + GatedFusion`.
- loss: `CE(final_logits, label) + 0.03 * CE(aux_c3, label) + 0.03 * CE(aux_c4, label)`
- purpose: repeat the TGGA c3/c4 candidate after run01's promising but unstable result.
- architecture: `DFormerv2_S + TGGA(c3,c4) + ResNet-18 DepthEncoder + GatedFusion + SimpleFPNDecoder`
- settings: `batch_size=2`, `max_epochs=50`, `lr=6e-5`, `num_workers=4`, `early_stop_patience=30`
- pretrained: `C:/Users/qintian/Desktop/qintian/dformer_work/checkpoints/pretrained/DFormerv2_Small_pretrained.pth`
- recorded validation epochs: `50`
- best val/mIoU: `0.517437` at epoch `49`
- last val/mIoU: `0.486566`
- best val/loss: `0.995109` at epoch `10`
- last val/loss: `1.387486`
- mean val/mIoU over last 10 epochs: `0.501329`
- mean val/mIoU over last 5 epochs: `0.501959`
- post-best mean val/mIoU: `0.486566`
- final train/loss: `0.160672`
- final train/main_loss: `0.140603`
- final train/tgga_aux_loss_c3: `0.301237`
- final train/tgga_aux_loss_c4: `0.367719`
- final train/tgga_beta_c3: `0.039473`
- final train/tgga_beta_c4: `0.022773`
- final train/tgga_gate_c3_mean: `0.409689`
- final train/tgga_gate_c4_mean: `0.126628`
- final train/tgga_gate_c3_std: `0.346383`
- final train/tgga_gate_c4_std: `0.010253`
- comparison clean 10-run RGB-D baseline mean best: `0.517397`
- comparison clean 10-run RGB-D baseline std: `0.004901`
- delta vs clean baseline mean: `+0.000040` (`+0.008` baseline std units)
- delta vs PMAD w0.15 mean: `-0.003358`
- late-curve check: epoch 41-50 val/mIoU = `0.510171, 0.512682, 0.516483, 0.495031, 0.469128, 0.487177, 0.507587, 0.511026, 0.517437, 0.486566`. The best epoch is epoch 49, then epoch 50 drops by `0.030871`.
- checkpoint: `checkpoints/dformerv2_tgga_c34_beta002_aux003_detachsem_simplefpn_v2_run02/dformerv2_tgga_c34_beta002_aux003_detachsem_simplefpn_v2-epoch=48-val_mIoU=0.5174.pt`
- evidence: `miou_list/dformerv2_tgga_c34_beta002_aux003_detachsem_simplefpn_v2_run02.md`
- conclusion: **not a stable improvement.** Run02 only matches the clean baseline mean on best mIoU and repeats the late-collapse pattern, so it weakens the case for TGGA c3/c4 as a main method.

## 2026-05-12 TGGA diagnostic variant implementation

- type: code implementation note, not a training run.
- implemented future diagnostic models:
  - `dformerv2_tgga_c4only_beta002_aux003_detachsem_simplefpn_v1`
  - `dformerv2_tgga_c34_weakc3_beta001_c4beta002_aux003_detachsem_v1`
- purpose: localize whether original TGGA run01 instability comes from c3 high-resolution residual/gate noise, and test a weaker c3 variant if c4-only is more stable.
- no mIoU result yet.

## 2026-05-12 code cleanup

- type: code organization cleanup, not a training run.
- branch: `cleanup/archive-failed-modules`.
- active registry now exposes `dformerv2_mid_fusion`, `dformerv2_tgga_c34_beta002_aux003_detachsem_simplefpn_v2`, `dformerv2_geometry_primary_teacher`, and `dformerv2_primkd_logit_only`, plus legacy `early` / `mid_fusion`.
- archived/default-hidden branches: DGBF, CGPC, SGBR-Lite, CGCD/ClassContext, context decoder/PPM, FFT freq enhance, FFT HiLo, and depth FFT select.
- TGGA remains active but unstable after run01-run02; next priority is diagnostic rather than blind repeat.
- No mIoU result, checkpoint, TensorBoard log, or evidence file changed.

## 2026-05-12 dformerv2_tgga_c34_beta002_aux003_detachsem_simplefpn_v2_run01

- model: `dformerv2_tgga_c34_beta002_aux003_detachsem_simplefpn_v2`
- method: Task-Guided Geometry Calibration Adapter on DFormerV2 c3/c4 before external `DepthEncoder + GatedFusion`.
- loss: `CE(final_logits, label) + 0.03 * CE(aux_c3, label) + 0.03 * CE(aux_c4, label)`
- purpose: test whether task-guided semantic/high-frequency geometry calibration before GatedFusion can improve the stable DFormerv2 mid-fusion baseline without changing decoder, fusion, optimizer, data, PMAD/KD, or training recipe.
- architecture: `DFormerv2_S + TGGA(c3,c4) + ResNet-18 DepthEncoder + GatedFusion + SimpleFPNDecoder`
- settings: `batch_size=2`, `max_epochs=50`, `lr=6e-5`, `num_workers=4`, `early_stop_patience=30`
- pretrained: `C:/Users/qintian/Desktop/qintian/dformer_work/checkpoints/pretrained/DFormerv2_Small_pretrained.pth`
- recorded validation epochs: `50`
- best val/mIoU: `0.522206` at epoch `48`
- last val/mIoU: `0.489865`
- best val/loss: `1.001215` at epoch `10`
- last val/loss: `1.384536`
- mean val/mIoU over last 10 epochs: `0.510627`
- mean val/mIoU over last 5 epochs: `0.512473`
- best-epoch local 5-epoch window mean: `0.517497`
- post-best mean val/mIoU: `0.503293`
- final train/loss: `0.183592`
- final train/main_loss: `0.162046`
- final train/tgga_aux_loss_c3: `0.327891`
- final train/tgga_aux_loss_c4: `0.390297`
- final train/tgga_beta_c3: `0.035080`
- final train/tgga_beta_c4: `0.023389`
- final train/tgga_gate_c3_mean: `0.351956`
- final train/tgga_gate_c4_mean: `0.131228`
- final train/tgga_gate_c3_std: `0.305273`
- final train/tgga_gate_c4_std: `0.016975`
- comparison clean 10-run RGB-D baseline mean best: `0.517397`
- comparison clean 10-run RGB-D baseline std: `0.004901`
- comparison clean 10-run RGB-D baseline mean + 1 std: `0.522298`
- comparison clean 10-run RGB-D baseline best single: `0.524425`
- comparison PMAD logit-only w0.15 5-run mean best: `0.520795`
- comparison bounded class-context 5-run mean best: `0.515986`
- comparison SGBR-Lite run01 best: `0.510159`
- delta vs clean baseline mean: `+0.004809` (`+0.981` baseline std units)
- delta vs clean baseline mean + 1 std: `-0.000092`
- delta vs clean baseline best single: `-0.002219`
- delta vs PMAD w0.15 mean: `+0.001411`
- delta vs bounded class-context mean: `+0.006220`
- delta vs SGBR-Lite best: `+0.012047`
- late-curve check: epoch 41-50 val/mIoU = `0.491948, 0.505812, 0.514443, 0.514902, 0.516806, 0.516127, 0.517443, 0.522206, 0.516722, 0.489865`. The best epoch is epoch 48, not the final epoch; the run rises to a strong late high point and then drops sharply at epoch 50.
- checkpoint: `checkpoints/dformerv2_tgga_c34_beta002_aux003_detachsem_simplefpn_v2_run01/dformerv2_tgga_c34_beta002_aux003_detachsem_simplefpn_v2-epoch=47-val_mIoU=0.5222.pt`
- evidence: `miou_list/dformerv2_tgga_c34_beta002_aux003_detachsem_simplefpn_v2_run01.md`
- conclusion: **promising but unstable single-run result.** TGGA run01 beats the clean baseline mean and PMAD 5-run mean, and is almost exactly at the baseline mean + 1 std threshold. However, the high point appears late and is followed by a final collapse, so this is not yet stable evidence. A repeat is justified before claiming improvement.

## 2026-05-12 dformerv2_sgbr_decoder_w010_b005_run01

- model: `dformerv2_sgbr_decoder`
- method: SGBR-Lite decoder, semantic-uncertainty gated raw-depth Sobel boundary residual.
- loss: `CE(final_logits, label) + 0.1 * CE(aux_logits, label)`
- SGBR settings: `sgbr_aux_weight=0.1`, `sgbr_beta_init=0.05`, `sgbr_beta_max=0.2`
- purpose: test whether decoder-side semantic-guided depth-boundary residual refinement can improve the DFormerv2 mid-fusion baseline without changing encoder or GatedFusion.
- architecture: `DFormerv2_S + ResNet-18 DepthEncoder + GatedFusion + SGBRFPNDecoder`
- settings: `batch_size=2`, `max_epochs=50`, `lr=6e-5`, `num_workers=4`, `early_stop_patience=30`
- pretrained: `C:/Users/qintian/Desktop/qintian/dformer_work/checkpoints/pretrained/DFormerv2_Small_pretrained.pth`
- recorded validation epochs: `50`
- best val/mIoU: `0.510159` at epoch `40`
- last val/mIoU: `0.502075`
- best val/loss: `1.046134` at epoch `9`
- last val/loss: `1.279392`
- mean val/mIoU over last 10 epochs: `0.495112`
- final train/loss: `0.148493`
- final train/final_loss: `0.134872`
- final train/aux_loss: `0.136213`
- final train/sgbr_beta: `0.075018`
- comparison clean 10-run RGB-D baseline mean best: `0.517397`
- comparison clean 10-run RGB-D baseline std: `0.004901`
- comparison clean 10-run RGB-D baseline mean + 1 std: `0.522298`
- comparison clean 10-run RGB-D baseline best single: `0.524425`
- comparison bounded class-context 5-run mean best: `0.515986`
- comparison PMAD logit-only w0.15 5-run mean best: `0.520795`
- delta vs clean baseline mean: `-0.007238` (`-1.477` baseline std units)
- delta vs clean baseline mean + 1 std: `-0.012139`
- delta vs clean baseline best single: `-0.014266`
- delta vs bounded class-context mean: `-0.005827`
- delta vs PMAD w0.15 mean: `-0.010636`
- checkpoint: `checkpoints/dformerv2_sgbr_decoder_w010_b005_run01/dformerv2_sgbr_decoder-epoch=39-val_mIoU=0.5102.pt`
- evidence: `miou_list/dformerv2_sgbr_decoder_w010_b005_run01.md`
- conclusion: **negative result.** SGBR-Lite is clearly below the clean baseline mean and below previous decoder/KD candidates. The bounded residual did not run away (`sgbr_beta` ended at `0.075018`), so the failure mode is not instability but insufficient useful signal from the semantic-uncertainty/depth-edge residual branch.

## 2026-05-12 dformerv2_class_context_decoder_bounded_a02 run01-run05 summary

- model: `dformerv2_class_context_decoder`
- loss: `CE(final_logits, label) + 0.2 * CE(aux_logits, label)`
- class-context settings: `class_context_channels=64`, `class_context_aux_weight=0.2`, `class_context_alpha_init=0.1`, `class_context_alpha_max=0.2`
- purpose: five-run repeat for bounded-alpha CGCD / OCR-lite decoder.
- architecture: `DFormerv2_S + ResNet-18 DepthEncoder + GatedFusion + ClassContextFPNDecoder`
- settings: `batch_size=2`, `max_epochs=50`, `lr=6e-5`, `num_workers=4`, `early_stop_patience=30`
- pretrained: `C:/Users/qintian/Desktop/qintian/dformer_work/checkpoints/pretrained/DFormerv2_Small_pretrained.pth`
- completed runs: `5`
- dformerv2_class_context_decoder_bounded_a02_run01: best val/mIoU `0.525156` at epoch 46, last `0.469674`, final alpha `0.134989`
- dformerv2_class_context_decoder_bounded_a02_run02: best val/mIoU `0.511353` at epoch 34, last `0.498829`, final alpha `0.134747`
- dformerv2_class_context_decoder_bounded_a02_run03: best val/mIoU `0.511318` at epoch 49, last `0.505855`, final alpha `0.133999`
- dformerv2_class_context_decoder_bounded_a02_run04: best val/mIoU `0.514017` at epoch 43, last `0.488192`, final alpha `0.135494`
- dformerv2_class_context_decoder_bounded_a02_run05: best val/mIoU `0.518087` at epoch 50, last `0.518087`, final alpha `0.135144`
- mean best val/mIoU: `0.515986`
- population std best val/mIoU: `0.005208`
- mean last val/mIoU: `0.496127`
- mean last-10 val/mIoU: `0.504101`
- best single run: `0.525156` (`run01`)
- worst single run: `0.511318` (`run03`)
- comparison clean 10-run RGB-D baseline mean best: `0.517397`
- comparison clean 10-run RGB-D baseline std: `0.004901`
- comparison clean 10-run RGB-D baseline mean + 1 std: `0.522298`
- comparison clean 10-run RGB-D baseline best single: `0.524425`
- comparison unbounded class-context run01 best: `0.519807`
- comparison PMAD logit-only w0.15 5-run mean best: `0.520795`
- mean delta vs clean baseline mean: `-0.001411` (`-0.288` baseline std units)
- mean delta vs clean baseline mean + 1 std: `-0.006312`
- mean delta vs PMAD w0.15 mean: `-0.004809`
- mean delta vs unbounded class-context run01: `-0.003821`
- runs above baseline mean: `2/5`
- runs above baseline mean + 1 std: `1/5`
- runs above baseline best single: `1/5`
- evidence: `miou_list/dformerv2_class_context_decoder_bounded_a02_run01.md`, `miou_list/dformerv2_class_context_decoder_bounded_a02_run02.md`, `miou_list/dformerv2_class_context_decoder_bounded_a02_run03.md`, `miou_list/dformerv2_class_context_decoder_bounded_a02_run04.md`, `miou_list/dformerv2_class_context_decoder_bounded_a02_run05.md`, and `miou_list/dformerv2_class_context_decoder_bounded_a02_run01_run05_summary.md`
- conclusion: **mixed repeated-run result.** Bounded alpha fixes the runaway-alpha failure mode and gives one strong run, but the five-run mean remains below the clean baseline mean and below PMAD. This should not be claimed as a stable improvement.

## 2026-05-12 dformerv2_class_context_decoder_run01

- model: `dformerv2_class_context_decoder`
- loss: `CE(final_logits, label) + 0.2 * CE(aux_logits, label)`
- class-context settings: `class_context_channels=64`, `class_context_aux_weight=0.2`, `class_context_alpha_init=0.1`
- purpose: test whether a lightweight OCR-style class-context decoder can improve the DFormerv2 mid-fusion baseline without changing encoder or GatedFusion.
- architecture: `DFormerv2_S + ResNet-18 DepthEncoder + GatedFusion + ClassContextFPNDecoder`
- settings: `batch_size=2`, `max_epochs=50`, `lr=6e-5`, `num_workers=4`, `early_stop_patience=30`
- pretrained: `C:/Users/qintian/Desktop/qintian/dformer_work/checkpoints/pretrained/DFormerv2_Small_pretrained.pth`
- recorded validation epochs: 50
- best val/mIoU: `0.519807` at epoch 46
- last val/mIoU: `0.503151`
- best val/loss: `1.055177` at epoch 10
- last val/loss: `1.279098`
- mean val/mIoU over last 10 epochs: `0.501142`
- final train/loss: `0.221005`
- final train/final_loss: `0.183129`
- final train/aux_loss: `0.189379`
- final train/context_alpha: `0.621710`
- comparison clean 10-run RGB-D baseline mean best: `0.517397`
- comparison clean 10-run RGB-D baseline std: `0.004901`
- comparison clean 10-run RGB-D baseline mean + 1 std: `0.522298`
- comparison clean 10-run RGB-D baseline best single: `0.524425`
- comparison PMAD logit-only w0.15 5-run mean best: `0.520795`
- comparison CGPC c3 best: `0.515838`
- comparison CGPC c4 best: `0.512659`
- delta vs clean baseline mean: `+0.002410` (`+0.492` baseline std units)
- delta vs clean baseline mean + 1 std: `-0.002491`
- delta vs clean baseline best single: `-0.004618`
- delta vs PMAD w0.15 mean: `-0.000988`
- delta vs CGPC c3: `+0.003969`
- delta vs CGPC c4: `+0.007148`
- checkpoint: `checkpoints/dformerv2_class_context_decoder_run01/dformerv2_class_context_decoder-epoch=45-val_mIoU=0.5198.pt`
- evidence: `miou_list/dformerv2_class_context_decoder_run01.md`
- conclusion: **marginal positive but unstable single-run result.** The class-context decoder beats the clean baseline mean and clearly outperforms CGPC c3/c4, but the gain is below one baseline standard deviation and the late-epoch curve is unstable. Do not claim stable improvement from one run.

## 2026-05-11 dformerv2_mid_fusion_cgpc_w001_t01_c4_detach_run01

- model: `dformerv2_mid_fusion`
- loss: `CE + 0.01 * CGPCLoss`
- CGPC settings: `stage=c4`, `temperature=0.1`, `min_pixels_per_class=10`, `max_pixels_per_class=128`, `detach_prototype=True`
- purpose: single-variable CGPC stage ablation from c3 to c4.
- architecture: unchanged `DFormerv2_S + ResNet-18 DepthEncoder + GatedFusion + SimpleFPNDecoder`
- settings: `batch_size=2`, `max_epochs=50`, `lr=6e-5`, `num_workers=4`, `early_stop_patience=30`
- pretrained: `C:/Users/qintian/Desktop/qintian/dformer_work/checkpoints/pretrained/DFormerv2_Small_pretrained.pth`
- recorded validation epochs: 50
- best val/mIoU: `0.512659` at epoch 49
- last val/mIoU: `0.507206`
- best val/loss: `1.034627` at epoch 11
- last val/loss: `1.292995`
- mean val/mIoU over last 10 epochs: `0.501490`
- final train/loss: `0.143598`
- final train/seg_loss: `0.139954`
- final train/cgpc_loss: `0.364393`
- final cgpc_num_classes: `9.790932`
- final cgpc_num_queries: `508.498749`
- comparison clean 10-run RGB-D baseline mean best: `0.517397`
- comparison clean 10-run RGB-D baseline std: `0.004901`
- comparison clean 10-run RGB-D baseline mean + 1 std: `0.522298`
- comparison clean 10-run RGB-D baseline best single: `0.524425`
- comparison CGPC c3 best: `0.515838`
- comparison DGBF depth_semantic best: `0.513194`
- delta vs clean baseline mean: `-0.004738` (`-0.967` baseline std units)
- delta vs clean baseline mean + 1 std: `-0.009639`
- delta vs clean baseline best single: `-0.011766`
- delta vs CGPC c3: `-0.003179`
- delta vs DGBF depth_semantic: `-0.000536`
- checkpoint: `checkpoints/dformerv2_mid_fusion_cgpc_w001_t01_c4_detach_run01/dformerv2_mid_fusion-epoch=48-val_mIoU=0.5127.pt`
- evidence: `miou_list/dformerv2_mid_fusion_cgpc_w001_t01_c4_detach_run01.md`
- conclusion: **negative result.** CGPC c4 underperforms both the clean CE baseline and the CGPC c3 run. The c4 setting has active CGPC optimization but lower class/query coverage than c3, and the semantic-stage hypothesis is not supported by this run.

## 2026-05-11 dformerv2_mid_fusion_cgpc_w001_t01_c3_detach_run01

- model: `dformerv2_mid_fusion`
- loss: `CE + 0.01 * CGPCLoss`
- CGPC settings: `stage=c3`, `temperature=0.1`, `min_pixels_per_class=10`, `max_pixels_per_class=128`, `detach_prototype=True`
- purpose: first class-label-guided prototype contrastive loss run using GatedFusion fused c3 features.
- architecture: unchanged `DFormerv2_S + ResNet-18 DepthEncoder + GatedFusion + SimpleFPNDecoder`
- settings: `batch_size=2`, `max_epochs=50`, `lr=6e-5`, `num_workers=4`, `early_stop_patience=30`
- pretrained: `C:/Users/qintian/Desktop/qintian/dformer_work/checkpoints/pretrained/DFormerv2_Small_pretrained.pth`
- recorded validation epochs: 50
- best val/mIoU: `0.515838` at epoch 50
- last val/mIoU: `0.515838`
- best val/loss: `1.023663` at epoch 11
- last val/loss: `1.225397`
- mean val/mIoU over last 10 epochs: `0.502118`
- final train/loss: `0.139299`
- final train/seg_loss: `0.135577`
- final train/cgpc_loss: `0.372201`
- final cgpc_num_classes: `13.440806`
- final cgpc_num_queries: `1120.790894`
- comparison clean 10-run RGB-D baseline mean best: `0.517397`
- comparison clean 10-run RGB-D baseline std: `0.004901`
- comparison clean 10-run RGB-D baseline mean + 1 std: `0.522298`
- comparison clean 10-run RGB-D baseline best single: `0.524425`
- delta vs clean baseline mean: `-0.001559` (`-0.318` baseline std units)
- delta vs clean baseline mean + 1 std: `-0.006460`
- delta vs clean baseline best single: `-0.008587`
- checkpoint: `checkpoints/dformerv2_mid_fusion_cgpc_w001_t01_c3_detach_run01/dformerv2_mid_fusion-epoch=49-val_mIoU=0.5158.pt`
- evidence: `miou_list/dformerv2_mid_fusion_cgpc_w001_t01_c3_detach_run01.md`
- conclusion: **neutral-to-negative result.** CGPC c3 is much healthier than the failed DGBF setting and ends close to the CE baseline, but it does not exceed the clean baseline mean. Diagnostics show healthy class/query coverage, so this is not a sampling failure.

## 2026-05-11 dformerv2_mid_fusion_dgbf_a1_g2_depthsem_run01

- model: `dformerv2_mid_fusion`
- loss: `DGBFLoss`
- DGBF settings: `alpha=1.0`, `gamma=2.0`, `mode=depth_semantic`
- purpose: first output-level Depth-Geometry Boundary Focal Loss run on the clean DFormerV2 mid-fusion baseline.
- architecture: unchanged `DFormerv2_S + ResNet-18 DepthEncoder + GatedFusion + SimpleFPNDecoder`
- settings: `batch_size=2`, `max_epochs=50`, `lr=6e-5`, `num_workers=4`, `early_stop_patience=30`
- pretrained: `C:/Users/qintian/Desktop/qintian/dformer_work/checkpoints/pretrained/DFormerv2_Small_pretrained.pth`
- recorded validation epochs: 50
- best val/mIoU: `0.513194` at epoch 49
- last val/mIoU: `0.443671`
- best val/loss: `1.030464` at epoch 12
- last val/loss: `1.385053`
- mean val/mIoU over last 10 epochs: `0.497684`
- final train/loss: `0.190071`
- final DGBF boundary mean: `0.002385`
- final DGBF boundary max: `0.946904`
- final DGBF weight mean: `1.000793`
- final DGBF weight max: `1.728474`
- comparison clean 10-run RGB-D baseline mean best: `0.517397`
- comparison clean 10-run RGB-D baseline std: `0.004901`
- comparison clean 10-run RGB-D baseline mean + 1 std: `0.522298`
- comparison clean 10-run RGB-D baseline best single: `0.524425`
- delta vs clean baseline mean: `-0.004203` (`-0.857` baseline std units)
- delta vs clean baseline mean + 1 std: `-0.009104`
- delta vs clean baseline best single: `-0.011231`
- checkpoint: `checkpoints/dformerv2_mid_fusion_dgbf_a1_g2_depthsem_run01/dformerv2_mid_fusion-epoch=48-val_mIoU=0.5132.pt`
- evidence: `miou_list/dformerv2_mid_fusion_dgbf_a1_g2_depthsem_run01.md`
- conclusion: **negative result.** The first DGBF `depth_semantic` setting underperforms the clean CE baseline and shows a late collapse at epoch 50. The boundary/weight averages are extremely close to zero/one, so this configuration behaves almost like CE on most pixels.

## 2026-05-11 PMAD logit-only KD weight sweep summary

- model: `dformerv2_primkd_logit_only`
- purpose: complete PMAD logit-only KD weight sweep after run01/run02 showed positive w=0.15 signal.
- student: `DFormerv2_S + ResNet-18 DepthEncoder + GatedFusion + SimpleFPNDecoder`
- teacher: frozen `dformerv2_geometry_primary_teacher_run01`, `DFormerv2_S(rgb, real_depth) + SimpleFPNDecoder`
- teacher checkpoint: `checkpoints/dformerv2_geometry_primary_teacher_run01/dformerv2_geometry_primary_teacher-epoch=37-val_mIoU=0.5168.pt`
- settings: `batch_size=2`, `max_epochs=50`, `lr=6e-5`, `num_workers=4`, `early_stop_patience=30`, `loss_type=ce`
- KD settings: `kd_temperature=4.0`, logit-only, no feature KD
- pretrained: `C:/Users/qintian/Desktop/qintian/dformer_work/checkpoints/pretrained/DFormerv2_Small_pretrained.pth`
- completed runs: 8 total (run01-run05 at w=0.15, run01 at w=0.175, run01 at w=0.20, plus earlier w=0.10 run01)
- baseline reference: clean 10-run RGB-D baseline mean best `0.517397`, std `0.004901`, mean+1std `0.522298`, best single `0.524425`

### w=0.15 T=4.0 5-run results

| run | best val/mIoU | epoch | last val/mIoU | delta vs baseline mean |
|-----|--------------|-------|---------------|----------------------|
| run01 | 0.522998 | 48 | 0.513176 | +0.005601 |
| run02 | 0.520144 | 47 | 0.498182 | +0.002747 |
| run03 | 0.522600 | 41 | unknown | +0.005203 |
| run04 | 0.524028 | 45 | 0.490264 | +0.006631 |
| run05 | 0.514204 | 41 | 0.505725 | -0.003193 |
| **5-run mean** | **0.520795** | | | **+0.003398** |
| **5-run std** | **0.003799** | | | |

- 5-run mean delta vs baseline mean: `+0.003398` (`+0.693` baseline std units)
- 5-run mean delta vs baseline mean + 1 std: `-0.001503`
- 5-run best single: `0.524028` (run04), delta vs baseline best single: `-0.000397`
- runs above baseline mean: 4/5 (run01, run02, run03, run04)
- runs above baseline mean + 1 std: 2/5 (run03, run04)

### w=0.175 T=4.0 run01

- best val/mIoU: `0.518158` at epoch 49, last `0.491036`
- delta vs baseline mean: `+0.000761` (`+0.155` std units)
- conclusion: near baseline, no advantage over w=0.15

### w=0.20 T=4.0 run01

- best val/mIoU: `0.514454` at epoch 40, last `0.513074`
- delta vs baseline mean: `-0.002943` (`-0.600` std units)
- conclusion: negative, below baseline mean

### KD weight ablation summary

| kd_weight | runs | mean best mIoU | delta vs baseline | verdict |
|-----------|------|---------------|-------------------|---------|
| 0.10 | 1 | 0.5101 | -0.0073 (-1.50 std) | negative |
| 0.15 | 5 | 0.5208 | +0.0034 (+0.69 std) | marginal positive |
| 0.175 | 1 | 0.5182 | +0.0008 (+0.16 std) | neutral |
| 0.20 | 1 | 0.5145 | -0.0029 (-0.60 std) | negative |

- conclusion: **w=0.15 is the best KD weight.** The 5-run mean 0.5208 beats the baseline mean by +0.69 std and is close to baseline mean + 1 std. 4/5 runs exceed baseline mean. Higher weights (0.175, 0.20) show diminishing returns. PMAD logit-only at w=0.15 is a marginal positive candidate for the paper.
- next step: PMAD logit-only is sufficient as a marginal contribution. Do not add feature KD. Document as ablation study in paper.

## 2026-05-11 dformerv2_primkd_logit_only_w020_t4_run01

- model: `dformerv2_primkd_logit_only`
- purpose: PMAD / PrimKD logit-only KD weight ablation at w=0.20.
- student: `DFormerv2_S + ResNet-18 DepthEncoder + GatedFusion + SimpleFPNDecoder`
- teacher: frozen `dformerv2_geometry_primary_teacher_run01`
- KD settings: `kd_weight=0.20`, `kd_temperature=4.0`, logit-only
- recorded validation epochs: 50
- best val/mIoU: `0.514454` at epoch 40
- last val/mIoU: `0.513074`
- mean val/mIoU over last 10 epochs: `0.500622`
- best val/loss: `1.083174` at epoch 6
- last val/loss: `1.257040`
- final train/loss: `0.228650`
- final train/ce_loss: `0.153539`
- final train/kd_loss: `0.375556`
- delta vs clean baseline mean: `-0.002943` (`-0.600` std units)
- delta vs clean baseline mean + 1 std: `-0.007844`
- delta vs w=0.15 5-run mean: `-0.006341`
- checkpoint: `checkpoints/dformerv2_primkd_logit_only_w020_t4_run01/dformerv2_primkd_logit_only-epoch=39-val_mIoU=0.5145.pt`
- evidence: `miou_list/dformerv2_primkd_logit_only_w020_t4_run01.md`
- conclusion: **negative.** w=0.20 underperforms baseline mean. KD weight too high.

## 2026-05-11 dformerv2_primkd_logit_only_w0175_t4_run01

- model: `dformerv2_primkd_logit_only`
- purpose: PMAD / PrimKD logit-only KD weight ablation at w=0.175.
- student: `DFormerv2_S + ResNet-18 DepthEncoder + GatedFusion + SimpleFPNDecoder`
- teacher: frozen `dformerv2_geometry_primary_teacher_run01`
- KD settings: `kd_weight=0.175`, `kd_temperature=4.0`, logit-only
- recorded validation epochs: 50
- best val/mIoU: `0.518158` at epoch 49
- last val/mIoU: `0.491036`
- mean val/mIoU over last 10 epochs: `0.498970`
- best val/loss: `1.095414` at epoch 8
- last val/loss: `1.303892`
- final train/loss: `0.251676`
- final train/ce_loss: `0.173178`
- final train/kd_loss: `0.448565`
- delta vs clean baseline mean: `+0.000761` (`+0.155` std units)
- delta vs clean baseline mean + 1 std: `-0.004140`
- delta vs w=0.15 5-run mean: `-0.002637`
- checkpoint: `checkpoints/dformerv2_primkd_logit_only_w0175_t4_run01/dformerv2_primkd_logit_only-epoch=48-val_mIoU=0.5182.pt`
- evidence: `miou_list/dformerv2_primkd_logit_only_w0175_t4_run01.md`
- conclusion: **neutral.** Near baseline mean but no advantage over w=0.15.

## 2026-05-11 dformerv2_primkd_logit_only_w015_t4_run05

- model: `dformerv2_primkd_logit_only`
- purpose: 5th repeat of w=0.15 PMAD logit-only KD for stability statistics.
- student: `DFormerv2_S + ResNet-18 DepthEncoder + GatedFusion + SimpleFPNDecoder`
- teacher: frozen `dformerv2_geometry_primary_teacher_run01`
- KD settings: `kd_weight=0.15`, `kd_temperature=4.0`, logit-only
- recorded validation epochs: 50
- best val/mIoU: `0.514204` at epoch 41
- last val/mIoU: `0.505725`
- mean val/mIoU over last 10 epochs: `0.503790`
- best val/loss: `1.068434` at epoch 8
- last val/loss: `1.274728`
- final train/loss: `0.209402`
- final train/ce_loss: `0.149067`
- final train/kd_loss: `0.402230`
- delta vs clean baseline mean: `-0.003193` (`-0.651` std units)
- delta vs clean baseline mean + 1 std: `-0.008094`
- delta vs w=0.15 4-run mean (excl run05): `-0.009791`
- checkpoint: `checkpoints/dformerv2_primkd_logit_only_w015_t4_run05/dformerv2_primkd_logit_only-epoch=40-val_mIoU=0.5142.pt`
- evidence: `miou_list/dformerv2_primkd_logit_only_w015_t4_run05.md`
- conclusion: **negative outlier.** This run pulled the 5-run mean down. Below baseline mean. The w=0.15 signal is not perfectly stable 鈥?1/5 runs falls below baseline.

## 2026-05-11 dformerv2_primkd_logit_only_w015_t4_run04

- model: `dformerv2_primkd_logit_only`
- purpose: 4th repeat of w=0.15 PMAD logit-only KD for stability statistics.
- student: `DFormerv2_S + ResNet-18 DepthEncoder + GatedFusion + SimpleFPNDecoder`
- teacher: frozen `dformerv2_geometry_primary_teacher_run01`
- KD settings: `kd_weight=0.15`, `kd_temperature=4.0`, logit-only
- recorded validation epochs: 50
- best val/mIoU: `0.524028` at epoch 45
- last val/mIoU: `0.490264`
- mean val/mIoU over last 10 epochs: `0.504944`
- best val/loss: `1.061741` at epoch 7
- last val/loss: `1.274921`
- final train/loss: `0.230699`
- final train/ce_loss: `0.163558`
- final train/kd_loss: `0.447604`
- delta vs clean baseline mean: `+0.006631` (`+1.353` std units)
- delta vs clean baseline mean + 1 std: `+0.001730`
- delta vs clean baseline best single: `-0.000397`
- checkpoint: `checkpoints/dformerv2_primkd_logit_only_w015_t4_run04/dformerv2_primkd_logit_only-epoch=44-val_mIoU=0.5240.pt`
- evidence: `miou_list/dformerv2_primkd_logit_only_w015_t4_run04.md`
- conclusion: **strong positive.** Best w=0.15 run, exceeds baseline mean + 1 std, nearly matches baseline best single. Shows PMAD can produce near-peak baseline performance.

## 2026-05-11 dformerv2_primkd_logit_only_w015_t4_run03

- model: `dformerv2_primkd_logit_only`
- purpose: 3rd repeat of w=0.15 PMAD logit-only KD for stability statistics.
- student: `DFormerv2_S + ResNet-18 DepthEncoder + GatedFusion + SimpleFPNDecoder`
- teacher: frozen `dformerv2_geometry_primary_teacher_run01`
- KD settings: `kd_weight=0.15`, `kd_temperature=4.0`, logit-only
- recorded validation epochs: unknown (TensorBoard validation log incomplete)
- best val/mIoU: `0.522600` at epoch 41 (from checkpoint filename)
- delta vs clean baseline mean: `+0.005203` (`+1.062` std units)
- delta vs clean baseline mean + 1 std: `+0.000302`
- checkpoint: `checkpoints/dformerv2_primkd_logit_only_w015_t4_run03/dformerv2_primkd_logit_only-epoch=41-val_mIoU=0.5226.pt`
- evidence: `miou_list/dformerv2_primkd_logit_only_w015_t4_run03.md`
- conclusion: **positive.** Exceeds baseline mean + 1 std. Consistent with run01/run04.

## 2026-05-10 dformerv2_primkd_logit_only_w015_t4_run02

- model: `dformerv2_primkd_logit_only`
- purpose: repeat `kd_weight=0.15` PMAD / PrimKD logit-only run to test whether the positive run01 signal is reproducible.
- student: `DFormerv2_S + ResNet-18 DepthEncoder + GatedFusion + SimpleFPNDecoder`
- teacher: frozen `dformerv2_geometry_primary_teacher_run01`, `DFormerv2_S(rgb, real_depth) + SimpleFPNDecoder`
- teacher checkpoint: `checkpoints/dformerv2_geometry_primary_teacher_run01/dformerv2_geometry_primary_teacher-epoch=37-val_mIoU=0.5168.pt`
- settings: `batch_size=2`, `max_epochs=50`, `lr=6e-5`, `num_workers=4`, `early_stop_patience=30`, `loss_type=ce`
- KD settings: `kd_weight=0.15`, `kd_temperature=4.0`, logit-only, no feature KD
- pretrained: `C:/Users/qintian/Desktop/qintian/dformer_work/checkpoints/pretrained/DFormerv2_Small_pretrained.pth`
- recorded validation epochs: 50
- best val/mIoU: `0.520144` at epoch 47
- last val/mIoU: `0.498182`
- best val/loss: `1.065779` at epoch 8
- last val/loss: `1.277715`
- mean val/mIoU over last 10 epochs: `0.502338`
- final train/loss: `0.295255`
- final train/ce_loss: `0.206779`
- final train/kd_loss: `0.589839`
- comparison clean 10-run RGB-D baseline mean best: `0.517397`
- comparison clean 10-run RGB-D baseline std: `0.004901`
- comparison clean 10-run RGB-D baseline mean + 1 std: `0.522298`
- comparison clean 10-run RGB-D baseline best single: `0.524425`
- comparison geometry-primary teacher best: `0.516824`
- comparison `dformerv2_primkd_logit_only_w015_t4_run01` best: `0.522998`
- comparison `dformerv2_primkd_logit_only_w010_t4_run01` best: `0.510068`
- current `kd_weight=0.15` 2-run mean best: `0.521571`
- current `kd_weight=0.15` 2-run population std: `0.001427`
- delta vs clean baseline mean: `+0.002747` (`+0.560` baseline std units)
- delta vs clean baseline mean + 1 std: `-0.002154`
- delta vs clean baseline best single: `-0.004281`
- delta vs teacher best: `+0.003320`
- delta vs `kd_weight=0.15` run01: `-0.002854`
- delta vs `kd_weight=0.10` run01: `+0.010075`
- checkpoint: `checkpoints/dformerv2_primkd_logit_only_w015_t4_run02/dformerv2_primkd_logit_only-epoch=46-val_mIoU=0.5201.pt`
- evidence: `miou_list/dformerv2_primkd_logit_only_w015_t4_run02.md`
- conclusion: **positive but not strong repeat.** The second `kd_weight=0.15` run beats the clean baseline mean and is much healthier than `kd_weight=0.10`, but it does not cross the pre-defined strong-signal threshold. The two-run mean is positive (`0.521571`) but still below `mean+1std=0.522298`.
- next step: run `dformerv2_primkd_logit_only_w015_t4_run03` before adding feature KD. If the 3-run mean stays `>=0.519`, PMAD can be treated as a marginal positive candidate; if it crosses `0.522298`, it becomes a strong main-result candidate. The late epoch-49 collapse should be noted as stability risk.

## 2026-05-10 dformerv2_primkd_logit_only_w010_t4_run01

- model: `dformerv2_primkd_logit_only`
- purpose: PMAD / PrimKD logit-only KD weight ablation after the positive `kd_weight=0.15` single run.
- student: `DFormerv2_S + ResNet-18 DepthEncoder + GatedFusion + SimpleFPNDecoder`
- teacher: frozen `dformerv2_geometry_primary_teacher_run01`, `DFormerv2_S(rgb, real_depth) + SimpleFPNDecoder`
- teacher checkpoint: `checkpoints/dformerv2_geometry_primary_teacher_run01/dformerv2_geometry_primary_teacher-epoch=37-val_mIoU=0.5168.pt`
- settings: `batch_size=2`, `max_epochs=50`, `lr=6e-5`, `num_workers=4`, `early_stop_patience=30`, `loss_type=ce`
- KD settings: `kd_weight=0.10`, `kd_temperature=4.0`, logit-only, no feature KD
- pretrained: `C:/Users/qintian/Desktop/qintian/dformer_work/checkpoints/pretrained/DFormerv2_Small_pretrained.pth`
- recorded validation epochs: 50
- best val/mIoU: `0.510068` at epoch 50
- last val/mIoU: `0.510068`
- best val/loss: `1.079473` at epoch 12
- last val/loss: `1.232582`
- mean val/mIoU over last 10 epochs: `0.500400`
- final train/loss: `0.197795`
- final train/ce_loss: `0.151020`
- final train/kd_loss: `0.467753`
- comparison clean 10-run RGB-D baseline mean best: `0.517397`
- comparison clean 10-run RGB-D baseline std: `0.004901`
- comparison clean 10-run RGB-D baseline mean + 1 std: `0.522298`
- comparison clean 10-run RGB-D baseline best single: `0.524425`
- comparison geometry-primary teacher best: `0.516824`
- comparison `dformerv2_primkd_logit_only_w015_t4_run01` best: `0.522998`
- delta vs clean baseline mean: `-0.007329` (`-1.495` baseline std units)
- delta vs clean baseline mean + 1 std: `-0.012230`
- delta vs clean baseline best single: `-0.014357`
- delta vs teacher best: `-0.006756`
- delta vs `kd_weight=0.15` PMAD run: `-0.012930`
- checkpoint: `checkpoints/dformerv2_primkd_logit_only_w010_t4_run01/dformerv2_primkd_logit_only-epoch=49-val_mIoU=0.5101.pt`
- evidence: `miou_list/dformerv2_primkd_logit_only_w010_t4_run01.md`
- conclusion: **negative KD-weight ablation.** Reducing KD from `0.15` to `0.10` removes the positive PMAD signal and falls well below the clean baseline mean. This does not support the hypothesis that weaker logit KD is safer for the usable-but-not-strong geometry-primary teacher.
- next step: do not repeat `kd_weight=0.10` and do not add feature KD on top of this setting. If continuing PMAD, the only decision-value next run is `kd_weight=0.20` or a repeat of the current best `kd_weight=0.15`; stop the logit-KD branch if those do not recover the `0.522+` signal.

## 2026-05-10 dformerv2_primkd_logit_only_w015_t4_run01

- model: `dformerv2_primkd_logit_only`
- purpose: Phase 1 PMAD / PrimKD logit-only distillation with geometry-primary teacher.
- student: `DFormerv2_S + ResNet-18 DepthEncoder + GatedFusion + SimpleFPNDecoder`
- teacher: frozen `dformerv2_geometry_primary_teacher_run01`, `DFormerv2_S(rgb, real_depth) + SimpleFPNDecoder`
- teacher checkpoint: `checkpoints/dformerv2_geometry_primary_teacher_run01/dformerv2_geometry_primary_teacher-epoch=37-val_mIoU=0.5168.pt`
- settings: `batch_size=2`, `max_epochs=50`, `lr=6e-5`, `num_workers=4`, `early_stop_patience=30`, `loss_type=ce`
- KD settings: `kd_weight=0.15`, `kd_temperature=4.0`, logit-only, no feature KD
- pretrained: `C:/Users/qintian/Desktop/qintian/dformer_work/checkpoints/pretrained/DFormerv2_Small_pretrained.pth`
- recorded validation epochs: 50
- best val/mIoU: `0.522998` at epoch `48` (checkpoint filename epoch `47`)
- last val/mIoU: `0.513176`
- best val/loss: `1.060515` at epoch `7`
- last val/loss: `1.203945`
- mean val/mIoU over last 10 epochs: `0.504683`
- final train/loss_epoch: `0.246043`
- final train/ce_loss_epoch: `0.172287`
- final train/kd_loss_epoch: `0.491706`
- comparison clean 10-run RGB-D baseline mean best: `0.517397`
- comparison clean 10-run RGB-D baseline std: `0.004901`
- comparison clean 10-run RGB-D baseline mean + 1 std: `0.522298`
- comparison clean 10-run RGB-D baseline best single: `0.524425`
- comparison repeat5 RGB-D baseline mean best: `0.511893`
- comparison geometry-primary teacher best: `0.516824`
- delta vs clean baseline mean: `+0.005601` (`+1.143` baseline std units)
- delta vs clean baseline mean + 1 std: `+0.000700`
- delta vs clean baseline best single: `-0.001427`
- delta vs repeat5 mean: `+0.011105`
- delta vs teacher best: `+0.006174`
- checkpoint: `checkpoints/dformerv2_primkd_logit_only_w015_t4_run01/dformerv2_primkd_logit_only-epoch=47-val_mIoU=0.5230.pt`
- evidence: `miou_list/dformerv2_primkd_logit_only_w015_t4_run01.md`
- conclusion: **positive single-run PMAD signal.** Logit-only PMAD with `kd_weight=0.15` exceeds the clean baseline mean by more than 1 std and is above the pre-defined strong-signal threshold, though it does not exceed the best clean baseline single run. This is the best PMAD evidence so far but not yet a stable improvement claim.
- next step: do not add feature KD yet. First confirm the logit-only effect with decision-value ablations: run `kd_weight=0.10` and/or `kd_weight=0.20`, then repeat the best setting for 3 runs if it remains above baseline mean + 1 std.

## 2026-05-10 dformerv2_geometry_primary_teacher_run01

- model: `dformerv2_geometry_primary_teacher`
- purpose: Phase 0 geometry-primary teacher sanity check before PMAD / PrimKD logit distillation.
- architecture: `DFormerv2_S(rgb, real_depth) + SimpleFPNDecoder`; no extra `DepthEncoder + GatedFusion` branch.
- settings: `batch_size=2`, `max_epochs=50`, `lr=6e-5`, `num_workers=4`, `early_stop_patience=30`, `loss_type=ce`
- pretrained: `C:/Users/qintian/Desktop/qintian/dformer_work/checkpoints/pretrained/DFormerv2_Small_pretrained.pth`
- recorded validation epochs: 50
- best val/mIoU: `0.516824` at epoch `38` (checkpoint filename epoch `37`)
- last val/mIoU: `0.509223`
- best val/loss: `1.032507` at epoch `8`
- last val/loss: `1.263402`
- mean val/mIoU over last 10 epochs: `0.504379`
- teacher usability threshold: `0.515000`
- strong teacher threshold (clean baseline mean + 1 std): `0.522298`
- comparison clean 10-run RGB-D baseline mean best: `0.517397`
- comparison clean 10-run RGB-D baseline std: `0.004901`
- comparison constant-zero teacher best: `0.488489`
- delta vs teacher usability threshold: `+0.001824`
- delta vs clean baseline mean: `-0.000573` (`-0.117` baseline std units)
- delta vs constant-zero teacher: `+0.028335`
- checkpoint: `checkpoints/dformerv2_geometry_primary_teacher_run01/dformerv2_geometry_primary_teacher-epoch=37-val_mIoU=0.5168.pt`
- evidence: `miou_list/dformerv2_geometry_primary_teacher_run01.md`
- conclusion: **usable teacher.** The geometry-primary teacher passes the minimum `0.515` gate and is essentially tied with the full RGB-D baseline mean, while improving over the failed constant-zero teacher by `+0.028335`. This confirms that real DFormerV2 depth geometry prior is necessary for the teacher.
- next step: proceed to Phase 1 PMAD logit-only with this checkpoint. Use conservative KD settings because the teacher is usable but not strong: `kd_weight=0.15`, `kd_temperature=4.0`. Do not repeat teacher now; only revisit teacher repeats if PMAD shows a positive signal.

## 2026-05-10 dformerv2_rgb_teacher_constdepth_run01

- model: `dformerv2_rgb_teacher_constdepth`
- purpose: Phase 0 RGB-only teacher sanity check before PMAD / PrimKD logit distillation.
- architecture: `DFormerv2_S + SimpleFPNDecoder`; model-internal constant zero depth replaces real depth, so DFormerV2 geometry prior degenerates to position-only prior.
- settings: `batch_size=2`, `max_epochs=50`, `lr=6e-5`, `num_workers=4`, `early_stop_patience=30`, `loss_type=ce`
- pretrained: `C:/Users/qintian/Desktop/qintian/dformer_work/checkpoints/pretrained/DFormerv2_Small_pretrained.pth`
- recorded validation epochs: 50
- best val/mIoU: `0.488489` at epoch `43`
- last val/mIoU: `0.456266`
- best val/loss: `1.102055` at epoch `7`
- last val/loss: `1.454060`
- mean val/mIoU over last 10 epochs: `0.468347`
- teacher usability threshold: `0.515000`
- comparison clean 10-run RGB-D baseline mean best: `0.517397`
- comparison clean 10-run RGB-D baseline std: `0.004901`
- delta vs teacher usability threshold: `-0.026511`
- delta vs clean baseline mean: `-0.028908` (`-5.898` baseline std units)
- checkpoint: `checkpoints/dformerv2_rgb_teacher_constdepth_run01/dformerv2_rgb_teacher_constdepth-epoch=42-val_mIoU=0.4885.pt`
- evidence: `miou_list/dformerv2_rgb_teacher_constdepth_run01.md`
- conclusion: **teacher sanity failed.** The constant-depth RGB-only DFormerV2 teacher is far below the minimum `0.515` gate and far below the RGB-D baseline. It should not be used as the PMAD teacher because logit distillation would anchor the RGB-D student to a weaker decision boundary.
- next step: do not run formal `dformerv2_primkd_logit_only` with this checkpoint. Either fix the teacher design/training first or pause PMAD. The most likely issue is that zeroing depth removes useful DFormerV2 geometry information while the teacher still lacks the RGB-D baseline's depth branch and fusion path.

## 2026-05-10 dformerv2_mid_fusion_gate_baseline_repeat5 summary

- model: `dformerv2_mid_fusion`
- change: baseline sanity check 鈥?5 repeated runs of the exact same baseline config after C4 PPM and CE+Dice experiments both yielded ~0.507
- purpose: verify whether the baseline itself has shifted, or whether the ~0.507 results are genuine negatives
- settings: `batch_size=2`, `max_epochs=50`, `lr=6e-5`, `num_workers=4`, `early_stop_patience=30`, `loss_type=ce`
- pretrained: `C:/Users/qintian/Desktop/qintian/dformer_work/checkpoints/pretrained/DFormerv2_Small_pretrained.pth`
- code diff vs original baseline: none (`dformerv2_mid_fusion` path untouched; all recent changes are purely additive to new model classes)
- run01: best val/mIoU `0.509013` at epoch 32, last `0.496441`, best val/loss `1.000347` at epoch 10
- run02: best val/mIoU `0.510994` at epoch 45, last `0.475698`, best val/loss `1.039783` at epoch 12
- run03: best val/mIoU `0.510176` at epoch 46, last `0.466466`, best val/loss `1.028617` at epoch 13
- run04: best val/mIoU `0.517698` at epoch 42, last `0.512032`, best val/loss `1.011931` at epoch 10
- run05: best val/mIoU `0.511585` at epoch 39, last `0.476304`, best val/loss `1.024783` at epoch 11
- repeat5 mean best val/mIoU: `0.511893`
- repeat5 population std: `0.003028`
- repeat5 min best: `0.509013` (run01)
- repeat5 max best: `0.517698` (run04)
- repeat5 mean last val/mIoU: `0.485388`
- repeat5 mean best val/loss: `1.021093`
- original 10-run baseline mean best: `0.517397`
- original 10-run baseline std: `0.004901`
- original 10-run baseline best single: `0.524425`
- delta repeat5 mean vs original mean: `-0.005504` (1.12 original std units)
- evidence: `miou_list/dformerv2_mid_fusion_gate_baseline_repeat5_run01.md` through `run05.md`, and `miou_list/dformerv2_mid_fusion_gate_baseline_repeat5_summary.md`
- conclusion: **the repeat5 baseline is ~0.005 below the original 10-run mean.** No run exceeded 0.518; the upper tail (0.519-0.524) observed in the original 10 runs is missing. The distribution is compressed (std=0.003 vs 0.005). This suggests either random variance with only 5 samples, or a mild environmental shift. Relative to the repeat5 mean (0.5119), the CE+Dice (0.5070) and C4 PPM (0.5073) experiments are only ~1.6蟽 below 鈥?still negative, but less anomalous than when measured against the original mean. The baseline code is confirmed clean; the gap is likely due to seed/environment variance rather than code contamination.
- next step: the two 0.507 experiments remain negative relative to both baselines. Use the repeat5 mean (0.5119) as the more conservative reference for future comparisons. If further experiments also fall below 0.510, investigate environmental factors (GPU state, driver version, data integrity).

## 2026-05-10 dformerv2_context_decoder_c4ppm_run01

- model: `dformerv2_context_decoder`
- change: decoder-side lightweight context refinement with a C4 PPM block before FPN lateral4.
- settings: `batch_size=2`, `max_epochs=50`, `lr=6e-5`, `num_workers=4`, `early_stop_patience=30`, `loss_type=ce`
- pretrained: `C:/Users/qintian/Desktop/qintian/dformer_work/checkpoints/pretrained/DFormerv2_Small_pretrained.pth`
- recorded validation epochs: 50
- best val/mIoU: 0.507293 at epoch 49
- last val/mIoU: 0.507293
- best val/loss: 1.042966 at epoch 11
- last val/loss: 1.266029
- mean val/mIoU over last 10 epochs: 0.498246
- comparison clean 10-run baseline mean best: 0.517397
- delta vs baseline mean best: -0.010104
- comparison clean 10-run baseline std: 0.004901
- delta in baseline std units: -2.062
- comparison clean 10-run baseline best single: 0.524425
- delta vs baseline best single: -0.017132
- evidence: `miou_list/dformerv2_context_decoder_c4ppm_run01.md`
- conclusion: **negative single-run result.** C4 PPM context decoder underperforms the clean baseline by 2.06 std. Best mIoU 0.5073 is essentially identical to CE+Dice's 0.5070, both well below the baseline mean 0.5174. The PPM context block does not help this architecture. val/loss rises from 1.04 (epoch 11) to 1.27 (epoch 49) while train/loss decreases to 0.14, showing moderate overfitting but less severe than CE+Dice.
- next step: do not claim as improvement. The decoder-side PPM context refinement does not produce gains on this architecture. At this point, all tested modification categories (fusion replacement, FFT enhancement, auxiliary loss, loss recipe, decoder context) have failed to beat the clean baseline. Consider whether the architecture is already near-optimal for NYUDepthV2, or whether a fundamentally different approach is needed.

## 2026-05-09 dformerv2_mid_fusion_ce_dice_w05_run01

- model: `dformerv2_mid_fusion`
- change: CE + Dice loss recipe; `loss_type=ce_dice`, `dice_weight=0.5`.
- settings: `batch_size=2`, `max_epochs=50`, `lr=6e-5`, `num_workers=4`, `early_stop_patience=30`, `loss_type=ce_dice`, `dice_weight=0.5`
- pretrained: `C:/Users/qintian/Desktop/qintian/dformer_work/checkpoints/pretrained/DFormerv2_Small_pretrained.pth`
- training loss: `CrossEntropyLoss(ignore_index=255) + 0.5 * DiceLoss(ignore_index=255)`.
- validation loss: fixed CE for comparability with previous logs.
- validation metric: `val/mIoU` unchanged.
- model structure: unchanged `DFormerV2 primary branch + DepthEncoder + GatedFusion + SimpleFPNDecoder`.
- recorded validation epochs: 50
- best val/mIoU: 0.507000 at epoch 41
- last val/mIoU: 0.489077
- best val/loss: 1.037181 at epoch 6
- last val/loss: 1.830893
- mean val/mIoU over last 10 epochs: 0.498237
- comparison clean 10-run baseline mean best: 0.517397
- delta vs baseline mean best: -0.010397
- comparison clean 10-run baseline std: 0.004901
- delta in baseline std units: -2.121
- comparison clean 10-run baseline best single: 0.524425
- delta vs baseline best single: -0.017425
- evidence: `miou_list/dformerv2_mid_fusion_ce_dice_w05_run01.md`
- conclusion: **negative single-run result.** CE + Dice (w=0.5) significantly underperforms the clean CE baseline. Best mIoU 0.5070 is 2.12 baseline std below the baseline mean. Training shows clear overfitting: val/loss rises from 1.037 (epoch 6) to 1.831 (epoch 49) while train/loss continues decreasing from 1.069 to 0.282. The Dice component appears to destabilize late training, causing val/mIoU to plateau in the 0.48-0.51 range instead of the baseline's 0.50-0.52 range.
- next step: do not claim as improvement. The CE + Dice recipe is harmful for this architecture. If pursuing loss experiments, test Dice weight 0.1 or 0.2 (weaker Dice); or test FocalLoss to see if class imbalance correction helps. Otherwise, remain on pure CE baseline.

## 2026-05-09 dformerv2_fft_freq_enhance_hh_w1111_c025_g01 3-run summary

- model: `dformerv2_fft_freq_enhance`
- change: 3-run repeat of FFT high-frequency enhancement with cutoff=0.25, gamma_init=0.1.
- run01: best val/mIoU `0.522688` at epoch `41`, last `0.514651`, delta vs baseline mean `+0.005291`
- run02: best val/mIoU `0.5159` at epoch `38`, last `0.443`, delta vs baseline mean `-0.001497`
- run03: best val/mIoU `0.5145` at epoch `42`, last `0.489`, delta vs baseline mean `-0.002897`
- 3-run mean best val/mIoU: `0.517696`
- 3-run population std: `0.003664`
- 3-run mean delta vs clean 10-run baseline mean: `+0.000299` (0.061 std)
- conclusion: **not a stable improvement**. The 3-run mean 0.517696 is essentially identical to the baseline mean 0.517397. run01 was a high-variance outlier; run02 and run03 both fell below baseline. The FFT freq_enhance direction does not produce consistent gains.
- next step: deprioritize FFT freq_enhance. The initial positive signal was statistical noise. Consider whether any other direction warrants exploration, or whether the current baseline is already near-optimal for this architecture.

## 2026-05-09 dformerv2_fft_freq_enhance_hh_w1111_c025_g01_run02

- model: `dformerv2_fft_freq_enhance`
- change: repeat of FFT freq_enhance with same settings as run01.
- settings: `batch_size=2`, `max_epochs=50`, `lr=6e-5`, `num_workers=4`, `early_stop_patience=30`, `cutoff_ratio=0.25`, `gamma_init=0.1`
- recorded validation epochs: `50`
- best val/mIoU: `0.5159` at recorded epoch `38`
- last val/mIoU: `0.443`
- comparison clean 10-run GatedFusion baseline mean best: `0.517397`
- delta vs clean 10-run baseline mean: `-0.001497`
- comparison run01 best: `0.522688`
- delta vs run01: `-0.006788`
- evidence: `miou_list/dformerv2_fft_freq_enhance_hh_w1111_c025_g01_run02.md`
- conclusion: negative result. Below baseline mean. Severe late collapse (best 0.5159 鈫?last 0.443).

## 2026-05-09 dformerv2_fft_freq_enhance_hh_w1111_c025_g01_run03

- model: `dformerv2_fft_freq_enhance`
- change: repeat of FFT freq_enhance with same settings as run01.
- settings: `batch_size=2`, `max_epochs=50`, `lr=6e-5`, `num_workers=4`, `early_stop_patience=30`, `cutoff_ratio=0.25`, `gamma_init=0.1`
- recorded validation epochs: `50`
- best val/mIoU: `0.5145` at recorded epoch `42`
- last val/mIoU: `0.489`
- comparison clean 10-run GatedFusion baseline mean best: `0.517397`
- delta vs clean 10-run baseline mean: `-0.002897`
- comparison run01 best: `0.522688`
- delta vs run01: `-0.008188`
- evidence: `miou_list/dformerv2_fft_freq_enhance_hh_w1111_c025_g01_run03.md`
- conclusion: negative result. Below baseline mean. Late collapse (best 0.5145 鈫?last 0.489).

## 2026-05-09 dformerv2_fft_hilo_enhance_w1111_c025_ah01_al003_am05_run01

- model: `dformerv2_fft_hilo_enhance`
- change: FFT low/high dual-band gated residual enhancement on both primary/DFormerV2 features and aligned depth features before GatedFusion.
- settings: `batch_size=2`, `max_epochs=50`, `lr=6e-5`, `num_workers=4`, `early_stop_patience=30`, `cutoff_ratio=0.25`, `alpha_high_init=0.10`, `alpha_low_init=0.03`, `alpha_max=0.5`, `hilo_stage_weights=1,1,1,1`
- pretrained: `C:/Users/qintian/Desktop/qintian/dformer_work/checkpoints/pretrained/DFormerv2_Small_pretrained.pth`
- total params: 46.6M
- recorded validation epochs: `50`
- best val/mIoU: `0.519128` at recorded epoch `41`
- last val/mIoU: `0.518313`
- best val/loss: `1.017970` at recorded epoch `9`
- train/loss_epoch: first `2.251478`, last `0.134208`
- mean val/mIoU over last 10 epochs (40-49): `0.508414`
- comparison clean 10-run GatedFusion baseline mean best: `0.517397`
- delta vs clean 10-run baseline mean: `+0.001731`
- comparison clean 10-run baseline std: `0.004901`
- delta in baseline std units: `+0.353`
- comparison clean 10-run baseline best single run: `0.524425`
- delta vs clean baseline best single run: `-0.005297`
- comparison dformerv2_fft_freq_enhance g01 best: `0.522688`
- delta vs freq_enhance g01: `-0.003560`
- evidence: `miou_list/dformerv2_fft_hilo_enhance_w1111_c025_ah01_al003_am05_run01.md`
- conclusion: positive single-run signal but marginal. Best val/mIoU 0.519128 beats the clean 10-run baseline mean by +0.001731 (0.35 std), but does not beat the baseline best single run 0.524425. The last-epoch mIoU 0.518313 is very close to the best, indicating stable late training without collapse. However, this result is weaker than the original dformerv2_fft_freq_enhance (gamma=0.1) single run 0.522688. The HiLo dual-band design with alpha_low=0.03, alpha_high=0.10 did not outperform the simpler high-frequency-only freq_enhance design. Training shows significant oscillation in val/mIoU throughout (drops at epochs 21, 29-30, 36, 42-43), suggesting the dual-band enhancement introduces more instability than the single-band design.
- next step: do not claim as improvement. The HiLo design adds complexity without clear gain over the simpler freq_enhance. If pursuing this direction, test alpha_low_init=0 (disable low-freq enhancement) to isolate whether the low-frequency branch hurts; or test higher alpha_high_init=0.15 to match freq_enhance's effective gamma range.

## 2026-05-09 dformerv2_fft_hilo_enhance implementation

- model: `dformerv2_fft_hilo_enhance`
- status: code implemented; training completed (see run01 entry above).

## 2026-05-09 dformerv2_cm_infonce_c34_lam005_t01_s256_run01

- model: `dformerv2_cm_infonce`
- change: training-only c3+c4 one-way depth-to-primary InfoNCE contrastive auxiliary loss on top of the unchanged DFormerv2 mid-fusion inference path.
- settings: `batch_size=2`, `max_epochs=50`, `lr=6e-5`, `num_workers=4`, `early_stop_patience=30`, `lambda_contrast=0.005`, `contrast_temperature=0.1`, `contrast_proj_dim=64`, `contrast_sample_points=256`, `contrast_stage_weights=0,0,1,1`
- pretrained: `C:/Users/qintian/Desktop/qintian/dformer_work/checkpoints/pretrained/DFormerv2_Small_pretrained.pth`
- recorded validation epochs: `50`
- best val/mIoU: `0.514461` at recorded epoch `46`
- last val/mIoU: `0.498469`
- train/contrast_loss_epoch: first `5.574991`, last `3.089162`
- contrast loss dropped 45% from init, indicating InfoNCE is learning cross-modal alignment.
- lambda-weighted contrast contribution: first `0.027875`, last `0.015446`
- train/seg_loss_epoch: first `2.231843`, last `0.154907`
- comparison clean 10-run GatedFusion baseline mean best: `0.517397`
- delta vs clean 10-run baseline mean: `-0.002936`
- comparison clean 10-run baseline std: `0.004901`
- delta in baseline std units: `-0.599`
- comparison clean 10-run baseline best single run: `0.524425`
- delta vs clean baseline best single run: `-0.009964`
- late collapse: epoch 46 `0.514461` 鈫?epoch 49 `0.498469`, drop `0.015992` in 3 epochs.
- evidence: `miou_list/dformerv2_cm_infonce_c34_lam005_t01_s256_run01.md`
- conclusion: negative single-run result. InfoNCE contrast loss converges (5.57鈫?.09, 45% drop), but the alignment signal does not improve validation mIoU above the clean repeated baseline mean. The run shows late collapse similar to other auxiliary loss experiments. The contrast loss is learning, but the learned cross-modal alignment is either not beneficial for segmentation or too weak to overcome the noise it introduces.
- next step: do not claim as improvement. If continuing InfoNCE, test `lambda_contrast=0.01` or `0.02` to see if stronger contrast signal helps; alternatively test `contrast_stage_weights=0,1,1,1` to include c2. If both negative, pivot away from contrastive losses entirely.

## 2026-05-08 dformerv2_cm_infonce implementation

- model: `dformerv2_cm_infonce`
- status: code implemented; waiting for formal training.
- purpose: add a training-only cross-modal InfoNCE auxiliary loss while keeping the DFormerv2 mid-fusion inference path unchanged.
- inference path: `DFormerV2_S + DepthEncoder + GatedFusion + SimpleFPNDecoder`.
- feature source: reuses `DFormerV2MidFusionSegmentor.extract_features(rgb, depth)` to obtain DFormerV2 primary c1-c4 features, aligned depth c1-c4 features, and fused features.
- contrast direction: one-way depth query to primary key.
- key/query design: `k = primary_proj(P.detach())`, `q = depth_proj(D)`.
- gradient boundary: the DFormerV2 primary encoder is protected from contrast loss gradients, but the primary projection head still receives contrast loss gradients.
- stage setting: first configuration uses c3+c4 only with `contrast_stage_weights=0,0,1,1`.
- defaults: `lambda_contrast=0.005`, `contrast_temperature=0.1`, `contrast_proj_dim=64`, `contrast_sample_points=256`.
- training loss: `L_total = L_seg + lambda_contrast * L_contrast`.
- code evidence: `src/models/contrastive_loss.py`, `src/models/mid_fusion.py`, and `train.py`.
- result: no mIoU yet; do not cite as an experimental improvement until a completed run has TensorBoard evidence and a `miou_list` record.

## 2026-05-08 dformerv2_depth_fft_select_c030_run01

- model: `dformerv2_depth_fft_select`
- change: FADC-style internal depth FFT low/high frequency selection after DepthEncoder c2/c3/c4; c1 skipped.
- settings: `batch_size=2`, `max_epochs=50`, `lr=6e-5`, `num_workers=4`, `early_stop_patience=30`, `cutoff_ratio=0.30`
- pretrained: `C:/Users/qintian/Desktop/qintian/dformer_work/checkpoints/pretrained/DFormerv2_Small_pretrained.pth`
- recorded validation epochs: `50`
- best val/mIoU: `0.513871` at recorded epoch `43`
- last val/mIoU: `0.482797`
- best val/loss: `1.025829` at recorded epoch `11`
- train/loss_epoch: first `2.230345`, last `0.180115`
- mean val/mIoU over last 10 epochs: `0.494112`
- comparison clean 10-run GatedFusion baseline mean best: `0.517397`
- delta vs clean 10-run baseline mean: `-0.003526`
- comparison clean 10-run baseline std: `0.004901`
- delta in baseline std units: `-0.719`
- comparison clean 10-run baseline best single run: `0.524425`
- delta vs clean baseline best single run: `-0.010554`
- checkpoint diagnostics: bias-implied selection weights stayed near identity: c2 low/high `0.997994/1.007397`, c3 low/high `0.993755/1.011458`, c4 low/high `0.993354/1.007414`.
- checkpoint weight norms for `DepthFrequencySelect` remained small: c2 low/high `0.258338/0.393132`, c3 low/high `0.522824/0.737450`, c4 low/high `0.691974/0.940259`.
- evidence: `miou_list/dformerv2_depth_fft_select_c030_run01.md`
- conclusion: negative single-run result. The internal depth FFT selection branch remained close to identity and did not beat the clean repeated GatedFusion baseline. This suggests that encoder-internal FFT selection at abstract ResNet feature stages is not currently a promising main path.
- next step: do not continue deepening this encoder-internal selection branch. If pursuing frequency modeling, the stronger evidence remains the post-encoder/pre-fusion `dformerv2_fft_freq_enhance` setting with `cutoff_ratio=0.25`, `gamma_init=0.1`, which should be repeated before new complexity is added.

## 2026-05-08 dformerv2_depth_fft_select implementation

- model: `dformerv2_depth_fft_select`
- status: code implemented; waiting for formal training.
- purpose: test FADC-style internal depth frequency selection by inserting FFT low/high selection into the DepthEncoder stage flow before GatedFusion.
- position: `DepthEncoder layer2/layer3/layer4 -> DepthFrequencySelect -> next stage and returned depth feature`; c1 is skipped.
- frequency decomposition: spatial `torch.fft.fft2` over H/W, `torch.fft.fftshift`, hard circular low-frequency mask in the Fourier plane, `torch.fft.ifftshift`, and `torch.fft.ifft2(...).real`.
- formula: `D_low = IFFT2(M_low * FFT2(D)).real`, `D_high = D - D_low`, `D_out = D + (w_low - 1) * D_low + (w_high - 1) * D_high`.
- selection weights: `w_low = sigmoid(DWConv(D)) * 2`, `w_high = sigmoid(DWConv(D)) * 2`.
- initialization: both low/high depthwise convolutions are zero-initialized, so the initial module is identity.
- first configuration: `cutoff_ratio=0.30`.
- training loss: segmentation loss only; no auxiliary loss is added.
- code evidence: `src/models/depth_fft_select.py`, `src/models/mid_fusion.py`, and `train.py`.
- result: no mIoU yet; do not cite as an experimental improvement until a completed run has TensorBoard evidence and a `miou_list` record.

## 2026-05-08 dformerv2_fft_freq_enhance_hh_w1111_c025_g01_run01

- model: `dformerv2_fft_freq_enhance`
- change: FFT-based modality-wise frequency enhancement before GatedFusion on both primary/DFormerV2 features and aligned depth features.
- settings: `batch_size=2`, `max_epochs=50`, `lr=6e-5`, `num_workers=4`, `early_stop_patience=30`, `cutoff_ratio=0.25`, `gamma_init=0.1`
- pretrained: `C:/Users/qintian/Desktop/qintian/dformer_work/checkpoints/pretrained/DFormerv2_Small_pretrained.pth`
- recorded validation epochs: `50`
- best val/mIoU: `0.522688` at recorded epoch `41`
- last val/mIoU: `0.514651`
- best val/loss: `1.015986` at recorded epoch `13`
- train/loss_epoch: first `2.242228`, last `0.139953`
- comparison clean 10-run GatedFusion baseline mean best: `0.517397`
- delta vs clean 10-run baseline mean: `+0.005291`
- comparison clean 10-run baseline std: `0.004901`
- comparison clean 10-run baseline best single run: `0.524425`
- delta vs clean 10-run baseline best single run: `-0.001737`
- evidence: `miou_list/dformerv2_fft_freq_enhance_hh_w1111_c025_g01_run01.md`
- conclusion: positive single-run signal. The run beats the clean 10-run GatedFusion baseline mean by slightly more than one baseline standard deviation, but it does not beat the strongest clean baseline single run. Do not claim stable improvement until repeated runs confirm the mean.
- next step: repeat the same setting for at least 5 runs. If the repeated mean stays above `0.517397`, then test a small sweep around `gamma_init=0.05/0.1` and `cutoff_ratio=0.20/0.25/0.30`.

## 2026-05-08 dformerv2_fft_freq_enhance_hh_w1111_c025_g02_run01

- model: `dformerv2_fft_freq_enhance`
- change: same FFT enhancement as g01 but with higher `gamma_init=0.20` to test whether a stronger initial frequency enhancement push helps.
- settings: `batch_size=2`, `max_epochs=50`, `lr=6e-5`, `num_workers=4`, `early_stop_patience=30`, `cutoff_ratio=0.25`, `gamma_init=0.2`
- pretrained: `C:/Users/qintian/Desktop/qintian/dformer_work/checkpoints/pretrained/DFormerv2_Small_pretrained.pth`
- recorded validation epochs: `50`
- best val/mIoU: `0.515696` at recorded epoch `47`
- last val/mIoU: `0.476463`
- best val/loss: `1.032634` at recorded epoch `6`
- comparison g01 best (gamma=0.1): `0.522688`
- delta vs g01: `-0.006992`
- comparison clean 10-run GatedFusion baseline mean best: `0.517397`
- delta vs clean 10-run baseline mean: `-0.001701`
- evidence: `miou_list/dformerv2_fft_freq_enhance_hh_w1111_c025_g02_run01.md`
- conclusion: negative result. gamma=0.2 performs worse than gamma=0.1 and falls below the clean baseline mean. The run peaks very late (epoch 47) and then collapses sharply (epoch 50: 0.476463), indicating that stronger initial gamma destabilizes late training. gamma=0.1 remains the better setting. Do not further increase gamma_init.

## 2026-05-08 dformerv2_fft_freq_enhance implementation

- model: `dformerv2_fft_freq_enhance`
- status: code implemented; waiting for formal training.
- purpose: add FFT-based modality-wise frequency enhancement before GatedFusion while keeping the DFormerv2 + DepthEncoder + GatedFusion + SimpleFPNDecoder segmentation structure.
- position: `dformer_feats, aligned_depth -> FFTFrequencyEnhance -> GatedFusion -> SimpleFPNDecoder`.
- objects: both primary/DFormerV2 features and aligned depth features are enhanced at all four stages.
- frequency decomposition: spatial `torch.fft.fft2` over H/W, `torch.fft.fftshift`, hard circular low-frequency mask in the Fourier plane, `torch.fft.ifftshift`, and `torch.fft.ifft2(...).real`.
- formula: `X = FFT2(F)`, `F_low = IFFT2(M_low * X).real`, `F_high = F - F_low`, `F_out = F + gamma * sigmoid(Conv([F, F_high])) * Clean(F_high)`.
- first configuration: `cutoff_ratio=0.25`, `gamma_init=0.05`.
- training loss: segmentation loss only; no auxiliary loss is added.
- code evidence: `src/models/freq_enhance.py`, `src/models/mid_fusion.py`, and `train.py`.
- reference-code motivation: FADC `FrequencySelection` frequency branch and FreqFusion high/low-frequency fusion motivation; no mmcv/MMSegmentation modules were imported.
- verification: `compileall`, `train.py --help`, standalone FFT module backward test, and `224x224` model smoke test all passed.
- parameter count: baseline `dformerv2_mid_fusion` `41,046,082`; `dformerv2_fft_freq_enhance` `43,136,970`; added parameters `2,090,888`.
- result: no mIoU yet; do not cite as an experimental improvement until a completed run has TensorBoard evidence and a `miou_list` record.

## 2026-05-08 dformerv2_feat_maskrec_c34_w0011_lam01_run01

- model: `dformerv2_feat_maskrec_c34`
- change: training-only c3+c4 feature-level cross-modal mask reconstruction auxiliary loss on top of the unchanged DFormerv2 mid-fusion inference path.
- settings: `batch_size=2`, `max_epochs=50`, `lr=6e-5`, `num_workers=4`, `early_stop_patience=30`, `lambda_mask=0.1`, `mask_ratio_depth=0.30`, `mask_ratio_primary=0.15`, `maskrec_alpha=0.5`, `maskrec_loss_type=smooth_l1`, `maskrec_stage_weights=0,0,1,1`
- pretrained: `C:/Users/qintian/Desktop/qintian/dformer_work/checkpoints/pretrained/DFormerv2_Small_pretrained.pth`
- recorded validation epochs: `50`
- best val/mIoU: `0.515327` at recorded epoch `33`
- last val/mIoU: `0.501496`
- best val/loss: `1.016409` at recorded epoch `15`
- train/maskrec_loss_epoch: first `0.440276`, last `0.339426`
- lambda-weighted maskrec contribution: first `0.044028`, last `0.033943`
- maskrec/stage1 and maskrec/stage2 stayed `0`, confirming `maskrec_stage_weights=0,0,1,1` used c3+c4 only.
- comparison clean 10-run GatedFusion baseline mean best: `0.517397`
- delta vs clean 10-run baseline mean: `-0.002070`
- comparison clean 10-run baseline std: `0.004901`
- comparison clean 10-run baseline best single run: `0.524425`
- evidence: `miou_list/dformerv2_feat_maskrec_c34_w0011_lam01_run01.md`
- conclusion: negative single-run result. The c3+c4 mask reconstruction auxiliary loss optimizes normally, but the supervised reconstruction target does not improve validation mIoU in this run and ends below the clean repeated GatedFusion baseline mean.
- next step: do not claim this as an improvement. If continuing mask reconstruction, test whether a lower `lambda_mask=0.01` is less disruptive or whether c4-only `maskrec_stage_weights=0,0,0,1` is safer; otherwise pivot away from reconstruction losses.

## 2026-05-08 dformerv2_feat_maskrec_c34 implementation

- model: `dformerv2_feat_maskrec_c34`
- status: code implemented; waiting for formal training.
- purpose: add feature-level cross-modal mask reconstruction auxiliary loss while keeping the DFormerv2 + DepthEncoder + GatedFusion + SimpleFPNDecoder inference structure unchanged.
- training loss: `L_total = L_seg + lambda_mask * maskrec_loss`.
- maskrec loss: `maskrec_loss = sum_i w_i * (depth_rec_i + maskrec_alpha * primary_rec_i) / sum_i w_i`.
- stage control: `--maskrec_stage_weights` explicitly controls c1-c4 participation; `0,0,1,1` means c3+c4 only.
- default settings: `lambda_mask=0.01`, `mask_ratio_depth=0.30`, `mask_ratio_primary=0.15`, `maskrec_alpha=0.5`, `maskrec_loss_type=smooth_l1`; formal experiment commands must explicitly pass `--maskrec_stage_weights`.
- primary-to-depth: mask depth feature locations, reconstruct depth from `[primary_feat, masked_depth_feat]`, and supervise only masked depth locations.
- depth-to-primary: mask primary/DFormerv2 feature locations, reconstruct primary from `[depth_feat, masked_primary_feat]`, and supervise only masked primary locations.
- target features are detached; source features and masked target inputs are not detached.
- validation and inference remain ordinary segmentation forward passes.
- code evidence: `src/models/mask_reconstruction_loss.py`, `src/models/mid_fusion.py`, and `train.py`.
- reference-code motivation: MultiMAE masked multi-modal reconstruction and output-adapter organization; M3AE masked/missing modality reconstruction training organization.
- result: no mIoU yet; do not cite as an experimental improvement until a completed run has TensorBoard evidence and a `miou_list` record.

## 2026-05-08 dformerv2_ms_freqcov aggressive sweep summary

- model: `dformerv2_ms_freqcov`
- change: training-only c1-c4 multi-scale frequency covariance auxiliary loss on top of the unchanged DFormerv2 mid-fusion inference path.
- baseline for comparison: clean 10-run `dformerv2_mid_fusion` GatedFusion baseline mean best val/mIoU `0.517397`, population std `0.004901`, best single run `0.524425`.
- completed runs: `7` / `7`; every listed run has `50` recorded validation epochs.
- `dformerv2_ms_freqcov_run01`: `lambda_freq=0.01`, `freq_stage_weights=1,1,1,1`, best val/mIoU `0.520539` at epoch `50`, last `0.520539`, delta vs baseline mean `+0.003142`.
- `dformerv2_ms_freqcov_lam01_run01`: `lambda_freq=0.1`, `freq_stage_weights=1,1,1,1`, best val/mIoU `0.516227` at epoch `40`, last `0.499021`, delta vs baseline mean `-0.001170`.
- `dformerv2_ms_freqcov_lam1_run01`: `lambda_freq=1.0`, `freq_stage_weights=1,1,1,1`, best val/mIoU `0.518769` at epoch `45`, last `0.512625`, delta vs baseline mean `+0.001372`.
- `dformerv2_ms_freqcov_stage3412_run01`: `lambda_freq=0.1`, `freq_stage_weights=0.5,1,1,2`, best val/mIoU `0.515543` at epoch `44`, last `0.496894`, delta vs baseline mean `-0.001854`.
- `dformerv2_ms_freqcov_stage3412_lam1_run01`: `lambda_freq=1.0`, `freq_stage_weights=0.5,1,1,2`, best val/mIoU `0.514508` at epoch `50`, last `0.514508`, delta vs baseline mean `-0.002889`.
- `dformerv2_ms_freqcov_stage234_lam1_run01`: `lambda_freq=1.0`, `freq_stage_weights=0,0.5,1,2`, best val/mIoU `0.520060` at epoch `40`, last `0.505907`, delta vs baseline mean `+0.002663`.
- `dformerv2_ms_freqcov_stage3412_lam2_run01`: `lambda_freq=2.0`, `freq_stage_weights=0.5,1,1,2`, best val/mIoU `0.504229` at epoch `44`, last `0.502087`, delta vs baseline mean `-0.013168`.
- sweep mean best val/mIoU: `0.515697`.
- sweep population std best val/mIoU: `0.005143`.
- evidence: `miou_list/dformerv2_ms_freqcov_run01.md`, `miou_list/dformerv2_ms_freqcov_lam01_run01.md`, `miou_list/dformerv2_ms_freqcov_lam1_run01.md`, `miou_list/dformerv2_ms_freqcov_stage3412_run01.md`, `miou_list/dformerv2_ms_freqcov_stage3412_lam1_run01.md`, `miou_list/dformerv2_ms_freqcov_stage234_lam1_run01.md`, `miou_list/dformerv2_ms_freqcov_stage3412_lam2_run01.md`, and `miou_list/dformerv2_ms_freqcov_aggressive_sweep_summary.md`.
- conclusion: all planned freqcov jobs finished. The default weak run and stage234-lam1 run show positive single-run signals, but the aggressive sweep mean is below the clean baseline mean and the best freqcov run does not beat the clean baseline best single run. `lambda_freq=2.0` is clearly negative. Do not claim freqcov as a stable improvement yet.
- next step: if continuing freqcov, only test focused repeats of the two non-negative settings (`lambda=0.01 uniform` and `lambda=1.0 weights=0,0.5,1,2`). Otherwise pivot to a different training-only auxiliary target, such as conservative mask reconstruction, because simply increasing covariance weight did not produce a stable gain.

## 2026-05-07 dformerv2_ms_freqcov_run01

- model: `dformerv2_ms_freqcov`
- change: DFormerv2 mid-fusion baseline plus training-only c1-c4 multi-scale frequency covariance auxiliary loss.
- settings: `batch_size=2`, `max_epochs=50`, `lr=6e-5`, `num_workers=4`, `early_stop_patience=30`, `lambda_freq=0.01`, `freq_eta=1.0`, `freq_proj_dim=64`, `freq_kernel_size=3`, `freq_stage_weights=1,1,1,1`
- pretrained: `C:/Users/qintian/Desktop/qintian/dformer_work/checkpoints/pretrained/DFormerv2_Small_pretrained.pth`
- recorded validation epochs: `50`
- best val/mIoU: `0.520539` at recorded epoch `50`
- last val/mIoU: `0.520539`
- best val/loss: `1.005932` at recorded epoch `10`
- train/freqcov_loss_epoch: first `0.001377`, last `0.000002`
- comparison clean 10-run GatedFusion baseline mean best: `0.517397`
- delta vs clean 10-run baseline mean: `+0.003142`
- comparison clean 10-run baseline std: `0.004901`
- evidence: `miou_list/dformerv2_ms_freqcov_run01.md`
- conclusion: promising single-run result. The run beats the clean 10-run GatedFusion baseline mean, but the margin is smaller than one baseline standard deviation, so it cannot yet be claimed as a stable improvement.
- next step: run repeated seeds for `dformerv2_ms_freqcov`; at minimum run01-run05 before promoting it as a main candidate.

## 2026-05-07 dformerv2_mid_fusion_gate_baseline clean ten-run summary

- model: `dformerv2_mid_fusion`
- change: baseline DFormerv2 mid-fusion with original `GatedFusion`; no architecture change.
- included complete runs: `dformerv2_mid_fusion_gate_baseline_run01` through `run09`, plus `dformerv2_mid_fusion_gate_baseline_run10_retry`
- excluded run: `dformerv2_mid_fusion_gate_baseline_run10`, because it recorded only `43` validation epochs.
- mean best val/mIoU: `0.517397`
- population std best val/mIoU: `0.004901`
- mean last val/mIoU: `0.507137`
- evidence: `miou_list/dformerv2_mid_fusion_gate_baseline_summary_run01_09_run10_retry.md`
- conclusion: this clean ten-run statistic supersedes the earlier nine-complete-run plus partial-run baseline summary for future comparisons.

## 2026-05-07 dformerv2_ms_freqcov implementation

- model: `dformerv2_ms_freqcov`
- status: code implemented; waiting for formal training.
- purpose: add c1-c4 multi-scale feature-level frequency covariance auxiliary loss while keeping the DFormerv2 + DepthEncoder + GatedFusion + SimpleFPNDecoder inference structure unchanged.
- training loss: `L_total = L_seg + lambda_freq * L_freqcov`.
- default settings: `lambda_freq=0.01`, `freq_eta=1.0`, `freq_proj_dim=64`, `freq_kernel_size=3`, `freq_stage_weights=1,1,1,1`.
- code evidence: `src/models/freq_cov_loss.py`, `src/models/mid_fusion.py`, and `train.py`.
- result: no mIoU yet; do not cite as an experimental improvement until a completed run has TensorBoard evidence and a `miou_list` record.

## 2026-05-07 dformerv2_mid_fusion_gate_baseline run01-run10 summary

- model: `dformerv2_mid_fusion`
- change: baseline DFormerv2 mid-fusion with the original `GatedFusion`; no architecture change.
- settings: `batch_size=2`, `max_epochs=50`, `lr=6e-5`, `num_workers=4`, `early_stop_patience=30`
- pretrained: `C:/Users/qintian/Desktop/qintian/dformer_work/checkpoints/pretrained/DFormerv2_Small_pretrained.pth`
- complete runs: `9`
- partial runs: `1`
- dformerv2_mid_fusion_gate_baseline_run01: best val/mIoU `0.517621` at recorded epoch `49`, last `0.516805`, best val/loss `1.018484` at recorded epoch `13`
- dformerv2_mid_fusion_gate_baseline_run02: best val/mIoU `0.518642` at recorded epoch `49`, last `0.512856`, best val/loss `1.028006` at recorded epoch `11`
- dformerv2_mid_fusion_gate_baseline_run03: best val/mIoU `0.514432` at recorded epoch `40`, last `0.506427`, best val/loss `1.017887` at recorded epoch `10`
- dformerv2_mid_fusion_gate_baseline_run04: best val/mIoU `0.518596` at recorded epoch `38`, last `0.492355`, best val/loss `1.017909` at recorded epoch `9`
- dformerv2_mid_fusion_gate_baseline_run05: best val/mIoU `0.524425` at recorded epoch `50`, last `0.524425`, best val/loss `1.017244` at recorded epoch `9`
- dformerv2_mid_fusion_gate_baseline_run06: best val/mIoU `0.519506` at recorded epoch `31`, last `0.507040`, best val/loss `1.009728` at recorded epoch `9`
- dformerv2_mid_fusion_gate_baseline_run07: best val/mIoU `0.505699` at recorded epoch `45`, last `0.502685`, best val/loss `1.022507` at recorded epoch `10`
- dformerv2_mid_fusion_gate_baseline_run08: best val/mIoU `0.517555` at recorded epoch `49`, last `0.516317`, best val/loss `1.006927` at recorded epoch `11`
- dformerv2_mid_fusion_gate_baseline_run09: best val/mIoU `0.514622` at recorded epoch `49`, last `0.469588`, best val/loss `1.006296` at recorded epoch `9`
- dformerv2_mid_fusion_gate_baseline_run10: partial only, recorded `43` validation epochs; best recorded val/mIoU `0.514412` at recorded epoch `38`, last recorded `0.503475`, best val/loss `1.029115` at recorded epoch `10`
- complete-run mean best val/mIoU: `0.516789`
- complete-run population std best val/mIoU: `0.004795`
- complete-run mean last val/mIoU: `0.505389`
- partial-inclusive mean best val/mIoU: `0.516551`
- partial-inclusive population std best val/mIoU: `0.004604`
- partial-inclusive mean last val/mIoU: `0.505197`
- comparison previous baseline mean best: `0.513406`
- complete-run delta vs previous baseline mean: `+0.003383`
- evidence: `miou_list/dformerv2_mid_fusion_gate_baseline_run01.md`, `miou_list/dformerv2_mid_fusion_gate_baseline_run02.md`, `miou_list/dformerv2_mid_fusion_gate_baseline_run03.md`, `miou_list/dformerv2_mid_fusion_gate_baseline_run04.md`, `miou_list/dformerv2_mid_fusion_gate_baseline_run05.md`, `miou_list/dformerv2_mid_fusion_gate_baseline_run06.md`, `miou_list/dformerv2_mid_fusion_gate_baseline_run07.md`, `miou_list/dformerv2_mid_fusion_gate_baseline_run08.md`, `miou_list/dformerv2_mid_fusion_gate_baseline_run09.md`, `miou_list/dformerv2_mid_fusion_gate_baseline_run10.md`, and `miou_list/dformerv2_mid_fusion_gate_baseline_summary_run01_10.md`
- conclusion: the current GatedFusion baseline is stronger than the earlier baseline estimate. Use the nine complete runs as the valid repeated baseline; keep run10 as partial evidence only and do not count it as a completed repeated run.
- next step: if a clean ten-run statistic is needed, rerun only `dformerv2_mid_fusion_gate_baseline_run10` in a fresh directory or overwrite it intentionally after backing up the partial log.

## 2026-05-06 dformerv2_guided_depth_adapter_simple run01-run06 summary

- model: `dformerv2_guided_depth_adapter_simple`
- change: Part 1-only DFormer-guided depth adapter simple fusion. It reuses stage-wise guided depth adaptation and removes Full++ rectification and attention aggregation.
- settings: `batch_size=2`, `max_epochs=50`, `lr=6e-5`, `num_workers=4`, `early_stop_patience=30`
- pretrained: `C:/Users/qintian/Desktop/qintian/dformer_work/checkpoints/pretrained/DFormerv2_Small_pretrained.pth`
- completed runs: `6`
- dformerv2_guided_depth_adapter_simple_run01: best val/mIoU `0.511741` at recorded epoch `37`, last `0.490230`, best val/loss `1.017727` at recorded epoch `9`
- dformerv2_guided_depth_adapter_simple_run02: best val/mIoU `0.518872` at recorded epoch `48`, last `0.481400`, best val/loss `1.035822` at recorded epoch `10`
- dformerv2_guided_depth_adapter_simple_run03: best val/mIoU `0.513250` at recorded epoch `45`, last `0.502194`, best val/loss `1.024035` at recorded epoch `8`
- dformerv2_guided_depth_adapter_simple_run04: best val/mIoU `0.512987` at recorded epoch `50`, last `0.512987`, best val/loss `1.028134` at recorded epoch `9`
- dformerv2_guided_depth_adapter_simple_run05: best val/mIoU `0.510183` at recorded epoch `47`, last `0.484418`, best val/loss `1.029133` at recorded epoch `9`
- dformerv2_guided_depth_adapter_simple_run06: best val/mIoU `0.506865` at recorded epoch `46`, last `0.453570`, best val/loss `1.036385` at recorded epoch `12`
- mean best val/mIoU: `0.512316`
- population std best val/mIoU: `0.003626`
- mean last val/mIoU: `0.487466`
- comparison baseline mean best: `0.513406`
- mean delta vs baseline mean: `-0.001090`
- comparison Full++ mean best: `0.511379`
- mean delta vs Full++ mean best: `+0.000937`
- comparison DGC-AF++ mean best: `0.511418`
- mean delta vs DGC-AF++ mean best: `+0.000898`
- evidence: `miou_list/dformerv2_guided_depth_adapter_simple_run01.md`, `miou_list/dformerv2_guided_depth_adapter_simple_run02.md`, `miou_list/dformerv2_guided_depth_adapter_simple_run03.md`, `miou_list/dformerv2_guided_depth_adapter_simple_run04.md`, `miou_list/dformerv2_guided_depth_adapter_simple_run05.md`, `miou_list/dformerv2_guided_depth_adapter_simple_run06.md`, and `miou_list/dformerv2_guided_depth_adapter_simple_summary_run01_06.md`
- conclusion: near-baseline but negative repeated-run result. The Part 1-only branch is healthier than Full++ and DGC-AF++ on mean best val/mIoU, but six runs do not beat the repeated DFormerv2 mid-fusion baseline mean.
- next step: do not claim this branch as a stable paper improvement. Keep it as evidence that early DFormer-guided depth adaptation is less harmful than the full chain, but the current adapter/residual design still needs either stronger stability control or a more conservative ablation before promotion.

## 2026-05-06 dformerv2_guided_depth_adapter_simple run01-run03 summary

- model: `dformerv2_guided_depth_adapter_simple`
- change: Part 1-only DFormer-guided depth adapter simple fusion. It reuses stage-wise guided depth adaptation and removes Full++ rectification and attention aggregation.
- settings: `batch_size=2`, `max_epochs=50`, `lr=6e-5`, `num_workers=4`, `early_stop_patience=30`
- pretrained: `C:/Users/qintian/Desktop/qintian/dformer_work/checkpoints/pretrained/DFormerv2_Small_pretrained.pth`
- completed runs: `3`
- dformerv2_guided_depth_adapter_simple_run01: best val/mIoU `0.511741` at recorded epoch `37`, last `0.490230`, best val/loss `1.017727` at recorded epoch `9`
- dformerv2_guided_depth_adapter_simple_run02: best val/mIoU `0.518872` at recorded epoch `48`, last `0.481400`, best val/loss `1.035822` at recorded epoch `10`
- dformerv2_guided_depth_adapter_simple_run03: best val/mIoU `0.513250` at recorded epoch `45`, last `0.502194`, best val/loss `1.024035` at recorded epoch `8`
- mean best val/mIoU: `0.514621`
- population std best val/mIoU: `0.003069`
- mean last val/mIoU: `0.491274`
- comparison baseline mean best: `0.513406`
- mean delta vs baseline mean: `+0.001215`
- comparison Full++ mean best: `0.511379`
- mean delta vs Full++ mean best: `+0.003242`
- comparison DGC-AF++ mean best: `0.511418`
- mean delta vs DGC-AF++ mean best: `+0.003203`
- evidence: `miou_list/dformerv2_guided_depth_adapter_simple_run01.md`, `miou_list/dformerv2_guided_depth_adapter_simple_run02.md`, `miou_list/dformerv2_guided_depth_adapter_simple_run03.md`, and `miou_list/dformerv2_guided_depth_adapter_simple_summary_run01_03.md`
- conclusion: promising three-run result. The Part 1-only simple branch is above the repeated DFormerv2 mid-fusion baseline mean and clearly above Full++ / DGC-AF++ means, suggesting the guided depth adapter is the useful part while later rectification/attention aggregation likely adds instability.
- next step: do not finalize the paper claim from only three runs. Extend to five repeated runs before promoting this as the main candidate; if the five-run mean remains above `0.513406`, keep Part 1-only as the primary new branch and treat Full++ as a negative complexity ablation.

## 2026-05-06 dformerv2_guided_depth_comp_fusion run01-run05 summary

- model: `dformerv2_guided_depth_comp_fusion`
- change: DFormer-guided depth rectification and complementary fusion with stage-wise guided depth adaptation, asymmetric complementary rectification, and attention complementary aggregation.
- settings: `batch_size=2`, `max_epochs=50`, `lr=6e-5`, `num_workers=4`, `early_stop_patience=30`
- pretrained: `C:/Users/qintian/Desktop/qintian/dformer_work/checkpoints/pretrained/DFormerv2_Small_pretrained.pth`
- completed runs: `5`
- dformerv2_guided_depth_comp_fusion_run01: best val/mIoU `0.508875` at recorded epoch `46`, last `0.481020`, best val/loss `1.052861` at recorded epoch `8`
- dformerv2_guided_depth_comp_fusion_run02: best val/mIoU `0.512506` at recorded epoch `39`, last `0.507708`, best val/loss `1.024278` at recorded epoch `9`
- dformerv2_guided_depth_comp_fusion_run03: best val/mIoU `0.508686` at recorded epoch `49`, last `0.483381`, best val/loss `1.040731` at recorded epoch `7`
- dformerv2_guided_depth_comp_fusion_run04: best val/mIoU `0.510958` at recorded epoch `45`, last `0.505383`, best val/loss `1.007499` at recorded epoch `10`
- dformerv2_guided_depth_comp_fusion_run05: best val/mIoU `0.515870` at recorded epoch `45`, last `0.507223`, best val/loss `1.059600` at recorded epoch `7`
- mean best val/mIoU: `0.511379`
- population std best val/mIoU: `0.002651`
- mean last val/mIoU: `0.496943`
- comparison baseline mean best: `0.513406`
- mean delta vs baseline mean: `-0.002027`
- comparison DGC-AF++ mean best: `0.511418`
- mean delta vs DGC-AF++ mean best: `-0.000039`
- evidence: `miou_list/dformerv2_guided_depth_comp_fusion_run01.md`, `miou_list/dformerv2_guided_depth_comp_fusion_run02.md`, `miou_list/dformerv2_guided_depth_comp_fusion_run03.md`, `miou_list/dformerv2_guided_depth_comp_fusion_run04.md`, `miou_list/dformerv2_guided_depth_comp_fusion_run05.md`, and `miou_list/dformerv2_guided_depth_comp_fusion_summary_run01_05.md`
- conclusion: negative repeated-run result. The new three-stage guided depth fusion has one strong run, but the five-run mean remains below the repeated DFormerv2 mid-fusion baseline and is essentially tied with DGC-AF++.
- next step: do not claim this branch as a paper improvement. Keep it as a meaningful negative ablation showing that stage-wise guided depth adaptation alone does not solve the instability; next direction should reduce complexity and isolate Part 1 / Part 2 / Part 3 separately.

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

