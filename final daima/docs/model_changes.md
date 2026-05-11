# Model Changes

## 2026-05-11 CGPC Loss Minimal Implementation

- Added `src/losses/cgpc_loss.py` with `CGPCLoss`.
- Added `CGPCLoss` export in `src/losses/__init__.py`.
- CGPC is a class-label-guided prototype contrast loss over fused features, not modality-to-modality alignment.
- First implementation supports single-stage fused features only: `c2`, `c3`, or `c4`; default planned stage is `c3`.
- Prototype construction is batch-local, class-label guided, sampled per class, and detached by default.
- Did not add projection heads, memory banks, hard negative mining, full pixel-pairwise contrast, c3/c4 averaging, depth-to-primary contrast, or tri-prototype contrast.
- Updated only `LitDFormerV2MidFusion` in `src/models/mid_fusion.py`: when `cgpc_weight=0`, `training_step()` immediately returns `super().training_step(...)`, preserving existing CE/DGBF paths; when `cgpc_weight>0`, it uses `self.model.extract_features()` to get fused features, decodes logits, computes segmentation loss, then adds `cgpc_weight * cgpc_loss`.
- Added CLI args in `train.py`: `--cgpc_weight`, `--cgpc_temperature`, `--cgpc_stage`, `--cgpc_min_pixels_per_class`, and `--cgpc_max_pixels_per_class`.
- Did not modify `BaseLitSeg`, `DFormerV2MidFusionSegmentor.forward()`, `GatedFusion`, DFormerV2 attention, PrimKD, teacher models, DGBF loss, decoder, or data module.
- Verification: `train.py --help` lists CGPC args; `CGPCLoss` imports and forward/backward passes on synthetic tensors; `dformerv2_mid_fusion` builds with `cgpc_weight=0.0` and `cgpc_weight=0.01`; `compileall` passed.

## 2026-05-11 DGBF Loss Minimal Implementation

- Added `src/losses/dgbf_loss.py` with `DGBFLoss`.
- Added `DGBFLoss` export in `src/losses/__init__.py`.
- Extended `BaseLitSeg` with `loss_type="dgbf"` while keeping `loss_type="ce"` behavior unchanged.
- DGBF computes CE per pixel, a Sobel depth edge map, a valid semantic boundary map, optional focal modulation, and a final weighted CE loss.
- Supported DGBF modes: `depth_semantic`, `semantic_only`, `depth_only`, `focal_only`, and `none`.
- Added DGBF training logs: `train/dgbf_boundary_mean`, `train/dgbf_boundary_max`, `train/dgbf_weight_mean`, and `train/dgbf_weight_max`.
- Added CLI args in `train.py`: `--dgbf_alpha`, `--dgbf_gamma`, and `--dgbf_mode`; `--loss_type` now supports `dgbf`.
- Updated only `LitDFormerV2MidFusion.__init__` to pass DGBF parameters into `BaseLitSeg`; `DFormerV2MidFusionSegmentor.forward()` remains unchanged.
- Did not implement GSFR, FFT, GSA prior supervision, DFormerV2 attention changes, dataset changes, teacher changes, PMAD changes, or model-structure changes.
- Verification: `train.py --help` lists DGBF args; `DGBFLoss` imports and forward/backward passes on synthetic tensors; `dgbf_mode=none` exactly matches CE in a synthetic check; `dformerv2_mid_fusion` builds with both `loss_type=ce` and `loss_type=dgbf`.

## 2026-05-10 Geometry-Primary Teacher Update

- Replaced the PMAD teacher candidate with a geometry-primary DFormerV2 teacher.
- Renamed the active teacher registry entry to `dformerv2_geometry_primary_teacher`.
- Updated `src/models/teacher_model.py`: `DFormerV2GeometryPrimaryTeacherSegmentor` now uses `DFormerv2_S(rgb, depth) + SimpleFPNDecoder` and no longer constructs constant zero depth.
- Updated `src/models/primkd_lit.py`: the frozen teacher forward now receives real batch depth through `self.teacher(rgb, depth)`.
- Updated `train.py` imports, `MODEL_REGISTRY`, and `build_model()` branch for the new teacher name.
- Motivation: the previous `dformerv2_rgb_teacher_constdepth_run01` failed because zero depth removed DFormerV2's depth geometry prior; this update preserves DFormerV2 geometry self-attention while still excluding the extra `DepthEncoder + GatedFusion` branch.
- Did not change `dformerv2_mid_fusion`, `BaseLitSeg`, DFormerV2 attention internals, decoder, data module, feature KD, reliability, or MixPrompt.
- The old `dformerv2_rgb_teacher_constdepth_run01` result remains a failed teacher-sanity record and should not be merged with future geometry-primary teacher results.

## 2026-05-10 PMAD Phase 0-1 Minimal Implementation

- Added `src/models/teacher_model.py` with `DFormerV2RGBTeacherSegmentor` and `LitDFormerV2RGBTeacher`.
- Registered new model `dformerv2_rgb_teacher_constdepth` in `train.py`.
- The RGB teacher uses `DFormerv2_S + SimpleFPNDecoder` and constructs constant zero depth inside `forward`, so real batch depth is not used.
- Added `src/models/primkd_lit.py` with `LitDFormerV2PrimKD`.
- Registered new model `dformerv2_primkd_logit_only` in `train.py`.
- PMAD Phase 1 is logit-only: `CE(student_logits, label) + kd_weight * KL(student/T, teacher/T) * T^2`, with `ignore_index=255` pixels excluded from KD.
- Added CLI args `--teacher_ckpt`, `--kd_weight`, `--kd_temperature`, and `--save_student_only`.
- Added optional checkpoint export support in `DirectStateDictCheckpoint`: when `--save_student_only` is set and the LightningModule exposes `export_state_dict()`, only the student model state is saved.
- Did not change `dformerv2_mid_fusion`, `BaseLitSeg`, DFormerV2 attention, dataset/dataloader, decoder, feature KD, reliability gating, or MixPrompt.
- Verification: `compileall` passed; `train.py --help` lists both new model names and new args under the `qintian-rgbd` environment; 1-epoch smoke tests for `dformerv2_rgb_teacher_constdepth` and `dformerv2_primkd_logit_only` produced checkpoints. Smoke mIoU values are not formal results and must not be cited.

## 2026-05-09 dformerv2_context_decoder

- Added `PPMContextBlock` and `ContextFPNDecoder` in `src/models/decoder.py`.
- Added `DFormerV2ContextDecoderSegmentor` and `LitDFormerV2ContextDecoder` in `src/models/mid_fusion.py`.
- Added model entry `dformerv2_context_decoder` in `train.py`.
- The new model keeps the same encoder and fusion path as `dformerv2_mid_fusion`: `DFormerV2_S + DepthEncoder + GatedFusion`.
- The only structural change is decoder-side context refinement: fused `c4` is passed through a lightweight PPM-style context block before the original FPN top-down path.
- `SimpleFPNDecoder` remains unchanged and is still used by the clean baseline.
- This is not a fusion replacement, not FFT enhancement, and not an auxiliary loss.
- Loss, metrics, checkpoint monitor, dataset, dataloader, optimizer, and validation logic are unchanged.
- First planned setting: `pool_scales=(1,2,3,6)`, `alpha_init=0.1`, `branch_channels=max(C//4,64)`, `loss_type=ce`.
- Status: code implemented; waiting for smoke test and formal training.

## 2026-05-09 CE + Dice Loss Recipe Support

- Added `src/losses/dice_loss.py` with multiclass `DiceLoss` and `CEDiceLoss`.
- Added `src/losses/__init__.py` to export the loss classes.
- Added `loss_type` and `dice_weight` support to `BaseLitSeg`.
- Default `loss_type=ce` keeps the previous baseline behavior: `CrossEntropyLoss(ignore_index=255)`.
- New `loss_type=ce_dice` uses `CE + dice_weight * DiceLoss`, with default `dice_weight=0.5`.
- Training loss is selectable through `train.py`; validation loss remains fixed CE for comparability with historical logs.
- `val/mIoU`, checkpoint monitor, model forward paths, decoder, GatedFusion, FFT modules, dataset, and dataloader are unchanged.
- This is a logits-level segmentation loss recipe, not a restoration of archived feature-level auxiliary losses from `feiqi/losses/`.

## 2026-05-09 Archive Deprecated Auxiliary Loss Modules

- Moved `src/models/freq_cov_loss.py` → `feiqi/losses/freq_cov_loss.py`
- Moved `src/models/mask_reconstruction_loss.py` → `feiqi/losses/mask_reconstruction_loss.py`
- Moved `src/models/contrastive_loss.py` → `feiqi/losses/contrastive_loss.py`
- Removed from `train.py`: MODEL_REGISTRY entries `dformerv2_ms_freqcov`, `dformerv2_feat_maskrec_c34`, `dformerv2_cm_infonce`; their argparse arguments; and their `build_model` branches.
- Removed from `src/models/mid_fusion.py`: imports of `CrossModalInfoNCELoss`, `MultiScaleFrequencyCovarianceLoss`, `FeatureMaskReconstructionLoss`; and classes `LitDFormerV2MSFreqCov`, `LitDFormerV2FeatMaskRecC34`, `LitDFormerV2CMInfoNCE`.
- Active MODEL_REGISTRY now: `early`, `mid_fusion`, `dformerv2_mid_fusion`, `dformerv2_depth_fft_select`, `dformerv2_fft_freq_enhance`, `dformerv2_fft_hilo_enhance`.
- Kept all experiment records in `miou_list/` and `docs/experiment_log.md` unchanged.
- Archive README at `feiqi/losses/README.md`.

## 2026-05-09 DFormerv2 FFT HiLo Enhancement

- Added experiment entry `dformerv2_fft_hilo_enhance`.
- Added `src/models/fft_hilo_enhance.py` with `FFTHiLoEnhance`.
- Added `DFormerV2FFTHiLoEnhanceSegmentor` and `LitDFormerV2FFTHiLoEnhance` in `src/models/mid_fusion.py`.
- The module is an inference-path FFT low/high dual-band residual enhancement inserted after encoder feature extraction and before `GatedFusion`.
- Both primary/DFormerV2 features and aligned depth features are enhanced at all four stages by default.
- Formula: `x_low = IFFT2(M_low * FFT2(x)).real`, `x_high = x - x_low`, `out = x + alpha_low * gate_low([x,x_low,x_high]) * clean_low(x_low) + alpha_high * gate_high([x,x_low,x_high]) * clean_high(x_high)`.
- Alpha values are bounded: `alpha = alpha_max * sigmoid(raw_alpha)`. Raw alpha is initialized by inverse sigmoid, not by directly assigning the desired alpha value.
- First planned configuration: `cutoff_ratio=0.25`, `alpha_high_init=0.10`, `alpha_low_init=0.03`, `alpha_max=0.5`, `hilo_stage_weights=1,1,1,1`.
- `gate_low` and `gate_high` are zero-initialized so initial gates are `0.5`; `clean_low` and `clean_high` use near-zero normal initialization with `std=1e-4`.
- This branch adds no auxiliary loss and does not stack with freqcov, maskrec, or InfoNCE.
- Did not modify DFormerV2, `DepthEncoder`, `GatedFusion`, `SimpleFPNDecoder`, `BaseLitSeg`, validation, inference logic beyond this model path, dataset, or dataloader.
- Kept `dformerv2_mid_fusion`, `dformerv2_fft_freq_enhance`, `dformerv2_depth_fft_select`, `dformerv2_ms_freqcov`, `dformerv2_feat_maskrec_c34`, and `dformerv2_cm_infonce` unchanged.
- Status: code implemented; waiting for formal training.

## 2026-05-08 DFormerv2 Cross-Modal InfoNCE Auxiliary Loss

- Added experiment entry `dformerv2_cm_infonce`.
- Added `src/models/contrastive_loss.py` with `CrossModalInfoNCELoss`.
- Added `LitDFormerV2CMInfoNCE` in `src/models/mid_fusion.py`.
- The experiment keeps the inference path unchanged: `DFormerV2_S + DepthEncoder + GatedFusion + SimpleFPNDecoder`.
- Training uses a one-way depth-to-primary contrastive auxiliary loss on aligned c3/c4 features by default.
- Key/query design: `k = primary_proj(P.detach())`, `q = depth_proj(D)`.
- This protects the DFormerV2 primary encoder from contrast loss gradients while still allowing the primary projection head, depth projection head, and DepthEncoder to learn.
- Primary and depth features use separate per-stage 1x1 projection heads because their channels are not assumed to be shared.
- Positives are same-batch, same-spatial-location primary/depth pairs sampled with a shared spatial index set; negatives are all other sampled keys in the `[B*S, B*S]` similarity matrix.
- Defaults: `lambda_contrast=0.005`, `contrast_temperature=0.1`, `contrast_proj_dim=64`, `contrast_sample_points=256`, `contrast_stage_weights=0,0,1,1`.
- Did not modify DFormerV2, `DepthEncoder`, `GatedFusion`, `SimpleFPNDecoder`, `BaseLitSeg`, validation, inference, dataset, or dataloader.
- Status: code implemented; waiting for formal training.
- Result note for `dformerv2_cm_infonce_c34_lam005_t01_s256_run01`: best val/mIoU `0.514461`, below clean 10-run baseline mean `0.517397` by `0.002936`; negative single-run result.

## 2026-05-08 DFormerv2 Depth FFT Frequency Selection

- Added experiment entry `dformerv2_depth_fft_select`.
- Added `src/models/depth_fft_select.py` with `DepthFrequencySelect` and `DepthEncoderFFTSelect`.
- Added `DFormerV2DepthFFTSelectSegmentor` and `LitDFormerV2DepthFFTSelect` in `src/models/mid_fusion.py`.
- The new depth encoder keeps the original ResNet-18 depth branch structure and inserts internal FFT low/high frequency selection after c2, c3, and c4. c1 is skipped.
- The enhanced depth feature is appended as the stage output and also passed into the next ResNet stage, making this an internal DepthEncoder enhancement rather than an encoder-output adapter.
- `DepthFrequencySelect` performs spatial FFT decomposition with `torch.fft.fft2`, `torch.fft.fftshift`, a hard circular low-frequency mask, `torch.fft.ifftshift`, and `torch.fft.ifft2(...).real`.
- The selection formula is `D_out = D + (w_low - 1) * D_low + (w_high - 1) * D_high`, where `w_low = sigmoid(DWConv(D)) * 2` and `w_high = sigmoid(DWConv(D)) * 2`.
- The low/high depthwise selection convolutions are zero-initialized, so the initial output is exactly identity: `w_low = w_high = 1` and `D_out = D`.
- First planned configuration: `cutoff_ratio=0.30`, c2/c3/c4 enabled, segmentation loss only.
- Did not modify DFormerV2, `GatedFusion`, `SimpleFPNDecoder`, `BaseLitSeg`, loss, training step, validation step, dataset, or dataloader.
- Kept `dformerv2_mid_fusion`, `dformerv2_fft_freq_enhance`, `dformerv2_ms_freqcov`, and `dformerv2_feat_maskrec_c34` unchanged.
- Status: code implemented; waiting for formal training.

## 2026-05-08 DFormerv2 FFT-Based Frequency Enhancement

- Added experiment entry `dformerv2_fft_freq_enhance`.
- Added `src/models/freq_enhance.py` with `FrequencyClean` and `FFTFrequencyEnhance`.
- Added `DFormerV2FFTFreqEnhanceSegmentor` and `LitDFormerV2FFTFreqEnhance` in `src/models/mid_fusion.py`.
- The module performs true spatial Fourier decomposition with `torch.fft.fft2`, `torch.fft.fftshift`, a hard circular low-frequency mask in the Fourier plane, `torch.fft.ifftshift`, and `torch.fft.ifft2(...).real`.
- Formula: `X = FFT2(F)`, `F_low = IFFT2(M_low * X).real`, `F_high = F - F_low`, `F_out = F + gamma * sigmoid(Conv([F, F_high])) * Clean(F_high)`.
- `FrequencyClean` is a single near-zero-initialized `Conv2d(channels, channels, 1, bias=False)` and only acts on the FFT-derived high-frequency component.
- The enhancement is inserted before `GatedFusion` for both primary/DFormerV2 features and aligned depth features.
- The segmentation path is `DFormerV2_S + DepthEncoder -> FFTFrequencyEnhance per modality and stage -> GatedFusion -> SimpleFPNDecoder`.
- The first configuration is `cutoff_ratio=0.25`, `gamma_init=0.05`, four stages enabled through four `ModuleList` modules, and segmentation loss only.
- This is not an auxiliary loss, channel-only attention, SE block, AvgPool high-pass approximation, FreqFusion module import, FADC AdaptiveDilatedConv, CARAFE, new decoder, or new backbone.
- Kept `dformerv2_mid_fusion`, `dformerv2_ms_freqcov`, and `dformerv2_feat_maskrec_c34` unchanged.
- Actual parameter count: baseline `dformerv2_mid_fusion` `41,046,082`; `dformerv2_fft_freq_enhance` `43,136,970`; added parameters `2,090,888`.
- Verification passed: `compileall`, `train.py --help`, standalone FFT module backward test, and `224x224` model smoke test.

## 2026-05-08 DFormerv2 Feature-Level Mask Reconstruction Auxiliary Loss

- Added experiment entry `dformerv2_feat_maskrec_c34`.
- Added `src/models/mask_reconstruction_loss.py` with `FeatureMaskReconstructionLoss`.
- Added `LitDFormerV2FeatMaskRecC34` in `src/models/mid_fusion.py`.
- The experiment reuses `DFormerV2MidFusionSegmentor.extract_features(rgb, depth)` to obtain DFormerv2 primary c1-c4 features, aligned DepthEncoder c1-c4 features, and original GatedFusion fused features.
- The segmentation inference path remains unchanged: `DFormerV2_S + DepthEncoder + GatedFusion + SimpleFPNDecoder`.
- The auxiliary loss is training-only and creates reconstruction heads for c1-c4.
- `maskrec_stage_weights` now explicitly controls which stages affect the total loss; weight `0` stages are skipped and logged as zero.
- Example stage weights: `0,0,1,1` for c3+c4, `0,0,0,1` for c4 only, `0,0.5,1,1` for weak c2 plus c3+c4, and `1,1,1,1` for all stages.
- Primary-to-depth direction: mask the depth feature, concatenate `[primary_feat, masked_depth_feat]`, predict the original depth feature, and compute loss only on masked depth locations.
- Depth-to-primary direction: mask the primary/DFormerv2 feature, concatenate `[depth_feat, masked_primary_feat]`, predict the original primary feature, and compute loss only on masked primary locations.
- Reconstruction targets are detached; source features and masked target inputs are not detached, so the auxiliary loss can update the encoders and reconstruction heads.
- Training loss: `L_total = L_seg + lambda_mask * sum_i w_i * (depth_rec_i + maskrec_alpha * primary_rec_i) / sum_i w_i`.
- Defaults: `lambda_mask=0.01`, `mask_ratio_depth=0.30`, `mask_ratio_primary=0.15`, `maskrec_alpha=0.5`, `maskrec_loss_type=smooth_l1`; formal experiment commands must explicitly pass `--maskrec_stage_weights`.
- Kept `dformerv2_mid_fusion` and `dformerv2_ms_freqcov` unchanged.

## 2026-05-07 DFormerv2 Multi-Scale Frequency Covariance Auxiliary Loss

- Added experiment entry `dformerv2_ms_freqcov`.
- Added `src/models/freq_cov_loss.py` with `MultiScaleFrequencyCovarianceLoss`.
- Added `DFormerV2MidFusionSegmentor.extract_features(rgb, depth)` so training can access DFormerv2 c1-c4 features, aligned DepthEncoder c1-c4 features, and fused GatedFusion features without changing baseline inference behavior.
- Added `LitDFormerV2MSFreqCov` in `src/models/mid_fusion.py`.
- Training loss: `L_total = L_seg + lambda_freq * sum_i w_i * (L_high_cov_i + eta * L_low_cov_i) / sum_i w_i`.
- Frequency split is training-only and uses `avg_pool2d` smoothing: `low = avg_pool2d(x)`, `high = x - low`.
- Each stage has separate 1x1 projection heads for DFormerv2 features and DepthEncoder features before covariance calculation.
- Covariance calculation centers projected BCHW features after flattening to `[B*H*W, C]`, then computes `cov = z.T @ z / (N - 1)`.
- The segmentation inference path remains `DFormerV2_S + DepthEncoder + GatedFusion + SimpleFPNDecoder`.
- Kept `dformerv2_mid_fusion` unchanged as the baseline registry entry.
- Did not restore DGC-AF++, Full++, guided adapter simple, or archived modules to the main training path.

## 2026-05-06 Archive Deprecated Fusion Modules Outside Main Path

- Created `feiqi/` as the archive folder for deprecated fusion experiment modules.
- Moved `src/models/feiqi_fusion_blocks.py` to `feiqi/feiqi_fusion_blocks.py` and changed its relative imports to absolute `src.models.*` imports for archive readability and compilation.
- Moved the guided-depth Full++ and Part 1-only simple ablation classes out of `src/models/mid_fusion.py` into `feiqi/feiqi_guided_depth_blocks.py`.
- Added `feiqi/README.md` to mark the folder as archived reference code only.
- Restored `src/models/mid_fusion.py` to the baseline main path classes only: `GatedFusion`, `MidFusionSegmentor`, `LitMidFusion`, `DFormerV2MidFusionSegmentor`, and `LitDFormerV2MidFusion`.
- Restored active `train.py` model entries to `early`, `mid_fusion`, and `dformerv2_mid_fusion` only.
- Did not delete experiment records, `miou_list`, docs, or checkpoints.

## 2026-05-06 DFormerv2 Guided Depth Adapter Simple Fusion

- Added active model entry `dformerv2_guided_depth_adapter_simple`.
- Added `DFormerGuidedDepthAdapterSimpleFusion`, `DFormerV2GuidedDepthAdapterSimpleSegmentor`, and `LitDFormerV2GuidedDepthAdapterSimple` in `src/models/mid_fusion.py`.
- Reused the existing `DFormerGuidedDepthAdapter` and `GuidedDepthEncoder` from the Full++ branch.
- Design: keep only Part 1 guided depth adaptation, then use a minimal primary-preserving residual fusion: `primary_feat + gamma * SimpleAdapter([guided_depth_feat, abs(primary_feat - guided_depth_feat)])`.
- Removed Full++ components from this ablation path: asymmetric complementary rectification, channel attention, spatial attention, support/conflict logic, relation selection, CSG, GRM, and ARD.
- Did not use `GatedFusion`, `g * primary + (1 - g) * depth`, full `QK^T` attention, token flattening, `HW x HW` attention, `grid_sample`, flow warp, deformable attention, or archived experimental modules.
- Did not modify DFormerv2 encoder, the original `DepthEncoder`, `SimpleFPNDecoder`, loss, optimizer, or scheduler.
- Active model entries after this change: `early`, `mid_fusion`, `dformerv2_mid_fusion`, `dformerv2_guided_depth_comp_fusion`, `dformerv2_guided_depth_adapter_simple`.
- Parameter size: original `dformerv2_mid_fusion` fusion blocks `4,182,720`; Full++ total model `38,565,130`; simple guided depth adapters `709,504`; simple fusion blocks `136,640`; simple total model `37,709,506`.
- Verification: `py_compile` passed for `src/models/mid_fusion.py` and `train.py`; CPU smoke test output shape was `(1, 40, 64, 64)`.

## 2026-05-06 DFormerv2 Guided Depth Rectification and Complementary Fusion

- Added active model entry `dformerv2_guided_depth_comp_fusion`.
- Added `DFormerGuidedDepthAdapter`, `GuidedDepthEncoder`, `AsymmetricComplementaryRectification`, `AttentionComplementaryAggregation`, `DFormerGuidedDepthCompFusionBlock`, `DFormerV2GuidedDepthCompFusionSegmentor`, and `LitDFormerV2GuidedDepthCompFusion` in `src/models/mid_fusion.py`.
- Design: DFormerv2 primary features guide DepthEncoder stage outputs first, then a primary-preserving asymmetric rectification and lightweight attention complementary aggregation produce fused stage features.
- `GuidedDepthEncoder` is a wrapper around the original `DepthEncoder`; it does not modify the original `DepthEncoder` class or feed adapted depth features back into later ResNet stages.
- Fusion output is strictly primary-preserving: `primary_feat + small complementary residual`.
- Residual scales are small-initialized: adapter `beta=1e-3`, depth rectification `alpha_d=1e-3`, primary rectification `alpha_p=1e-4`, aggregation `gamma=1e-3`.
- Did not use `GatedFusion`, `g * primary + (1 - g) * depth`, full `QK^T` attention, token flattening, `HW x HW` attention, `grid_sample`, flow warp, deformable attention, or archived DGC-AF/CSG/GRM-ARD modules.
- Did not modify DFormerv2 encoder, the original `DepthEncoder`, `SimpleFPNDecoder`, loss, optimizer, or scheduler.
- Active model entries after this change: `early`, `mid_fusion`, `dformerv2_mid_fusion`, `dformerv2_guided_depth_comp_fusion`.
- Parameter size: original `dformerv2_mid_fusion` fusion blocks `4,182,720`; new guided depth adapters `709,504`; new rectification blocks `716,032`; new aggregation blocks `276,232`; new total model `38,565,130`.
- Verification: `py_compile` passed for `src/models/mid_fusion.py` and `train.py`; CPU smoke test output shape was `(1, 40, 64, 64)`.

## 2026-05-06 Main Training Path Cleanup After DGC-AF Ablations

- Cleaned `src/models/mid_fusion.py` so it only contains the active ResNet mid-fusion baseline and DFormerv2 mid-fusion baseline classes.
- Moved PG-SparseComp, DGC-AF Full, DGC-AF++, and DGC-AF++ CSG modules and their historical segmentor/Lightning wrappers into `src/models/fusion_blocks.py` as archived experimental code.
- No GRM-ARD implementation was present in the current source snapshot, so no GRM-ARD class was moved.
- Removed PG-SparseComp, DGC-AF Full, DGC-AF++, and DGC-AF++ CSG imports and registry entries from `train.py`.
- Active model entries after cleanup: `early`, `mid_fusion`, `dformerv2_mid_fusion`.
- `dformerv2_mid_fusion` remains the only active model that accepts `--dformerv2_pretrained`.
- Kept experiment records, `miou_list`, and checkpoints untouched.
- Verification: `py_compile` passed for `src/models/mid_fusion.py`, `src/models/fusion_blocks.py`, and `train.py`.

## 2026-05-04 DFormerv2 DGC-AF Plus CSG

- Added `--model dformerv2_dgc_af_plus_csg`.
- Added `DFormerGuidedCyclicAttentionFusionPlusCSG`, `DFormerV2DGCAFPlusCSGSegmentor`, and `LitDFormerV2DGCAFPlusCSG` in `src/models/mid_fusion.py`.
- Kept `dformerv2_dgc_af_plus` unchanged as the current main self-designed branch.
- Motivation: use high-level DFormerv2 c4 semantics to guide each stage's DGC-AF++ residual generation, without reintroducing GRM-ARD.
- CSG generator: `semantic_context = AdaptiveAvgPool2d(1) -> Conv1x1(c4,c4) -> ReLU -> Conv1x1(c4,c4) -> ReLU`.
- Per-stage semantic gates: `semantic_gate_i = Sigmoid(Conv1x1(c4_channels, stage_channels)(global_context))`.
- Fusion injection: `p_guided = primary_feat * (1 + semantic_alpha * (2 * semantic_gate - 1))`, with `semantic_alpha=1e-3`.
- `p_guided` is used only inside DGC-AF++ residual generation; final output remains `primary_feat + residual`.
- Did not use `GatedFusion`, `g * primary + (1 - g) * depth`, full `QK^T` attention, token flattening, `HW x HW` attention, `grid_sample`, flow warp, deformable attention, GRM, ARD, or budget gate.
- Did not modify DFormerv2 encoder, DepthEncoder, SimpleFPNDecoder, loss, optimizer, or scheduler.
- Parameter size: fusion blocks `2,794,752` params; semantic guidance `1,015,808` params; fusion plus semantic guidance `3,810,560` params; total model `40,673,922` params.
- Comparison: CSG fusion plus semantic guidance remains below original `dformerv2_mid_fusion` GatedFusion fusion params `4,182,720`.
- Active model entries after this change: `early`, `mid_fusion`, `dformerv2_mid_fusion`, `dformerv2_pg_sparse_comp_fusion`, `dformerv2_dgc_af_full`, `dformerv2_dgc_af_plus`, `dformerv2_dgc_af_plus_csg`.
- Verification: `py_compile` passed for `src/models/mid_fusion.py` and `train.py`.
- Result note for `dformerv2_dgc_af_plus_csg_run02`: best val/mIoU `0.506402`, below baseline mean `0.513406` by `0.007004` and below DGC-AF++ run01 by `0.007182`; do not promote this branch.

## 2026-05-04 DFormerv2 DGC-AF Plus GRM-ARD

- Added `--model dformerv2_dgc_af_plus_grm_ard`.
- Added `DFormerGuidedCyclicAttentionFusionPlusGRMARD`, `DFormerV2DGCAFPlusGRMARDSegmentor`, and `LitDFormerV2DGCAFPlusGRMARD` in `src/models/mid_fusion.py`.
- Source-code ideas checked before implementation:
  - `C:/Users/qintian/Desktop/qintian/ref_codes/CAFuser/cafuser/modeling/modality_fusion/querry_guided_addition.py`
  - `C:/Users/qintian/Desktop/qintian/ref_codes/CAFuser/cafuser/modeling/modality_fusion/querry_guided_pca.py`
  - `C:/Users/qintian/Desktop/qintian/ref_codes/CAFuser/cafuser/modeling/modality_fusion/prallel_cross_attention.py`
  - `C:/Users/qintian/Desktop/qintian/ref_codes/CAFuser/cafuser/modeling/feature_adapter/mlp_learnable_ratio.py`
  - `C:/Users/qintian/Desktop/qintian/ref_codes/CAFuser/cafuser/modeling/condition_classifier/transformer.py`
  - `C:/Users/qintian/Desktop/qintian/ref_codes/Mul-VMamba/semseg/models/modules/ffm.py`
  - `C:/Users/qintian/Desktop/qintian/ref_codes/Mul-VMamba/semseg/models/modules/crossatt.py`
  - `C:/Users/qintian/Desktop/qintian/ref_codes/Mul-VMamba/semseg/models/modules/mspa.py`
  - `C:/Users/qintian/Desktop/qintian/ref_codes/CMFormer/model/utils/MS_CAC.py`
  - `C:/Users/qintian/Desktop/qintian/ref_codes/CMFormer/model/utils/GFA.py`
- Kept DFormerv2 primary features as the main path and used auxiliary depth only through controlled residual compensation.
- Did not use `GatedFusion`, `g * primary + (1 - g) * depth`, full `QK^T` attention, token flattening, `HW x HW` attention, `grid_sample`, flow warp, deformable attention, stochastic hard drop, Bernoulli masks, hard threshold, top-k, or straight-through estimators.
- New over DGC-AF++: depthwise multi-scale relation context, primary-guided residual anchor, residual mixture gate, residual budget gate, and soft adaptive bad residual suppression.
- Multi-scale relation context: `d_rel = d_rel + eta * 0.5 * (DWConv3x3(d_rel) + DWConv5x5(d_rel))`, with `eta=1e-3`.
- Primary-guided anchor: `guided_select * guided_update` from `d_rect` and `abs(p - d_rect)`.
- Residual mixture: softmax over reshaped `[B, 3C, H, W] -> [B, 3, C, H, W]` logits to mix guided/support/detail residual candidates.
- Residual-safe budget: `budget = 1 + budget_alpha * (2 * budget_raw - 1)`, with `budget_alpha=1e-3`.
- ARD: `keep_score = 1 - bad_strength * bad_prob`, with `bad_strength=1e-3`; this is soft suppression, not data augmentation.
- Final fusion form: `p + budget * keep_score * (w_guided * gamma_g * guided_select * guided_update + w_support * gamma_s * support_select * support_update + w_detail * gamma_d * detail_select * detail_update)`.
- Initial scales: `beta=1e-3`, `eta=1e-3`, `gamma_g=5e-4`, `gamma_s=1e-3`, `gamma_d=2e-4`, `sparse_k=5.0`, `sparse_tau=0`, `detail_k=3.0`, `detail_tau=0.1`, `noise_token=0`, `budget_alpha=1e-3`, `bad_strength=1e-3`.
- Parameter size: fusion blocks `3,365,136` params; total model `40,228,498` params.
- Fusion comparison: `817,584` fewer fusion params than original `dformerv2_mid_fusion` GatedFusion (`4,182,720`) and `571,344` more than DGC-AF++ (`2,793,792`).
- Active model entries after this change: `early`, `mid_fusion`, `dformerv2_mid_fusion`, `dformerv2_pg_sparse_comp_fusion`, `dformerv2_dgc_af_full`, `dformerv2_dgc_af_plus`, `dformerv2_dgc_af_plus_grm_ard`.
- Verification: `py_compile` passed for `src/models/mid_fusion.py` and `train.py`.
- Result note for `dformerv2_dgc_af_plus_grm_ard_run01`: best val/mIoU `0.507743`, below baseline mean `0.513406` by `0.005663` and below DGC-AF++ run01 by `0.005841`; do not promote this branch.

## 2026-05-04 DFormerv2 DGC-AF Plus

- Added `--model dformerv2_dgc_af_plus`.
- Added `DFormerGuidedCyclicAttentionFusionPlus`, `DFormerV2DGCAFPlusSegmentor`, and `LitDFormerV2DGCAFPlus` in `src/models/mid_fusion.py`.
- Kept DFormerv2 primary features as the main path and used auxiliary depth only through primary-preserving residual updates.
- Did not use `GatedFusion`, `g * primary + (1 - g) * depth`, full `QK^T` token attention, token flattening, `HW x HW` attention, `grid_sample`, flow warp, or deformable attention.
- Channel sizes come from `self.rgb_encoder.out_channels` and `self.depth_encoder.out_channels`; no hard-coded stage channels.
- Asymmetric depth rectification: `d_rect = d + beta * rectify_gate * rectify_update`, with `beta=1e-3`.
- Gemini-style relation selection: relation logits reshape from `[B, 2C, H, W]` to `[B, 2, C, H, W]`, softmax on relation dimension, then `d_rel = r_depth * v_d + r_noise * noise_token`.
- Soft sparse selection: `support_select = sigmoid(sparse_k * (support + r_depth - conflict - sparse_tau))`, with `sparse_k=5.0` and `sparse_tau=0`.
- Detail selection: `detail_select = sigmoid(detail_k * (conflict - support - detail_tau))`, with `detail_k=3.0` and `detail_tau=0.1`.
- Final fusion form: `p + gamma_s * support_select * support_update + gamma_d * detail_select * detail_update`, with `gamma_s=1e-3` and `gamma_d=2e-4`.
- Parameter size: fusion blocks `2,793,792` params; total model `39,657,154` params.
- Active model entries after this change: `early`, `mid_fusion`, `dformerv2_mid_fusion`, `dformerv2_pg_sparse_comp_fusion`, `dformerv2_dgc_af_full`, `dformerv2_dgc_af_plus`.
- Verification: `py_compile` passed for `src/models/mid_fusion.py` and `train.py`.
- Result note for repeated validation: run01-run04 mean best val/mIoU is `0.511418`, below baseline mean `0.513406` by `0.001988`; do not promote DGC-AF++ as a stable improvement.

## 2026-05-04 DFormerv2 DGC-AF Full

- Added `--model dformerv2_dgc_af_full`.
- Added `DFormerGuidedCyclicAttentionFusion`, `DFormerV2DGCAFFullSegmentor`, and `LitDFormerV2DGCAFFull` in `src/models/mid_fusion.py`.
- Kept DFormerv2 primary features as the main path and used auxiliary depth only through residual compensation.
- Did not use `GatedFusion`, `g * primary + (1 - g) * depth`, full `QK^T` token attention, DFormerv2 geometry self-attention, or raw depth geometry priors.
- Channel sizes come from `self.rgb_encoder.out_channels` and `self.depth_encoder.out_channels`; no hard-coded stage channels.
- Fusion form: `p + gamma * sparse_mask * residual_adapter(cat(d_attn, abs(p - d_attn)))`.
- Depth cleaning is residual-safe: `d_clean = d * (1 + beta * (2 * clean_gate - 1))`, with `beta=1e-3`.
- Sparse mask is soft and differentiable: `sigmoid(sparse_k * (support - conflict - sparse_tau))`, with `sparse_k=5.0` and `sparse_tau=0`.
- Parameter size: fusion blocks `2,435,776` params; total model `39,299,138` params.
- Active model entries after this change: `early`, `mid_fusion`, `dformerv2_mid_fusion`, `dformerv2_pg_sparse_comp_fusion`, `dformerv2_dgc_af_full`.
- Verification: `py_compile` passed for `src/models/mid_fusion.py` and `train.py`.
- Result note for `dformerv2_dgc_af_full_run01`: best val/mIoU `0.512766`, below baseline mean `0.513406` by `0.000640`, but above PG-SparseComp run01 by `0.001288`.

## 2026-05-04 DFormerv2 Primary-Guided Sparse Depth Compensation

- Added `--model dformerv2_pg_sparse_comp_fusion`.
- Added `DepthReliabilityHead`, `PrimaryGuidedSparseDepthCompensation`, `DFormerV2PGSparseCompSegmentor`, and `LitDFormerV2PGSparseComp` in `src/models/mid_fusion.py`.
- Kept DFormerv2 features as the primary path and used DepthEncoder features only as a gated residual compensation signal.
- Did not use `GatedFusion`, full `QK^T` token attention, or DFormerv2 geometry self-attention.
- Channel sizes come from `self.rgb_encoder.out_channels` and `self.depth_encoder.out_channels`; no hard-coded `[64, 128, 256, 512]`.
- Fusion form: `p + gamma * spatial_reliability * channel_reliability * residual_adapter(cat(depth_proj, abs(p - depth_proj)))`, with `gamma=1e-3`.
- Parameter size: fusion blocks `2,020,884` params; total model `38,884,246` params.
- Active model entries after this change: `early`, `mid_fusion`, `dformerv2_mid_fusion`, `dformerv2_pg_sparse_comp_fusion`.
- Verification: `py_compile` passed for `src/models/mid_fusion.py` and `train.py`.
- Result note for `dformerv2_pg_sparse_comp_fusion_run01`: best val/mIoU `0.511478`, below baseline mean `0.513406`; close but not a valid improvement.

## 2026-05-04 DFormerv2 SA-Gate One-Way Fusion

- Added `--model dformerv2_sagate_fusion`.
- Source idea: `FilterLayer` and `FSP` from SA-Gate.
- Original file: `C:/Users/qintian/Desktop/qintian/ref_codes/RGBD_Semantic_Segmentation_PyTorch-master/model/SA-Gate.nyu/net_util.py`.
- Copied hyperparameters: `reduction=16`, `AdaptiveAvgPool2d(1)`, two `Linear` layers, `ReLU(inplace=True)`, and `Sigmoid`.
- Kept DFormerv2 features as the primary path and used DepthEncoder features only as auxiliary support.
- Did not copy the original symmetric `SAGate` softmax aggregation because it mixes RGB and HHA as equal branches.
- Final fusion form: `primary_feat + gamma * auxiliary_residual`, with `gamma=1e-3`.
- Parameter size: `37,634,870` params, about `150.54 MB` with fp32 parameters.
- Active model entries after this change: `early`, `mid_fusion`, `dformerv2_mid_fusion`, `dformerv2_sagate_fusion`.
- Verification: `py_compile` passed for `src/models/fusion_blocks.py`, `src/models/mid_fusion.py`, and `train.py`.
- Result note: five-run mean best val/mIoU is `0.513216`, so this branch is lighter but not a stable improvement over `dformerv2_mid_fusion`.

## 2026-05-04 DFormerv2 SA-Gate Token Selection Fusion

- Added `--model dformerv2_sagate_token_fusion`.
- Source idea: keep SA-Gate `FilterLayer/FSP`, replace the simple spatial gate with TokenFusion-style dynamic token scoring.
- TokenFusion original files:
  - `C:/Users/qintian/Desktop/qintian/ref_codes/TokenFusion/semantic_segmentation/models/mix_transformer.py`
  - `C:/Users/qintian/Desktop/qintian/ref_codes/TokenFusion/semantic_segmentation/models/modules.py`
- Copied TokenFusion score structure: `LayerNorm -> Linear(C,C) -> GELU -> Linear(C,C/2) -> GELU -> Linear(C/2,C/4) -> GELU -> Linear(C/4,2) -> LogSoftmax`.
- Copied mask logic: `softmax(score.reshape(B, -1, 2), dim=2)[:, :, 0]`.
- Did not copy hard `TokenExchange(mask_threshold=0.02)` because the first V2 uses soft selection to reduce instability.
- Adaptation: flattened BCHW selector features into tokens for scoring, then reshaped the soft mask back to `[B,1,H,W]`.
- Selector input was updated from primary-only to `primary_feat`, `aux_support`, and `abs(primary_feat - aux_support)` so the gate can judge depth support quality.
- Final fusion form remains `primary_feat + gamma * auxiliary_residual`, with `gamma=1e-3`.
- Parameter size: `38,902,954` params, about `155.61 MB` with fp32 parameters; fusion blocks are about `8.16 MB`, below original GatedFusion's about `16.73 MB`.
- Active model entries after this change: `early`, `mid_fusion`, `dformerv2_mid_fusion`, `dformerv2_sagate_fusion`, `dformerv2_sagate_token_fusion`.
- Verification: `py_compile` passed for `src/models/fusion_blocks.py`, `src/models/mid_fusion.py`, and `train.py`.
- Result note for `dformerv2_sagate_token_fusion_run01`: best val/mIoU `0.509558`, below baseline mean `0.513406`; not a stable improvement.

## 2026-05-04 Restore Clean Baseline After SA-Gate Ablations

- Removed `dformerv2_sagate_fusion` and `dformerv2_sagate_token_fusion` from the active training path.
- Kept experiment records in `miou_list` and `docs/experiment_log.md`.
- Removed SA-Gate and TokenFusion experimental imports/classes from `src/models/mid_fusion.py`.
- Restored `src/models/fusion_blocks.py` to a placeholder file.
- Active model entries after restore: `early`, `mid_fusion`, `dformerv2_mid_fusion`.
- `dformerv2_mid_fusion` still supports `--dformerv2_pretrained`.
- Verification: `py_compile` passed for `src/models/fusion_blocks.py`, `src/models/mid_fusion.py`, and `train.py`.

## 2026-05-04 DFormerv2 Gated Co-Attention Residual Fusion

- Added `--model dformerv2_gated_coattn_res_fusion`.
- Kept the original `GatedFusion` path intact inside the new module.
- Added a lightweight correction branch after `base = refine(g * primary + (1 - g) * depth_proj)`.
- Context input: `primary_feat`, `depth_proj`, `base`, and `abs(primary_feat - depth_proj)`.
- Correction branch: channel attention, spatial attention, then `Conv1x1(4C -> C//8) + BN + ReLU + DWConv3x3 + BN + ReLU + Conv1x1(C//8 -> C) + BN`.
- Fusion output: `base + gamma * correction`, with `gamma=1e-3`.
- Direct code copy: none. The implementation only borrows local attention/fusion ideas from ACNet, SGACNet, and FAFNet.
- Parameter size: `41,665,250` total params, about `166.66 MB` with fp32 parameters; fusion blocks are `4.802M` params, about `19.21 MB`.
- Baseline comparison: original `dformerv2_mid_fusion` fusion blocks are `4.183M` params, about `16.73 MB`.
- Active model entries after this change: `early`, `mid_fusion`, `dformerv2_mid_fusion`, `dformerv2_gated_coattn_res_fusion`.
- Verification: `py_compile` passed for `src/models/mid_fusion.py` and `train.py`.
- Result note for `dformerv2_gated_coattn_res_fusion_run01`: manually stopped after 20 recorded validation epochs; best recorded val/mIoU `0.483357`, far below baseline mean `0.513406`.

## 2026-05-04 Restore Clean Baseline After Gated Co-Attention Residual Ablation

- Removed `dformerv2_gated_coattn_res_fusion` from the active training path.
- Kept experiment records in `miou_list` and `docs/experiment_log.md`.
- Removed `GatedCoAttentionResidualFusion`, `DFormerV2GatedCoAttnResFusionSegmentor`, and `LitDFormerV2GatedCoAttnResFusion` from `src/models/mid_fusion.py`.
- Active model entries after restore: `early`, `mid_fusion`, `dformerv2_mid_fusion`.
- `dformerv2_mid_fusion` still supports `--dformerv2_pretrained`.
- Verification: `py_compile` passed for `src/models/fusion_blocks.py`, `src/models/mid_fusion.py`, and `train.py`.
