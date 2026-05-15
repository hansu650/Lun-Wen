# Model Changes

## 2026-05-16 R040 c4 Low-Rank Depth Prompt

- Added `C4LowRankDepthPromptFusion` in `src/models/mid_fusion.py` for the experiment branch.
- Added `DFormerV2C4LowRankDepthPromptSegmentor` in `src/models/mid_fusion.py`.
- Added `LitDFormerV2C4LowRankDepthPrompt` in `src/models/mid_fusion.py`.
- Registered `dformerv2_c4_lowrank_depth_prompt` in `train.py`.
- The experiment preserved c1-c3 original `GatedFusion` and replaced only c4 with a prompt-conditioned version of the original gate/refine logic.
- The c4 prompt branch projects depth to c4 RGB/DFormerv2 channels, builds low-rank channel and spatial bases from `[depth_proj, abs(rgb-depth_proj)]`, projects the prompt with a zero-initialized `1x1`, and adds it to c4 RGB/DFormerv2 features before fusion.
- Fixed `rank=8`; no CLI sweep knob was added.
- Logged `train/c4_prompt_abs`, `train/c4_prompt_raw_abs`, `train/c4_prompt_gate_mean`, and `train/c4_prompt_gate_std`.
- Smoke verification confirmed c1-c3 original `GatedFusion`, c4 `C4LowRankDepthPromptFusion`, finite real-batch CE, nonzero prompt/depth/gate/refine gradients, and unchanged DFormerv2 pretrained load stats.
- Full-train result: best val/mIoU `0.527946` at validation epoch `37`, last val/mIoU `0.524679`, best-to-last drop `0.003267`.
- Decision: reject as active mainline because it remains below the stage threshold `0.53` and below R016 `0.541121`.
- Cleanup: remove `dformerv2_c4_lowrank_depth_prompt` from the active registry after recording evidence; archive the implementation snippet under `feiqi/failed_experiments_r040_20260516/`.
- No dataset split, dataloader, augmentation, evaluation metric, mIoU calculation, loss, optimizer, scheduler, batch size, epoch count, learning rate, worker count, early stopping, DFormerv2-S level, pretrained loading, checkpoint artifacts, dataset files, pretrained weights, or TensorBoard event files were changed.

## 2026-05-16 R039 MIIM-lite c4 Global-Local Residual

- Added `MIIMC4LiteFusion` in `src/models/mid_fusion.py` for the experiment branch.
- Added `DFormerV2MIIMC4LiteSegmentor` in `src/models/mid_fusion.py`.
- Added `LitDFormerV2MIIMC4Lite` in `src/models/mid_fusion.py`.
- Registered `dformerv2_miim_c4_lite` in `train.py`.
- The experiment preserved c1-c3 original `GatedFusion` and replaced only c4 with a base `GatedFusion` plus tiny MIIM-lite residual.
- The MIIM-lite residual used a global channel gate from pooled `[rgb, depth_proj, abs(rgb-depth_proj), base]` and a local `1x1 -> depthwise 7x7 -> 1x1` update.
- Fixed `alpha_max=0.05`; the initial effective alpha was about `0.0025`; no CLI sweep knob was added.
- Logged `train/miim_c4_alpha`, `train/miim_c4_gate_mean`, `train/miim_c4_gate_std`, and `train/miim_c4_update_abs`.
- Smoke verification confirmed c1-c3 original `GatedFusion`, c4 `MIIMC4LiteFusion`, finite real-batch CE, nonzero MIIM gradients, and unchanged DFormerv2 pretrained load stats.
- Full-train result: best val/mIoU `0.534131` at validation epoch `41`, last val/mIoU `0.509767`, best-to-last drop `0.024364`.
- Decision: reject as active mainline because it remains below R016 `0.541121` and worsens late collapse.
- Cleanup: remove `dformerv2_miim_c4_lite` from the active registry after recording evidence; archive the implementation snippet under `feiqi/failed_experiments_r039_20260516/`.
- No dataset split, dataloader, augmentation, evaluation metric, mIoU calculation, loss, optimizer, scheduler, batch size, epoch count, learning rate, worker count, early stopping, DFormerv2-S level, pretrained loading, checkpoint artifacts, dataset files, pretrained weights, or TensorBoard event files were changed.

## 2026-05-16 R038 DSCF-lite c4-only Fusion

- Added `DSCFC4LiteFusion` in `src/models/mid_fusion.py`.
- Added `DFormerV2DSCFC4LiteSegmentor` in `src/models/mid_fusion.py`.
- Added `LitDFormerV2DSCFC4Lite` in `src/models/mid_fusion.py`.
- Registered `dformerv2_dscf_c4_lite` in `train.py`.
- The new entry preserves c1-c3 original `GatedFusion` and replaces only c4 with dynamic sparse depth sampling over four learned offset branches.
- `grid_sample` uses `align_corners=True`, `padding_mode="border"`, and normalized offsets consistent with pixel offsets.
- Offset bias uses four small directional seeds to avoid exact K-sample collapse; sample logits start uniform.
- Logged `train/dscf_c4_offset_abs` and `train/dscf_c4_weight_entropy`.
- Smoke verification confirmed non-collapsed K branches and nonzero gradients for offset, sample weight, gate, refine, and depth projection.
- Full-train result: best val/mIoU `0.530810` at validation epoch `38`, last val/mIoU `0.530308`, best-to-last drop `0.000502`.
- Decision: reject as active mainline because it remains below R016 `0.541121`.
- Cleanup: remove `dformerv2_dscf_c4_lite` from the active registry after recording evidence; archive the implementation snippet under `feiqi/failed_experiments_r038_20260516/`.
- No dataset split, dataloader, augmentation, evaluation metric, mIoU calculation, loss, optimizer, scheduler, batch size, epoch count, learning rate, worker count, early stopping, DFormerv2-S level, pretrained loading, checkpoint artifacts, dataset files, pretrained weights, or TensorBoard event files were changed.

## 2026-05-15 R037 DGL Minimal Gradient Disentanglement

- Added `DFormerV2DGLMinimalSegmentor` in `src/models/mid_fusion.py`.
- Added `LitDFormerV2DGLMinimal` in `src/models/mid_fusion.py`.
- Registered `dformerv2_dgl_minimal` in `train.py`.
- The new entry preserves the R016 inference path shape and returns only fused logits at validation/inference time.
- During training, fused logits are computed from detached primary/depth features so fused CE updates fusion/decoder but not DFormerv2/DepthEncoder.
- Added training-only `primary_aux_decoder` and `depth_aux_decoder`; their CE losses train the encoders and aux heads without updating fusion/decoder.
- Fixed DGL aux weight is `0.03`; no CLI sweep knob was added.
- Smoke verification confirmed the intended gradient routing on a real NYU CUDA batch.
- Full-train result: best val/mIoU `0.534656` at validation epoch `42`, last val/mIoU `0.530153`, best-to-last drop `0.004503`.
- Decision: reject as active mainline because it remains below R016 `0.541121`.
- Cleanup: remove `dformerv2_dgl_minimal` from the active registry after recording evidence; archive the implementation snippet under `feiqi/failed_experiments_r037_20260515/`.
- No dataset split, dataloader, augmentation, evaluation metric, mIoU calculation, optimizer, scheduler, batch size, epoch count, learning rate, worker count, early stopping, DFormerv2-S level, pretrained loading, checkpoint artifacts, dataset files, pretrained weights, or TensorBoard event files were changed.

## 2026-05-15 R036 c3/c4 Bounded Depth Residual

- Added `GatedFusionC34BoundedDepthResidual` in `src/models/mid_fusion.py`.
- Added `DFormerV2C34BoundedDepthResidualSegmentor` in `src/models/mid_fusion.py`.
- Added `LitDFormerV2C34BoundedDepthResidual` in `src/models/mid_fusion.py`.
- Registered `dformerv2_c34_bounded_depth_residual` in `train.py`.
- The new fusion path preserves original `GatedFusion` at c1/c2 and replaces only c3/c4 with `base + alpha * residual`, where `alpha <= 0.05` and the final residual projection is zero-initialized.
- The baseline `dformerv2_mid_fusion` entry and original `GatedFusion` class remain unchanged.
- Smoke verification confirmed exact initial parity with the base fusion path, finite real-batch loss `3.837778`, nonzero c3/c4 residual gradients, and successful DFormerv2 pretrained loading.
- Full-train result: best val/mIoU `0.539790` at validation epoch `44`, last val/mIoU `0.528882`, best-to-last drop `0.010908`.
- c3 residual alpha first/last: `0.025097` / `0.026970`; c4 residual alpha first/last: `0.025034` / `0.025553`.
- Decision: reject as active mainline because it remains below R016 `0.541121`.
- Cleanup: remove `dformerv2_c34_bounded_depth_residual` from the active registry after recording evidence; archive the implementation snippet under `feiqi/failed_experiments_r036_20260515/`.
- No dataset split, dataloader, augmentation, evaluation metric, mIoU calculation, loss, optimizer, scheduler, batch size, epoch count, learning rate, worker count, early stopping, DFormerv2-S level, pretrained loading, DepthEncoder structure, SimpleFPNDecoder, checkpoint artifacts, dataset files, pretrained weights, or TensorBoard event files were changed.

## 2026-05-15 R035 Gate Balance Regularizer

- Added `GatedFusionWithBalanceStats` in `src/models/mid_fusion.py`.
- Added `DFormerV2GateBalanceRegSegmentor` in `src/models/mid_fusion.py`.
- Added `LitDFormerV2GateBalanceReg` in `src/models/mid_fusion.py`.
- Registered `dformerv2_gate_balance_reg` in `train.py`.
- The new fusion module preserves the original `GatedFusion` computation and stores each stage gate for training-only logging and regularization.
- The Lightning wrapper adds `0.01 * mean((gate_mean - 0.5)^2)` to CE during training and logs `train/gate_balance_loss`, `train/gate_mean_c1..c4`, and `train/gate_std_c1..c4`.
- The baseline `dformerv2_mid_fusion` entry and original `GatedFusion` class remain unchanged.
- Smoke verification confirmed finite real-batch loss `3.750904`, gate means near `0.5`, nonzero `depth_proj` gradient, and successful DFormerv2 pretrained loading.
- Full-train result: best val/mIoU `0.529498` at validation epoch `38`, last val/mIoU `0.521308`, best-to-last drop `0.008190`.
- Decision: reject as active mainline because it falls below the `0.53` stage threshold and far below R016 `0.541121`.
- Cleanup: removed `dformerv2_gate_balance_reg` from the active registry after recording evidence; archived the implementation snippet under `feiqi/failed_experiments_r035_20260515/`.
- No dataset split, dataloader, augmentation, evaluation metric, mIoU calculation, optimizer, scheduler, batch size, epoch count, learning rate, worker count, early stopping, DFormerv2-S level, pretrained loading, DepthEncoder structure, SimpleFPNDecoder, checkpoint artifacts, dataset files, pretrained weights, or TensorBoard event files were changed.

## 2026-05-15 R034 MASG Gate-Only Depth Stop-Gradient

- Added `GatedFusionGateStopGrad` in `src/models/mid_fusion.py`.
- Added `DFormerV2MASGFusionSegmentor` in `src/models/mid_fusion.py`.
- Added `LitDFormerV2MASGFusion` in `src/models/mid_fusion.py`.
- Registered `dformerv2_masg_fusion` in `train.py`.
- The new fusion module preserves the existing `GatedFusion` topology but computes the gate from `torch.cat([rgb_feat, d.detach()], dim=1)`, where `d = depth_proj(depth_feat)`.
- The depth projection and depth encoder still receive gradients through the fused value path `g * rgb_feat + (1 - g) * d`; only the depth projection's gate-input route is detached.
- The baseline `dformerv2_mid_fusion` entry and original `GatedFusion` class remain unchanged.
- Smoke verification confirmed real-batch logits `(1, 40, 480, 640)`, CE loss `3.733857`, nonzero `depth_proj` gradient, nonzero gate gradient, and successful DFormerv2 pretrained loading.
- Full-train result: best val/mIoU `0.539322` at validation epoch `40`, last val/mIoU `0.518738`, best-to-last drop `0.020584`.
- Decision: reject as active mainline because it remains below R016 `0.541121` and worsens late instability.
- Cleanup: removed `dformerv2_masg_fusion` from the active registry after recording evidence; archived the implementation snippet under `feiqi/failed_experiments_r034_20260515/`.
- No dataset split, dataloader, augmentation, evaluation metric, mIoU calculation, loss, optimizer, scheduler, batch size, epoch count, learning rate, worker count, early stopping, DFormerv2-S level, pretrained loading, DepthEncoder structure, SimpleFPNDecoder, checkpoint artifacts, dataset files, pretrained weights, or TensorBoard event files were changed.

## 2026-05-15 Post-R033 Mainline Cleanup

- Removed non-mainline R019/R020/R025/R026/R027/R030/R031/R032/R033 entries from the active `train.py` registry.
- Removed the corresponding inactive experimental classes from `src/models/mid_fusion.py` and `src/models/decoder.py`.
- Archived code summaries/snippets under `feiqi/failed_experiments_r019_r033_20260515/`.
- Kept active entries focused on `dformerv2_mid_fusion` (R016 current best), `dformerv2_ham_decoder` (R022 stable reference), and `dformerv2_geometry_primary_ham_decoder` (R024 stable structure diagnostic).
- Kept all evidence in `docs/`, `miou_list/`, `reports/`, `metrics/`, and `experiments/`; checkpoint and TensorBoard event files remain untracked and were not committed.
- No dataset split, dataloader, augmentation, evaluation metric, mIoU calculation, loss, optimizer, scheduler, batch size, epoch count, learning rate, worker count, early stopping, DFormerv2-S level, pretrained loading, checkpoint artifacts, dataset files, pretrained weights, or TensorBoard event files were changed.

## 2026-05-15 R033 SimpleFPN Ham Logit Fusion

- Added `SimpleFPNHamLogitFusionDecoder` in `src/models/decoder.py`.
- Added `DFormerV2SimpleFPNHamLogitFusionSegmentor` in `src/models/mid_fusion.py`.
- Added `LitDFormerV2SimpleFPNHamLogitFusion` in `src/models/mid_fusion.py`.
- Registered `dformerv2_simplefpn_ham_logit_fusion` in `train.py`.
- The new decoder keeps the corrected R016 fusion/base path and computes `simple_fpn_logits + alpha * ham_logits`, where `alpha = sigmoid(ham_logit_logit)`.
- `ham_logit_logit` is initialized to `-2.944439`, giving initial `alpha` about `0.05`.
- Logged `train/ham_logit_alpha` for audit without changing loss or optimizer behavior.
- The corrected baseline `dformerv2_mid_fusion` remains unchanged.
- Smoke verification confirmed decoder type `SimpleFPNHamLogitFusionDecoder`, initial alpha `0.050000`, logits `(2, 40, 480, 640)`, CE loss `3.807222`, alpha gradient `0.010959`, Ham classifier gradient sum `15.861186`, SimpleFPN classifier gradient sum `76.114777`, and peak memory about `5775.9 MB`.
- Full-train result: best val/mIoU `0.533020` at validation epoch `49`, last val/mIoU `0.528883`, ham logit alpha first/last `0.050669` / `0.090593`.
- Decision: reject as an active mainline improvement because it remains below R016 `0.541121`; archive or keep only as partial-positive Ham-complementarity evidence.
- No dataset split, dataloader, augmentation, evaluation metric, mIoU calculation, loss, optimizer, scheduler, batch size, epoch count, learning rate, worker count, early stopping, DFormerv2-S level, pretrained loading, pretrained DFormerv2 weights, pretrained DepthEncoder weights, GatedFusion equations, checkpoint artifacts, or TensorBoard event files were changed.

## 2026-05-15 R032 SimpleFPN C1 Detail Gate

- Added `SimpleFPNDecoderC1DetailGate` in `src/models/decoder.py`.
- Added `DFormerV2SimpleFPNC1DetailGateSegmentor` in `src/models/mid_fusion.py`.
- Added `LitDFormerV2SimpleFPNC1DetailGate` in `src/models/mid_fusion.py`.
- Registered `dformerv2_simplefpn_c1_detail_gate` in `train.py`.
- The new decoder keeps the SimpleFPN topology but computes `p1 = alpha * lateral1(c1) + upsample(p2)`, with `alpha = sigmoid(c1_detail_logit)`.
- `c1_detail_logit` is initialized to `6.906755`, giving initial `alpha` about `0.999`.
- Logged `train/c1_detail_alpha` for audit without changing loss or optimizer behavior.
- The corrected baseline `dformerv2_mid_fusion` remains unchanged.
- Smoke verification confirmed decoder type `SimpleFPNDecoderC1DetailGate`, initial alpha `0.999000`, logits `(2, 40, 480, 640)`, CE loss `3.659918`, nonzero `c1_detail_logit` gradient, classifier gradient sum `75.462433`, and peak memory about `5739.9 MB`.
- Full-train result: best val/mIoU `0.536603` at validation epoch `50`, last val/mIoU `0.536603`, alpha first/last `0.998994` / `0.998770`.
- Decision: keep as partial-positive evidence but do not promote; alpha barely moved and the peak remains below R016 `0.541121`.
- No dataset split, dataloader, augmentation, evaluation metric, mIoU calculation, loss, optimizer, scheduler, batch size, epoch count, learning rate, worker count, early stopping, DFormerv2-S level, pretrained loading, pretrained DFormerv2 weights, pretrained DepthEncoder weights, GatedFusion equations, checkpoint artifacts, or TensorBoard event files were changed.

## 2026-05-15 R031 SimpleFPN Classifier Dropout

- Added `SimpleFPNDecoderWithClassifierDropout` in `src/models/decoder.py`.
- Added `DFormerV2SimpleFPNClassifierDropoutSegmentor` in `src/models/mid_fusion.py`.
- Added `LitDFormerV2SimpleFPNClassifierDropout` in `src/models/mid_fusion.py`.
- Registered `dformerv2_simplefpn_classifier_dropout` in `train.py`.
- The new decoder subclasses the existing SimpleFPN behavior and applies `Dropout2d(0.1)` immediately before `classifier`.
- The corrected baseline `dformerv2_mid_fusion` remains unchanged.
- Smoke verification confirmed decoder type `SimpleFPNDecoderWithClassifierDropout`, dropout `p=0.1`, logits `(2, 40, 480, 640)`, CE loss `3.702577`, classifier gradient sum `76.829285`, and peak memory about `5720.9 MB`.
- Full-train result: best val/mIoU `0.531544` at validation epoch `40`, last val/mIoU `0.525760`.
- Decision: reject classifier dropout on the SimpleFPN path as a main direction because it remains below R016 `0.541121`.
- No dataset split, dataloader, augmentation, evaluation metric, mIoU calculation, loss, optimizer, scheduler, batch size, epoch count, learning rate, worker count, early stopping, DFormerv2-S level, pretrained loading, pretrained DFormerv2 weights, pretrained DepthEncoder weights, GatedFusion equations, checkpoint artifacts, or TensorBoard event files were changed.

## 2026-05-15 R030 GatedFusion Residual-Top

- Added `GatedFusionResidualTop` in `src/models/mid_fusion.py`.
- Added `DFormerV2GatedFusionResidualTopSegmentor` in `src/models/mid_fusion.py`.
- Added `LitDFormerV2GatedFusionResidualTop` in `src/models/mid_fusion.py`.
- Registered `dformerv2_gated_fusion_residual_top` in `train.py`.
- The new fusion module preserves the original `GatedFusion` computation as `base`, then predicts a residual correction from `cat(rgb_feat, depth_proj, base, abs(rgb_feat - depth_proj))`.
- The final residual `Conv2d` weight and bias are zero-initialized, so the initial output exactly equals the `GatedFusion` base.
- Smoke verification confirmed all four fusions are `GatedFusionResidualTop`, initial base max diffs are `[0.0, 0.0, 0.0, 0.0]`, final residual conv gradients are nonzero after backward, logits are `(2, 40, 480, 640)`, CE loss is `3.742640`, and peak memory is about `5956.9 MB`.
- Full-train result: best val/mIoU `0.536454` at validation epoch `42`, last val/mIoU `0.529803`.
- Decision: keep as partial-positive evidence but do not promote; all-stage residual-top correction remains below R016 `0.541121`.
- No dataset split, dataloader, augmentation, evaluation metric, mIoU calculation, loss, optimizer, scheduler, batch size, epoch count, learning rate, worker count, early stopping, DFormerv2-S level, pretrained loading, pretrained DFormerv2 weights, pretrained DepthEncoder weights, SimpleFPNDecoder, checkpoint artifacts, or TensorBoard event files were changed.

## 2026-05-15 R027 Primary Residual Depth Injection

- Added `PrimaryResidualDepthInjection` in `src/models/mid_fusion.py`.
- Added `DFormerV2PrimaryResidualDepthInjectionSegmentor` in `src/models/mid_fusion.py`.
- Added `LitDFormerV2PrimaryResidualDepthInjection` in `src/models/mid_fusion.py`.
- Registered `dformerv2_primary_residual_depth` in `train.py`.
- The new fusion module projects DepthEncoder features to the DFormerv2 channel width, computes `abs(rgb_feat - depth_proj)`, and predicts a residual correction from `cat(depth_proj, abs_diff)`.
- The final residual `Conv2d` weight and bias are zero-initialized, so the initial fused feature is exactly `rgb_feat`.
- This model intentionally replaces all four `GatedFusion` modules to test one fusion-form hypothesis; it does not change the decoder, loss, optimizer, scheduler, data path, or DFormerv2-S pretrained loading.
- Smoke verification confirmed all four fusions are `PrimaryResidualDepthInjection`, initial identity max diffs are `[0.0, 0.0, 0.0, 0.0]`, final residual conv gradients are nonzero after backward, logits are `(2, 40, 480, 640)`, CE loss is `3.815064`, and peak memory is about `5730.3 MB`.
- Full-train result: best val/mIoU `0.536739` at validation epoch `41`, last val/mIoU `0.505286`.
- Decision: keep as partial-positive evidence but do not use it as the next base; replacing R016 `GatedFusion` produces a high peak but unstable late behavior.
- No dataset split, dataloader, augmentation, evaluation metric, mIoU calculation, loss, optimizer, scheduler, batch size, epoch count, learning rate, worker count, early stopping, DFormerv2-S level, pretrained loading, pretrained DFormerv2 weights, pretrained DepthEncoder weights, SimpleFPNDecoder, checkpoint artifacts, or TensorBoard event files were changed.

## 2026-05-15 R026 Official-Style Local Module Init

- Added `init_official_style_local_modules` in `src/models/mid_fusion.py`.
- Added `DFormerV2OfficialInitLocalModulesSegmentor` in `src/models/mid_fusion.py`.
- Added `LitDFormerV2OfficialInitLocalModules` in `src/models/mid_fusion.py`.
- Registered `dformerv2_official_init_local_modules` in `train.py`.
- The new model keeps `dformerv2_mid_fusion` unchanged and uses a separate model name to avoid silently changing the corrected baseline entry.
- Initialization is applied only to `self.fusions` and `self.decoder`: Conv2d Kaiming normal fan-in/relu, BatchNorm2d `eps=1e-3`, `momentum=0.1`, weight `1`, bias `0`.
- Smoke verification confirmed fusion/decoder BN modules changed to `eps=0.001`, DepthEncoder BN stayed at `1e-5` and training mode, logits `(2, 40, 480, 640)`, CE loss `3.754482`, and peak memory about `5724.3 MB`.
- Full-train result: best val/mIoU `0.507906` at validation epoch `33`, last val/mIoU `0.499770`.
- Decision: reject this initialization direction; do not use it as a base for future fusion experiments.
- No dataset split, dataloader, augmentation, evaluation metric, mIoU calculation, loss, optimizer, scheduler, batch size, epoch count, learning rate, worker count, early stopping, DFormerv2-S level, pretrained loading, pretrained DFormerv2 weights, pretrained DepthEncoder weights, GatedFusion equations, SimpleFPNDecoder topology, checkpoint artifacts, or TensorBoard event files were changed.

## 2026-05-15 R025 DepthEncoder BN Eval

- Added `DFormerV2DepthEncoderBNEvalSegmentor` in `src/models/mid_fusion.py`.
- Added `LitDFormerV2DepthEncoderBNEval` in `src/models/mid_fusion.py`.
- Registered `dformerv2_depth_encoder_bn_eval` in `train.py`.
- The new segmentor subclasses the corrected mid-fusion model and overrides `train(mode=True)` to call `super().train(mode)`, then set only `self.depth_encoder` `nn.BatchNorm2d` modules to eval mode.
- DepthEncoder BN affine parameters remain trainable; no parameters are frozen and optimizer construction is unchanged.
- Verification before full train: syntax compile, `train.py --help`, real-batch CUDA forward/backward smoke, and read-only planner review passed. Smoke confirmed 20 DepthEncoder BN modules in eval mode, 8 fusion BN modules still in train mode, DepthEncoder BN affine gradients present, logits `(2, 40, 480, 640)`, CE loss `3.640506`, and peak memory about `5724.2 MB`.
- Full-train result: best val/mIoU `0.532572` at validation epoch `47`, last val/mIoU `0.496030`.
- Decision: do not promote BN eval as a stable base because late collapse remains severe and the peak is below R016/R022.
- No dataset split, dataloader, augmentation, evaluation metric, mIoU calculation, loss, optimizer, scheduler, batch size, epoch count, learning rate, worker count, early stopping, DFormerv2-S level, pretrained loading, DepthEncoder architecture, GatedFusion equations, SimpleFPNDecoder, checkpoint artifacts, or TensorBoard event files were changed.

## 2026-05-15 R024 Geometry-Primary Ham Decoder Entry

- Added `DFormerV2GeometryPrimaryHamDecoderSegmentor` in `src/models/teacher_model.py`.
- Added `LitDFormerV2GeometryPrimaryHamDecoder` in `src/models/teacher_model.py`.
- Registered `dformerv2_geometry_primary_ham_decoder` in `train.py`.
- The new model uses `Dformerv2_S(rgb, depth)` features directly with the existing `OfficialHamDecoder`; it does not instantiate the external ResNet-18 DepthEncoder or GatedFusion stack.
- Existing `OfficialHamDecoder` is reused unchanged, including the R022 `Dropout2d(0.1)` classifier parity fix.
- Verification before full train: syntax compile, `train.py --help`, registry lookup, real-batch CUDA forward/backward smoke, and static reviewer passed. Smoke confirmed no `depth_encoder`, no `fusions`, decoder `OfficialHamDecoder`, dropout `0.1`, logits `(2, 40, 480, 640)`, CE loss `3.837853`, and peak memory about `4662.6 MB`.
- Full-train result: best val/mIoU `0.530186` at validation epoch `45`, last val/mIoU `0.529383`.
- Decision: keep as a stable structure diagnostic, but do not promote it as the corrected baseline because it remains below R016 `0.541121` and R022 `0.534332`.
- No dataset split, dataloader, augmentation, evaluation metric, mIoU calculation, loss, optimizer, scheduler, batch size, epoch count, learning rate, worker count, early stopping, DFormerv2-S level, pretrained loading, OfficialHamDecoder internals, checkpoint artifacts, or TensorBoard event files were changed.

## 2026-05-15 R022 Ham Classifier Dropout Parity Fix

- Updated `OfficialHamDecoder` in `src/models/decoder.py` to add `self.dropout = nn.Dropout2d(0.1)`.
- The classifier now uses `self.classifier(self.dropout(x))`, matching the official `BaseDecodeHead.cls_seg()` dropout behavior.
- No registry, model entry, encoder, DFormerv2-S level, pretrained loading, DepthEncoder, GatedFusion, label mapping, depth normalization, dataset split, loader, augmentation, eval metric, mIoU calculation, optimizer, scheduler, batch size, epoch count, learning rate, worker count, early stopping, checkpoint artifact, or TensorBoard event file changed.
- Verification before full train: `compileall`, `train.py --help`, real-batch CUDA forward/backward smoke, and static reviewer passed. Smoke confirmed `Dropout2d(p=0.1)` and logits `(2, 40, 480, 640)`.
- Full-train result: best val/mIoU `0.534332` at validation epoch `50`, above R021 `0.527353` but below R016 `0.541121`.
- Decision: keep as a partial-positive Ham parity variant, but do not promote it as the corrected baseline. Further Ham micro-fixes are lower value than corrected-contract PMAD teacher refresh.

## 2026-05-15 R021 LightHam-Like Decoder

- Added `OfficialHamDecoder`, `NMF2D`, `Hamburger`, and `ConvBNReLU` in `src/models/decoder.py`.
- Added `DFormerV2HamDecoderSegmentor` and `LitDFormerV2HamDecoder` in `src/models/mid_fusion.py`.
- Registered `dformerv2_ham_decoder` in `train.py`.
- The new model keeps DFormerv2-S, pretrained loading, external ResNet-18 DepthEncoder, GatedFusion, label mapping, depth normalization, data/eval path, optimizer, scheduler, batch size, epoch count, learning rate, worker count, and early stopping unchanged.
- The decoder uses the official DFormer Ham input pattern, c2/c3/c4, with NMF defaults equivalent to `MD_S=1`, `MD_R=64`, `TRAIN_STEPS=6`, and `EVAL_STEPS=7`.
- Audit caveat: R021 omits official `BaseDecodeHead.cls_seg()` `Dropout2d(0.1)`, so this code is LightHam-like and not strict official Ham parity.
- Verification before full train: `compileall`, `train.py --help`, real-batch CUDA forward/backward smoke, and static reviewer passed. Smoke produced logits `(2, 40, 480, 640)`, CE loss `4.244293`, and peak memory about `5035.6 MB`.
- Full-train result: best val/mIoU `0.527353` at validation epoch `39`, last val/mIoU `0.501377`, below R016 `0.541121`.
- Decision: do not promote this no-dropout Ham decoder as a successful method. A single R022 dropout parity fix is justified; if it fails, archive the Ham decoder branch and move on.

## 2026-05-15 R020 Branch-Specific Depth Blend Adapter

- Added `DFormerV2BranchDepthBlendAdapterSegmentor` in `src/models/mid_fusion.py`.
- Added `LitDFormerV2BranchDepthBlendAdapter` in `src/models/mid_fusion.py`.
- Registered `dformerv2_branch_depth_blend_adapter` in `train.py`.
- The DFormerv2 geometry branch remains unchanged and receives R016 official normalized depth.
- The external DepthEncoder branch receives `(1-alpha)*depth + alpha*depth01`, where `depth01 = clamp(depth * 0.28 + 0.48, 0, 1)`.
- `alpha = sigmoid(depth_blend_logit)` is a single learnable scalar initialized to about `0.05`, and `train/depth_blend_alpha` is logged for audit.
- `dformerv2_mid_fusion` and `dformerv2_branch_depth_adapter` remain unchanged.
- Verification: `compileall`, `train.py --help`, and CUDA forward smoke passed. Smoke confirmed initial alpha `0.050000`, blended DepthEncoder input range `[-1.628571, 1.814286]`, and logits `(2, 40, 480, 640)`.
- Full-train result: best val/mIoU `0.532924` at validation epoch `41`, last val/mIoU `0.503238`, alpha last `0.051455`.
- Decision: keep as a partial-positive stabilization variant, but do not promote it as the corrected baseline because it remains below R016 `0.541121`.
- No dataset split, dataloader, augmentation, evaluation metric, mIoU calculation, loss, optimizer, scheduler, batch size, epoch count, learning rate, worker count, early stopping, DFormerv2-S level, pretrained loading, DepthEncoder architecture, GatedFusion, SimpleFPNDecoder, checkpoint artifacts, or TensorBoard event files were changed.

## 2026-05-14 R019 Branch-Specific Depth Adapter

- Added `DFormerV2BranchDepthAdapterSegmentor` in `src/models/mid_fusion.py`.
- Added `LitDFormerV2BranchDepthAdapter` in `src/models/mid_fusion.py`.
- Registered `dformerv2_branch_depth_adapter` in `train.py`.
- The DFormerv2 geometry branch remains unchanged and receives R016 official normalized depth.
- The external DepthEncoder branch receives `torch.clamp(depth * 0.28 + 0.48, min=0.0, max=1.0)` inside the model.
- `dformerv2_mid_fusion` default behavior remains unchanged.
- Parameter size remains effectively unchanged at `41.0 M`; no new trainable adapter layers were added.
- Verification: `compileall`, `train.py --help`, and CUDA forward smoke passed. Smoke confirmed DFormer depth range `[-1.714286, 1.857143]`, DepthEncoder adapter range `[0.000000, 1.000000]`, and logits `(2, 40, 480, 640)`.
- Full-train result: best val/mIoU `0.532539` at validation epoch `46`, but last val/mIoU `0.495229`, below R016 `0.541121` and unstable.
- Decision: keep as an active partial-positive research variant for follow-up design, but do not promote it as the corrected baseline or main result.
- No dataset split, dataloader, augmentation, evaluation metric, mIoU calculation, loss, optimizer, scheduler, batch size, epoch count, learning rate, worker count, early stopping, DFormerv2-S level, pretrained loading, DepthEncoder architecture, GatedFusion, SimpleFPNDecoder, checkpoint artifacts, or TensorBoard event files were changed.

## 2026-05-14 R018 DropPath 0.25 Contract Negative Archive

- Temporarily added `drop_path_rate` passthrough to `DFormerV2MidFusionSegmentor` in `src/models/mid_fusion.py`.
- Temporarily registered `dformerv2_mid_fusion_dpr025` in `train.py`, with `drop_path_rate=0.25`, while preserving the default `dformerv2_mid_fusion` path.
- Smoke checks passed: `compileall`, `train.py --help`, real-batch CUDA forward, and DropPath inspection (`29` DropPath modules, max/last `0.25`).
- Full-train retry1 result: best val/mIoU `0.526282` at validation epoch `46`, below the R016 corrected baseline `0.541121`.
- Decision: do not keep this active code change. `src/models/mid_fusion.py` and `train.py` are restored to the R017/R016 mainline state.
- Archived the failed code diff under `feiqi/failed_experiments_r014_plus_20260514/R018_droppath025_contract.md`.
- No retained model structure, DFormerv2-S level, pretrained loading, DepthEncoder, GatedFusion, SimpleFPNDecoder, data preprocessing, optimizer, scheduler, batch size, epoch count, learning rate, worker count, early stopping, split files, validation loader behavior, checkpoint artifacts, or TensorBoard event files were changed in the mainline.

## 2026-05-14 R017 RGB/BGR Contract Negative Archive

- Temporarily removed `cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)` in `src/data_module.py` to test the official DFormer NYUDepthV2 BGR input contract.
- Full-train result: best val/mIoU `0.529090` at validation epoch `38`, below the R016 corrected baseline `0.541121`.
- Decision: do not keep this active code change. `src/data_module.py` is restored to the R016 RGB input path.
- Archived the failed code diff under `feiqi/failed_experiments_r014_plus_20260514/R017_rgb_bgr_contract.md`.
- No model structure, DFormerv2_S, pretrained loading, DepthEncoder, GatedFusion, SimpleFPNDecoder, optimizer, scheduler, batch size, epoch count, learning rate, worker count, early stopping, label mapping, depth normalization, split files, validation loader behavior, checkpoint artifacts, or TensorBoard event files were changed in the retained mainline.

## 2026-05-14 R016 Official Depth Normalization Contract

- Added `DFORMER_DEPTH_MEAN = 0.48`, `DFORMER_DEPTH_STD = 0.28`, and `normalize_nyu_depth_to_dformer()` in `src/data_module.py`.
- The data module now treats `depth` as an Albumentations `mask` target so RGB ImageNet Normalize does not touch it.
- Depth is manually normalized with official DFormer modal_x semantics: `raw / 255.0`, then `(x - 0.48) / 0.28`.
- This is an input/preprocessing contract alignment, not a model-structure innovation.
- Did not modify model structure, DFormerv2_S, pretrained loading, DepthEncoder, GatedFusion, SimpleFPNDecoder, optimizer, scheduler, batch size, epoch count, learning rate, worker count, early stopping, label mapping, split files, validation loader behavior, checkpoint artifacts, or TensorBoard event files.
- Verification before full train: `py_compile`, `train.py --help`, real-batch stats, and CUDA forward smoke passed. Real-batch depth range was `[-1.714286, 1.857143]`, matching the DFormer modal_x range.
- Full-train result: retry1 best val/mIoU `0.541121` at validation epoch `49`, last val/mIoU `0.527420`, with checkpoint `checkpoints/R016_depth_norm_official_baseline_retry1/dformerv2_mid_fusion-epoch=48-val_mIoU=0.5411.pt`.
- Decision: keep this official depth normalization contract as the current strongest corrected baseline. Cite DFormer for this preprocessing protocol; do not claim it as a proposed method.

## 2026-05-14 R015 Official Label/Ignore Contract Reset

- Added `map_nyu40_labels_to_train_ids()` in `src/data_module.py`.
- The data module now maps NYUDepthV2 raw labels using official DFormer semantics: raw `0` becomes `255` ignore, raw `1..40` becomes train ids `0..39`.
- Updated `src/utils/metrics.py` so `sanitize_labels()` no longer applies the old dynamic `min>=1` decrement. It now expects train-id labels and masks invalid values while preserving `255` ignore.
- This changes label/ignore semantics and therefore resets the baseline coordinate system. It is not a direct model improvement claim.
- Did not modify model structure, DFormerv2_S, pretrained loading, DepthEncoder, GatedFusion, SimpleFPNDecoder, optimizer, scheduler, batch size, epoch count, learning rate, worker count, early stopping, data augmentation, split files, or validation loader behavior.
- Verification before full train: `py_compile`, `train.py --help`, label unit mapping, and real-batch forward smoke passed. The real batch preserved class `39` and mapped ignore pixels to `255`.
- Full-train result: best val/mIoU `0.537398` at validation epoch `45`, last val/mIoU `0.499418`, with checkpoint `checkpoints/R015_label_ignore_official_baseline/dformerv2_mid_fusion-epoch=44-val_mIoU=0.5374.pt`.
- Decision: keep this label/ignore contract reset as the new official-label baseline. It is a data-label contract correction, not a model-structure change.

## 2026-05-14 Mainline Cleanup Before R014

- Cleaned the active registry before the next goal-driven experiment.
- Active `train.py` now exposes only `dformerv2_mid_fusion`, `dformerv2_tgga_c4only_beta002_aux003_detachsem_simplefpn_v1`, `dformerv2_geometry_primary_teacher`, `dformerv2_primkd_logit_only`, and the legacy `early` / `mid_fusion` entries.
- Removed TGGA c3/c4, no-aux c3/c4, and weak-c3 TGGA from the active registry because R004/R005 show c4 is safer and c3 remains risky.
- Archived the pre-cleanup TGGA c3 implementations and registry snapshot under `feiqi/failed_experiments_r001_r013_20260514/`.
- Moved inactive frequency modules `depth_fft_select.py`, `freq_enhance.py`, and `fft_hilo_enhance.py` from `src/models/` to `feiqi/failed_experiments_r001_r013_20260514/`.
- Kept all experiment evidence in docs, `miou_list`, reports, metrics, and experiment ledgers.
- Merged the R010/R012/R013 evidence ledgers without promoting R013 LMLP decoder code into the active model path.
- Fixed the active training entrypoint to pass `TQDMProgressBar` to the Trainer and to reconfigure stdout/stderr as UTF-8 on Windows; the progress bar remains enabled.
- No dataset, dataloader, augmentation, evaluation metric, mIoU calculation, optimizer, scheduler, batch size, epoch count, learning rate, worker count, checkpoint artifact, dataset, pretrained weight, or TensorBoard event file was changed.

## 2026-05-13 R013 LMLP Decoder Head

- Added `MLPProjection` and `LMLPDecoder` in `src/models/decoder.py`.
- Added `DFormerV2LMLPDecoderSegmentor` and `LitDFormerV2LMLPDecoder` in `src/models/mid_fusion.py`.
- Registered `dformerv2_lmlp_decoder` in `train.py`.
- Motivation: test a DFormer/SegFormer-style c2-c4 MLP decoder head as a clean alternative to SimpleFPN top-down additive fusion.
- Decoder behavior: project c2/c3/c4 fused features to `embed_dim=768`, upsample c3/c4 to c2 resolution, concatenate, apply `1x1` fuse + BN + ReLU + dropout + classifier, then upsample logits to the input image size.
- Did not modify DFormerv2_S, pretrained loading, DepthEncoder, GatedFusion, dataset/data module, mIoU/eval logic, optimizer, scheduler, batch size, epoch count, learning rate, early stopping, PMAD, or TGGA.
- Verification: `py_compile` passed for `train.py`, `src/models/decoder.py`, and `src/models/mid_fusion.py`; `train.py --help` lists `dformerv2_lmlp_decoder`.
- Real-batch smoke check on NYUDepthv2 with `batch_size=2` produced logits `(2, 40, 480, 640)`, CE loss `3.821265`, and optimizer `AdamW(lr=6e-5, weight_decay=0.01)`.
- Result note for `dformerv2_lmlp_decoder_run01`: best val/mIoU `0.517981`, only `+0.000584` above the clean baseline mean and below baseline mean + 1 std, with final val/mIoU `0.490231`; do not promote this decoder.

## 2026-05-13 R010 Training Entrypoint UTF-8/TQDM Fix

- No model structure was changed for R010.
- `train.py` now reconfigures `stdout` and `stderr` to UTF-8 with replacement so Lightning/Rich/TQDM teardown does not crash after a completed Windows run because of GBK encoding.
- `train.py` now passes the full callback list returned by `build_callbacks(...)` into the Trainer, preserving `TQDMProgressBar` while keeping the existing checkpoint and early-stop callbacks.
- No dataset, dataloader, augmentation, evaluation metric, mIoU calculation, optimizer, scheduler, batch size, epoch count, learning rate, worker count, checkpoint artifact, dataset, pretrained weight, or TensorBoard event file was changed.
- Result note for `dformerv2_primkd_logit_only_w015_t4_run06_retry1`: best val/mIoU `0.527469`, below the `0.53` goal but above the prior PMAD w0.15/T4 best single `0.524028`. This is a partial positive repeat, not a model-structure change or a success claim.

## 2026-05-13 R001-R005 Pause Cleanup / feiqi Archive

- Paused the goal-driven loop after R005 and cleaned active code back to the `main` registry/state.
- Archived R001-R003 failed implementation snapshots under `feiqi/experiments_20260513/`:
  - `primkd_failed_variants_r001_r003.py` for PMAD boundary/confidence KD and correct-and-entropy KD.
  - `decoder_with_r002_freqfpn.py` and `mid_fusion_with_r002_freqfpn.py` for the frequency-aware FPN decoder path.
  - `train_registry_r001_r005_before_cleanup.py` as the pre-cleanup registry snapshot.
- Active `train.py`, `src/models/decoder.py`, `src/models/mid_fusion.py`, and `src/models/primkd_lit.py` were restored to the `main` code state. This keeps the clean baseline, existing TGGA diagnostics, geometry-primary teacher, and PMAD logit-only active while removing R001-R003 failed variants from the active training path.
- No dataset, dataloader, augmentation, evaluation metric, mIoU calculation, optimizer, scheduler, batch size, epoch count, learning rate, worker count, checkpoint artifact, dataset, pretrained weight, or TensorBoard event file was changed.
- Result boundary: R004 c4-only remains the strongest loop signal (`0.522849`) but is below `0.53`; R005 weak-c3 (`0.518253`) does not improve R004.

## 2026-05-13 R003 Correct-and-Entropy-Selective PMAD KD

- Implemented `dformerv2_primkd_correct_entropy` as a separate model name.
- Added `LitDFormerV2PrimKDCorrectEntropy` in `src/models/primkd_lit.py`; it inherits the existing PMAD student/teacher setup from `LitDFormerV2PrimKD`.
- The student remains `DFormerv2_S + ResNet-18 DepthEncoder + GatedFusion + SimpleFPNDecoder`.
- The frozen teacher remains `DFormerV2GeometryPrimaryTeacherSegmentor`; teacher checkpoint loading and `export_state_dict()` behavior are unchanged.
- Training loss remains `CE(student, label) + kd_weight * KL(student, teacher)`, but KL is applied only to valid training pixels where the teacher prediction equals the sanitized label and normalized teacher entropy is `<= --kd_entropy_threshold`.
- The R003 threshold is `--kd_entropy_threshold 0.25`; selected-pixel KL is normalized by the number of valid pixels, so selecting fewer pixels reduces total KD strength rather than amplifying selected pixels.
- Added training logs: `train/kd_mask_ratio`, `train/kd_entropy_mean`, `train/kd_entropy_selected_mean`, `train/kd_teacher_valid_acc`, `train/kd_teacher_selected_acc`, and `train/kd_selected_kl`.
- Registered `dformerv2_primkd_correct_entropy` in `train.py` and added `--kd_entropy_threshold`; `dformerv2_primkd_logit_only` and `dformerv2_primkd_boundary_conf` remain unchanged.
- Fixed recipe remains external to the model: `batch_size=2`, `max_epochs=50`, `lr=6e-5`, `num_workers=4`, `early_stop_patience=30`, `loss_type=ce`.
- Verification: `py_compile` passed for `train.py` and `src/models/primkd_lit.py`; `train.py --help` lists the model and arg; a real one-batch smoke check gave `kd_mask_ratio=0.895539`, teacher trainable params `0`, and optimizer `AdamW(lr=6e-5, weight_decay=0.01)`.
- Result note for `dformerv2_primkd_correct_entropy_w015_t4_h025_run01`: best val/mIoU `0.516597`, below the clean baseline mean `0.517397` and below PMAD w0.15/T4 mean `0.520795`; do not promote this model.

## 2026-05-13 R002 Frequency-Aware FPN Decoder

- Implemented `dformerv2_freqfpn_decoder` as a separate model name.
- Added `FrequencyAwareTopDownFuse` and `FrequencyAwareFPNDecoder` in `src/models/decoder.py`.
- Added `DFormerV2FreqFPNDecoderSegmentor` and `LitDFormerV2FreqFPNDecoder` in `src/models/mid_fusion.py`.
- Registered `dformerv2_freqfpn_decoder` in `train.py`; the clean `dformerv2_mid_fusion` path and `SimpleFPNDecoder` remain unchanged.
- The model keeps the same encoder/fusion path as the clean baseline: `DFormerV2_S + ResNet-18 DepthEncoder + GatedFusion`.
- The only structural change is decoder top-down fusion: each top-down step uses low-frequency residual correction from 5x5 average low-pass features and high-frequency residual correction from 3x3 high-pass features.
- Correction projections are zero-initialized, so the initial `FrequencyAwareTopDownFuse` output is exactly `hr_feat + bilinear(lr_feat)` before training.
- This is not PMAD/KD, not an auxiliary loss, not boundary loss, not FADC, not an imported FreqFusion/CARAFE module, and not a backbone or GatedFusion change.
- Fixed recipe remains external to the model: `batch_size=2`, `max_epochs=50`, `lr=6e-5`, `num_workers=4`, `early_stop_patience=30`, `loss_type=ce`.
- Verification: `py_compile` passed for `train.py`, `src/models/decoder.py`, and `src/models/mid_fusion.py`; `train.py --help` lists the model; a tensor-level fuse check showed initial identity delta `0.0`; decoder smoke output shape was `(1, 40, 480, 640)`.
- Result note for `dformerv2_freqfpn_decoder_run01`: best val/mIoU `0.516915`, below the clean baseline mean `0.517397` and below PMAD w0.15/T4 mean `0.520795`; do not promote this decoder.

## 2026-05-12 R001 PMAD Boundary/Confidence-Selective KD

- Implemented `dformerv2_primkd_boundary_conf` as a separate model name.
- Added `LitDFormerV2PrimKDBoundaryConf` in `src/models/primkd_lit.py`; it inherits the existing PMAD logit-only student/teacher setup from `LitDFormerV2PrimKD`.
- The student remains `DFormerv2_S + ResNet-18 DepthEncoder + GatedFusion + SimpleFPNDecoder`.
- The frozen teacher remains `DFormerV2GeometryPrimaryTeacherSegmentor`; teacher checkpoint loading and `export_state_dict()` behavior are unchanged.
- Training loss remains `CE(student, label) + kd_weight * KL(student, teacher)`, but the KL term is now weighted by a deterministic trust mask from teacher confidence and ground-truth semantic boundary pixels.
- Boundary mask is computed only inside the training loss from sanitized labels; it does not change dataset split, loader behavior, augmentation, validation, test, metric, or inference.
- Registered `dformerv2_primkd_boundary_conf` in `train.py`; `dformerv2_primkd_logit_only` remains unchanged.
- Fixed recipe remains external to the model: `batch_size=2`, `max_epochs=50`, `lr=6e-5`, `num_workers=4`, `early_stop_patience=30`, `loss_type=ce`.
- Verification: `py_compile` passed for `train.py` and `src/models/primkd_lit.py`; a small tensor-level `selective_kl_loss` check returned finite loss and logging stats.
- Result note for `dformerv2_primkd_boundary_conf_w015_t4_run01`: best val/mIoU `0.511646`, below the clean baseline mean `0.517397` and below PMAD w0.15/T4 mean `0.520795`; do not promote this model.

## 2026-05-12 TGGA C3/C4 No-Aux Semantic-Gradient Diagnostic

- Implemented `dformerv2_tgga_c34_noaux_semgrad_beta002_simplefpn_v1`.
- Added `detach_semantic` to `TGGABlock`; default remains `True`, so the original `dformerv2_tgga_c34_beta002_aux003_detachsem_simplefpn_v2`, c4-only, and weak-c3 variants are unchanged.
- The no-aux diagnostic keeps TGGA on DFormerV2 c3/c4 with `beta_init=0.02`, `beta_max=0.1`.
- Removed auxiliary CE from the training objective: `L_total = CE(final_logits, label)`.
- Because the original semantic cue is detached and would become randomly fixed without aux CE, this diagnostic sets `detach_semantic=False` so the final segmentation loss can train the semantic cue through the gate path.
- Logged detached diagnostic auxiliary CE as `train/tgga_aux_ce_c3_diag` and `train/tgga_aux_ce_c4_diag`; these are not added to the loss.
- Purpose: separate aux-CE conflict from TGGA gate/residual instability after TGGA run01-run02 showed weak positive best mIoU but severe late collapse.
- Result note for run01: best val/mIoU `0.512152`, below clean baseline mean by `0.005245`; this is a negative diagnostic, not an improvement.

## 2026-05-12 TGGA Diagnostic Variants

- Implemented `dformerv2_tgga_c4only_beta002_aux003_detachsem_simplefpn_v1`.
- Implemented `dformerv2_tgga_c34_weakc3_beta001_c4beta002_aux003_detachsem_v1`.
- Added `gate_bias_init` to `TGGABlock` with default `-2.0`, preserving original TGGA behavior.
- c4-only applies TGGA only to DFormerV2 c4 and uses only c4 auxiliary CE.
- weak-c3 keeps TGGA on c3/c4 but uses weaker c3 settings: `beta_init=0.01`, `beta_max=0.05`, `gate_bias_init=-3.0`.
- No training result is implied by this code change.
- Original TGGA c34 model, `dformerv2_mid_fusion`, PMAD, and geometry-primary teacher remain unchanged.

## 2026-05-12 cleanup/archive-failed-modules

- Cleaned the default `train.py` active registry.
- Moved active TGGA implementation from `src/models/mid_fusion.py` to `src/models/tgga_adapter.py` without changing TGGA forward, loss, or logging behavior.
- Kept `dformerv2_mid_fusion` baseline active and slimmed `src/models/mid_fusion.py` back to baseline/legacy mid-fusion classes.
- Moved archived decoder blocks from `src/models/decoder.py` to `feiqi/models/archived_decoders.py`; active `src/models/decoder.py` now keeps only `SimpleFPNDecoder`.
- Kept `dformerv2_tgga_c34_beta002_aux003_detachsem_simplefpn_v2` active for run02/run03.
- Kept `dformerv2_geometry_primary_teacher` and `dformerv2_primkd_logit_only` active for PMAD.
- Removed DGBF/CGPC/SGBR/CGCD/FFT/depth FFT/context decoder models and parameters from the default `train.py` path.
- Added `docs/ACTIVE_STATUS.md`, `docs/cleanup_notes.md`, and updated `feiqi/README.md`.
- No experimental result changed.

## 2026-05-12 TGGA C3/C4 Minimal Structure Experiment

- Added `TGGABlock` in `src/models/mid_fusion.py` originally; after cleanup, active TGGA code lives in `src/models/tgga_adapter.py`.
- Added `DFormerV2TGGAC34Beta002Aux003DetachSemSimpleFPNV2Segmentor` and `LitDFormerV2TGGAC34Beta002Aux003DetachSemSimpleFPNV2`.
- Registered new model name `dformerv2_tgga_c34_beta002_aux003_detachsem_simplefpn_v2` in `train.py`.
- TGGA is inserted only after DFormerV2-S outputs `c3` and `c4`, before the existing external `DepthEncoder + GatedFusion` path.
- The baseline path is otherwise preserved: `DFormerv2_S + ResNet-18 DepthEncoder + GatedFusion + SimpleFPNDecoder`.
- TGGA gate cues are task-guided and detached from the auxiliary semantic head: uncertainty, semantic edge, RGB Sobel edge, depth Sobel edge, and RGB-depth edge difference.
- Residual branch uses depthwise convolution, GroupNorm, GELU, and pointwise convolution.
- Residual strength is bounded as `effective_beta = 0.1 * tanh(raw_beta)`, initialized to approximately `0.02`.
- Training loss for this model only is `CE(final_logits, label) + 0.03 * CE(aux_c3, label) + 0.03 * CE(aux_c4, label)`.
- The first version supports only `--loss_type ce`; it does not combine with DGBF, CGPC, PMAD/KD, LightHam, Dice, Lovasz, Focal, boundary CE, or deep supervision.
- Did not modify `DFormerV2MidFusionSegmentor.forward()`, DFormerV2-S encoder internals, `DepthEncoder`, `GatedFusion`, `SimpleFPNDecoder`, `BaseLitSeg`, data module, optimizer, learning rate, epoch count, or augmentation.
- Verification: `py_compile` passed for `train.py`, `src/models/mid_fusion.py`, and `src/models/decoder.py`; `train.py --help` lists the new model; dummy `240x320` forward/backward passed with final logits `[1,40,240,320]`, c3 aux logits `[1,40,15,20]`, c4 aux logits `[1,40,8,10]`, initial beta values near `0.02`, and both TGGA raw beta gradients present. Loading the old `LitDFormerV2MidFusion` state dict into the new Lit model with `strict=False` produced no unexpected keys and missing keys only under `model.tgga_c3.*` and `model.tgga_c4.*`.

## 2026-05-12 SGBR-Lite Decoder Minimal Implementation

- Added `SGBRBlock` and `SGBRFPNDecoder` in `src/models/decoder.py`.
- Added `DFormerV2SGBRDecoderSegmentor` and `LitDFormerV2SGBRDecoder` in `src/models/mid_fusion.py`.
- Registered new model name `dformerv2_sgbr_decoder` in `train.py`.
- Added CLI args: `--sgbr_aux_weight`, `--sgbr_beta_init`, and `--sgbr_beta_max`.
- The new model keeps `DFormerv2_S + DepthEncoder + GatedFusion` unchanged and replaces only the decoder with a semantic-guided boundary residual decoder.
- Decoder flow: fused features -> FPN `p1` -> auxiliary logits -> prediction uncertainty -> raw-depth Sobel edge -> gated residual refinement -> final logits.
- Residual strength is bounded as `beta = beta_max * sigmoid(raw_beta)`, with first planned setting `beta_init=0.05`, `beta_max=0.2`.
- Training loss for the first version is `CE(final_logits, label) + sgbr_aux_weight * CE(aux_logits, label)`.
- The first version supports only `--loss_type ce`; it does not combine with DGBF, CGPC, PMAD, teacher KD, FFT, CGCD, or feature-level contrastive losses.
- Did not modify `DFormerV2MidFusionSegmentor.forward()`, `GatedFusion`, DFormerV2 attention, `BaseLitSeg`, PrimKD, teacher models, DGBF, CGPC, dataset, or data module.

## 2026-05-12 Class Context Decoder Bounded Alpha Update

- Updated `ClassContextBlock` in `src/models/decoder.py` to use bounded residual strength for class-context refinement.
- Replaced the previous unconstrained trainable `alpha` with `alpha = alpha_max * sigmoid(raw_alpha)`.
- Added `alpha_max` support through `ClassContextFPNDecoder`, `DFormerV2ClassContextDecoderSegmentor`, `LitDFormerV2ClassContextDecoder`, and `train.py`.
- Added CLI arg `--class_context_alpha_max`, default `0.2`.
- Default initialization remains effectively unchanged: `class_context_alpha_init=0.1` and `class_context_alpha_max=0.2` initialize `raw_alpha=0`, so actual alpha starts at `0.1`.
- Motivation: `dformerv2_class_context_decoder_run01` reached best val/mIoU `0.519807`, but `context_alpha` increased from `0.111641` to `0.621710` and late validation became unstable. The bounded update directly limits the refinement branch strength.
- Did not modify `ContextFPNDecoder` / `PPMContextBlock`, `DFormerV2MidFusionSegmentor.forward()`, GatedFusion, DFormerV2 attention, BaseLitSeg, PMAD, teacher models, DGBF, CGPC, or the data module.

## 2026-05-11 Class-Guided Context Decoder Minimal Implementation

- Added `ClassContextBlock` and `ClassContextFPNDecoder` in `src/models/decoder.py`.
- Added `DFormerV2ClassContextDecoderSegmentor` and `LitDFormerV2ClassContextDecoder` in `src/models/mid_fusion.py`.
- Registered new model name `dformerv2_class_context_decoder` in `train.py`.
- Added CLI args: `--class_context_channels`, `--class_context_aux_weight`, and `--class_context_alpha_init`.
- The new model keeps `DFormerv2_S + DepthEncoder + GatedFusion` unchanged and replaces only the decoder with a lightweight class-context FPN decoder.
- Decoder flow: fused features -> FPN `p1` -> auxiliary logits -> class probability map -> class context prototypes -> pixel-to-class attention -> refined `p1` -> final logits.
- Training loss for the first version is `CE(final_logits, label) + class_context_aux_weight * CE(aux_logits, label)`.
- The first version supports only `--loss_type ce`; it does not combine with DGBF, CGPC, PMAD, FFT, PPM/ASPP, Lovasz, or feature-level contrastive losses.
- Did not modify `DFormerV2MidFusionSegmentor.forward()`, `GatedFusion`, DFormerV2 attention, `BaseLitSeg`, PrimKD, teacher models, DGBF, CGPC, decoder baselines, or the data module.
- Verification: `py_compile` passed for `train.py`, `src/models/decoder.py`, and `src/models/mid_fusion.py`; `train.py --help` lists the new model and args; `ClassContextFPNDecoder` and `ClassContextBlock` import; random-tensor forward returns final logits and auxiliary logits with expected shape.

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
