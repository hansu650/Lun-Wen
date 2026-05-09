# Experiment Log

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
- conclusion: negative result. Below baseline mean. Severe late collapse (best 0.5159 → last 0.443).

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
- conclusion: negative result. Below baseline mean. Late collapse (best 0.5145 → last 0.489).

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
- late collapse: epoch 46 `0.514461` → epoch 49 `0.498469`, drop `0.015992` in 3 epochs.
- evidence: `miou_list/dformerv2_cm_infonce_c34_lam005_t01_s256_run01.md`
- conclusion: negative single-run result. InfoNCE contrast loss converges (5.57→3.09, 45% drop), but the alignment signal does not improve validation mIoU above the clean repeated baseline mean. The run shows late collapse similar to other auxiliary loss experiments. The contrast loss is learning, but the learned cross-modal alignment is either not beneficial for segmentation or too weak to overcome the noise it introduces.
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
