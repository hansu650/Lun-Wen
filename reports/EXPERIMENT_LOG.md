# Experiment Loop Log

This is the top-level ledger for the Goal-Driven RGB-D mIoU loop.

Current phase: R026 official-style local init is negative; next step is primary-preserving residual depth injection.

Stage goal: `val/mIoU >= 0.53` under the active fixed recipe.

Final goal: `val/mIoU >= 0.56`.

Baseline reference:

- Active baseline: `dformerv2_mid_fusion`
- Clean 10-run mean best val/mIoU: `0.517397`
- Population std: `0.004901`
- Best single baseline run: `0.524425`
- Existing evidence: `final daima/miou_list/dformerv2_mid_fusion_gate_baseline_summary_run01_09_run10_retry.md`

## Entries

### 2026-05-15 R026 Result: Official-Style Local Init Negative

- `R026_official_init_local_modules_run01` completed 50 validation epochs with exit code `0`.
- Best val/mIoU: `0.507906` at validation epoch `33`; last val/mIoU: `0.499770`.
- Last-5 mean val/mIoU: `0.496476`; last-10 mean val/mIoU: `0.495483`; best-to-last drop: `0.008136`.
- Evidence: `final daima/miou_list/R026_official_init_local_modules_run01.md`.
- Checkpoint: `final daima/checkpoints/R026_official_init_local_modules_run01/dformerv2_official_init_local_modules-epoch=32-val_mIoU=0.5079.pt`.
- TensorBoard event: `final daima/checkpoints/R026_official_init_local_modules_run01/lightning_logs/version_0/events.out.tfevents.1778803189.Administrator.35684.0`.
- Decision: negative. Official-style init of the local random modules should not be continued.
- Next: test primary-preserving residual depth injection initialized as DFormerv2 identity.

### 2026-05-15 R025 Result: DepthEncoder BN Eval Peak Positive, Stability Negative

- `R025_depth_encoder_bn_eval_run01` completed 50 validation epochs with exit code `0`.
- Best val/mIoU: `0.532572` at validation epoch `47`; last val/mIoU: `0.496030`.
- Last-5 mean val/mIoU: `0.520333`; last-10 mean val/mIoU: `0.517969`; best-to-last drop: `0.036541`.
- Evidence: `final daima/miou_list/R025_depth_encoder_bn_eval_run01.md`.
- Checkpoint: `final daima/checkpoints/R025_depth_encoder_bn_eval_run01/dformerv2_depth_encoder_bn_eval-epoch=46-val_mIoU=0.5326.pt`.
- TensorBoard event: `final daima/checkpoints/R025_depth_encoder_bn_eval_run01/lightning_logs/version_0/events.out.tfevents.1778798018.Administrator.26772.0`.
- Decision: partial-positive peak but unstable. Do not use BN eval as the next base.
- Next: test official-style initialization of only local random modules (`GatedFusion` + `SimpleFPNDecoder`).

### 2026-05-15 R024 Result: Raw DFormerv2-S + Ham Stable Positive Below Corrected Baseline

- `R024_geometry_primary_ham_decoder_run01` completed 50 validation epochs with exit code `0`.
- Best val/mIoU: `0.530186` at validation epoch `45`; last val/mIoU: `0.529383`.
- Last-5 mean val/mIoU: `0.521843`; last-10 mean val/mIoU: `0.522327`; best-to-last drop: `0.000803`.
- Evidence: `final daima/miou_list/R024_geometry_primary_ham_decoder_run01.md`.
- Checkpoint: `final daima/checkpoints/R024_geometry_primary_ham_decoder_run01/dformerv2_geometry_primary_ham_decoder-epoch=44-val_mIoU=0.5302.pt`.
- TensorBoard event: `final daima/checkpoints/R024_geometry_primary_ham_decoder_run01/lightning_logs/version_0/events.out.tfevents.1778793121.Administrator.20368.0`.
- Decision: stable positive diagnostic above `0.53`, but below R022 `0.534332` and R016 `0.541121`. Raw DFormerv2-S + Ham does not replace the local external-fusion path.
- Next: stop Ham micro-fixes and test a single stability hypothesis on the stronger corrected mid-fusion path.

### 2026-05-15 R023 Result: Corrected Geometry-Primary Teacher Negative Gate

- `R023_geometry_primary_teacher_corrected_contract_run01` completed 50 validation epochs with exit code `0`.
- Best val/mIoU: `0.524498` at validation epoch `43`; last val/mIoU: `0.507023`.
- Last-5 mean val/mIoU: `0.510467`; last-10 mean val/mIoU: `0.512531`; best-to-last drop: `0.017475`.
- Evidence: `final daima/miou_list/R023_geometry_primary_teacher_corrected_contract_run01.md`.
- Checkpoint: `final daima/checkpoints/R023_geometry_primary_teacher_corrected_contract_run01/dformerv2_geometry_primary_teacher-epoch=42-val_mIoU=0.5245.pt`.
- TensorBoard event: `final daima/checkpoints/R023_geometry_primary_teacher_corrected_contract_run01/lightning_logs/version_0/events.out.tfevents.1778787920.Administrator.37368.0`.
- Decision: negative teacher gate. The refreshed teacher is below R016 `0.541121` and below `0.53`, so corrected PMAD from this checkpoint is low value.
- Next: run raw `DFormerv2_S + OfficialHamDecoder` without external DepthEncoder/GatedFusion to isolate the remaining structure-contract gap.

### 2026-05-15 R022 Result: Ham Dropout Parity Partial Positive

- `R022_ham_dropout_parity_run01` completed 50 validation epochs with exit code `0`.
- Best val/mIoU: `0.534332` at validation epoch `50`; last val/mIoU: `0.534332`.
- Last-5 mean val/mIoU: `0.527687`; last-10 mean val/mIoU: `0.512629`; best-to-last drop: `0.000000`.
- Evidence: `final daima/miou_list/R022_ham_dropout_parity_run01.md`.
- Checkpoint: `final daima/checkpoints/R022_ham_dropout_parity_run01/dformerv2_ham_decoder-epoch=49-val_mIoU=0.5343.pt`.
- TensorBoard event: `final daima/checkpoints/R022_ham_dropout_parity_run01/lightning_logs/version_0/events.out.tfevents.1778782418.Administrator.36360.0`.
- Decision: partial-positive parity fix. It improves R021 and R020 but remains below R016 `0.541121`; do not claim it as the corrected baseline.
- Next: run corrected-contract geometry-primary teacher refresh before any further PMAD student experiment.

### 2026-05-15 R021 Result: LightHam-Like Decoder Negative

- `R021_official_ham_decoder_parity_run01` completed 50 validation epochs with exit code `0`.
- Best val/mIoU: `0.527353` at validation epoch `39`; last val/mIoU: `0.501377`.
- Last-5 mean val/mIoU: `0.503158`; last-10 mean val/mIoU: `0.506140`; best-to-last drop: `0.025976`.
- Evidence: `final daima/miou_list/R021_official_ham_decoder_parity_run01.md`.
- Checkpoint: `final daima/checkpoints/R021_official_ham_decoder_parity_run01/dformerv2_ham_decoder-epoch=38-val_mIoU=0.5274.pt`.
- TensorBoard event: `final daima/checkpoints/R021_official_ham_decoder_parity_run01/lightning_logs/version_0/events.out.tfevents.1778777116.Administrator.12456.0`.
- Decision: negative relative to R016 `0.541121`. The implementation is LightHam-like, not strict official Ham parity, because it omits official `Dropout2d(0.1)` before classification.
- Next: run one minimal R022 dropout parity fix; if it remains below R016/R020, stop Ham decoder work and move to corrected-contract PMAD teacher refresh.

### 2026-05-15 R020 Result: Branch-Specific Depth Blend Adapter Stabilization Signal

- `R020_branch_depth_blend_adapter_run01` completed 50 validation epochs with exit code `0`.
- Best val/mIoU: `0.532924` at validation epoch `41`; last val/mIoU: `0.503238`.
- Last-5 mean val/mIoU: `0.520456`; last-10 mean val/mIoU: `0.516804`; best-to-last drop: `0.029686`.
- Alpha first/last: `0.050022` / `0.051455`.
- Evidence: `final daima/miou_list/R020_branch_depth_blend_adapter_run01.md`.
- Checkpoint: `final daima/checkpoints/R020_branch_depth_blend_adapter_run01/dformerv2_branch_depth_blend_adapter-epoch=40-val_mIoU=0.5329.pt`.
- TensorBoard event: `final daima/checkpoints/R020_branch_depth_blend_adapter_run01/lightning_logs/version_0/events.out.tfevents.1778771221.Administrator.41764.0`.
- Decision: keep as partial-positive stabilization evidence, not as a new corrected baseline. It is below R016 `0.541121`.
- Next: target late stability directly, use a richer branch-specific adapter, or run official Ham parity audit for reference-gap diagnosis.

### 2026-05-14 R019 Result: Branch-Specific Depth Adapter Partial Positive, Unstable

- `R019_branch_depth_adapter_run01` completed 50 validation epochs with exit code `0`.
- Best val/mIoU: `0.532539` at validation epoch `46`; last val/mIoU: `0.495229`.
- Last-5 mean val/mIoU: `0.509575`; last-10 mean val/mIoU: `0.518038`; best-to-last drop: `0.037311`.
- Evidence: `final daima/miou_list/R019_branch_depth_adapter_run01.md`.
- Checkpoint: `final daima/checkpoints/R019_branch_depth_adapter_run01/dformerv2_branch_depth_adapter-epoch=45-val_mIoU=0.5325.pt`.
- TensorBoard event: `final daima/checkpoints/R019_branch_depth_adapter_run01/lightning_logs/version_0/events.out.tfevents.1778765914.Administrator.27112.0`.
- Decision: keep as partial-positive original-method evidence, not as a new corrected baseline. It is below R016 `0.541121` and has severe late collapse.
- Next: stabilize the branch-specific depth path or run official Ham parity audit; do not blindly repeat the exact R019 setting.

### 2026-05-14 R018 Result: Official DropPath 0.25 Contract Negative

- `R018_dformerv2_mid_fusion_dpr025_retry1` completed 50 validation epochs with exit code `0`.
- Best val/mIoU: `0.526282` at validation epoch `46`; last val/mIoU: `0.522893`.
- Last-5 mean val/mIoU: `0.512694`; last-10 mean val/mIoU: `0.513363`; best-to-last drop: `0.003389`.
- Evidence: `final daima/miou_list/R018_dformerv2_mid_fusion_dpr025_retry1.md`.
- Checkpoint: `final daima/checkpoints/R018_dformerv2_mid_fusion_dpr025_retry1/dformerv2_mid_fusion_dpr025-epoch=45-val_mIoU=0.5263.pt`.
- TensorBoard event: `final daima/checkpoints/R018_dformerv2_mid_fusion_dpr025_retry1/lightning_logs/version_0/events.out.tfevents.1778760450.Administrator.7836.0`.
- Decision: reject the `drop_path_rate=0.25` contract gate for the local mid-fusion adaptation. It is `-0.014839` below R016 `0.541121`.
- Code handling: the failed active model-entry diff was archived under `final daima/feiqi/failed_experiments_r014_plus_20260514/R018_droppath025_contract.md`, and `src/models/mid_fusion.py` / `train.py` are restored to the corrected mainline state.
- Process note: the first foreground launch reached 42 validation epochs but hung after stdout/progress pipe timeout; retry1 is the valid full-train evidence.
- Next: keep R016 as corrected baseline. If still pursuing `0.56`, choose between official Ham parity audit, corrected-contract PMAD teacher refresh, or branch-specific depth input adaptation.

### 2026-05-14 R017 Result: Official RGB/BGR Contract Negative

- `R017_rgb_bgr_official_contract` completed 50 validation epochs with exit code `0`.
- Best val/mIoU: `0.529090` at validation epoch `38`; last val/mIoU: `0.523078`.
- Last-5 mean val/mIoU: `0.494251`; last-10 mean val/mIoU: `0.506949`; best-to-last drop: `0.006011`.
- Evidence: `final daima/miou_list/R017_rgb_bgr_official_contract.md`.
- Checkpoint: `final daima/checkpoints/R017_rgb_bgr_official_contract/dformerv2_mid_fusion-epoch=37-val_mIoU=0.5291.pt`.
- TensorBoard event: `final daima/checkpoints/R017_rgb_bgr_official_contract/lightning_logs/version_0/events.out.tfevents.1778750750.Administrator.38244.0`.
- Decision: reject the BGR input contract for this local adaptation. It is `-0.012031` below R016 `0.541121`.
- Code handling: the failed active `BGR2RGB` removal was archived under `final daima/feiqi/failed_experiments_r014_plus_20260514/R017_rgb_bgr_contract.md`, and `src/data_module.py` is restored to the R016 RGB path.
- Next: keep R016 as corrected baseline and test DFormerv2-S official `drop_path_rate=0.25`.

### 2026-05-14 R017 Approval: Official RGB/BGR Channel Contract

- Branch: `exp/R017-rgb-bgr-contract-v1`.
- Hypothesis: after official label and depth contracts, RGB channel order should match official DFormer NYUDepthV2 input behavior.
- Planned model: `dformerv2_mid_fusion`.
- Planned run: `R017_rgb_bgr_official_contract`; checkpoint directory `checkpoints/R017_rgb_bgr_official_contract`.
- Code scope: `final daima/src/data_module.py` only.
- Official source: DFormer `RGBXDataset.py` uses `rgb_mode = "BGR"` for non-SUNRGBD datasets and does not convert BGR to RGB in that mode.
- Fixed recipe remains `batch_size=2`, `max_epochs=50`, `lr=6e-5`, `num_workers=4`, `early_stop_patience=30`, `loss_type=ce`, no scheduler.
- Claim boundary: this is official input-contract alignment, not a novel method.
- Status: approved for smoke test and one full train.

### 2026-05-14 R016 Result: Official Depth Normalization Raises Baseline To 0.541121

- `R016_depth_norm_official_baseline_retry1` completed 50 validation epochs with exit code `0`.
- Best val/mIoU: `0.541121` at validation epoch `49`; last val/mIoU: `0.527420`.
- Last-5 mean val/mIoU: `0.535500`; last-10 mean val/mIoU: `0.524063`; best-to-last drop: `0.013702`.
- Evidence: `final daima/miou_list/R016_depth_norm_official_baseline_retry1.md`.
- Checkpoint: `final daima/checkpoints/R016_depth_norm_official_baseline_retry1/dformerv2_mid_fusion-epoch=48-val_mIoU=0.5411.pt`.
- TensorBoard event: `final daima/checkpoints/R016_depth_norm_official_baseline_retry1/lightning_logs/version_0/events.out.tfevents.1778745208.Administrator.36016.0`.
- Decision: positive official-contract alignment; this is the strongest current run and improves over R015 by `+0.003723`.
- Claim boundary: this is not a novel method. It is DFormer official modal_x/depth preprocessing alignment and should be cited as such.
- Process note: the first R016 launch was interrupted after 47 validation epochs by a Windows/Intel `forrtl` window-close event because the command window was closed. Retry1 is the valid full-train evidence.
- Next: continue toward `0.56`, with `RGB/BGR` input contract as the next highest-priority contract gate unless a stronger evidence-backed candidate appears.

### 2026-05-14 R016 Approval: Official Depth Normalization Contract

- Branch: `exp/R016-depth-norm-contract-v1`.
- Hypothesis: after R015 aligned the official NYU label/ignore contract, depth input should also follow the official DFormer modal_x normalization contract.
- Planned model: `dformerv2_mid_fusion`.
- Planned run: `R016_depth_norm_official_baseline`; checkpoint directory `checkpoints/R016_depth_norm_official_baseline`.
- Code scope: `final daima/src/data_module.py` only.
- Official source: DFormer normalizes `modal_x` with mean `[0.48, 0.48, 0.48]` and std `[0.28, 0.28, 0.28]`, using raw `/255.0` first.
- Fixed recipe remains `batch_size=2`, `max_epochs=50`, `lr=6e-5`, `num_workers=4`, `early_stop_patience=30`, `loss_type=ce`, no scheduler.
- Smoke status: `py_compile`, `train.py --help`, real-batch stats, and CUDA forward sanity passed.
- Status: full train running; no result claim yet.

### 2026-05-14 R015 Result: Official-Label Baseline Reaches 0.53 Stage Target

- `R015_label_ignore_official_baseline` completed 50 validation epochs with exit code `0`.
- Best val/mIoU: `0.537398` at validation epoch `45`; last val/mIoU: `0.499418`.
- Last-5 mean val/mIoU: `0.520010`; last-10 mean val/mIoU: `0.520691`; best-to-last drop: `0.037981`.
- Evidence: `final daima/miou_list/R015_label_ignore_official_baseline.md`.
- Checkpoint: `final daima/checkpoints/R015_label_ignore_official_baseline/dformerv2_mid_fusion-epoch=44-val_mIoU=0.5374.pt`.
- TensorBoard event: `final daima/checkpoints/R015_label_ignore_official_baseline/lightning_logs/version_0/events.out.tfevents.1778734783.Administrator.15996.0`.
- Decision: the fixed-recipe `0.53` stage target is met under the official-label contract. This becomes the new baseline coordinate system; do not mix it as a direct improvement claim against old-contract results.
- Next: continue toward `0.56` with official contract alignment, starting with depth normalization as a single isolated hypothesis.

### 2026-05-14 Cleanup: nyu056 Mainline

- Branch: `cleanup/nyu056-mainline`.
- Purpose: merge useful R010/R012/R013 evidence into the mainline while removing failed experiment code from active training choices.
- Active code kept: clean baseline, PMAD logit-only, geometry-primary teacher, and TGGA c4-only reuse unit.
- Archived code: TGGA c3/c4 and weak-c3 snapshot, pre-cleanup registry snapshot, depth FFT select, FFT frequency enhance, and FFT HiLo under `final daima/feiqi/failed_experiments_r001_r013_20260514/`.
- Evidence retained: experiment docs, `miou_list`, reports, metrics, and experiment ledgers.
- Entry point fix: keep `TQDMProgressBar` enabled and configure stdout/stderr to UTF-8 for Windows/Rich output.
- Next experiment: `R014_pmad_logit015_t4_tgga_c4only` on `exp/R014-pmad-tgga-c4-v1`.

### 2026-05-14 R015 Approval: Label/Ignore Official Contract Baseline Reset

- Branch: `exp/R015-label-ignore-contract-v1`.
- Hypothesis: official DFormer NYU label mapping `0 -> 255 ignore`, `1..40 -> 0..39` is required before judging the gap to official DFormerv2-S results.
- Planned model: `dformerv2_mid_fusion`.
- Planned run: `R015_label_ignore_official_baseline`; checkpoint directory `checkpoints/R015_label_ignore_official_baseline`.
- Code scope: `final daima/src/data_module.py` and `final daima/src/utils/metrics.py`.
- Contract boundary: this is a baseline reset, not a direct old-baseline improvement claim.
- Fixed recipe remains `batch_size=2`, `max_epochs=50`, `lr=6e-5`, `num_workers=4`, `early_stop_patience=30`, `loss_type=ce`, no scheduler.
- Forbidden-change check: no split-file, augmentation, loader-size, model, optimizer, scheduler, epoch, batch, lr, worker, checkpoint-artifact, dataset, pretrained-weight, or TensorBoard-log change is approved.
- Status: approved for smoke test and one full train after reviewer/reproducer checks.

### 2026-05-12 R001 Approval: Boundary/Confidence-Selective PMAD KD

- Approved one experiment on branch `exp/R001-pmad-selective-kd-v1`.
- Hypothesis: boundary/confidence-selective PMAD logit KD can keep the positive `dformerv2_primkd_logit_only` w0.15/T4 signal while reducing harmful teacher transfer in uncertain or non-boundary pixels.
- Reason: PMAD logit-only w0.15/T4 is the strongest repeat-backed current direction with five-run mean best val/mIoU `0.520795`; broad fusion replacements, frequency/auxiliary losses, TGGA diagnostics, and decoder/context changes are negative or unstable.
- Planned model name: `dformerv2_primkd_boundary_conf`.
- Planned run name: `w015_t4_run01`; checkpoint directory `checkpoints/dformerv2_primkd_boundary_conf_w015_t4_run01`.
- Fixed recipe remains `batch_size=2`, `max_epochs=50`, `lr=6e-5`, `num_workers=4`, `early_stop_patience=30`, `loss_type=ce`, no scheduler.
- Forbidden-change check: no dataset split, dataloader, augmentation, validation, metric, mIoU, optimizer, scheduler, epoch, batch, lr, worker, checkpoint-artifact, dataset, pretrained-weight, or TensorBoard-log change is approved.
- Status: approved for implementation and one full train after dry-check.

### 2026-05-13 R001 Result: Negative

- `dformerv2_primkd_boundary_conf_w015_t4_run01` completed 50 validation epochs.
- best val/mIoU: `0.511646` at epoch `50`; last val/mIoU: `0.511646`.
- Result is below the clean 10-run baseline mean `0.517397` and below PMAD logit-only w0.15/T4 five-run mean `0.520795`.
- Evidence: `final daima/miou_list/dformerv2_primkd_boundary_conf_w015_t4_run01.md`.
- Report: `reports/R001-pmad-selective-kd-v1.md`.
- Diagnostic: final `train/kd_mask_ratio` is `0.998182`, so confidence threshold `0.40` was effectively non-selective.
- Audit: code review `approved`; reproducer/report audit `audit_passed_no_rerun`.
- Decision: reject this exact setting; continue the loop after audit with the next highest-decision-value candidate.

### 2026-05-13 R002 Approval: Frequency-Aware FPN Decoder

- Approved one experiment on branch `exp/R002-freqfpn-decoder-v1`.
- Hypothesis: frequency-aware top-down decoder fusion can reduce boundary displacement and feature-frequency mismatch in `SimpleFPNDecoder`.
- Reason: R001 PMAD refinement failed, while the literature/idea scan ranked FreqFusion-style decoder fusion as the strongest remaining non-overlapping hypothesis. This tests decoder top-down fusion directly rather than repeating auxiliary loss, KD, or pre-fusion FFT directions.
- Planned model name: `dformerv2_freqfpn_decoder`.
- Planned run name: `run01`; checkpoint directory `checkpoints/dformerv2_freqfpn_decoder_run01`.
- Fixed recipe remains `batch_size=2`, `max_epochs=50`, `lr=6e-5`, `num_workers=4`, `early_stop_patience=30`, `loss_type=ce`, no scheduler.
- Forbidden-change check: no dataset split, dataloader, augmentation, validation, metric, mIoU, optimizer, scheduler, epoch, batch, lr, worker, checkpoint-artifact, dataset, pretrained-weight, or TensorBoard-log change is approved.
- Status: approved for implementation and one full train after dry-check.

### 2026-05-13 R002 Result: Negative

- `dformerv2_freqfpn_decoder_run01` completed 50 validation epochs.
- best val/mIoU: `0.516915` at epoch `44`; last val/mIoU: `0.486524`.
- Result is slightly below the clean 10-run baseline mean `0.517397` and below PMAD logit-only w0.15/T4 five-run mean `0.520795`.
- Evidence: `final daima/miou_list/dformerv2_freqfpn_decoder_run01.md`.
- Report: `reports/R002-freqfpn-decoder-v1.md`.
- Diagnostic: late instability after epoch 47; final val/mIoU drops to `0.486524`.
- Audit: code review `approved`; reproducer/report audit `audit_passed_no_rerun`.
- Decision: reject this exact decoder; continue the loop after audit with the next highest-decision-value candidate.

### 2026-05-13 R003 Approval: Correct-and-Entropy-Selective PMAD KD

- Approved one experiment on branch `exp/R003-pmad-correct-entropy-kd-v1`.
- Hypothesis: PMAD logit KD can avoid harmful teacher transfer by distilling only pixels where the frozen teacher is both label-correct and low-entropy during training.
- Reason: PMAD w0.15/T4 remains the best repeat-backed direction with five-run mean best val/mIoU `0.520795`; R001 only falsified a weak selector because `kd_mask_ratio` stayed near `0.998`, while R002 decoder frequency fusion did not improve over baseline.
- Planned model name: `dformerv2_primkd_correct_entropy`.
- Planned run name: `w015_t4_h025_run01`; checkpoint directory `checkpoints/dformerv2_primkd_correct_entropy_w015_t4_h025_run01`.
- Fixed recipe remains `batch_size=2`, `max_epochs=50`, `lr=6e-5`, `num_workers=4`, `early_stop_patience=30`, `loss_type=ce`, no scheduler.
- Selector settings: teacher argmax must match the sanitized training label and normalized teacher entropy must be `<=0.25`; no boundary override or boundary boost.
- Smoke check on one real train batch: `kd_mask_ratio=0.895539`, `kd_entropy_mean=0.054600`, `kd_teacher_valid_acc=0.922844`, teacher trainable params `0`, optimizer `AdamW(lr=6e-5, weight_decay=0.01)`.
- Forbidden-change check: no dataset split, dataloader, augmentation, validation, metric, mIoU, optimizer, scheduler, epoch, batch, lr, worker, checkpoint-artifact, dataset, pretrained-weight, or TensorBoard-log change is approved.
- Status: approved for one full train after dry-check and smoke-check.

### 2026-05-13 R003 Result: Negative

- `dformerv2_primkd_correct_entropy_w015_t4_h025_run01` completed 50 validation epochs.
- best val/mIoU: `0.516597` at epoch `50`; last val/mIoU: `0.516597`.
- Result is slightly below the clean 10-run baseline mean `0.517397` and below PMAD logit-only w0.15/T4 five-run mean `0.520795`.
- Evidence: `final daima/miou_list/dformerv2_primkd_correct_entropy_w015_t4_h025_run01.md`.
- Report: `reports/R003-pmad-correct-entropy-kd-v1.md`.
- Diagnostic: final `kd_mask_ratio=0.910636`, final `kd_teacher_selected_acc=1.000000`; the selector was meaningful but did not recover the original PMAD signal.
- Audit: code review `approved`; reproducer/report audit `audit_passed_no_rerun`.
- Decision: reject this exact PMAD filtering setting; continue the loop after audit with the next highest-decision-value candidate.

### 2026-05-13 R004 Approval: TGGA C4-Only Diagnostic

- Approved one experiment on branch `exp/R004-tgga-c4only-diagnostic-v1`.
- Hypothesis: TGGA c4-only can retain high-level semantic/geometry calibration while removing the c3 high-resolution gate/residual path that may cause c3/c4 TGGA late collapse.
- Reason: PMAD filtering failed in R001 and R003; R002 decoder frequency fusion failed; original TGGA c3/c4 had the strongest remaining weak signal but collapsed late, and the already implemented c4-only diagnostic cleanly tests whether c3 is the unsafe component.
- Planned model name: `dformerv2_tgga_c4only_beta002_aux003_detachsem_simplefpn_v1`.
- Planned run name: `run01`; checkpoint directory `checkpoints/dformerv2_tgga_c4only_beta002_aux003_detachsem_simplefpn_v1_run01`.
- Fixed recipe remains `batch_size=2`, `max_epochs=50`, `lr=6e-5`, `num_workers=4`, `early_stop_patience=30`, `loss_type=ce`, no scheduler.
- No code change is approved for this round; the model is already implemented and registered.
- Forbidden-change check: no dataset split, dataloader, augmentation, validation, metric, mIoU, optimizer, scheduler, epoch, batch, lr, worker, checkpoint-artifact, dataset, pretrained-weight, TensorBoard-log, or code change is approved.
- Status: approved for one full train after dry-check.

### 2026-05-13 R004 Result: Partial Positive, Below Goal

- `dformerv2_tgga_c4only_beta002_aux003_detachsem_simplefpn_v1_run01` completed 50 validation epochs.
- best val/mIoU: `0.522849` at epoch `42`; last val/mIoU: `0.509320`.
- Result is above the clean 10-run baseline mean `0.517397`, above baseline mean + 1 std `0.522298` by `0.000551`, and above PMAD logit-only w0.15/T4 five-run mean `0.520795`, but below the `0.53` goal and below the best single clean baseline run `0.524425`.
- Evidence: `final daima/miou_list/dformerv2_tgga_c4only_beta002_aux003_detachsem_simplefpn_v1_run01.md`.
- Report: `reports/R004-tgga-c4only-diagnostic-v1.md`.
- Diagnostic: c4 gate remains conservative but opens gradually; final `gate_c4_mean=0.130742`, `gate_c4_std=0.014394`, `tgga_beta_c4=0.022874`.
- Late-curve caveat: final val/mIoU is `0.013529` below the best epoch.
- Audit: code review `approved_on_staged_diff`; reproducer/report audit `audit_passed_no_rerun`.
- Decision: keep as the strongest current diagnostic signal, reject it as a goal-completing method, and continue the loop with the next highest-decision-value candidate.

### 2026-05-13 R005 Approval: TGGA Weak-C3 + C4 Diagnostic

- Approved one experiment on branch `exp/R005-tgga-weakc3-v1`.
- Hypothesis: TGGA weak-c3 plus c4 can retain the R004 c4-only calibration signal while reintroducing a conservative c3 detail path without the original c3/c4 high-resolution gate instability.
- Reason: R004 is the strongest loop result so far (`0.522849`) but c4 alone is still below `0.53` and drops late; the already implemented weak-c3 variant is the narrowest follow-up that tests whether a reduced c3 gate can add detail without repeating the original c3 instability.
- Planned model name: `dformerv2_tgga_c34_weakc3_beta001_c4beta002_aux003_detachsem_v1`.
- Planned run name: `run01`; checkpoint directory `checkpoints/dformerv2_tgga_c34_weakc3_beta001_c4beta002_aux003_detachsem_v1_run01`.
- Fixed recipe remains `batch_size=2`, `max_epochs=50`, `lr=6e-5`, `num_workers=4`, `early_stop_patience=30`, `loss_type=ce`, no scheduler.
- No code change is approved for this round; the model is already implemented and registered.
- Forbidden-change check: no dataset split, dataloader, augmentation, validation, metric, mIoU, optimizer, scheduler, epoch, batch, lr, worker, checkpoint-artifact, dataset, pretrained-weight, TensorBoard-log, or code change is approved.
- Status: approved for one full train after dry-check.

### 2026-05-13 R005 Result: Weak Positive, Below Goal

- `dformerv2_tgga_c34_weakc3_beta001_c4beta002_aux003_detachsem_v1_run01` completed 50 validation epochs.
- best val/mIoU: `0.518253` at epoch `43`; last val/mIoU: `0.514908`.
- Result is only `+0.000856` above the clean 10-run baseline mean `0.517397`, below baseline mean + 1 std `0.522298`, below PMAD logit-only w0.15/T4 five-run mean `0.520795`, below R004 c4-only `0.522849`, and below the `0.53` goal.
- Evidence: `final daima/miou_list/dformerv2_tgga_c34_weakc3_beta001_c4beta002_aux003_detachsem_v1_run01.md`.
- Report: `reports/R005-tgga-weakc3-v1.md`.
- Diagnostic: final c3 gate still opens substantially (`gate_c3_mean=0.293138`, `gate_c3_std=0.331622`), while c4 remains conservative (`gate_c4_mean=0.131140`, `gate_c4_std=0.012837`).
- Process note: `Trainer.fit` reached `max_epochs=50`; after metric/checkpoint writing, Rich progress teardown raised a Windows GBK `UnicodeEncodeError`.
- Audit: code review `approved_current_diff`; reproducer/report audit `audit_passed_no_rerun`.
- Decision: reject weak-c3 as a goal path, keep R004 c4-only as the better TGGA diagnostic, and pause the experiment loop for user/external review.

### 2026-05-13 Pause Cleanup: Archive Failed Loop Code

- Paused the experiment loop after R005 per user instruction; no new training is approved after this point.
- Useful code status: TGGA diagnostic variants, including c4-only and weak-c3, already exist in the `main` active code state.
- Archived failed R001-R003 implementation snapshots under `final daima/feiqi/experiments_20260513/`.
- Restored active `train.py`, `src/models/decoder.py`, `src/models/mid_fusion.py`, and `src/models/primkd_lit.py` to the `main` code state so failed R001-R003 variants are no longer active registry entries on the pause branch.
- Kept all R001-R005 reports, mIoU details, metrics, and coordination records for evidence and review.
- Did not stage or modify checkpoint files, TensorBoard event files, datasets, or pretrained weights.

### 2026-05-13 R010 Approval: PMAD Logit-Only w0.15/T4 Repeat

- Approved one repeat experiment on branch `exp/R010-primkd-logit-w015-repeat-v1`.
- Hypothesis: PMAD logit-only w0.15/T4 remains the strongest repeat-backed positive KD direction and may produce a high-tail run while adding one more stability sample.
- Reason: prior PMAD logit-only w0.15/T4 five-run mean best was `0.520795`, stronger than recent R001-R005 refinements except the single R004 TGGA diagnostic.
- Planned model name: `dformerv2_primkd_logit_only`.
- Planned run name: `w015_t4_run06`; completed evidence uses retry checkpoint directory `checkpoints/dformerv2_primkd_logit_only_w015_t4_run06_retry1` because the first launch failed before validation.
- Fixed recipe remains `batch_size=2`, `max_epochs=50`, `lr=6e-5`, `num_workers=4`, `early_stop_patience=30`, `loss_type=ce`, no scheduler.
- Forbidden-change check: no dataset split, dataloader, augmentation, validation, metric, mIoU, optimizer, scheduler, epoch, batch, lr, worker, checkpoint-artifact, dataset, pretrained-weight, TensorBoard-log, backbone, or encoder change is approved.

### 2026-05-13 R010 Result: Partial Positive, Below Goal

- `dformerv2_primkd_logit_only_w015_t4_run06_retry1` completed 50 validation epochs.
- best val/mIoU: `0.527469` at epoch `49`; last val/mIoU: `0.526316`.
- Result is above the clean 10-run baseline mean `0.517397`, above baseline mean + 1 std `0.522298`, above the clean baseline best single `0.524425`, and above the prior PMAD w0.15/T4 best single `0.524028`, but below the `0.53` goal.
- Evidence: `final daima/miou_list/dformerv2_primkd_logit_only_w015_t4_run06_retry1.md`.
- Report: `reports/R010-primkd-logit-w015-repeat-v1.md`.
- Diagnostic: updated PMAD w0.15/T4 six-run mean best is `0.521907`; this supports PMAD as a durable marginal-positive direction, not a solved result.
- Process note: first non-retry launch stopped during epoch 0 with `forrtl error (200): program aborting due to window-CLOSE event`; no `val/mIoU` was recorded and it is excluded.
- Audit: static code review `approved_current_diff`; evidence/report audit `PASS`; reproducer audit `audit_passed_no_rerun`.
- Decision: keep as partial positive evidence, reject as goal-completing method, and continue the loop with a distinct next hypothesis.

### 2026-05-13 R012 Approval: PMAD Logit-Only w0.15/T4 Repeat Run07

- Approved one repeat experiment on branch `exp/R012-primkd-logit-w015-repeat-v2`.
- Hypothesis: a second fixed-recipe PMAD logit-only w0.15/T4 repeat can test whether the R010 high-tail result is reproducible and may reach the `>=0.53` target without changing model structure or the training recipe.
- Reason: R010 reached best val/mIoU `0.527469`, the closest current evidence-backed result to the target; the updated six-run PMAD w0.15/T4 mean best is `0.521907`.
- Planned model name: `dformerv2_primkd_logit_only`.
- Planned run name: `w015_t4_run07`; checkpoint directory `checkpoints/dformerv2_primkd_logit_only_w015_t4_run07`.
- Fixed recipe remains `batch_size=2`, `max_epochs=50`, `lr=6e-5`, `num_workers=4`, `early_stop_patience=30`, `loss_type=ce`, no scheduler.
- No code change is approved for this round; the model is already implemented and registered.
- Launch note: use a hidden `cmd.exe /c` script so the Lightning/TQDM progress stream remains in stdout while avoiding the previous Windows `forrtl error (200): program aborting due to window-CLOSE event`.
- Forbidden-change check: no dataset split, dataloader, augmentation, validation, metric, mIoU, optimizer, scheduler, epoch, batch, lr, worker, checkpoint-artifact, dataset, pretrained-weight, TensorBoard-log, backbone, encoder, or model-code change is approved.

### 2026-05-13 R012 Result: Negative

- `dformerv2_primkd_logit_only_w015_t4_run07` completed 50 validation epochs.
- best val/mIoU: `0.516967` at epoch `43`; last val/mIoU: `0.508205`.
- Result is slightly below the clean 10-run baseline mean `0.517397`, below baseline mean + 1 std `0.522298`, below the prior PMAD w0.15/T4 five-run mean `0.520795`, far below R010 run06_retry1 `0.527469`, and below the `0.53` goal.
- Evidence: `final daima/miou_list/dformerv2_primkd_logit_only_w015_t4_run07.md`.
- Report: `reports/R012-primkd-logit-w015-repeat-v2.md`.
- Diagnostic: updated PMAD w0.15/T4 seven-run mean best is `0.521201` with population std `0.004148`; only `1/7` runs exceed the clean baseline best single.
- Process note: the hidden `cmd.exe /c` launch completed with exit code `0`; `Trainer.fit` reached `max_epochs=50`; no Windows/Rich teardown crash occurred.
- Audit: static review `PASS`; evidence/report audit `PASS`; reproducer audit `audit_passed_no_rerun`.
- Decision: reject this repeat as a goal path. Stop blind PMAD w0.15/T4 repeats and pivot to a distinct high-decision-value hypothesis if the loop continues.

### 2026-05-13 R013 Approval: LMLP Decoder Head

- Approved one experiment on branch `exp/R013-lmlp-decoder-v1`.
- Hypothesis: a DFormer/SegFormer-style c2-c4 LMLP decoder head can test whether SimpleFPN top-down additive fusion is limiting the current DFormerv2 RGB-D fused features.
- Paper/code evidence: SegFormer All-MLP decoder (`https://arxiv.org/abs/2105.15203`, `https://github.com/NVlabs/SegFormer`) and DFormer local reference `ref_codes/DFormer/models/decoders/LMLPDecoder.py`.
- Planned model name: `dformerv2_lmlp_decoder`.
- Planned run name: `run01`; checkpoint directory `checkpoints/dformerv2_lmlp_decoder_run01`.
- Planned minimal change: add c2/c3/c4 MLP projections, upsample to c2, concatenate, `1x1` fuse + BN + ReLU + dropout + classifier, then upsample logits to input size.
- Fixed recipe remains `batch_size=2`, `max_epochs=50`, `lr=6e-5`, `num_workers=4`, `early_stop_patience=30`, `loss_type=ce`, no scheduler.
- Forbidden-change check: no dataset split, dataloader, augmentation, validation, metric, mIoU, optimizer, scheduler, epoch, batch, lr, worker, checkpoint-artifact, dataset, pretrained-weight, TensorBoard-log, backbone, or encoder change is approved.

### 2026-05-13 R013 Result: Weak Near-Baseline, Below Goal

- `dformerv2_lmlp_decoder_run01` completed 50 validation epochs.
- best val/mIoU: `0.517981` at epoch `41`; last val/mIoU: `0.490231`.
- Result is only `+0.000584` above the clean 10-run baseline mean `0.517397`, below baseline mean + 1 std `0.522298`, below the clean baseline best single `0.524425`, below R004 c4-only TGGA `0.522849`, below R010 PMAD run06_retry1 `0.527469`, and below the `0.53` goal.
- Evidence: `final daima/miou_list/dformerv2_lmlp_decoder_run01.md`.
- Report: `reports/R013-lmlp-decoder-v1.md`.
- Late-curve caveat: best-to-last drop is `0.027750`.
- Process note: training completed with exit code `0`; `Trainer.fit` reached `max_epochs=50`.
- Audit: static review `PASS`; evidence/report audit `PASS`; reproducer audit `audit_passed_no_rerun`.
- Decision: reject this exact LMLP decoder as a goal path. Pause after R013 per user request; do not launch a next full train.

### 2026-05-12 Orchestrator Candidate Check

- Read orchestration rules, current reports, metrics, experiment coordination files, and active paper/result notes.
- `experiments/candidates.jsonl` contains only the schema record; no candidate experiment is available for approval.
- `experiments/queue.jsonl`, `experiments/completed.jsonl`, and `experiments/rejected.jsonl` also contain only schema records.
- `metrics/runs.jsonl` and `metrics/leaderboard.csv` contain no completed orchestration-loop run records.
- Decision: no experiment is approved and no queue record is appended. Wait for the Literature/Idea agent to write candidate records before selecting a next round.
- No training was run, and no model, training, data, metric, loader, augmentation, checkpoint, dataset, pretrained weight, or large log file was modified.

### 2026-05-12 Orchestration Setup

- Created the shared Goal-Driven experiment loop files.
- Added `codex/WINDOW_OPERATION_GUIDE.md` for concrete multi-window usage.
- No training was run.
- No model, training script, data, optimizer, scheduler, metric, loader, or augmentation code was modified.
