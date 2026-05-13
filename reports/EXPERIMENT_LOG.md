# Experiment Loop Log

This is the top-level ledger for the Goal-Driven RGB-D mIoU loop.

Current phase: active experiment loop. R002 has completed one full train and did not reach the target.

Goal: `val/mIoU >= 0.53`.

Baseline reference:

- Active baseline: `dformerv2_mid_fusion`
- Clean 10-run mean best val/mIoU: `0.517397`
- Population std: `0.004901`
- Best single baseline run: `0.524425`
- Existing evidence: `final daima/miou_list/dformerv2_mid_fusion_gate_baseline_summary_run01_09_run10_retry.md`

## Entries

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
