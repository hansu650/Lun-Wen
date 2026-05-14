# R023 Corrected-Contract Geometry-Primary Teacher Refresh

## Hypothesis

Refreshing the geometry-primary DFormerv2 teacher under the corrected R015/R016 label/depth contract can decide whether corrected-contract PMAD/KD deserves another student experiment.

## Implementation

- Branch: `exp/R023-corrected-contract-teacher-refresh-v1`
- Model: `dformerv2_geometry_primary_teacher`
- Run: `R023_geometry_primary_teacher_corrected_contract_run01`
- Code change: none. This run uses the existing `DFormerV2GeometryPrimaryTeacherSegmentor`.
- Fixed recipe preserved: batch size `2`, max epochs `50`, lr `6e-5`, num workers `4`, early-stop patience `30`, CE loss, AdamW, DFormerv2-S level, and the existing DFormerv2-S pretrained loading.
- No `teacher_ckpt`, `kd_weight`, `kd_temperature`, or `save_student_only` arguments were used, because R023 trains the teacher itself.

## Evidence

- Status: full train completed with exit code `0`; `Trainer.fit` reached `max_epochs=50`.
- TensorBoard event: `final daima/checkpoints/R023_geometry_primary_teacher_corrected_contract_run01/lightning_logs/version_0/events.out.tfevents.1778787920.Administrator.37368.0`
- Best checkpoint: `final daima/checkpoints/R023_geometry_primary_teacher_corrected_contract_run01/dformerv2_geometry_primary_teacher-epoch=42-val_mIoU=0.5245.pt`
- Per-epoch mIoU: `final daima/miou_list/R023_geometry_primary_teacher_corrected_contract_run01.md`

## Result

| Metric | Value |
|---|---:|
| recorded validation epochs | 50 |
| best val/mIoU | 0.524498 |
| best validation epoch | 43 |
| last val/mIoU | 0.507023 |
| last-5 mean val/mIoU | 0.510467 |
| last-10 mean val/mIoU | 0.512531 |
| best-to-last drop | 0.017475 |
| best val/loss | 0.988437 |
| best val/loss epoch | 10 |
| final train/loss_epoch | 0.092329 |

## Decision

R023 is a negative teacher gate. It is below the R016 corrected baseline `0.541121` by `0.016623`, below R022 `0.534332`, and below the `0.53` teacher gate.

Do not run corrected-contract PMAD from this teacher checkpoint. The next higher-value step is a non-KD structure isolation experiment: raw `DFormerv2_S(rgb, depth) -> OfficialHamDecoder`, without the external DepthEncoder/GatedFusion stack.

## Forbidden-Change Check

R023 did not modify dataset split, eval metric, mIoU calculation, val/test loader behavior, data augmentation, optimizer, scheduler, batch size, max epochs, learning rate, worker count, early-stop setting, DFormerv2-S level, pretrained loading, checkpoint artifacts, dataset files, pretrained weights, or TensorBoard event files.
