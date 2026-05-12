# R003 Correct-and-Entropy-Selective PMAD KD

## Verdict

- status: `completed_negative`
- model: `dformerv2_primkd_correct_entropy`
- branch: `exp/R003-pmad-correct-entropy-kd-v1`
- best val/mIoU: `0.516597` at epoch `50`
- last val/mIoU: `0.516597`
- target: `>= 0.530000`
- result: target not reached; slightly below clean 10-run baseline mean `0.517397`

## Hypothesis

PMAD logit KD would avoid harmful teacher transfer by distilling only training pixels where the frozen teacher is both label-correct and low-entropy, while keeping the student, teacher, CE loss, optimizer, data, validation metric, and fixed training recipe unchanged.

## Command

```powershell
cd "C:\Users\qintian\Desktop\qintian\final daima"

$env:PYTHONIOENCODING = "utf-8"
& "D:\Anaconda\envs\qintian-rgbd\python.exe" train.py `
  --model dformerv2_primkd_correct_entropy `
  --data_root C:/Users/qintian/Desktop/qintian/data/NYUDepthv2 `
  --num_classes 40 `
  --batch_size 2 `
  --max_epochs 50 `
  --lr 6e-5 `
  --num_workers 4 `
  --checkpoint_dir checkpoints/dformerv2_primkd_correct_entropy_w015_t4_h025_run01 `
  --early_stop_patience 30 `
  --devices 1 `
  --accelerator auto `
  --dformerv2_pretrained C:/Users/qintian/Desktop/qintian/dformer_work/checkpoints/pretrained/DFormerv2_Small_pretrained.pth `
  --loss_type ce `
  --teacher_ckpt checkpoints/dformerv2_geometry_primary_teacher_run01/dformerv2_geometry_primary_teacher-epoch=37-val_mIoU=0.5168.pt `
  --kd_weight 0.15 `
  --kd_temperature 4.0 `
  --kd_entropy_threshold 0.25 `
  --save_student_only
```

## Evidence

- mIoU detail: `final daima/miou_list/dformerv2_primkd_correct_entropy_w015_t4_h025_run01.md`
- TensorBoard event: `final daima/checkpoints/dformerv2_primkd_correct_entropy_w015_t4_h025_run01/lightning_logs/version_0/events.out.tfevents.1778613991.Administrator.34044.0`
- best checkpoint: `final daima/checkpoints/dformerv2_primkd_correct_entropy_w015_t4_h025_run01/dformerv2_primkd_correct_entropy-epoch=49-val_mIoU=0.5166.pt`
- process note: `Trainer.fit` reached `max_epochs=50`.

## Comparisons

- delta vs clean 10-run baseline mean `0.517397`: `-0.000800`
- delta vs clean baseline mean + 1 std `0.522298`: `-0.005701`
- delta vs PMAD logit-only w0.15/T4 five-run mean `0.520795`: `-0.004198`
- last-5 mean val/mIoU: `0.505977`
- last-10 mean val/mIoU: `0.500502`

## Diagnostics

- final train/kd_mask_ratio: `0.910636`
- final train/kd_entropy_mean: `0.047532`
- final train/kd_entropy_selected_mean: `0.030076`
- final train/kd_teacher_valid_acc: `0.932230`
- final train/kd_teacher_selected_acc: `1.000000`
- The selector is meaningful compared with R001 (`0.910636` vs `0.998182` mask ratio), but it does not improve the final result.

## Forbidden-Change Audit

- No dataset split, dataloader, validation/test loader, data augmentation, eval metric, or mIoU calculation changed.
- No optimizer, scheduler, batch size, epoch count, learning rate, worker count, or early-stop setting changed.
- Checkpoints, TensorBoard event logs, pretrained weights, datasets, and large logs are evidence only and must not be committed.

## Decision

Reject this exact correct-and-entropy selective PMAD setting. It should not be claimed as an improvement and should not be repeated unchanged.

## Audit

- code review: `approved`
- reproducer/report audit: `audit_passed_no_rerun`
- metric audit: TensorBoard event contains 50 `val/mIoU` records; best, last, last-5 mean, last-10 mean, best val/loss, final train metrics, and all per-epoch table rows match the report.
- commit hygiene requirement: stage only explicit R003 code/docs/metrics/report files; do not stage tracked checkpoint deletions or ignored checkpoint evidence.
