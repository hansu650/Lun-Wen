# R001 PMAD Boundary/Confidence-Selective KD

## Verdict

- status: `completed_negative`
- model: `dformerv2_primkd_boundary_conf`
- branch: `exp/R001-pmad-selective-kd-v1`
- best val/mIoU: `0.511646` at epoch `50`
- last val/mIoU: `0.511646`
- target: `>= 0.530000`
- result: target not reached; also below clean 10-run baseline mean `0.517397`

## Hypothesis

Boundary/confidence-selective PMAD logit KD would preserve the positive `dformerv2_primkd_logit_only` w0.15/T4 signal while reducing harmful teacher transfer in uncertain or non-boundary pixels.

## Command

```powershell
cd "C:\Users\qintian\Desktop\qintian\final daima"

& "D:\Anaconda\envs\qintian-rgbd\python.exe" train.py `
  --model dformerv2_primkd_boundary_conf `
  --data_root C:/Users/qintian/Desktop/qintian/data/NYUDepthv2 `
  --num_classes 40 `
  --batch_size 2 `
  --max_epochs 50 `
  --lr 6e-5 `
  --num_workers 4 `
  --checkpoint_dir checkpoints/dformerv2_primkd_boundary_conf_w015_t4_run01 `
  --early_stop_patience 30 `
  --devices 1 `
  --accelerator auto `
  --dformerv2_pretrained C:/Users/qintian/Desktop/qintian/dformer_work/checkpoints/pretrained/DFormerv2_Small_pretrained.pth `
  --loss_type ce `
  --teacher_ckpt checkpoints/dformerv2_geometry_primary_teacher_run01/dformerv2_geometry_primary_teacher-epoch=37-val_mIoU=0.5168.pt `
  --kd_weight 0.15 `
  --kd_temperature 4.0 `
  --save_student_only
```

## Evidence

- mIoU detail: `final daima/miou_list/dformerv2_primkd_boundary_conf_w015_t4_run01.md`
- TensorBoard event: `final daima/checkpoints/dformerv2_primkd_boundary_conf_w015_t4_run01/lightning_logs/version_0/events.out.tfevents.1778600707.Administrator.4516.0`
- best checkpoint: `final daima/checkpoints/dformerv2_primkd_boundary_conf_w015_t4_run01/dformerv2_primkd_boundary_conf-epoch=49-val_mIoU=0.5116.pt`
- process note: `Trainer.fit` reached `max_epochs=50`; after metrics/checkpoint writing, Rich progress teardown raised a Windows GBK `UnicodeEncodeError`.

## Comparisons

- delta vs clean 10-run baseline mean `0.517397`: `-0.005751`
- delta vs clean baseline mean + 1 std `0.522298`: `-0.010652`
- delta vs PMAD logit-only w0.15/T4 five-run mean `0.520795`: `-0.009149`
- last-5 mean val/mIoU: `0.507105`
- last-10 mean val/mIoU: `0.502303`

## Diagnostics

- final train/kd_mask_ratio: `0.998182`
- final train/kd_boundary_ratio: `0.061130`
- final train/kd_conf_mean: `0.938347`
- The confidence threshold `0.40` was not selective in practice; the KD mask stayed near `0.998`, so this run mainly tested confidence weighting plus boundary boosting.

## Forbidden-Change Audit

- No dataset split, dataloader, validation/test loader, data augmentation, eval metric, or mIoU calculation changed.
- No optimizer, scheduler, batch size, epoch count, learning rate, worker count, or early-stop setting changed.
- Checkpoints, TensorBoard event logs, pretrained weights, datasets, and large logs are evidence only and must not be committed.

## Decision

Reject this exact setting. Do not repeat `confidence_threshold=0.40` plus `boundary_boost=1.0` unchanged. The next round should move away from weakly selective PMAD weighting, likely toward the highest remaining decision-value candidate after audit.

## Audit

- code review: `approved`
- reproducer/report audit: `audit_passed_no_rerun`
- metric audit: TensorBoard event contains 50 `val/mIoU` records; best, last, last-5 mean, and last-10 mean match the report.
- commit hygiene requirement: stage only explicit R001 code/docs/metrics/report files; do not stage tracked checkpoint deletions or ignored checkpoint evidence.
