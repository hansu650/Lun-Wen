# R012 PMAD Logit-Only w0.15/T4 Repeat Run07

- branch: `exp/R012-primkd-logit-w015-repeat-v2`
- model: `dformerv2_primkd_logit_only`
- checkpoint_dir: `final daima/checkpoints/dformerv2_primkd_logit_only_w015_t4_run07`
- TensorBoard event: `final daima/checkpoints/dformerv2_primkd_logit_only_w015_t4_run07/lightning_logs/version_0/events.out.tfevents.1778675198.Administrator.22916.0`
- best checkpoint: `final daima/checkpoints/dformerv2_primkd_logit_only_w015_t4_run07/dformerv2_primkd_logit_only-epoch=42-val_mIoU=0.5170.pt`
- mIoU detail: `final daima/miou_list/dformerv2_primkd_logit_only_w015_t4_run07.md`

## Hypothesis

R010 showed a high-tail PMAD logit-only repeat at `0.527469`, close to the `0.53` target. R012 tests whether the same fixed-recipe PMAD setting (`kd_weight=0.15`, `kd_temperature=4.0`) reproduces that peak or reaches the target as run07.

## Fixed Recipe Check

- data root: `C:\Users\qintian\Desktop\qintian\data\NYUDepthv2`
- pretrained DFormerv2-S: `C:\Users\qintian\Desktop\qintian\dformer_work\checkpoints\pretrained\DFormerv2_Small_pretrained.pth`
- teacher checkpoint: `checkpoints\dformerv2_geometry_primary_teacher_run01\dformerv2_geometry_primary_teacher-epoch=37-val_mIoU=0.5168.pt`
- `batch_size=2`
- `max_epochs=50`
- `lr=6e-5`
- `num_workers=4`
- `early_stop_patience=30`
- `loss_type=ce`
- `accelerator=gpu`, `devices=1`
- `kd_weight=0.15`, `kd_temperature=4.0`, `--save_student_only`

No dataset split, dataloader, validation/test loader, augmentation, mIoU calculation, optimizer, scheduler, batch size, epoch count, learning rate, backbone, encoder, or model code was changed for this round.

## Training Command

```powershell
cd "C:\Users\qintian\Desktop\qintian\final daima"
& "D:\Anaconda\envs\qintian-rgbd\python.exe" train.py `
  --model dformerv2_primkd_logit_only `
  --data_root "C:\Users\qintian\Desktop\qintian\data\NYUDepthv2" `
  --num_classes 40 `
  --batch_size 2 `
  --max_epochs 50 `
  --lr 6e-5 `
  --num_workers 4 `
  --early_stop_patience 30 `
  --accelerator gpu `
  --devices 1 `
  --dformerv2_pretrained "C:\Users\qintian\Desktop\qintian\dformer_work\checkpoints\pretrained\DFormerv2_Small_pretrained.pth" `
  --teacher_ckpt "checkpoints\dformerv2_geometry_primary_teacher_run01\dformerv2_geometry_primary_teacher-epoch=37-val_mIoU=0.5168.pt" `
  --kd_weight 0.15 `
  --kd_temperature 4.0 `
  --loss_type ce `
  --save_student_only `
  --checkpoint_dir ".\checkpoints\dformerv2_primkd_logit_only_w015_t4_run07"
```

## Result

- recorded validation epochs: `50`
- best val/mIoU: `0.516967` at epoch `43`
- last val/mIoU: `0.508205`
- last-5 mean val/mIoU: `0.496441`
- last-10 mean val/mIoU: `0.503120`
- best val/loss: `1.039913` at epoch `8`
- final train/loss: `0.216768`
- final train/ce_loss: `0.153825`
- final train/kd_loss: `0.419615`
- process status: exit code `0`; `Trainer.fit` reached `max_epochs=50`; no Windows/Rich teardown crash.

## Comparison

- clean 10-run baseline mean best: `0.517397`
- clean 10-run baseline std: `0.004901`
- clean 10-run baseline mean + 1 std: `0.522298`
- clean baseline best single: `0.524425`
- prior PMAD w0.15/T4 five-run mean best: `0.520795`
- R010 PMAD run06_retry1 best: `0.527469`
- delta vs clean baseline mean: `-0.000430` (`-0.088` baseline std units)
- delta vs prior PMAD five-run mean: `-0.003828`
- delta vs R010 run06_retry1: `-0.010502`
- gap to `0.53`: `-0.013033`

## Decision

R012 is a negative repeat. It completed cleanly but did not reproduce R010's high-tail result, did not exceed the clean baseline mean, and did not approach the `0.53` goal. More blind repeats of PMAD logit-only w0.15/T4 are low-value; the next experiment should use a distinct hypothesis.

## Audit Notes

- TensorBoard event contains 50 `val/mIoU` scalar points.
- Best checkpoint filename matches the TensorBoard best to four decimals.
- Static review: `PASS`.
- Evidence/report audit: `PASS`.
- Reproducer audit: `audit_passed_no_rerun`.
- Checkpoints and TensorBoard events are evidence only and must not be staged.
- `agent_workspace` run scripts/logs are temporary and must not be staged.
