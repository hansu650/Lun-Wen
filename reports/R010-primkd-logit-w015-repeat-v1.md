# R010 PMAD Logit-Only w0.15/T4 Repeat

- branch: `exp/R010-primkd-logit-w015-repeat-v1`
- model: `dformerv2_primkd_logit_only`
- checkpoint_dir: `final daima/checkpoints/dformerv2_primkd_logit_only_w015_t4_run06_retry1`
- TensorBoard event: `final daima/checkpoints/dformerv2_primkd_logit_only_w015_t4_run06_retry1/lightning_logs/version_0/events.out.tfevents.1778662831.Administrator.24204.0`
- best checkpoint: `final daima/checkpoints/dformerv2_primkd_logit_only_w015_t4_run06_retry1/dformerv2_primkd_logit_only-epoch=48-val_mIoU=0.5275.pt`
- mIoU detail: `final daima/miou_list/dformerv2_primkd_logit_only_w015_t4_run06_retry1.md`

## Hypothesis

The repeat-backed PMAD logit-only setting (`kd_weight=0.15`, `kd_temperature=4.0`) is the strongest current positive KD direction and may produce a high-tail run approaching the `0.53` goal while improving stability evidence.

## Fixed Recipe Check

- `batch_size=2`
- `max_epochs=50`
- `lr=6e-5`
- `num_workers=4`
- `early_stop_patience=30`
- `loss_type=ce`
- optimizer/scheduler unchanged through `BaseLitSeg`
- dataset split, dataloader, augmentation, validation, test, metric, and mIoU code unchanged
- DFormerv2_S backbone/pretrained handling unchanged

## Command

The completed retry used:

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
  --checkpoint_dir ".\checkpoints\dformerv2_primkd_logit_only_w015_t4_run06_retry1"
```

## Result

- recorded validation epochs: `50`
- best val/mIoU: `0.527469` at epoch `49`
- last val/mIoU: `0.526316`
- last-5 mean val/mIoU: `0.519330`
- last-10 mean val/mIoU: `0.516229`
- best val/loss: `1.064507` at epoch `7`
- final train/loss: `0.209408`
- final train/ce_loss: `0.149212`
- final train/kd_loss: `0.401306`
- TensorBoard exit evidence: `Trainer.fit` reached `max_epochs=50`; launch script exit code `0`

## Comparison

- clean 10-run baseline mean best: `0.517397`
- clean 10-run baseline std: `0.004901`
- clean 10-run baseline mean + 1 std: `0.522298`
- clean 10-run baseline best single: `0.524425`
- prior PMAD w0.15/T4 five-run mean best: `0.520795`
- prior PMAD w0.15/T4 best single: `0.524028`
- delta vs clean baseline mean: `+0.010072` (`+2.055` std units)
- delta vs clean baseline mean + 1 std: `+0.005171`
- delta vs clean baseline best single: `+0.003044`
- delta vs prior PMAD five-run mean: `+0.006674`
- delta vs prior PMAD best single: `+0.003441`
- gap to `0.53`: `-0.002531`
- updated PMAD w0.15/T4 six-run mean best: `0.521907`
- updated PMAD w0.15/T4 six-run population std: `0.004073`

## Process Notes

- The first non-retry `run06` launch stopped during epoch 0 with Windows `forrtl error (200): program aborting due to window-CLOSE event`; it produced no `val/mIoU` and is excluded from all result claims.
- The retry used a hidden `cmd.exe` script under `agent_workspace` to keep the process alive while preserving TQDM progress in the stdout log.
- Progress text in redirected logs shows mojibake for progress-bar glyphs, but TensorBoard scalars and checkpoint evidence are valid.

## Audit Notes

- Static review found no forbidden dataset/eval/metric/loader/augmentation/optimizer/scheduler/backbone change and returned `approved_current_diff`.
- Evidence/report audit confirmed 50 TensorBoard `val/mIoU` points, best `0.527469` at epoch 49, last `0.526316`, and bounded partial-positive claims.
- Reproducer confirmed the active retry command, qintian-rgbd Python, unique checkpoint directory, KD settings, `--save_student_only`, and returned `audit_passed_no_rerun`.
- The repository still has pre-existing checkpoint/TensorBoard deletion noise; those files must not be staged.

## Decision

R010 is a partial positive repeat and the best single orchestration-loop run so far, but it is below the `0.53` success threshold. Do not stop the Goal-Driven loop or claim target success. Use it as stronger evidence that PMAD logit-only is marginally useful, then pivot to a distinct high-value hypothesis.
