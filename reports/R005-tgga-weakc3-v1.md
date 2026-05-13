# R005-tgga-weakc3-v1

## Dry Check

- branch: `exp/R005-tgga-weakc3-v1`
- hypothesis: TGGA weak-c3 plus c4 can retain the R004 c4-only calibration signal while reintroducing a conservative c3 detail path without the original c3/c4 high-resolution gate instability.
- model: `dformerv2_tgga_c34_weakc3_beta001_c4beta002_aux003_detachsem_v1`
- run name: `run01`
- checkpoint dir: `checkpoints/dformerv2_tgga_c34_weakc3_beta001_c4beta002_aux003_detachsem_v1_run01`
- code change: none; the model was already implemented and registered before this round.
- fixed recipe: `batch_size=2`, `max_epochs=50`, `lr=6e-5`, `num_workers=4`, `early_stop_patience=30`, `loss_type=ce`, no scheduler.
- forbidden-change boundary: no dataset split, loader, augmentation, validation/test path, eval metric, mIoU calculation, optimizer, scheduler, epoch, batch size, learning rate, worker count, early-stop setting, checkpoint artifact, dataset, pretrained weight, TensorBoard event, or code change was approved.

## Command

```powershell
D:\Anaconda\envs\qintian-rgbd\python.exe train.py `
  --model dformerv2_tgga_c34_weakc3_beta001_c4beta002_aux003_detachsem_v1 `
  --data_root C:/Users/qintian/Desktop/qintian/data/NYUDepthv2 `
  --num_classes 40 `
  --batch_size 2 `
  --max_epochs 50 `
  --lr 6e-5 `
  --num_workers 4 `
  --checkpoint_dir checkpoints/dformerv2_tgga_c34_weakc3_beta001_c4beta002_aux003_detachsem_v1_run01 `
  --early_stop_patience 30 `
  --devices 1 `
  --accelerator auto `
  --dformerv2_pretrained C:/Users/qintian/Desktop/qintian/dformer_work/checkpoints/pretrained/DFormerv2_Small_pretrained.pth `
  --loss_type ce
```

## Evidence

- TensorBoard event: `final daima/checkpoints/dformerv2_tgga_c34_weakc3_beta001_c4beta002_aux003_detachsem_v1_run01/lightning_logs/version_0/events.out.tfevents.1778626367.Administrator.41400.0`
- best checkpoint: `final daima/checkpoints/dformerv2_tgga_c34_weakc3_beta001_c4beta002_aux003_detachsem_v1_run01/dformerv2_tgga_c34_weakc3_beta001_c4beta002_aux003_detachsem_v1-epoch=42-val_mIoU=0.5183.pt`
- mIoU detail: `final daima/miou_list/dformerv2_tgga_c34_weakc3_beta001_c4beta002_aux003_detachsem_v1_run01.md`
- stdout log: `agent_workspace/run_logs/R005-tgga-weakc3-v1.stdout.log`
- stderr log: `agent_workspace/run_logs/R005-tgga-weakc3-v1.stderr.log`
- process note: `Trainer.fit` reached `max_epochs=50`; after metric/checkpoint writing, Rich progress teardown raised a Windows GBK `UnicodeEncodeError`.

## Result

- recorded validation epochs: `50`
- best val/mIoU: `0.518253` at epoch `43`
- last val/mIoU: `0.514908`
- last-5 mean val/mIoU: `0.508991`
- last-10 mean val/mIoU: `0.507070`
- delta vs clean baseline mean `0.517397`: `+0.000856` (`+0.175` baseline std units)
- delta vs clean baseline mean + 1 std `0.522298`: `-0.004045`
- delta vs clean baseline best single `0.524425`: `-0.006172`
- delta vs PMAD w0.15/T4 five-run mean `0.520795`: `-0.002542`
- delta vs R004 TGGA c4-only `0.522849`: `-0.004596`
- best-to-last delta: `-0.003345`

## Decision

- status: `completed_weak_positive_below_goal`
- R005 is slightly above the clean baseline mean but below every stronger decision boundary: baseline mean + 1 std, PMAD w0.15/T4 mean, R004 c4-only, clean best single, and the `0.53` goal.
- The weak c3 gate still opens heavily by the end (`gate_c3_mean=0.293138`, `gate_c3_std=0.331622`), so the result does not rescue the c3 TGGA path.
- Do not continue the weak-c3 line unchanged. Keep R004 c4-only as the better TGGA diagnostic branch and pause before selecting another experiment.

## Audit

- code review status: `approved_current_diff`
- reproducer/report audit status: `audit_passed_no_rerun`
- audit note: full worktree still contains unrelated checkpoint/TensorBoard deletions and untracked reference/tool folders, but R005 result files do not include those artifacts.
