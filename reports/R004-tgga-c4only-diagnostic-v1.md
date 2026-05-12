# R004-tgga-c4only-diagnostic-v1

## Dry Check

- branch: `exp/R004-tgga-c4only-diagnostic-v1`
- hypothesis: TGGA c4-only can retain high-level semantic/geometry calibration while removing the c3 high-resolution gate/residual path that may cause c3/c4 TGGA late collapse.
- model: `dformerv2_tgga_c4only_beta002_aux003_detachsem_simplefpn_v1`
- run name: `run01`
- checkpoint dir: `checkpoints/dformerv2_tgga_c4only_beta002_aux003_detachsem_simplefpn_v1_run01`
- code change: none; the model was already implemented and registered before this round.
- fixed recipe: `batch_size=2`, `max_epochs=50`, `lr=6e-5`, `num_workers=4`, `early_stop_patience=30`, `loss_type=ce`, no scheduler.
- forbidden-change boundary: no dataset split, loader, augmentation, validation/test path, eval metric, mIoU calculation, optimizer, scheduler, epoch, batch size, learning rate, worker count, early-stop setting, checkpoint artifact, dataset, pretrained weight, TensorBoard event, or code change was approved.

## Command

```powershell
D:\Anaconda\envs\qintian-rgbd\python.exe train.py `
  --model dformerv2_tgga_c4only_beta002_aux003_detachsem_simplefpn_v1 `
  --data_root C:/Users/qintian/Desktop/qintian/data/NYUDepthv2 `
  --num_classes 40 `
  --batch_size 2 `
  --max_epochs 50 `
  --lr 6e-5 `
  --num_workers 4 `
  --checkpoint_dir checkpoints/dformerv2_tgga_c4only_beta002_aux003_detachsem_simplefpn_v1_run01 `
  --early_stop_patience 30 `
  --devices 1 `
  --accelerator auto `
  --dformerv2_pretrained C:/Users/qintian/Desktop/qintian/dformer_work/checkpoints/pretrained/DFormerv2_Small_pretrained.pth `
  --loss_type ce
```

## Evidence

- TensorBoard event: `final daima/checkpoints/dformerv2_tgga_c4only_beta002_aux003_detachsem_simplefpn_v1_run01/lightning_logs/version_0/events.out.tfevents.1778620555.Administrator.23676.0`
- best checkpoint: `final daima/checkpoints/dformerv2_tgga_c4only_beta002_aux003_detachsem_simplefpn_v1_run01/dformerv2_tgga_c4only_beta002_aux003_detachsem_simplefpn_v1-epoch=41-val_mIoU=0.5228.pt`
- mIoU detail: `final daima/miou_list/dformerv2_tgga_c4only_beta002_aux003_detachsem_simplefpn_v1_run01.md`
- stdout log: `agent_workspace/run_logs/R004-tgga-c4only-diagnostic-v1.stdout.log`
- stderr log: `agent_workspace/run_logs/R004-tgga-c4only-diagnostic-v1.stderr.log`
- process note: `Trainer.fit` reached `max_epochs=50`.

## Result

- recorded validation epochs: `50`
- best val/mIoU: `0.522849` at epoch `42`
- last val/mIoU: `0.509320`
- last-5 mean val/mIoU: `0.505936`
- last-10 mean val/mIoU: `0.510887`
- delta vs clean baseline mean `0.517397`: `+0.005452` (`+1.112` baseline std units)
- delta vs clean baseline mean + 1 std `0.522298`: `+0.000551`
- delta vs clean baseline best single `0.524425`: `-0.001576`
- delta vs PMAD w0.15/T4 five-run mean `0.520795`: `+0.002054`
- delta vs original TGGA c3/c4 run01 best `0.522206`: `+0.000643`
- best-to-last delta: `-0.013529`

## Decision

- status: `completed_partial_positive`
- R004 is the strongest orchestration-loop run so far and crosses clean baseline mean + 1 std, but it does not reach the `0.53` goal and still has late instability.
- The c4-only diagnostic supports the idea that c4-level TGGA calibration is safer than the original c3/c4 gate, but c4 alone is not enough.
- Do not claim goal success. Continue the loop with a single next hypothesis that uses this signal without reverting to already failed PMAD filtering or decoder frequency fusion.

## Audit

- code review status: `approved_on_staged_diff`
- reproducer/report audit status: `audit_passed_no_rerun`
- audit note: full worktree contains unrelated pre-existing checkpoint/TensorBoard deletions, but R004 staging is restricted to result docs, metrics, and coordination files only.
