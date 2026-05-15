# R033 SimpleFPN Ham Logit Fusion

## Hypothesis

SimpleFPN has the strongest corrected peak while Ham variants provide contextual logits. R033 keeps the corrected R016 fusion/base path and tests whether a small learned Ham logit residual can complement SimpleFPN logits.

## Configuration

- Branch: `exp/R033-simplefpn-ham-logit-fusion-v1`
- Model: `dformerv2_simplefpn_ham_logit_fusion`
- Run: `R033_simplefpn_ham_logit_fusion_run01`
- Main code change: `SimpleFPNHamLogitFusionDecoder`
- Output equation: `simple_fpn_logits + alpha * ham_logits`
- Initial alpha: `sigmoid(-2.944439) ~= 0.05`
- Fixed recipe: batch_size `2`, max_epochs `50`, lr `6e-5`, num_workers `4`, early_stop_patience `30`, loss `ce`
- DFormerv2-S pretrained path unchanged.

## Evidence

- TensorBoard event: `final daima/checkpoints/R033_simplefpn_ham_logit_fusion_run01/lightning_logs/version_0/events.out.tfevents.1778829755.Administrator.10268.0`
- Best checkpoint: `final daima/checkpoints/R033_simplefpn_ham_logit_fusion_run01/dformerv2_simplefpn_ham_logit_fusion-epoch=48-val_mIoU=0.5330.pt`
- Per-epoch mIoU: `final daima/miou_list/R033_simplefpn_ham_logit_fusion_run01.md`
- Exit code: `0`
- `Trainer.fit` stopped because `max_epochs=50` was reached.

## Result

| Metric | Value |
|---|---:|
| best val/mIoU | 0.533020 |
| best validation epoch | 49 |
| last val/mIoU | 0.528883 |
| last-5 mean val/mIoU | 0.527628 |
| last-10 mean val/mIoU | 0.519951 |
| best-to-last drop | 0.004137 |
| best val/loss | 0.977882 |
| final train/loss_epoch | 0.052197 |
| ham logit alpha first | 0.050669 |
| ham logit alpha last | 0.090593 |

## Interpretation

R033 is partial-positive but below the corrected baseline. It crosses `0.53`, and the Ham residual branch learned to open from `0.050669` to `0.090593`, but the best mIoU remains below R016 `0.541121` by `-0.008101` and below R027/R032/R030.

The result rejects logits-level SimpleFPN+Ham scalar residual fusion as an active mainline improvement. It should not be described as reaching the `0.56` target or improving over the corrected baseline.

## Contract Check

R033 did not modify dataset split, eval metric, mIoU calculation, val/test loader behavior, data augmentation, optimizer, scheduler, batch size, max epochs, learning rate, worker count, early-stop setting, DFormerv2-S level, pretrained loading, pretrained DFormerv2 weights, pretrained DepthEncoder weights, GatedFusion equations, checkpoint artifacts, dataset files, pretrained weights, or TensorBoard event files.
