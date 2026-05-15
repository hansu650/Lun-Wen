# R032 SimpleFPN C1 Detail Gate

## Hypothesis

The strongest corrected SimpleFPN path may be limited by noisy high-resolution `c1` detail. R032 keeps the SimpleFPN topology and adds only a learnable strength on the `lateral1(c1)` contribution, initialized near the baseline-equivalent value.

## Implementation

- Branch: `exp/R032-simplefpn-c1-detail-gate-v1`
- Model: `dformerv2_simplefpn_c1_detail_gate`
- Run: `R032_simplefpn_c1_detail_gate_run01`
- Code change: add `SimpleFPNDecoderC1DetailGate`, with `p1 = alpha * lateral1(c1) + upsample(p2)`.
- Initialization: `alpha = sigmoid(6.906755)`, initially about `0.999`.
- Audit logging: logs `train/c1_detail_alpha` once per epoch.
- Kept unchanged: baseline `dformerv2_mid_fusion`, DFormerv2-S backbone level, DFormerv2-S pretrained loading, ResNet-18 DepthEncoder, GatedFusion equations, CE loss, AdamW, batch size `2`, max epochs `50`, lr `6e-5`, num workers `4`, early-stop patience `30`, data split, loaders, augmentation, eval metric, and mIoU calculation.

## Evidence

- Status: full train completed with exit code `0`; `Trainer.fit` reached `max_epochs=50`.
- TensorBoard event: `final daima/checkpoints/R032_simplefpn_c1_detail_gate_run01/lightning_logs/version_0/events.out.tfevents.1778824352.Administrator.34648.0`
- Best checkpoint: `final daima/checkpoints/R032_simplefpn_c1_detail_gate_run01/dformerv2_simplefpn_c1_detail_gate-epoch=49-val_mIoU=0.5366.pt`
- Per-epoch mIoU: `final daima/miou_list/R032_simplefpn_c1_detail_gate_run01.md`

## Result

| Metric | Value |
|---|---:|
| recorded validation epochs | 50 |
| best val/mIoU | 0.536603 |
| best validation epoch | 50 |
| last val/mIoU | 0.536603 |
| last-5 mean val/mIoU | 0.505390 |
| last-10 mean val/mIoU | 0.509657 |
| best-to-last drop | 0.000000 |
| best val/loss | 0.965559 |
| best val/loss epoch | 9 |
| final train/loss_epoch | 0.061732 |
| c1 detail alpha first | 0.998994 |
| c1 detail alpha last | 0.998770 |

## Decision

R032 is partial-positive but below the corrected baseline. It crosses `0.53` and is close to R027/R030, but remains below R016 `0.541121` by `-0.004518` and below the final `0.56` goal.

The alpha moved only `-0.000224`, so this near-baseline c1 detail gate does not strongly support further c1-gate tuning. The next round should use a distinct complementarity hypothesis: fuse SimpleFPN logits with a small Ham-context logit branch.

## Forbidden-Change Check

R032 did not modify dataset split, eval metric, mIoU calculation, val/test loader behavior, data augmentation, optimizer, scheduler, batch size, max epochs, learning rate, worker count, early-stop setting, DFormerv2-S level, pretrained loading, pretrained DFormerv2 weights, pretrained DepthEncoder weights, GatedFusion equations, loss, checkpoint artifacts, dataset files, pretrained weights, or TensorBoard event files.
