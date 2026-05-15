# R038 DSCF C4 Lite V1

## Summary

- Branch: `exp/R038-dscf-c4-lite-v1`
- Model: `dformerv2_dscf_c4_lite`
- Run: `R038_dscf_c4_lite_run01`
- Status: `completed_negative_below_corrected_baseline`
- Hypothesis: KTB/CVPR 2025 DSCF-style dynamic sparse cross-modal sampling at c4 may improve high-level RGB-depth fusion over dense c4 GatedFusion.
- Code change: added a separate model entry replacing only c4 fusion; c1-c3 GatedFusion, SimpleFPNDecoder, loss, data/eval, DFormerv2-S, and fixed training recipe remain unchanged.
- Full train: completed 50 validation epochs with exit code `0`.

## Evidence

- TensorBoard event: `final daima/checkpoints\R038_dscf_c4_lite_run01\lightning_logs\version_0\events.out.tfevents.1778863547.Administrator.8372.0`
- Best checkpoint: `final daima/checkpoints\R038_dscf_c4_lite_run01\dformerv2_dscf_c4_lite-epoch=37-val_mIoU=0.5308.pt`
- Saved launch command: `final daima/checkpoints/R038_dscf_c4_lite_run01/run_r038.cmd`
- mIoU detail: `final daima/miou_list/R038_dscf_c4_lite_run01.md`

## Metrics

- Best val/mIoU: `0.530810` at validation epoch `38`
- Last val/mIoU: `0.530308`
- Last-5 mean val/mIoU: `0.526104`
- Last-10 mean val/mIoU: `0.522189`
- Best-to-last drop: `0.000502`
- Best val/loss: `0.936448` at validation epoch `10`
- Last val/loss: `1.218423`
- Final train/loss_epoch: `0.056458`
- DSCF c4 offset_abs first/last: `0.961821` / `1.675011`
- DSCF c4 weight_entropy first/last: `1.376656` / `1.336370`
- Delta vs R016 best `0.541121`: `-0.010311`

## Decision

R038 is negative below R016. The DSCF-lite branch trains and diagnostics show non-collapsed sampling, but the peak is lower than R016/R034/R036/R037. Do not tune K or offset scale as a micro-search; pivot to the next distinct candidate, likely HDBFormer MIIM-lite c4-only or a broader contract review.
