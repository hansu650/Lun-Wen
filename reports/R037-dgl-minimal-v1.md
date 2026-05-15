# R037 DGL Minimal V1

## Summary

- Branch: `exp/R037-dgl-minimal-v1`
- Model: `dformerv2_dgl_minimal`
- Run: `R037_dgl_minimal_run01`
- Status: `completed_partial_stable_below_corrected_baseline`
- Hypothesis: DGL-style gradient disentanglement may reduce multimodal optimization conflict by routing fused CE gradients only through fusion/decoder and routing encoder gradients through primary/depth auxiliary CE heads.
- Code change: added a separate model entry; validation/inference output path returns only fused logits and does not use the auxiliary heads.
- Full train: completed 50 validation epochs with exit code `0`.

## Evidence

- TensorBoard event: `final daima/checkpoints\R037_dgl_minimal_run01\lightning_logs\version_0\events.out.tfevents.1778857094.Administrator.1736.0`
- Best checkpoint: `final daima/checkpoints\R037_dgl_minimal_run01\dformerv2_dgl_minimal-epoch=41-val_mIoU=0.5347.pt`
- mIoU detail: `final daima/miou_list/R037_dgl_minimal_run01.md`

## Metrics

- Best val/mIoU: `0.534656` at validation epoch `42`
- Last val/mIoU: `0.530153`
- Last-5 mean val/mIoU: `0.526926`
- Last-10 mean val/mIoU: `0.526304`
- Best-to-last drop: `0.004503`
- Best val/loss: `0.949518` at validation epoch `12`
- Last val/loss: `1.118480`
- Final train/loss_epoch: `0.056264`
- DGL aux weight: `0.03`
- Fusion CE first/last: `2.268861` / `0.050335`
- Primary aux CE first/last: `2.388133` / `0.066309`
- Depth aux CE first/last: `2.517628` / `0.131319`
- Delta vs R016 best `0.541121`: `-0.006465`

## Smoke / Audit Notes

- `py_compile` and `train.py --help` passed.
- Real NYU CUDA smoke confirmed finite logits `(1, 40, 480, 640)`, unchanged DFormerv2 pretrained load stats, fused CE gradients zero to primary/depth encoders, fused CE gradients nonzero to fusion/decoder, aux CE gradients nonzero to primary/depth encoders and aux heads, and aux CE gradients zero to fusion/decoder.
- No dataset split, dataloader, augmentation, metric, optimizer, scheduler, batch size, epoch count, learning rate, early stopping, DFormerv2-S level, pretrained loading, or validation logic was changed.

## Decision

R037 is stable but below R016. The small best-to-last drop suggests DGL reduces late instability, but the peak is too low; do not promote this model or tune aux weight as a micro-search. Pivot to a distinct R038 fusion-operator hypothesis such as KTB DSCF-lite c4-only if continuing.
