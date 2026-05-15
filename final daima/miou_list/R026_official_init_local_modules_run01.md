# R026 Official Init Local Modules Run01 mIoU

- branch: `exp/R026-official-init-local-modules-v1`
- model: `dformerv2_official_init_local_modules`
- run: `R026_official_init_local_modules_run01`
- hypothesis: local random fusion/decoder modules may be under-initialized relative to the official DFormer decode head contract.
- implementation: official-style initialization is applied only to `GatedFusion` and `SimpleFPNDecoder`; pretrained DFormerv2 and DepthEncoder weights are untouched.
- status: full train completed with exit code `0`; `Trainer.fit` reached `max_epochs=50`.
- TensorBoard event: `checkpoints/R026_official_init_local_modules_run01/lightning_logs/version_0/events.out.tfevents.1778803189.Administrator.35684.0`
- best checkpoint: `checkpoints/R026_official_init_local_modules_run01/dformerv2_official_init_local_modules-epoch=32-val_mIoU=0.5079.pt`

## Summary

- recorded validation epochs: `50`
- best val/mIoU: `0.507906` at validation epoch `33`
- last val/mIoU: `0.499770`
- last-5 mean val/mIoU: `0.496476`
- last-10 mean val/mIoU: `0.495483`
- best-to-last drop: `0.008136`
- best val/loss: `1.073346` at validation epoch `5`
- final train/loss_epoch: `0.054818`
- comparison: below R016 `0.541121` by `0.033215` and below R025 `0.532572` by `0.024666`.
- decision: negative. Official-style init of local random modules hurts this pipeline and should not be continued.

## Per-Epoch val/mIoU

| validation epoch | val/mIoU |
|---:|---:|
| 1 | 0.188990 |
| 2 | 0.285925 |
| 3 | 0.344759 |
| 4 | 0.387152 |
| 5 | 0.419305 |
| 6 | 0.437324 |
| 7 | 0.445768 |
| 8 | 0.452809 |
| 9 | 0.464645 |
| 10 | 0.462506 |
| 11 | 0.472856 |
| 12 | 0.452405 |
| 13 | 0.476299 |
| 14 | 0.485243 |
| 15 | 0.474594 |
| 16 | 0.469684 |
| 17 | 0.470790 |
| 18 | 0.471872 |
| 19 | 0.482802 |
| 20 | 0.482549 |
| 21 | 0.488990 |
| 22 | 0.497487 |
| 23 | 0.491709 |
| 24 | 0.487931 |
| 25 | 0.498292 |
| 26 | 0.445757 |
| 27 | 0.486633 |
| 28 | 0.487630 |
| 29 | 0.500446 |
| 30 | 0.506067 |
| 31 | 0.504591 |
| 32 | 0.503945 |
| 33 | 0.507906 |
| 34 | 0.455814 |
| 35 | 0.477928 |
| 36 | 0.467490 |
| 37 | 0.497530 |
| 38 | 0.498324 |
| 39 | 0.503083 |
| 40 | 0.504576 |
| 41 | 0.499041 |
| 42 | 0.503381 |
| 43 | 0.496889 |
| 44 | 0.469765 |
| 45 | 0.503373 |
| 46 | 0.497338 |
| 47 | 0.489185 |
| 48 | 0.492323 |
| 49 | 0.503766 |
| 50 | 0.499770 |
