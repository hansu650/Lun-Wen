# R024 Geometry Primary Ham Decoder Run01 mIoU

- branch: `exp/R024-geometry-primary-ham-decoder-v1`
- model: `dformerv2_geometry_primary_ham_decoder`
- run: `R024_geometry_primary_ham_decoder_run01`
- hypothesis: raw DFormerv2-S features may match the official Ham decoder contract better than the local post-backbone external DepthEncoder/GatedFusion stack.
- implementation: `DFormerv2_S(rgb, depth) -> OfficialHamDecoder`, with no external DepthEncoder or GatedFusion.
- status: full train completed with exit code `0`; `Trainer.fit` reached `max_epochs=50`.
- TensorBoard event: `checkpoints/R024_geometry_primary_ham_decoder_run01/lightning_logs/version_0/events.out.tfevents.1778793121.Administrator.20368.0`
- best checkpoint: `checkpoints/R024_geometry_primary_ham_decoder_run01/dformerv2_geometry_primary_ham_decoder-epoch=44-val_mIoU=0.5302.pt`

## Summary

- recorded validation epochs: `50`
- best val/mIoU: `0.530186` at validation epoch `45`
- last val/mIoU: `0.529383`
- last-5 mean val/mIoU: `0.521843`
- last-10 mean val/mIoU: `0.522327`
- best-to-last drop: `0.000803`
- best val/loss: `1.062941` at validation epoch `8`
- final train/loss_epoch: `0.071329`
- comparison: above the fixed-recipe `0.53` stage threshold, below R022 `0.534332` by `0.004146`, and below R016 `0.541121` by `0.010935`.
- decision: stable but below the corrected baseline. Raw DFormerv2-S + Ham is not enough; the external fusion path still contributes useful performance.

## Per-Epoch val/mIoU

| validation epoch | val/mIoU |
|---:|---:|
| 1 | 0.139056 |
| 2 | 0.240687 |
| 3 | 0.317237 |
| 4 | 0.361415 |
| 5 | 0.387699 |
| 6 | 0.407392 |
| 7 | 0.428588 |
| 8 | 0.443435 |
| 9 | 0.450270 |
| 10 | 0.462728 |
| 11 | 0.467838 |
| 12 | 0.479195 |
| 13 | 0.437935 |
| 14 | 0.479394 |
| 15 | 0.485962 |
| 16 | 0.491956 |
| 17 | 0.488627 |
| 18 | 0.486001 |
| 19 | 0.489305 |
| 20 | 0.498021 |
| 21 | 0.499101 |
| 22 | 0.489491 |
| 23 | 0.484630 |
| 24 | 0.503256 |
| 25 | 0.507683 |
| 26 | 0.497613 |
| 27 | 0.508396 |
| 28 | 0.506546 |
| 29 | 0.510245 |
| 30 | 0.490497 |
| 31 | 0.509609 |
| 32 | 0.520473 |
| 33 | 0.505365 |
| 34 | 0.504268 |
| 35 | 0.506921 |
| 36 | 0.500957 |
| 37 | 0.519285 |
| 38 | 0.502606 |
| 39 | 0.515441 |
| 40 | 0.507383 |
| 41 | 0.518331 |
| 42 | 0.521396 |
| 43 | 0.517268 |
| 44 | 0.526878 |
| 45 | 0.530186 |
| 46 | 0.511697 |
| 47 | 0.522247 |
| 48 | 0.524969 |
| 49 | 0.520918 |
| 50 | 0.529383 |
