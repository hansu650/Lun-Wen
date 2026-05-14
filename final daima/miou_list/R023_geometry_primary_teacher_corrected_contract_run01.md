# R023 Geometry Primary Teacher Corrected Contract Run01 mIoU

- branch: `exp/R023-corrected-contract-teacher-refresh-v1`
- model: `dformerv2_geometry_primary_teacher`
- run: `R023_geometry_primary_teacher_corrected_contract_run01`
- hypothesis: refreshing the geometry-primary teacher under the corrected R015/R016 label/depth contract can decide whether corrected-contract PMAD deserves another student experiment.
- status: full train completed with exit code `0`; `Trainer.fit` reached `max_epochs=50`.
- TensorBoard event: `checkpoints/R023_geometry_primary_teacher_corrected_contract_run01/lightning_logs/version_0/events.out.tfevents.1778787920.Administrator.37368.0`
- best checkpoint: `checkpoints/R023_geometry_primary_teacher_corrected_contract_run01/dformerv2_geometry_primary_teacher-epoch=42-val_mIoU=0.5245.pt`

## Summary

- recorded validation epochs: `50`
- best val/mIoU: `0.524498` at validation epoch `43`
- last val/mIoU: `0.507023`
- last-5 mean val/mIoU: `0.510467`
- last-10 mean val/mIoU: `0.512531`
- best-to-last drop: `0.017475`
- best val/loss: `0.988437` at validation epoch `10`
- final train/loss_epoch: `0.092329`
- comparison: below R016 corrected baseline `0.541121` by `0.016623` and below the `0.53` PMAD-teacher gate.
- decision: negative teacher gate. Do not run corrected PMAD student from this teacher; pivot to a non-KD structure isolation experiment.

## Per-Epoch val/mIoU

| validation epoch | val/mIoU |
|---:|---:|
| 1 | 0.151010 |
| 2 | 0.223774 |
| 3 | 0.275515 |
| 4 | 0.330405 |
| 5 | 0.375746 |
| 6 | 0.405989 |
| 7 | 0.428819 |
| 8 | 0.444435 |
| 9 | 0.464256 |
| 10 | 0.488137 |
| 11 | 0.479618 |
| 12 | 0.471422 |
| 13 | 0.475075 |
| 14 | 0.491686 |
| 15 | 0.498388 |
| 16 | 0.485576 |
| 17 | 0.483773 |
| 18 | 0.506403 |
| 19 | 0.493791 |
| 20 | 0.495492 |
| 21 | 0.496666 |
| 22 | 0.494738 |
| 23 | 0.506694 |
| 24 | 0.506457 |
| 25 | 0.517007 |
| 26 | 0.511858 |
| 27 | 0.512718 |
| 28 | 0.466491 |
| 29 | 0.508717 |
| 30 | 0.507907 |
| 31 | 0.491761 |
| 32 | 0.495528 |
| 33 | 0.497686 |
| 34 | 0.514763 |
| 35 | 0.518866 |
| 36 | 0.513526 |
| 37 | 0.511630 |
| 38 | 0.447368 |
| 39 | 0.510707 |
| 40 | 0.516409 |
| 41 | 0.500777 |
| 42 | 0.509042 |
| 43 | 0.524498 |
| 44 | 0.521755 |
| 45 | 0.516908 |
| 46 | 0.516770 |
| 47 | 0.521004 |
| 48 | 0.510136 |
| 49 | 0.497401 |
| 50 | 0.507023 |
