# dformerv2_dgc_af_plus Run01-Run04 Summary

- model: `dformerv2_dgc_af_plus`
- status: 4 completed runs, each with 50 validation epochs
- settings: `batch_size=2`, `max_epochs=50`, `lr=6e-5`, `num_workers=4`, `early_stop_patience=30`
- mean best val/mIoU: `0.511418`
- population std best val/mIoU: `0.001379`
- min best val/mIoU: `0.510157`
- max best val/mIoU: `0.513584`
- comparison baseline mean best: `0.513406`
- mean delta vs baseline mean: `-0.001988`
- conclusion: repeated runs do not support promoting DGC-AF++ as a stable improvement over the repeated DFormerv2 mid-fusion baseline.

| run | recorded validation epochs | best val/mIoU | best epoch | last val/mIoU | best val/loss | best loss epoch | delta vs baseline mean | evidence |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| `dformerv2_dgc_af_plus_run01` | 50 | 0.513584 | 50 | 0.513584 | 1.034468 | 8 | +0.000178 | `miou_list/dformerv2_dgc_af_plus_run01.md` |
| `dformerv2_dgc_af_plus_run02` | 50 | 0.511645 | 48 | 0.505630 | 1.039455 | 10 | -0.001761 | `miou_list/dformerv2_dgc_af_plus_run02.md` |
| `dformerv2_dgc_af_plus_run03` | 50 | 0.510157 | 50 | 0.510157 | 1.045118 | 10 | -0.003249 | `miou_list/dformerv2_dgc_af_plus_run03.md` |
| `dformerv2_dgc_af_plus_run04` | 50 | 0.510288 | 50 | 0.510288 | 1.053412 | 9 | -0.003118 | `miou_list/dformerv2_dgc_af_plus_run04.md` |
