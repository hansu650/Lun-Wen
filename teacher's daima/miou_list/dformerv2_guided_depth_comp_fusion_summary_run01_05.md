# dformerv2_guided_depth_comp_fusion run01-run05 Summary

- model: `dformerv2_guided_depth_comp_fusion`
- settings: `batch_size=2, max_epochs=50, lr=6e-5, num_workers=4, early_stop_patience=30`
- pretrained: `C:/Users/qintian/Desktop/qintian/dformer_work/checkpoints/pretrained/DFormerv2_Small_pretrained.pth`
- completed runs: `5`
- mean best val/mIoU: `0.511379`
- population std best val/mIoU: `0.002651`
- min best val/mIoU: `0.508686`
- max best val/mIoU: `0.515870`
- mean last val/mIoU: `0.496943`
- comparison baseline mean best: `0.513406`
- mean delta vs baseline mean: `-0.002027`
- comparison DGC-AF++ mean best: `0.511418`
- mean delta vs DGC-AF++ mean best: `-0.000039`

| Run | Best val/mIoU | Best epoch | Last val/mIoU | Best val/loss | Best loss epoch | Detail file |
|---|---:|---:|---:|---:|---:|---|
| run01 | 0.508875 | 46 | 0.481020 | 1.052861 | 8 | `miou_list/dformerv2_guided_depth_comp_fusion_run01.md` |
| run02 | 0.512506 | 39 | 0.507708 | 1.024278 | 9 | `miou_list/dformerv2_guided_depth_comp_fusion_run02.md` |
| run03 | 0.508686 | 49 | 0.483381 | 1.040731 | 7 | `miou_list/dformerv2_guided_depth_comp_fusion_run03.md` |
| run04 | 0.510958 | 45 | 0.505383 | 1.007499 | 10 | `miou_list/dformerv2_guided_depth_comp_fusion_run04.md` |
| run05 | 0.515870 | 45 | 0.507223 | 1.059600 | 7 | `miou_list/dformerv2_guided_depth_comp_fusion_run05.md` |
