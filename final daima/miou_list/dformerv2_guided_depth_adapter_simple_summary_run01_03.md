# dformerv2_guided_depth_adapter_simple run01-run03 Summary

- model: `dformerv2_guided_depth_adapter_simple`
- settings: `batch_size=2, max_epochs=50, lr=6e-5, num_workers=4, early_stop_patience=30`
- pretrained: `C:/Users/qintian/Desktop/qintian/dformer_work/checkpoints/pretrained/DFormerv2_Small_pretrained.pth`
- completed runs: `3`
- mean best val/mIoU: `0.514621`
- population std best val/mIoU: `0.003069`
- min best val/mIoU: `0.511741`
- max best val/mIoU: `0.518872`
- mean last val/mIoU: `0.491274`
- comparison baseline mean best: `0.513406`
- mean delta vs baseline mean: `+0.001215`
- comparison Full++ mean best: `0.511379`
- mean delta vs Full++ mean best: `+0.003242`
- comparison DGC-AF++ mean best: `0.511418`
- mean delta vs DGC-AF++ mean best: `+0.003203`

| Run | Best val/mIoU | Best epoch | Last val/mIoU | Best val/loss | Best loss epoch | Detail file |
|---|---:|---:|---:|---:|---:|---|
| run01 | 0.511741 | 37 | 0.490230 | 1.017727 | 9 | `miou_list/dformerv2_guided_depth_adapter_simple_run01.md` |
| run02 | 0.518872 | 48 | 0.481400 | 1.035822 | 10 | `miou_list/dformerv2_guided_depth_adapter_simple_run02.md` |
| run03 | 0.513250 | 45 | 0.502194 | 1.024035 | 8 | `miou_list/dformerv2_guided_depth_adapter_simple_run03.md` |
