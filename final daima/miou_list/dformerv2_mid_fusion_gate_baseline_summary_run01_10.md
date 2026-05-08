# dformerv2_mid_fusion_gate_baseline run01-run10 summary

- model: `dformerv2_mid_fusion`
- fusion: baseline `GatedFusion` mid-fusion
- settings: `batch_size=2`, `max_epochs=50`, `lr=6e-5`, `num_workers=4`, `early_stop_patience=30`
- pretrained: `C:/Users/qintian/Desktop/qintian/dformer_work/checkpoints/pretrained/DFormerv2_Small_pretrained.pth`
- complete runs: `9`
- partial runs: `1`

| run | status | recorded epochs | best val/mIoU | best epoch | last val/mIoU | best val/loss | best loss epoch |
|---|---|---:|---:|---:|---:|---:|---:|
| run01 | complete | 50 | 0.517621 | 49 | 0.516805 | 1.018484 | 13 |
| run02 | complete | 50 | 0.518642 | 49 | 0.512856 | 1.028006 | 11 |
| run03 | complete | 50 | 0.514432 | 40 | 0.506427 | 1.017887 | 10 |
| run04 | complete | 50 | 0.518596 | 38 | 0.492355 | 1.017909 | 9 |
| run05 | complete | 50 | 0.524425 | 50 | 0.524425 | 1.017244 | 9 |
| run06 | complete | 50 | 0.519506 | 31 | 0.507040 | 1.009728 | 9 |
| run07 | complete | 50 | 0.505699 | 45 | 0.502685 | 1.022507 | 10 |
| run08 | complete | 50 | 0.517555 | 49 | 0.516317 | 1.006927 | 11 |
| run09 | complete | 50 | 0.514622 | 49 | 0.469588 | 1.006296 | 9 |
| run10 | partial | 43 | 0.514412 | 38 | 0.503475 | 1.029115 | 10 |

## Complete-run statistics

- mean best val/mIoU: `0.516789`
- population std best val/mIoU: `0.004795`
- mean last val/mIoU: `0.505389`

## Partial-inclusive reference

- mean best val/mIoU including partial run10: `0.516551`
- population std best val/mIoU including partial run10: `0.004604`
- mean last val/mIoU including partial run10: `0.505197`

## Evidence files

- `miou_list/dformerv2_mid_fusion_gate_baseline_run01.md`
- `miou_list/dformerv2_mid_fusion_gate_baseline_run02.md`
- `miou_list/dformerv2_mid_fusion_gate_baseline_run03.md`
- `miou_list/dformerv2_mid_fusion_gate_baseline_run04.md`
- `miou_list/dformerv2_mid_fusion_gate_baseline_run05.md`
- `miou_list/dformerv2_mid_fusion_gate_baseline_run06.md`
- `miou_list/dformerv2_mid_fusion_gate_baseline_run07.md`
- `miou_list/dformerv2_mid_fusion_gate_baseline_run08.md`
- `miou_list/dformerv2_mid_fusion_gate_baseline_run09.md`
- `miou_list/dformerv2_mid_fusion_gate_baseline_run10.md`
