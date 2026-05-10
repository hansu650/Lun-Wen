# dformerv2_mid_fusion_gate_baseline repeat5 summary

- model: `dformerv2_mid_fusion`
- fusion: baseline `GatedFusion` mid-fusion
- purpose: baseline sanity check after C4 PPM and CE+Dice experiments both yielded ~0.507
- settings: `batch_size=2`, `max_epochs=50`, `lr=6e-5`, `num_workers=4`, `early_stop_patience=30`, `loss_type=ce`
- pretrained: `C:/Users/qintian/Desktop/qintian/dformer_work/checkpoints/pretrained/DFormerv2_Small_pretrained.pth`
- complete runs: `5`
- code diff vs original baseline: none (purely additive changes to new model classes; `dformerv2_mid_fusion` path untouched)

| run | recorded epochs | best val/mIoU | best epoch | last val/mIoU | best val/loss | best loss epoch | mean last10 mIoU |
|---|---:|---:|---:|---:|---:|---:|---:|
| run01 | 50 | 0.509013 | 32 | 0.496441 | 1.000347 | 10 | 0.481051 |
| run02 | 50 | 0.510994 | 45 | 0.475698 | 1.039783 | 12 | 0.497885 |
| run03 | 50 | 0.510176 | 46 | 0.466466 | 1.028617 | 13 | 0.492659 |
| run04 | 50 | 0.517698 | 42 | 0.512032 | 1.011931 | 10 | 0.505787 |
| run05 | 50 | 0.511585 | 39 | 0.476304 | 1.024783 | 11 | 0.486855 |

## Repeat5 statistics

- mean best val/mIoU: `0.511893`
- population std best val/mIoU: `0.003028`
- min best: `0.509013` (run01)
- max best: `0.517698` (run04)
- mean last val/mIoU: `0.485388`
- mean best val/loss: `1.021093`
- mean best loss epoch: `11.2`

## Original 10-run baseline reference

- mean best val/mIoU: `0.517397`
- population std: `0.004901`
- best single: `0.524425` (run05)
- min best: `0.505699` (run07)

## Comparison

| metric | original 10-run | repeat 5-run | delta |
|--------|-----------------|-------------|-------|
| mean best mIoU | 0.517397 | 0.511893 | -0.005504 |
| std | 0.004901 | 0.003028 | -0.001873 |
| min best | 0.505699 | 0.509013 | +0.003314 |
| max best | 0.524425 | 0.517698 | -0.006727 |
| mean best loss | ~1.017 | 1.021 | +0.004 |
| mean best loss epoch | ~10 | 11.2 | +1.2 |

## Key observations

1. **Repeat5 mean (0.5119) is 0.0055 lower than original mean (0.5174)** — about 1.12 original std units below.
2. **No run exceeded 0.518**, while original had 5 runs above 0.517. The upper tail is missing.
3. **Best single repeat = 0.5177**, comparable to original mean — consistent with a downward shift.
4. **Lower tail is actually higher**: repeat min (0.5090) > original min (0.5057). The repeat distribution is compressed.
5. **Train loss slightly higher** (1.021 vs ~1.017), suggesting marginally worse optimization, but within noise.
6. **All runs completed 50 epochs** without early stopping — no divergence issue.
7. **Conclusion**: the repeat baseline is ~0.005 below the original. This gap is moderate (~1.1σ) and could be due to:
   - Random seed / initialization variance with only 5 runs (vs 10)
   - Environmental drift (GPU thermal state, background processes)
   - Non-determinism in cuDNN / data loading order
   - The two 0.507 experiments (CE+Dice, C4 PPM) are now only ~0.005 below repeat mean (~1.6σ), making them less anomalous relative to this repeat baseline

## Evidence files

- `miou_list/dformerv2_mid_fusion_gate_baseline_repeat5_run01.md`
- `miou_list/dformerv2_mid_fusion_gate_baseline_repeat5_run02.md`
- `miou_list/dformerv2_mid_fusion_gate_baseline_repeat5_run03.md`
- `miou_list/dformerv2_mid_fusion_gate_baseline_repeat5_run04.md`
- `miou_list/dformerv2_mid_fusion_gate_baseline_repeat5_run05.md`
