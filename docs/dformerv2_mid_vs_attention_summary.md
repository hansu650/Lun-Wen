# DFormerv2 Mid Fusion vs Attention Fusion Summary

- Date recorded: `2026-05-03`.
- Baseline: `dformerv2_mid_fusion` = DFormerv2_S + ResNet18 DepthEncoder + original GatedFusion + SimpleFPNDecoder.
- Attention: `dformerv2_attention_fusion` = DFormerv2_S + ResNet18 DepthEncoder + CrossModalReliabilityAttentionFusion + SimpleFPNDecoder.
- Pretrained: `C:\Users\qintian\Desktop\qintian\dformer_work\checkpoints\pretrained\DFormerv2_Small_pretrained.pth`.

## Baseline dformerv2_mid_fusion Repeat Runs

| run | best epoch | best val/mIoU | last val/mIoU | mIoU file |
|---|---:|---:|---:|---|
| dformerv2_mid_fusion_repeat_run01 | 49 | 0.513287 | 0.513287 | `miou_list\dformerv2_mid_fusion_repeat_run01.md` |
| dformerv2_mid_fusion_repeat_run02 | 38 | 0.509708 | 0.498344 | `miou_list\dformerv2_mid_fusion_repeat_run02.md` |
| dformerv2_mid_fusion_repeat_run03 | 44 | 0.515470 | 0.501154 | `miou_list\dformerv2_mid_fusion_repeat_run03.md` |
| dformerv2_mid_fusion_repeat_run04 | 46 | 0.515157 | 0.493394 | `miou_list\dformerv2_mid_fusion_repeat_run04.md` |

- Baseline mean best: `0.513406`.
- Baseline std best: `0.002647`.
- Baseline min best: `0.509708`.
- Baseline max best: `0.515470`.

## Attention dformerv2_attention_fusion Runs

| run | best epoch | best val/mIoU | last val/mIoU |
|---|---:|---:|---:|
| dformerv2_attention_fusion_run01 | 45 | 0.516979 | 0.508818 |
| dformerv2_attention_fusion_run02 | 46 | 0.507672 | 0.507041 |
| dformerv2_attention_fusion_run03 | 42 | 0.513333 | 0.487366 |
| dformerv2_attention_fusion_run04 | 49 | 0.518196 | 0.518196 |
| dformerv2_attention_fusion_run05 | 49 | 0.515997 | 0.515997 |

- Attention mean best: `0.514435`.
- Attention std best: `0.004184`.
- Attention min best: `0.507672`.
- Attention max best: `0.518196`.

## Comparison

- Mean best gain: `+0.001030`.
- Max best gain: `+0.002726`.
- Conclusion: attention fusion has a small positive average gain in these recorded runs.
