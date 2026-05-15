# R034 MASG Gated Fusion Archived Code

R034 tested a gate-only depth stop-gradient variant. The experiment completed a full train but stayed below R016 and was unstable, so the active registry was cleaned after recording evidence.

## Result

- Run: `R034_masg_gated_fusion_run01`
- Model during experiment: `dformerv2_masg_fusion`
- Best val/mIoU: `0.539322` at validation epoch `40`
- Last val/mIoU: `0.518738`
- Best-to-last drop: `0.020584`
- Decision: negative/unstable relative to R016 `0.541121`

## Archived Implementation

```python
class GatedFusionGateStopGrad(nn.Module):
    def __init__(self, rgb_channels, depth_channels):
        super().__init__()
        self.depth_proj = nn.Conv2d(depth_channels, rgb_channels, 1)
        self.gate = nn.Sequential(
            nn.Conv2d(rgb_channels * 2, rgb_channels, 1, bias=False),
            nn.BatchNorm2d(rgb_channels),
            nn.Sigmoid(),
        )
        self.refine = nn.Sequential(
            nn.Conv2d(rgb_channels, rgb_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(rgb_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, rgb_feat, depth_feat):
        d = self.depth_proj(depth_feat)
        g = self.gate(torch.cat([rgb_feat, d.detach()], dim=1))
        fused = g * rgb_feat + (1 - g) * d
        return self.refine(fused)
```

```python
class DFormerV2MASGFusionSegmentor(DFormerV2MidFusionSegmentor):
    def __init__(self, num_classes=40, dformerv2_pretrained=None):
        super().__init__(num_classes=num_classes, dformerv2_pretrained=dformerv2_pretrained)
        self.fusions = nn.ModuleList([
            GatedFusionGateStopGrad(rgb_ch, depth_ch)
            for rgb_ch, depth_ch in zip(self.rgb_encoder.out_channels, self.depth_encoder.out_channels)
        ])
```

`LitDFormerV2MASGFusion` followed the same Lightning wrapper pattern as `LitDFormerV2MidFusion`, and `train.py` temporarily registered `dformerv2_masg_fusion`.
