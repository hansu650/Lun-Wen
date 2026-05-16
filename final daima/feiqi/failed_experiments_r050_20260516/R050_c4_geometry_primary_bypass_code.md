# R050 C4 Geometry-Primary Bypass Archived Code

R050 was tested on branch `exp/R050-c4-geometry-primary-bypass-v1` as model `dformerv2_c4_geometry_primary_bypass`.

Result: best val/mIoU `0.533066`, last `0.526781`, below R016 `0.541121`. The active code was removed after recording evidence.

## Segmentor

```python
class DFormerV2C4GeometryPrimaryBypassSegmentor(nn.Module):
    def __init__(self, num_classes=40, dformerv2_pretrained=None):
        super().__init__()
        self.rgb_encoder = DFormerv2_S()
        if dformerv2_pretrained:
            self.pretrained_load_stats = load_dformerv2_pretrained(self.rgb_encoder, dformerv2_pretrained)
        else:
            self.pretrained_load_stats = None
        self.depth_encoder = DepthEncoder()
        self.fusions = nn.ModuleList([
            GatedFusion(rgb_ch, depth_ch)
            for rgb_ch, depth_ch in zip(self.rgb_encoder.out_channels[:3], self.depth_encoder.out_channels[:3])
        ])
        self.decoder = SimpleFPNDecoder(self.rgb_encoder.out_channels, num_classes=num_classes)

    def extract_features(self, rgb, depth):
        dformer_feats = self.rgb_encoder(rgb, depth)
        depth_feats = self.depth_encoder(depth)

        aligned_depth = []
        for rf, df in zip(dformer_feats[:3], depth_feats[:3]):
            if rf.shape[-2:] != df.shape[-2:]:
                df = F.interpolate(df, size=rf.shape[-2:], mode="bilinear", align_corners=False)
            aligned_depth.append(df)

        fused_feats = [
            fusion(r, d)
            for fusion, r, d in zip(self.fusions, dformer_feats[:3], aligned_depth)
        ]
        fused_feats.append(dformer_feats[3])
        return dformer_feats, aligned_depth, fused_feats

    def forward(self, rgb, depth):
        _, _, fused_feats = self.extract_features(rgb, depth)
        return self.decoder(fused_feats, input_size=rgb.shape[-2:])
```

## Lightning Wrapper

```python
class LitDFormerV2C4GeometryPrimaryBypass(BaseLitSeg):
    def __init__(
        self,
        num_classes=40,
        lr=1e-4,
        dformerv2_pretrained=None,
        loss_type: str = "ce",
        dice_weight: float = 0.5,
    ):
        super().__init__(
            num_classes=num_classes,
            lr=lr,
            loss_type=loss_type,
            dice_weight=dice_weight,
        )
        self.model = DFormerV2C4GeometryPrimaryBypassSegmentor(
            num_classes=num_classes,
            dformerv2_pretrained=dformerv2_pretrained,
        )

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=0.01)
```

## Registry Entry

```python
"dformerv2_c4_geometry_primary_bypass": LitDFormerV2C4GeometryPrimaryBypass
```
