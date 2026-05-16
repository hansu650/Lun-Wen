# R043 Archived Code: DepthGeo c4 Cue

Archived after full-train result:

- best val/mIoU: `0.535592` at validation epoch `42`
- last val/mIoU: `0.522214`
- status: partial geometry-cue signal below R041/R036/R016
- decision: remove from active `train.py` registry and do not tune Sobel/normal c4 cue micro-variants

## Model Entry

```python
class DepthGeometryC4CueFusion(nn.Module):
    def __init__(self, rgb_channels, depth_channels, reduction=4):
        super().__init__()
        hidden = max(rgb_channels // reduction, 32)
        self.depth_proj = nn.Conv2d(depth_channels, rgb_channels, 1)
        self.gate_conv = nn.Conv2d(rgb_channels * 2, rgb_channels, 1, bias=False)
        self.gate_bn = nn.BatchNorm2d(rgb_channels)
        self.geo_gate = nn.Sequential(
            nn.Conv2d(4, hidden, 3, padding=1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.GELU(),
            nn.Conv2d(hidden, rgb_channels, 1, bias=False),
        )
        nn.init.zeros_(self.geo_gate[-1].weight)
        self.refine = nn.Sequential(
            nn.Conv2d(rgb_channels, rgb_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(rgb_channels),
            nn.ReLU(inplace=True),
        )
        sobel_x = torch.tensor(
            [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]
        ).view(1, 1, 3, 3)
        sobel_y = torch.tensor(
            [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]]
        ).view(1, 1, 3, 3)
        self.register_buffer("sobel_x", sobel_x, persistent=False)
        self.register_buffer("sobel_y", sobel_y, persistent=False)
        self.last_geo_logit_abs = torch.tensor(0.0)
        self.last_gate_mean = torch.tensor(0.0)
        self.last_gate_std = torch.tensor(0.0)
        self.last_depth_edge_mean = torch.tensor(0.0)
        self.last_depth_edge_std = torch.tensor(0.0)

    def _geometry_cue(self, depth, target_size):
        dx = F.conv2d(depth, self.sobel_x.to(dtype=depth.dtype), padding=1)
        dy = F.conv2d(depth, self.sobel_y.to(dtype=depth.dtype), padding=1)
        mag = torch.sqrt(dx.square() + dy.square() + 1e-6)
        inv_norm = torch.rsqrt(dx.square() + dy.square() + 1.0)
        nx = -dx * inv_norm
        ny = -dy * inv_norm
        nz = inv_norm
        cue = torch.cat([nx, ny, nz, mag], dim=1)
        mean = cue.mean(dim=(2, 3), keepdim=True)
        std = cue.std(dim=(2, 3), keepdim=True).clamp_min(1e-6)
        cue = (cue - mean) / std
        cue = F.interpolate(cue, size=target_size, mode="bilinear", align_corners=False)
        self.last_depth_edge_mean = mag.detach().mean()
        self.last_depth_edge_std = mag.detach().std()
        return cue

    def forward(self, rgb_feat, depth_feat, depth):
        depth_projected = self.depth_proj(depth_feat)
        base_logit = self.gate_bn(self.gate_conv(torch.cat([rgb_feat, depth_projected], dim=1)))
        geo_logit = self.geo_gate(self._geometry_cue(depth, rgb_feat.shape[-2:]))
        gate = torch.sigmoid(base_logit + geo_logit)
        fused = gate * rgb_feat + (1 - gate) * depth_projected
        self.last_geo_logit_abs = geo_logit.detach().abs().mean()
        self.last_gate_mean = gate.detach().mean()
        self.last_gate_std = gate.detach().std()
        return self.refine(fused)
```

```python
class DFormerV2DepthGeoC4CueSegmentor(DFormerV2MidFusionSegmentor):
    def __init__(self, num_classes=40, dformerv2_pretrained=None):
        super().__init__(num_classes=num_classes, dformerv2_pretrained=dformerv2_pretrained)
        self.c4_fusion = DepthGeometryC4CueFusion(
            self.rgb_encoder.out_channels[3],
            self.depth_encoder.out_channels[3],
        )
        self.fusions[3] = nn.Identity()

    def extract_features(self, rgb, depth):
        dformer_feats = self.rgb_encoder(rgb, depth)
        depth_feats = self.depth_encoder(depth)

        aligned_depth = []
        for rf, df in zip(dformer_feats, depth_feats):
            if rf.shape[-2:] != df.shape[-2:]:
                df = F.interpolate(df, size=rf.shape[-2:], mode="bilinear", align_corners=False)
            aligned_depth.append(df)

        fused_feats = [
            fusion(r, d)
            for fusion, r, d in zip(self.fusions[:3], dformer_feats[:3], aligned_depth[:3])
        ]
        fused_feats.append(self.c4_fusion(dformer_feats[3], aligned_depth[3], depth))
        return dformer_feats, aligned_depth, fused_feats
```

```python
class LitDFormerV2DepthGeoC4Cue(BaseLitSeg):
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
        self.model = DFormerV2DepthGeoC4CueSegmentor(
            num_classes=num_classes,
            dformerv2_pretrained=dformerv2_pretrained,
        )

    def training_step(self, batch, batch_idx):
        loss = super().training_step(batch, batch_idx)
        c4_fusion = self.model.c4_fusion
        self.log("train/depthgeo_c4_geo_logit_abs", c4_fusion.last_geo_logit_abs, prog_bar=False, on_step=False, on_epoch=True)
        self.log("train/depthgeo_c4_gate_mean", c4_fusion.last_gate_mean, prog_bar=False, on_step=False, on_epoch=True)
        self.log("train/depthgeo_c4_gate_std", c4_fusion.last_gate_std, prog_bar=False, on_step=False, on_epoch=True)
        self.log("train/depthgeo_c4_edge_mean", c4_fusion.last_depth_edge_mean, prog_bar=False, on_step=False, on_epoch=True)
        self.log("train/depthgeo_c4_edge_std", c4_fusion.last_depth_edge_std, prog_bar=False, on_step=False, on_epoch=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=0.01)
```

## Former Registry Entry

```python
from src.models.mid_fusion import LitDFormerV2DepthGeoC4Cue

ACTIVE_MODEL_REGISTRY = {
    "dformerv2_depthgeo_c4_cue": LitDFormerV2DepthGeoC4Cue,
}

if args.model in {
    "dformerv2_depthgeo_c4_cue",
}:
    return model_cls(
        num_classes=args.num_classes,
        lr=args.lr,
        dformerv2_pretrained=args.dformerv2_pretrained,
        loss_type=args.loss_type,
    )
```
