# R042 DiffPixel C3-to-C4 Cue Archived Code

R042 tested a cross-stage RGB-depth differential cue: c1-c3 retained the original
`GatedFusion` outputs, while c4 gate logits received a zero-initialized cue
computed from c3 `[rgb-depth_proj, abs(rgb-depth_proj)]`.

The run completed 50 validation epochs with best val/mIoU `0.530729` and last
val/mIoU `0.458179`, below R041/R036/R016. The implementation is archived here
and removed from the active registry.

## Archived Module

```python
class C3ToC4DiffPixelCueFusion(nn.Module):
    def __init__(self, c4_rgb_channels, c4_depth_channels, c3_rgb_channels, c3_depth_channels, reduction=4):
        super().__init__()
        hidden = max(c4_rgb_channels // reduction, 32)
        self.c4_depth_proj = nn.Conv2d(c4_depth_channels, c4_rgb_channels, 1)
        self.c3_depth_proj = nn.Conv2d(c3_depth_channels, c3_rgb_channels, 1)
        self.gate_conv = nn.Conv2d(c4_rgb_channels * 2, c4_rgb_channels, 1, bias=False)
        self.gate_bn = nn.BatchNorm2d(c4_rgb_channels)
        self.c3_cue = nn.Sequential(
            nn.Conv2d(c3_rgb_channels * 2, hidden, 1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.GELU(),
            nn.Conv2d(hidden, c4_rgb_channels, 1, bias=False),
        )
        nn.init.zeros_(self.c3_cue[-1].weight)
        self.refine = nn.Sequential(
            nn.Conv2d(c4_rgb_channels, c4_rgb_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(c4_rgb_channels),
            nn.ReLU(inplace=True),
        )
        self.last_c3_cue_abs = torch.tensor(0.0)
        self.last_c3_diff_context_abs = torch.tensor(0.0)
        self.last_gate_mean = torch.tensor(0.0)
        self.last_gate_std = torch.tensor(0.0)

    def forward(self, c4_rgb, c4_depth, c3_rgb, c3_depth):
        c4_d = self.c4_depth_proj(c4_depth)
        c3_d = self.c3_depth_proj(c3_depth)
        c3_diff = c3_rgb - c3_d
        c3_context = torch.cat([c3_diff, c3_diff.abs()], dim=1)
        c3_logit = self.c3_cue(c3_context)
        c3_logit = F.interpolate(c3_logit, size=c4_rgb.shape[-2:], mode="bilinear", align_corners=False)
        base_logit = self.gate_bn(self.gate_conv(torch.cat([c4_rgb, c4_d], dim=1)))
        gate = torch.sigmoid(base_logit + c3_logit)
        fused = gate * c4_rgb + (1 - gate) * c4_d
        self.last_c3_cue_abs = c3_logit.detach().abs().mean()
        self.last_c3_diff_context_abs = c3_context.detach().abs().mean()
        self.last_gate_mean = gate.detach().mean()
        self.last_gate_std = gate.detach().std()
        return self.refine(fused)
```

```python
class DFormerV2DiffPixelC3ToC4CueSegmentor(DFormerV2MidFusionSegmentor):
    def __init__(self, num_classes=40, dformerv2_pretrained=None):
        super().__init__(num_classes=num_classes, dformerv2_pretrained=dformerv2_pretrained)
        self.c4_fusion = C3ToC4DiffPixelCueFusion(
            self.rgb_encoder.out_channels[3],
            self.depth_encoder.out_channels[3],
            self.rgb_encoder.out_channels[2],
            self.depth_encoder.out_channels[2],
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
        fused_c4 = self.c4_fusion(
            dformer_feats[3],
            aligned_depth[3],
            dformer_feats[2],
            aligned_depth[2],
        )
        fused_feats.append(fused_c4)
        return dformer_feats, aligned_depth, fused_feats
```

```python
class LitDFormerV2DiffPixelC3ToC4Cue(BaseLitSeg):
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
        self.model = DFormerV2DiffPixelC3ToC4CueSegmentor(
            num_classes=num_classes,
            dformerv2_pretrained=dformerv2_pretrained,
        )

    def training_step(self, batch, batch_idx):
        loss = super().training_step(batch, batch_idx)
        c4_fusion = self.model.c4_fusion
        self.log("train/c3toc4_cue_abs", c4_fusion.last_c3_cue_abs, prog_bar=False, on_step=False, on_epoch=True)
        self.log("train/c3toc4_diff_context_abs", c4_fusion.last_c3_diff_context_abs, prog_bar=False, on_step=False, on_epoch=True)
        self.log("train/c3toc4_gate_mean", c4_fusion.last_gate_mean, prog_bar=False, on_step=False, on_epoch=True)
        self.log("train/c3toc4_gate_std", c4_fusion.last_gate_std, prog_bar=False, on_step=False, on_epoch=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=0.01)
```

## Archived Registry Entry

```python
"dformerv2_diffpixel_c3toc4_cue": LitDFormerV2DiffPixelC3ToC4Cue
```

The model used the fixed recipe and checkpoint directory
`checkpoints/R042_diffpixel_c3toc4_cue_run01`.
