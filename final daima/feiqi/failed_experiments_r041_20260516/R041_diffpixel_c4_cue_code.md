# R041 DiffPixel C4 Cue Archived Code

R041 tested a DiffPixelFormer-style RGB-depth differential cue only at c4.
The run completed 50 validation epochs with best val/mIoU `0.537098` and last
val/mIoU `0.529552`, which is stronger than R040/R038/R037 but below R016
`0.541121`. The implementation is archived here and removed from the active
registry.

## Archived Module

```python
class DiffPixelC4CueFusion(nn.Module):
    def __init__(self, rgb_channels, depth_channels, reduction=4):
        super().__init__()
        hidden = max(rgb_channels // reduction, 32)
        self.depth_proj = nn.Conv2d(depth_channels, rgb_channels, 1)
        self.gate_conv = nn.Conv2d(rgb_channels * 2, rgb_channels, 1, bias=False)
        self.gate_bn = nn.BatchNorm2d(rgb_channels)
        self.diff_gate = nn.Sequential(
            nn.Conv2d(rgb_channels * 2, hidden, 1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.GELU(),
            nn.Conv2d(hidden, rgb_channels, 1, bias=False),
        )
        nn.init.zeros_(self.diff_gate[-1].weight)
        self.refine = nn.Sequential(
            nn.Conv2d(rgb_channels, rgb_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(rgb_channels),
            nn.ReLU(inplace=True),
        )
        self.last_diff_gate_abs = torch.tensor(0.0)
        self.last_gate_mean = torch.tensor(0.0)
        self.last_gate_std = torch.tensor(0.0)
        self.last_diff_context_abs = torch.tensor(0.0)

    def forward(self, rgb_feat, depth_feat):
        d = self.depth_proj(depth_feat)
        diff = rgb_feat - d
        diff_context = torch.cat([diff, diff.abs()], dim=1)
        base_logit = self.gate_bn(self.gate_conv(torch.cat([rgb_feat, d], dim=1)))
        diff_logit = self.diff_gate(diff_context)
        g = torch.sigmoid(base_logit + diff_logit)
        fused = g * rgb_feat + (1 - g) * d
        self.last_diff_gate_abs = diff_logit.detach().abs().mean()
        self.last_gate_mean = g.detach().mean()
        self.last_gate_std = g.detach().std()
        self.last_diff_context_abs = diff_context.detach().abs().mean()
        return self.refine(fused)
```

```python
class DFormerV2DiffPixelC4CueSegmentor(DFormerV2MidFusionSegmentor):
    def __init__(self, num_classes=40, dformerv2_pretrained=None):
        super().__init__(num_classes=num_classes, dformerv2_pretrained=dformerv2_pretrained)
        self.fusions[3] = DiffPixelC4CueFusion(
            self.rgb_encoder.out_channels[3],
            self.depth_encoder.out_channels[3],
        )
```

```python
class LitDFormerV2DiffPixelC4Cue(BaseLitSeg):
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
        self.model = DFormerV2DiffPixelC4CueSegmentor(
            num_classes=num_classes,
            dformerv2_pretrained=dformerv2_pretrained,
        )

    def training_step(self, batch, batch_idx):
        loss = super().training_step(batch, batch_idx)
        c4_fusion = self.model.fusions[3]
        self.log("train/diffpixel_c4_gate_mean", c4_fusion.last_gate_mean, prog_bar=False, on_step=False, on_epoch=True)
        self.log("train/diffpixel_c4_gate_std", c4_fusion.last_gate_std, prog_bar=False, on_step=False, on_epoch=True)
        self.log("train/diffpixel_c4_diff_gate_abs", c4_fusion.last_diff_gate_abs, prog_bar=False, on_step=False, on_epoch=True)
        self.log("train/diffpixel_c4_diff_context_abs", c4_fusion.last_diff_context_abs, prog_bar=False, on_step=False, on_epoch=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=0.01)
```

## Archived Registry Entry

```python
"dformerv2_diffpixel_c4_cue": LitDFormerV2DiffPixelC4Cue
```

The model used the fixed recipe and checkpoint directory
`checkpoints/R041_diffpixel_c4_cue_run01`.
