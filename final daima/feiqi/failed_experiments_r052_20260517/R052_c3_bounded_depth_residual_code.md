# R052 C3 Bounded Depth Residual Code Archive

Status: rejected as active mainline after full train.

Evidence:

- Run: `R052_c3_bounded_depth_residual_run01`
- Best val/mIoU: `0.535289` at validation epoch `31`
- Last val/mIoU: `0.515195`
- Best-to-last drop: `0.020095`
- c3 residual alpha first/last: `0.025108` / `0.026631`
- mIoU detail: `final daima/miou_list/R052_c3_bounded_depth_residual_run01.md`

Decision: negative below R016 `0.541121` and below R036 `0.539790`; do not keep in active registry or continue c3-only residual micro-search.

## Archived Implementation

```python
class GatedFusionC3BoundedDepthResidual(nn.Module):
    def __init__(self, rgb_channels, depth_channels, alpha_max=0.05):
        super().__init__()
        self.base = GatedFusion(rgb_channels, depth_channels)
        self.alpha_max = float(alpha_max)
        self.alpha_logit = nn.Parameter(torch.tensor(0.0))
        self.residual = nn.Sequential(
            nn.Conv2d(rgb_channels * 2, rgb_channels, 1, bias=False),
            nn.BatchNorm2d(rgb_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(rgb_channels, rgb_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(rgb_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(rgb_channels, rgb_channels, 1),
        )
        nn.init.zeros_(self.residual[-1].weight)
        nn.init.zeros_(self.residual[-1].bias)
        self.register_buffer("last_residual_abs", torch.zeros(()), persistent=False)

    def alpha(self):
        return self.alpha_max * torch.sigmoid(self.alpha_logit)

    def forward(self, rgb_feat, depth_feat):
        base = self.base(rgb_feat, depth_feat)
        d = self.base.depth_proj(depth_feat)
        residual = self.residual(torch.cat([d, torch.abs(rgb_feat - d)], dim=1))
        self.last_residual_abs.copy_(residual.detach().abs().mean())
        return base + self.alpha() * residual
```

```python
class DFormerV2C3BoundedDepthResidualSegmentor(DFormerV2MidFusionSegmentor):
    def __init__(self, num_classes=40, dformerv2_pretrained=None):
        super().__init__(num_classes=num_classes, dformerv2_pretrained=dformerv2_pretrained)
        self.fusions[2] = GatedFusionC3BoundedDepthResidual(
            self.rgb_encoder.out_channels[2],
            self.depth_encoder.out_channels[2],
            alpha_max=0.05,
        )
```

```python
class LitDFormerV2C3BoundedDepthResidual(BaseLitSeg):
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
        self.model = DFormerV2C3BoundedDepthResidualSegmentor(
            num_classes=num_classes,
            dformerv2_pretrained=dformerv2_pretrained,
        )

    def training_step(self, batch, batch_idx):
        loss = super().training_step(batch, batch_idx)
        c3_fusion = self.model.fusions[2]
        self.log("train/c3_residual_alpha", c3_fusion.alpha(), prog_bar=False, on_step=False, on_epoch=True)
        self.log("train/c3_residual_abs", c3_fusion.last_residual_abs, prog_bar=False, on_step=False, on_epoch=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=0.01)
```
