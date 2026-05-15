# R036 c3/c4 Bounded Depth Residual Code Archive

Status: rejected as active mainline after full train.

Evidence:

- Run: `R036_c34_bounded_depth_residual_run01`
- Best val/mIoU: `0.539790` at validation epoch `44`
- Last val/mIoU: `0.528882`
- Best-to-last drop: `0.010908`
- c3 residual alpha first/last: `0.025097` / `0.026970`
- c4 residual alpha first/last: `0.025034` / `0.025553`
- mIoU detail: `final daima/miou_list/R036_c34_bounded_depth_residual_run01.md`

Decision: partial-positive below R016 `0.541121`; do not keep in active registry or continue bounded-residual micro-search.

## Archived Implementation

```python
class GatedFusionC34BoundedDepthResidual(nn.Module):
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

    def forward(self, rgb_feat, depth_feat):
        base = self.base(rgb_feat, depth_feat)
        d = self.base.depth_proj(depth_feat)
        residual = self.residual(torch.cat([d, torch.abs(rgb_feat - d)], dim=1))
        alpha = self.alpha_max * torch.sigmoid(self.alpha_logit)
        return base + alpha * residual

    def alpha(self):
        return self.alpha_max * torch.sigmoid(self.alpha_logit)
```

```python
class DFormerV2C34BoundedDepthResidualSegmentor(DFormerV2MidFusionSegmentor):
    def __init__(self, num_classes=40, dformerv2_pretrained=None):
        super().__init__(num_classes=num_classes, dformerv2_pretrained=dformerv2_pretrained)
        self.fusions[2] = GatedFusionC34BoundedDepthResidual(
            self.rgb_encoder.out_channels[2],
            self.depth_encoder.out_channels[2],
            alpha_max=0.05,
        )
        self.fusions[3] = GatedFusionC34BoundedDepthResidual(
            self.rgb_encoder.out_channels[3],
            self.depth_encoder.out_channels[3],
            alpha_max=0.05,
        )
```

```python
class LitDFormerV2C34BoundedDepthResidual(BaseLitSeg):
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
        self.model = DFormerV2C34BoundedDepthResidualSegmentor(
            num_classes=num_classes,
            dformerv2_pretrained=dformerv2_pretrained,
        )

    def training_step(self, batch, batch_idx):
        loss = super().training_step(batch, batch_idx)
        self.log("train/c3_residual_alpha", self.model.fusions[2].alpha(), prog_bar=False, on_step=False, on_epoch=True)
        self.log("train/c4_residual_alpha", self.model.fusions[3].alpha(), prog_bar=False, on_step=False, on_epoch=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=0.01)
```
