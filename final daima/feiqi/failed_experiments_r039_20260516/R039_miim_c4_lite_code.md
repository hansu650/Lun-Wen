# R039 MIIM C4 Lite Archived Code

R039 tested a HDBFormer/MIIM-inspired c4-only global-local interaction residual.
The full train completed below R016, so the active registry was cleaned and this
implementation is archived here for reference.

## Fusion Module

```python
class MIIMC4LiteFusion(nn.Module):
    def __init__(self, rgb_channels, depth_channels, alpha_max=0.05, reduction=4):
        super().__init__()
        hidden_channels = max(rgb_channels // reduction, 32)
        context_channels = rgb_channels * 4
        self.alpha_max = float(alpha_max)
        self.alpha_logit = nn.Parameter(torch.tensor(-2.9444))
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
        self.global_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(context_channels, hidden_channels, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, rgb_channels, 1),
            nn.Sigmoid(),
        )
        self.local_update = nn.Sequential(
            nn.Conv2d(context_channels, hidden_channels, 1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.GELU(),
            nn.Conv2d(
                hidden_channels,
                hidden_channels,
                7,
                padding=3,
                groups=hidden_channels,
                bias=False,
            ),
            nn.BatchNorm2d(hidden_channels),
            nn.GELU(),
            nn.Conv2d(hidden_channels, rgb_channels, 1, bias=False),
        )
        self.last_alpha = None
        self.last_gate_mean = None
        self.last_gate_std = None
        self.last_update_abs = None

    def forward(self, rgb_feat, depth_feat):
        d = self.depth_proj(depth_feat)
        g = self.gate(torch.cat([rgb_feat, d], dim=1))
        fused = g * rgb_feat + (1 - g) * d
        base = self.refine(fused)
        context = torch.cat([rgb_feat, d, torch.abs(rgb_feat - d), base], dim=1)
        global_gate = self.global_gate(context)
        local_update = self.local_update(context)
        alpha = torch.sigmoid(self.alpha_logit) * self.alpha_max
        out = base + alpha * global_gate * local_update
        self.last_alpha = alpha.detach()
        self.last_gate_mean = global_gate.detach().mean()
        self.last_gate_std = global_gate.detach().std(unbiased=False)
        self.last_update_abs = local_update.detach().abs().mean()
        return out
```

## Segmentor Wrapper

```python
class DFormerV2MIIMC4LiteSegmentor(DFormerV2MidFusionSegmentor):
    def __init__(self, num_classes=40, dformerv2_pretrained=None):
        super().__init__(num_classes=num_classes, dformerv2_pretrained=dformerv2_pretrained)
        self.fusions[3] = MIIMC4LiteFusion(
            self.rgb_encoder.out_channels[3],
            self.depth_encoder.out_channels[3],
        )
```

## Lightning Wrapper

```python
class LitDFormerV2MIIMC4Lite(BaseLitSeg):
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
        self.model = DFormerV2MIIMC4LiteSegmentor(
            num_classes=num_classes,
            dformerv2_pretrained=dformerv2_pretrained,
        )

    def training_step(self, batch, batch_idx):
        loss = super().training_step(batch, batch_idx)
        c4_fusion = self.model.fusions[3]
        self.log("train/miim_c4_alpha", c4_fusion.last_alpha, prog_bar=False, on_step=False, on_epoch=True)
        self.log("train/miim_c4_gate_mean", c4_fusion.last_gate_mean, prog_bar=False, on_step=False, on_epoch=True)
        self.log("train/miim_c4_gate_std", c4_fusion.last_gate_std, prog_bar=False, on_step=False, on_epoch=True)
        self.log("train/miim_c4_update_abs", c4_fusion.last_update_abs, prog_bar=False, on_step=False, on_epoch=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=0.01)
```

## Registry Entry

```python
from src.models.mid_fusion import LitDFormerV2MIIMC4Lite

ACTIVE_MODEL_REGISTRY = {
    "dformerv2_miim_c4_lite": LitDFormerV2MIIMC4Lite,
}
```

## Result

- Best val/mIoU: `0.534131` at validation epoch `41`
- Last val/mIoU: `0.509767`
- Best-to-last drop: `0.024364`
- Decision: archive and remove from active registry; do not tune MIIM alpha/channel as a micro-search.
