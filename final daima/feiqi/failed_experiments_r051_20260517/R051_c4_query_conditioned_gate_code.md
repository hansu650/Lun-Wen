# R051 C4 Query-Conditioned Gate Archived Code

R051 was tested on branch `exp/R051-c4-query-conditioned-gate-v1` as model `dformerv2_c4_query_conditioned_gate`.

Result: best val/mIoU `0.536702`, last `0.507323`, below R016 `0.541121` and unstable. The active code was removed after recording evidence.

## Query-Conditioned Fusion

```python
class QueryConditionedGatedFusion(nn.Module):
    def __init__(self, rgb_channels, depth_channels, hidden_ratio=4):
        super().__init__()
        self.depth_proj = nn.Conv2d(depth_channels, rgb_channels, 1)
        self.gate_conv = nn.Conv2d(rgb_channels * 2, rgb_channels, 1, bias=False)
        self.gate_bn = nn.BatchNorm2d(rgb_channels)
        hidden_channels = max(rgb_channels // hidden_ratio, 32)
        self.query_delta = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(rgb_channels, hidden_channels, 1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, rgb_channels, 1, bias=True),
        )
        nn.init.zeros_(self.query_delta[-1].weight)
        nn.init.zeros_(self.query_delta[-1].bias)
        self.refine = nn.Sequential(
            nn.Conv2d(rgb_channels, rgb_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(rgb_channels),
            nn.ReLU(inplace=True),
        )
        self.last_delta_abs = None
        self.last_gate_mean = None
        self.last_gate_std = None

    def forward(self, rgb_feat, depth_feat):
        d = self.depth_proj(depth_feat)
        base_logit = self.gate_bn(self.gate_conv(torch.cat([rgb_feat, d], dim=1)))
        delta = self.query_delta(rgb_feat)
        g = torch.sigmoid(base_logit + delta)
        fused = g * rgb_feat + (1 - g) * d
        self.last_delta_abs = delta.detach().abs().mean()
        self.last_gate_mean = g.detach().mean()
        self.last_gate_std = g.detach().std()
        return self.refine(fused)
```

## Segmentor

```python
class DFormerV2C4QueryConditionedGateSegmentor(DFormerV2MidFusionSegmentor):
    def __init__(self, num_classes=40, dformerv2_pretrained=None):
        super().__init__(num_classes=num_classes, dformerv2_pretrained=dformerv2_pretrained)
        self.fusions[-1] = QueryConditionedGatedFusion(
            self.rgb_encoder.out_channels[-1],
            self.depth_encoder.out_channels[-1],
        )
```

## Lightning Wrapper

```python
class LitDFormerV2C4QueryConditionedGate(BaseLitSeg):
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
        self.model = DFormerV2C4QueryConditionedGateSegmentor(
            num_classes=num_classes,
            dformerv2_pretrained=dformerv2_pretrained,
        )

    def training_step(self, batch, batch_idx):
        loss = super().training_step(batch, batch_idx)
        c4_fusion = self.model.fusions[-1]
        self.log("train/qc_c4_delta_abs", c4_fusion.last_delta_abs, prog_bar=False, on_step=False, on_epoch=True)
        self.log("train/qc_c4_gate_mean", c4_fusion.last_gate_mean, prog_bar=False, on_step=False, on_epoch=True)
        self.log("train/qc_c4_gate_std", c4_fusion.last_gate_std, prog_bar=False, on_step=False, on_epoch=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=0.01)
```
