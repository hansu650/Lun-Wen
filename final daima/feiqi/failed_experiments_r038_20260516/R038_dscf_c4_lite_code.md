# R038 DSCF-lite c4-only Code Archive

Status: rejected as active mainline after full train.

Evidence:

- Run: `R038_dscf_c4_lite_run01`
- Best val/mIoU: `0.530810` at validation epoch `38`
- Last val/mIoU: `0.530308`
- Best-to-last drop: `0.000502`
- DSCF c4 offset_abs first/last: `0.961821` / `1.675011`
- DSCF c4 weight_entropy first/last: `1.376656` / `1.336370`
- mIoU detail: `final daima/miou_list/R038_dscf_c4_lite_run01.md`

Decision: stable but below R016 `0.541121`; do not keep in active registry or tune K/offset scale.

## Archived Implementation

```python
class DSCFC4LiteFusion(nn.Module):
    def __init__(self, rgb_channels, depth_channels, num_samples=4, offset_scale=2.0):
        super().__init__()
        self.num_samples = int(num_samples)
        self.offset_scale = float(offset_scale)
        self.depth_proj = nn.Conv2d(depth_channels, rgb_channels, 1)
        self.offset = nn.Conv2d(rgb_channels * 2, self.num_samples * 2, 3, padding=1)
        self.sample_weight = nn.Conv2d(rgb_channels * 2, self.num_samples, 1)
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
        nn.init.zeros_(self.offset.weight)
        nn.init.zeros_(self.offset.bias)
        offset_bias = torch.tensor(
            [-0.25, 0.0, 0.25, 0.0, 0.0, -0.25, 0.0, 0.25],
            dtype=self.offset.bias.dtype,
        )
        self.offset.bias.data.copy_(offset_bias[: self.num_samples * 2])
        nn.init.zeros_(self.sample_weight.weight)
        nn.init.zeros_(self.sample_weight.bias)
        self.last_offset_abs = None
        self.last_weight_entropy = None

    def _base_grid(self, batch_size, height, width, device, dtype):
        y = torch.linspace(-1.0, 1.0, height, device=device, dtype=dtype)
        x = torch.linspace(-1.0, 1.0, width, device=device, dtype=dtype)
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        grid = torch.stack([xx, yy], dim=-1)
        return grid.unsqueeze(0).unsqueeze(1).expand(batch_size, self.num_samples, height, width, 2)

    def _sample_depth(self, depth_feat, offsets):
        b, c, h, w = depth_feat.shape
        offsets = offsets.view(b, self.num_samples, 2, h, w)
        offsets = torch.tanh(offsets) * self.offset_scale
        offset_x = offsets[:, :, 0] * (2.0 / max(w - 1, 1))
        offset_y = offsets[:, :, 1] * (2.0 / max(h - 1, 1))
        offset_grid = torch.stack([offset_x, offset_y], dim=-1)
        grid = self._base_grid(b, h, w, depth_feat.device, depth_feat.dtype) + offset_grid
        depth_rep = depth_feat.unsqueeze(1).expand(b, self.num_samples, c, h, w).reshape(b * self.num_samples, c, h, w)
        sampled = F.grid_sample(
            depth_rep,
            grid.reshape(b * self.num_samples, h, w, 2),
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        )
        return sampled.view(b, self.num_samples, c, h, w), offsets

    def forward(self, rgb_feat, depth_feat):
        d = self.depth_proj(depth_feat)
        context = torch.cat([rgb_feat, d], dim=1)
        sampled, offsets = self._sample_depth(d, self.offset(context))
        weights = torch.softmax(self.sample_weight(context), dim=1)
        sparse_depth = (sampled * weights.unsqueeze(2)).sum(dim=1)
        g = self.gate(torch.cat([rgb_feat, sparse_depth], dim=1))
        fused = g * rgb_feat + (1 - g) * sparse_depth
        self.last_offset_abs = offsets.detach().abs().mean()
        self.last_weight_entropy = (-(weights.detach() * weights.detach().clamp_min(1e-6).log()).sum(dim=1)).mean()
        return self.refine(fused)
```

```python
class DFormerV2DSCFC4LiteSegmentor(DFormerV2MidFusionSegmentor):
    def __init__(self, num_classes=40, dformerv2_pretrained=None):
        super().__init__(num_classes=num_classes, dformerv2_pretrained=dformerv2_pretrained)
        self.fusions[3] = DSCFC4LiteFusion(
            self.rgb_encoder.out_channels[3],
            self.depth_encoder.out_channels[3],
            num_samples=4,
            offset_scale=2.0,
        )
```

```python
class LitDFormerV2DSCFC4Lite(BaseLitSeg):
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
        self.model = DFormerV2DSCFC4LiteSegmentor(
            num_classes=num_classes,
            dformerv2_pretrained=dformerv2_pretrained,
        )

    def training_step(self, batch, batch_idx):
        loss = super().training_step(batch, batch_idx)
        dscf = self.model.fusions[3]
        self.log("train/dscf_c4_offset_abs", dscf.last_offset_abs, prog_bar=False, on_step=False, on_epoch=True)
        self.log("train/dscf_c4_weight_entropy", dscf.last_weight_entropy, prog_bar=False, on_step=False, on_epoch=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=0.01)
```
