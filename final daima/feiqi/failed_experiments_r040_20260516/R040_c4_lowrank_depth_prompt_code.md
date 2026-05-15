# R040 C4 Low-Rank Depth Prompt Archived Code

R040 tested a MixPrompt-style c4-only low-rank depth prompt before the original
`GatedFusion` logic. The full train completed below `0.53` and below R016, so
the active registry was cleaned and this implementation is archived here for
reference.

## Fusion Module

```python
class C4LowRankDepthPromptFusion(nn.Module):
    def __init__(self, rgb_channels, depth_channels, rank=8):
        super().__init__()
        self.rank = int(rank)
        hidden = max(rgb_channels // 4, 32)
        self.depth_proj = nn.Conv2d(depth_channels, rgb_channels, 1)
        self.channel_basis = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(rgb_channels * 2, hidden, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, rgb_channels * self.rank, 1, bias=True),
        )
        self.spatial_basis = nn.Sequential(
            nn.Conv2d(rgb_channels * 2, hidden, 1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.GELU(),
            nn.Conv2d(hidden, self.rank, 3, padding=1, bias=True),
        )
        self.prompt_proj = nn.Conv2d(rgb_channels, rgb_channels, 1, bias=False)
        nn.init.zeros_(self.prompt_proj.weight)
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
        self.last_prompt_abs = torch.tensor(0.0)
        self.last_prompt_raw_abs = torch.tensor(0.0)
        self.last_gate_mean = torch.tensor(0.0)
        self.last_gate_std = torch.tensor(0.0)

    def forward(self, rgb_feat, depth_feat):
        d = self.depth_proj(depth_feat)
        prompt_input = torch.cat([d, (rgb_feat - d).abs()], dim=1)
        b, c, h, w = rgb_feat.shape
        channel = self.channel_basis(prompt_input).view(b, c, self.rank, 1, 1)
        spatial = self.spatial_basis(prompt_input).view(b, 1, self.rank, h, w)
        prompt_raw = (channel * spatial).sum(dim=2) / (self.rank ** 0.5)
        prompt = self.prompt_proj(prompt_raw)
        prompted_rgb = rgb_feat + prompt
        g = self.gate(torch.cat([prompted_rgb, d], dim=1))
        fused = g * prompted_rgb + (1 - g) * d
        self.last_prompt_abs = prompt.detach().abs().mean()
        self.last_prompt_raw_abs = prompt_raw.detach().abs().mean()
        self.last_gate_mean = g.detach().mean()
        self.last_gate_std = g.detach().std()
        return self.refine(fused)
```

## Segmentor Wrapper

```python
class DFormerV2C4LowRankDepthPromptSegmentor(DFormerV2MidFusionSegmentor):
    def __init__(self, num_classes=40, dformerv2_pretrained=None):
        super().__init__(num_classes=num_classes, dformerv2_pretrained=dformerv2_pretrained)
        self.fusions[3] = C4LowRankDepthPromptFusion(
            self.rgb_encoder.out_channels[3],
            self.depth_encoder.out_channels[3],
            rank=8,
        )
```

## Lightning Wrapper

```python
class LitDFormerV2C4LowRankDepthPrompt(BaseLitSeg):
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
        self.model = DFormerV2C4LowRankDepthPromptSegmentor(
            num_classes=num_classes,
            dformerv2_pretrained=dformerv2_pretrained,
        )

    def training_step(self, batch, batch_idx):
        loss = super().training_step(batch, batch_idx)
        c4_fusion = self.model.fusions[3]
        self.log("train/c4_prompt_abs", c4_fusion.last_prompt_abs, prog_bar=False, on_step=False, on_epoch=True)
        self.log("train/c4_prompt_raw_abs", c4_fusion.last_prompt_raw_abs, prog_bar=False, on_step=False, on_epoch=True)
        self.log("train/c4_prompt_gate_mean", c4_fusion.last_gate_mean, prog_bar=False, on_step=False, on_epoch=True)
        self.log("train/c4_prompt_gate_std", c4_fusion.last_gate_std, prog_bar=False, on_step=False, on_epoch=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=0.01)
```

## Train Registry Entry

```python
ACTIVE_MODEL_REGISTRY = {
    "dformerv2_c4_lowrank_depth_prompt": LitDFormerV2C4LowRankDepthPrompt,
}
```

## Result

- Best val/mIoU: `0.527946` at validation epoch `37`
- Last val/mIoU: `0.524679`
- Best-to-last drop: `0.003267`
- Decision: negative below `0.53`; do not tune prompt rank/down-ratio/c4 scale.
