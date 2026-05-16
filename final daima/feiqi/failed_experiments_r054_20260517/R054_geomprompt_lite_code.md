# R054 GeomPrompt-Lite Code Archive

Status: rejected as active mainline after full train.

Evidence:

- Run: `R054_geomprompt_lite_run01`
- Best val/mIoU: `0.532737` at validation epoch `50`
- Last val/mIoU: `0.532737`
- Best-to-last drop: `0.000000`
- mIoU detail: `final daima/miou_list/R054_geomprompt_lite_run01.md`

Decision: negative below R016 `0.541121`, R036 `0.539790`, and R053 `0.536867`. The learned prompt stayed extremely small (`depth_prompt_alpha=0.000777`, `prompt_update_abs=0.000259` at epoch 50), so do not keep the registry entry or continue GeomPrompt-Lite alpha/hidden-size micro-tuning.

## Archived Implementation

```python
class DepthGeometryPrompt(nn.Module):
    def __init__(self, hidden_channels=16, alpha_max=0.10):
        super().__init__()
        self.alpha_max = float(alpha_max)
        self.alpha_logit = nn.Parameter(torch.zeros(()))
        self.prompt = nn.Sequential(
            nn.Conv2d(4, hidden_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, 1, 1),
        )
        self.register_buffer("last_alpha", torch.zeros(()), persistent=False)
        self.register_buffer("last_prompt_raw_abs", torch.zeros(()), persistent=False)
        self.register_buffer("last_prompt_update_abs", torch.zeros(()), persistent=False)

    def alpha(self):
        return self.alpha_max * torch.tanh(self.alpha_logit)

    def forward(self, rgb, depth):
        raw_prompt = torch.tanh(self.prompt(torch.cat([rgb, depth], dim=1)))
        alpha = self.alpha()
        update = alpha * raw_prompt
        self.last_alpha.copy_(alpha.detach())
        self.last_prompt_raw_abs.copy_(raw_prompt.detach().abs().mean())
        self.last_prompt_update_abs.copy_(update.detach().abs().mean())
        return depth + update
```

```python
class DFormerV2GeomPromptLiteSegmentor(DFormerV2MidFusionSegmentor):
    def __init__(self, num_classes=40, dformerv2_pretrained=None):
        super().__init__(num_classes=num_classes, dformerv2_pretrained=dformerv2_pretrained)
        self.depth_prompt = DepthGeometryPrompt(hidden_channels=16, alpha_max=0.10)

    def extract_features(self, rgb, depth):
        prompted_depth = self.depth_prompt(rgb, depth)
        dformer_feats = self.rgb_encoder(rgb, prompted_depth)
        depth_feats = self.depth_encoder(prompted_depth)

        aligned_depth = []
        for rf, df in zip(dformer_feats, depth_feats):
            if rf.shape[-2:] != df.shape[-2:]:
                df = F.interpolate(df, size=rf.shape[-2:], mode="bilinear", align_corners=False)
            aligned_depth.append(df)

        fused_feats = [fusion(r, d) for fusion, r, d in zip(self.fusions, dformer_feats, aligned_depth)]
        return dformer_feats, aligned_depth, fused_feats
```

```python
class LitDFormerV2GeomPromptLite(BaseLitSeg):
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
        self.model = DFormerV2GeomPromptLiteSegmentor(
            num_classes=num_classes,
            dformerv2_pretrained=dformerv2_pretrained,
        )

    def training_step(self, batch, batch_idx):
        loss = super().training_step(batch, batch_idx)
        depth_prompt = self.model.depth_prompt
        self.log("train/depth_prompt_alpha", depth_prompt.last_alpha, prog_bar=False, on_step=False, on_epoch=True)
        self.log("train/depth_prompt_raw_abs", depth_prompt.last_prompt_raw_abs, prog_bar=False, on_step=False, on_epoch=True)
        self.log("train/depth_prompt_update_abs", depth_prompt.last_prompt_update_abs, prog_bar=False, on_step=False, on_epoch=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=0.01)
```
