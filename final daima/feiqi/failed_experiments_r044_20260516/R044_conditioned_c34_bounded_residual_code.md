# R044 Archived Code: Conditioned C34 Bounded Residual

R044 was implemented on branch `exp/R044-conditioned-c34-bounded-residual-v1` as model `dformerv2_conditioned_c34_bounded_residual`.

Full-train result:

- Best val/mIoU: `0.535663` at validation epoch `49`
- Last val/mIoU: `0.520020`
- Best-to-last drop: `0.015643`
- Decision: do not keep active. This remained below R016 `0.541121`, R036 `0.539790`, and R041 `0.537098`.

This file preserves the failed implementation so the active registry can stay clean without losing the experiment code.

## Core Fusion Module

```python
class ImageConditionedC34BoundedDepthResidual(nn.Module):
    def __init__(self, rgb_channels, depth_channels, condition_channels, alpha_max=0.05):
        super().__init__()
        self.base = GatedFusion(rgb_channels, depth_channels)
        self.alpha_max = float(alpha_max)
        condition_hidden = max(condition_channels // 4, 32)
        self.stage_logit = nn.Parameter(torch.zeros(1, rgb_channels))
        self.channel_head = nn.Sequential(
            nn.Linear(condition_channels, condition_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(condition_hidden, rgb_channels),
        )
        self.residual = nn.Sequential(
            nn.Conv2d(rgb_channels * 2, rgb_channels, 1, bias=False),
            nn.BatchNorm2d(rgb_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(rgb_channels, rgb_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(rgb_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(rgb_channels, rgb_channels, 1),
        )
        nn.init.zeros_(self.channel_head[-1].weight)
        nn.init.zeros_(self.channel_head[-1].bias)
        nn.init.zeros_(self.residual[-1].weight)
        nn.init.zeros_(self.residual[-1].bias)
        self.last_alpha_mean = torch.tensor(0.0)
        self.last_alpha_std = torch.tensor(0.0)
        self.last_alpha_min = torch.tensor(0.0)
        self.last_alpha_max = torch.tensor(0.0)
        self.last_alpha_delta_abs = torch.tensor(0.0)
        self.last_residual_abs = torch.tensor(0.0)

    def forward(self, rgb_feat, depth_feat, condition_feat):
        base = self.base(rgb_feat, depth_feat)
        d = self.base.depth_proj(depth_feat)
        residual = self.residual(torch.cat([d, torch.abs(rgb_feat - d)], dim=1))
        condition = F.adaptive_avg_pool2d(condition_feat, output_size=1).flatten(1)
        alpha_logits = self.stage_logit + self.channel_head(condition)
        alpha = self.alpha_max * torch.sigmoid(alpha_logits).view(rgb_feat.size(0), rgb_feat.size(1), 1, 1)
        self.last_alpha_mean = alpha.detach().mean()
        self.last_alpha_std = alpha.detach().std()
        self.last_alpha_min = alpha.detach().amin()
        self.last_alpha_max = alpha.detach().amax()
        self.last_alpha_delta_abs = (alpha.detach() - (0.5 * self.alpha_max)).abs().mean()
        self.last_residual_abs = residual.detach().abs().mean()
        return base + alpha * residual
```

## Segmentor Wiring

```python
class DFormerV2ConditionedC34BoundedResidualSegmentor(DFormerV2MidFusionSegmentor):
    def __init__(self, num_classes=40, dformerv2_pretrained=None):
        super().__init__(num_classes=num_classes, dformerv2_pretrained=dformerv2_pretrained)
        condition_channels = self.rgb_encoder.out_channels[3]
        self.fusions[2] = ImageConditionedC34BoundedDepthResidual(
            self.rgb_encoder.out_channels[2],
            self.depth_encoder.out_channels[2],
            condition_channels=condition_channels,
            alpha_max=0.05,
        )
        self.fusions[3] = ImageConditionedC34BoundedDepthResidual(
            self.rgb_encoder.out_channels[3],
            self.depth_encoder.out_channels[3],
            condition_channels=condition_channels,
            alpha_max=0.05,
        )

    def extract_features(self, rgb, depth):
        dformer_feats = self.rgb_encoder(rgb, depth)
        depth_feats = self.depth_encoder(depth)

        aligned_depth = []
        for rf, df in zip(dformer_feats, depth_feats):
            if rf.shape[-2:] != df.shape[-2:]:
                df = F.interpolate(df, size=rf.shape[-2:], mode="bilinear", align_corners=False)
            aligned_depth.append(df)

        condition_feat = dformer_feats[3]
        fused_feats = [
            self.fusions[0](dformer_feats[0], aligned_depth[0]),
            self.fusions[1](dformer_feats[1], aligned_depth[1]),
            self.fusions[2](dformer_feats[2], aligned_depth[2], condition_feat),
            self.fusions[3](dformer_feats[3], aligned_depth[3], condition_feat),
        ]
        return dformer_feats, aligned_depth, fused_feats
```

## Lightning Wrapper

```python
class LitDFormerV2ConditionedC34BoundedResidual(BaseLitSeg):
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
        self.model = DFormerV2ConditionedC34BoundedResidualSegmentor(
            num_classes=num_classes,
            dformerv2_pretrained=dformerv2_pretrained,
        )

    def training_step(self, batch, batch_idx):
        loss = super().training_step(batch, batch_idx)
        c3_fusion = self.model.fusions[2]
        c4_fusion = self.model.fusions[3]
        self.log("train/c3_cond_alpha_mean", c3_fusion.last_alpha_mean, prog_bar=False, on_step=False, on_epoch=True)
        self.log("train/c3_cond_alpha_std", c3_fusion.last_alpha_std, prog_bar=False, on_step=False, on_epoch=True)
        self.log("train/c3_cond_alpha_min", c3_fusion.last_alpha_min, prog_bar=False, on_step=False, on_epoch=True)
        self.log("train/c3_cond_alpha_max", c3_fusion.last_alpha_max, prog_bar=False, on_step=False, on_epoch=True)
        self.log("train/c3_cond_alpha_delta_abs", c3_fusion.last_alpha_delta_abs, prog_bar=False, on_step=False, on_epoch=True)
        self.log("train/c3_cond_residual_abs", c3_fusion.last_residual_abs, prog_bar=False, on_step=False, on_epoch=True)
        self.log("train/c4_cond_alpha_mean", c4_fusion.last_alpha_mean, prog_bar=False, on_step=False, on_epoch=True)
        self.log("train/c4_cond_alpha_std", c4_fusion.last_alpha_std, prog_bar=False, on_step=False, on_epoch=True)
        self.log("train/c4_cond_alpha_min", c4_fusion.last_alpha_min, prog_bar=False, on_step=False, on_epoch=True)
        self.log("train/c4_cond_alpha_max", c4_fusion.last_alpha_max, prog_bar=False, on_step=False, on_epoch=True)
        self.log("train/c4_cond_alpha_delta_abs", c4_fusion.last_alpha_delta_abs, prog_bar=False, on_step=False, on_epoch=True)
        self.log("train/c4_cond_residual_abs", c4_fusion.last_residual_abs, prog_bar=False, on_step=False, on_epoch=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=0.01)
```

Registry entry used during the run:

```python
"dformerv2_conditioned_c34_bounded_residual": LitDFormerV2ConditionedC34BoundedResidual
```

