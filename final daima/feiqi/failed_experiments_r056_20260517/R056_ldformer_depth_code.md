# R056 LDFormer Depth Code Archive

Status: rejected as active mainline after full train.

Evidence:

- Run: `R056_ldformer_depth_run01`
- Best val/mIoU: `0.522759` at validation epoch `44`
- Last val/mIoU: `0.518073`
- Best-to-last drop: `0.004686`
- mIoU detail: `final daima/miou_list/R056_ldformer_depth_run01.md`

Decision: negative below R016 `0.541121`, R055 `0.531952`, and the 0.53 threshold. The thin LDFormer-style depth branch is much lighter than the pretrained ResNet-18 depth branch, but this standalone replacement loses too much representation capacity in the current fixed recipe. Do not keep the registry entry or tune width/dropout/stage variants immediately.

## Archived Implementation

```python
class LDFormerDepthStage(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1, groups=in_channels, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

    def forward(self, x):
        return self.block(x)
```

```python
class LDFormerDepthEncoder(nn.Module):
    """Lightweight depth branch inspired by HDBFormer LDFormer."""

    def __init__(self):
        super().__init__()
        self.stem = LDFormerDepthStage(1, 32)
        self.stage1 = LDFormerDepthStage(32, 64)
        self.stage2 = LDFormerDepthStage(64, 128)
        self.stage3 = LDFormerDepthStage(128, 256)
        self.stage4 = LDFormerDepthStage(256, 512)
        self.out_channels = [64, 128, 256, 512]

    def forward(self, x):
        feats = []
        x = self.stem(x)
        x = self.stage1(x); feats.append(x)
        x = self.stage2(x); feats.append(x)
        x = self.stage3(x); feats.append(x)
        x = self.stage4(x); feats.append(x)
        return feats
```

```python
class DFormerV2LDFormerDepthSegmentor(DFormerV2MidFusionSegmentor):
    def __init__(self, num_classes=40, dformerv2_pretrained=None):
        super().__init__(num_classes=num_classes, dformerv2_pretrained=dformerv2_pretrained)
        self.depth_encoder = LDFormerDepthEncoder()
        self.fusions = nn.ModuleList([
            GatedFusion(rgb_ch, depth_ch)
            for rgb_ch, depth_ch in zip(self.rgb_encoder.out_channels, self.depth_encoder.out_channels)
        ])
```

```python
class LitDFormerV2LDFormerDepth(BaseLitSeg):
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
        self.model = DFormerV2LDFormerDepthSegmentor(
            num_classes=num_classes,
            dformerv2_pretrained=dformerv2_pretrained,
        )

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=0.01)
```
