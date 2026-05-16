# R053 OCR-Lite Decoder Code Archive

Status: rejected as active mainline after full train.

Evidence:

- Run: `R053_ocr_lite_object_context_run01`
- Best val/mIoU: `0.536867` at validation epoch `49`
- Last val/mIoU: `0.522340`
- Best-to-last drop: `0.014527`
- mIoU detail: `final daima/miou_list/R053_ocr_lite_object_context_run01.md`

Decision: partial positive below R016 `0.541121`; do not keep in active registry or continue OCR-Lite width/context/dropout micro-search.

## Archived Implementation

```python
class OCRLiteDecoder(nn.Module):
    """SimpleFPN decoder with a lightweight object-context refinement branch."""

    def __init__(self, in_channels, out_channels=128, num_classes=40, context_channels=64):
        super().__init__()
        self.lateral4 = nn.Conv2d(in_channels[3], out_channels, 1)
        self.lateral3 = nn.Conv2d(in_channels[2], out_channels, 1)
        self.lateral2 = nn.Conv2d(in_channels[1], out_channels, 1)
        self.lateral1 = nn.Conv2d(in_channels[0], out_channels, 1)

        self.smooth = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.prior_classifier = nn.Conv2d(out_channels, num_classes, 1)
        self.query_proj = nn.Conv2d(out_channels, context_channels, 1, bias=False)
        self.key_proj = nn.Conv1d(out_channels, context_channels, 1, bias=False)
        self.value_proj = nn.Conv1d(out_channels, out_channels, 1, bias=False)
        self.context_scale = context_channels ** -0.5
        self.context_update = nn.Sequential(
            ConvBNReLU(out_channels * 2, out_channels, kernel_size=3, padding=1),
            nn.Conv2d(out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        self.classifier = nn.Conv2d(out_channels, num_classes, 1)
        self.register_buffer("last_context_update_abs", torch.zeros(1), persistent=False)
        self.register_buffer("last_prior_entropy", torch.zeros(1), persistent=False)
        self._init_context_update()

    def _init_context_update(self):
        final_conv = self.context_update[1]
        final_bn = self.context_update[2]
        nn.init.zeros_(final_conv.weight)
        nn.init.ones_(final_bn.weight)
        nn.init.zeros_(final_bn.bias)

    def _build_fpn_feature(self, features):
        c1, c2, c3, c4 = features
        p4 = self.lateral4(c4)
        p3 = self.lateral3(c3) + F.interpolate(p4, size=c3.shape[-2:], mode="bilinear", align_corners=False)
        p2 = self.lateral2(c2) + F.interpolate(p3, size=c2.shape[-2:], mode="bilinear", align_corners=False)
        p1 = self.lateral1(c1) + F.interpolate(p2, size=c1.shape[-2:], mode="bilinear", align_corners=False)
        return self.smooth(p1)

    def forward(self, features, input_size):
        p1 = self._build_fpn_feature(features)
        batch_size, channels, height, width = p1.shape
        num_pixels = height * width

        prior_logits = self.prior_classifier(p1)
        spatial_probs = F.softmax(prior_logits.view(batch_size, -1, num_pixels), dim=2)
        pixel_features = p1.view(batch_size, channels, num_pixels)
        object_context = torch.bmm(pixel_features, spatial_probs.transpose(1, 2))

        query = self.query_proj(p1).view(batch_size, -1, num_pixels).transpose(1, 2)
        key = self.key_proj(object_context)
        value = self.value_proj(object_context)
        affinity = torch.bmm(query, key) * self.context_scale
        attention = F.softmax(affinity, dim=2)
        context = torch.bmm(value, attention.transpose(1, 2)).view(batch_size, channels, height, width)

        context_update = self.context_update(torch.cat([p1, context], dim=1))
        refined = p1 + context_update
        logits = self.classifier(refined)
        return F.interpolate(logits, size=input_size, mode="bilinear", align_corners=False)
```

```python
class DFormerV2OCRLiteDecoderSegmentor(DFormerV2MidFusionSegmentor):
    def __init__(self, num_classes=40, dformerv2_pretrained=None):
        super().__init__(num_classes=num_classes, dformerv2_pretrained=dformerv2_pretrained)
        self.decoder = OCRLiteDecoder(
            self.rgb_encoder.out_channels,
            out_channels=128,
            num_classes=num_classes,
            context_channels=64,
        )
```

```python
class LitDFormerV2OCRLiteDecoder(BaseLitSeg):
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
        self.model = DFormerV2OCRLiteDecoderSegmentor(
            num_classes=num_classes,
            dformerv2_pretrained=dformerv2_pretrained,
        )

    def training_step(self, batch, batch_idx):
        loss = super().training_step(batch, batch_idx)
        decoder = self.model.decoder
        self.log("train/ocr_context_update_abs", decoder.last_context_update_abs, prog_bar=False, on_step=False, on_epoch=True)
        self.log("train/ocr_prior_entropy", decoder.last_prior_entropy, prog_bar=False, on_step=False, on_epoch=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=0.01)
```
