# R037 DGL Minimal Code Archive

Status: rejected as active mainline after full train.

Evidence:

- Run: `R037_dgl_minimal_run01`
- Best val/mIoU: `0.534656` at validation epoch `42`
- Last val/mIoU: `0.530153`
- Best-to-last drop: `0.004503`
- DGL aux weight: `0.03`
- mIoU detail: `final daima/miou_list/R037_dgl_minimal_run01.md`

Decision: stable but below R016 `0.541121`; do not keep in active registry or tune aux weight.

## Archived Implementation

```python
class DFormerV2DGLMinimalSegmentor(DFormerV2MidFusionSegmentor):
    def __init__(self, num_classes=40, dformerv2_pretrained=None):
        super().__init__(num_classes=num_classes, dformerv2_pretrained=dformerv2_pretrained)
        self.primary_aux_decoder = SimpleFPNDecoder(self.rgb_encoder.out_channels, num_classes=num_classes)
        self.depth_aux_decoder = SimpleFPNDecoder(self.depth_encoder.out_channels, num_classes=num_classes)

    def _encode_aligned(self, rgb, depth):
        rgb_feats = self.rgb_encoder(rgb, depth)
        depth_feats = self.depth_encoder(depth)
        aligned_depth = []
        for rf, df in zip(rgb_feats, depth_feats):
            if rf.shape[-2:] != df.shape[-2:]:
                df = F.interpolate(df, size=rf.shape[-2:], mode="bilinear", align_corners=False)
            aligned_depth.append(df)
        return rgb_feats, aligned_depth

    def _fused_logits(self, rgb_feats, aligned_depth, input_size):
        fused_feats = [
            fusion(r.detach(), d.detach())
            for fusion, r, d in zip(self.fusions, rgb_feats, aligned_depth)
        ]
        return self.decoder(fused_feats, input_size=input_size)

    def forward_with_aux(self, rgb, depth):
        rgb_feats, aligned_depth = self._encode_aligned(rgb, depth)
        logits = self._fused_logits(rgb_feats, aligned_depth, input_size=rgb.shape[-2:])
        primary_aux_logits = self.primary_aux_decoder(rgb_feats, input_size=rgb.shape[-2:])
        depth_aux_logits = self.depth_aux_decoder(aligned_depth, input_size=rgb.shape[-2:])
        return logits, primary_aux_logits, depth_aux_logits

    def forward(self, rgb, depth):
        rgb_feats, aligned_depth = self._encode_aligned(rgb, depth)
        return self._fused_logits(rgb_feats, aligned_depth, input_size=rgb.shape[-2:])
```

```python
class LitDFormerV2DGLMinimal(BaseLitSeg):
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
        self.dgl_aux_weight = 0.03
        self.model = DFormerV2DGLMinimalSegmentor(
            num_classes=num_classes,
            dformerv2_pretrained=dformerv2_pretrained,
        )

    def training_step(self, batch, batch_idx):
        rgb, depth, label = batch["rgb"], batch["depth"], batch["label"]
        label = sanitize_labels(label, num_classes=self.hparams.num_classes, ignore_index=255)
        logits, primary_aux_logits, depth_aux_logits = self.model.forward_with_aux(rgb, depth)
        fusion_loss = self.ce_criterion(logits, label)
        primary_aux_loss = self.ce_criterion(primary_aux_logits, label)
        depth_aux_loss = self.ce_criterion(depth_aux_logits, label)
        aux_loss = primary_aux_loss + depth_aux_loss
        loss = fusion_loss + self.dgl_aux_weight * aux_loss
        self.log("train/loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log("train/fusion_ce", fusion_loss, prog_bar=False, on_step=False, on_epoch=True)
        self.log("train/primary_aux_ce", primary_aux_loss, prog_bar=False, on_step=False, on_epoch=True)
        self.log("train/depth_aux_ce", depth_aux_loss, prog_bar=False, on_step=False, on_epoch=True)
        self.log("train/dgl_aux_loss", aux_loss, prog_bar=False, on_step=False, on_epoch=True)
        self.log("train/dgl_aux_weight", torch.tensor(self.dgl_aux_weight, device=loss.device), prog_bar=False, on_step=False, on_epoch=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=0.01)
```
