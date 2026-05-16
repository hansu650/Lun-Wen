# R045 c3/c4 Zero-Init Modality Adapter Archived Code

Status: `completed_negative_adapter_below_corrected_baseline`

R045 added `ZeroInitModalityAdapter`, `DFormerV2C34ZeroInitModalityAdapterSegmentor`, `LitDFormerV2C34ZeroInitModalityAdapter`, and the `dformerv2_c34_zero_init_modality_adapter` registry entry. The full train reached best val/mIoU `0.531454` and was rejected below R016, so this code is archived here and removed from the active mainline.

## Diff

```diff
diff --git a/final daima/src/models/mid_fusion.py b/final daima/src/models/mid_fusion.py
index 1c1ca54..dbcde44 100644
--- a/final daima/src/models/mid_fusion.py	
+++ b/final daima/src/models/mid_fusion.py	
@@ -95,6 +95,70 @@ class DFormerV2MidFusionSegmentor(nn.Module):
         return self.decoder(fused_feats, input_size=rgb.shape[-2:])
 
 
+class ZeroInitModalityAdapter(nn.Module):
+    def __init__(self, channels, reduction=4, scale=0.1):
+        super().__init__()
+        hidden_channels = max(channels // reduction, 16)
+        self.scale = float(scale)
+        self.adapter = nn.Sequential(
+            nn.Conv2d(channels, hidden_channels, 1),
+            nn.ReLU(inplace=True),
+            nn.Conv2d(hidden_channels, channels, 1),
+        )
+        nn.init.zeros_(self.adapter[-1].weight)
+        nn.init.zeros_(self.adapter[-1].bias)
+        self.last_delta_abs = None
+
+    def forward(self, x):
+        delta = self.adapter(x)
+        self.last_delta_abs = delta.detach().abs().mean()
+        return x + self.scale * delta
+
+
+class DFormerV2C34ZeroInitModalityAdapterSegmentor(DFormerV2MidFusionSegmentor):
+    def __init__(self, num_classes=40, dformerv2_pretrained=None):
+        super().__init__(num_classes=num_classes, dformerv2_pretrained=dformerv2_pretrained)
+        self.rgb_adapters = nn.ModuleList([
+            nn.Identity(),
+            nn.Identity(),
+            ZeroInitModalityAdapter(self.rgb_encoder.out_channels[2]),
+            ZeroInitModalityAdapter(self.rgb_encoder.out_channels[3]),
+        ])
+        self.depth_adapters = nn.ModuleList([
+            nn.Identity(),
+            nn.Identity(),
+            ZeroInitModalityAdapter(self.depth_encoder.out_channels[2]),
+            ZeroInitModalityAdapter(self.depth_encoder.out_channels[3]),
+        ])
+
+    def extract_features(self, rgb, depth):
+        dformer_feats = self.rgb_encoder(rgb, depth)
+        depth_feats = self.depth_encoder(depth)
+
+        adapted_rgb = []
+        adapted_depth = []
+        for idx, (rf, df) in enumerate(zip(dformer_feats, depth_feats)):
+            if rf.shape[-2:] != df.shape[-2:]:
+                df = F.interpolate(df, size=rf.shape[-2:], mode="bilinear", align_corners=False)
+            adapted_rgb.append(self.rgb_adapters[idx](rf))
+            adapted_depth.append(self.depth_adapters[idx](df))
+
+        fused_feats = [fusion(r, d) for fusion, r, d in zip(self.fusions, adapted_rgb, adapted_depth)]
+        return adapted_rgb, adapted_depth, fused_feats
+
+    def adapter_stats(self):
+        stats = {}
+        for name, adapter in (
+            ("rgb_c3", self.rgb_adapters[2]),
+            ("rgb_c4", self.rgb_adapters[3]),
+            ("depth_c3", self.depth_adapters[2]),
+            ("depth_c4", self.depth_adapters[3]),
+        ):
+            if adapter.last_delta_abs is not None:
+                stats[f"{name}_adapter_delta_abs"] = adapter.last_delta_abs
+        return stats
+
+
 class DFormerV2HamDecoderSegmentor(DFormerV2MidFusionSegmentor):
     def __init__(self, num_classes=40, dformerv2_pretrained=None):
         super().__init__(num_classes=num_classes, dformerv2_pretrained=dformerv2_pretrained)
@@ -128,6 +192,37 @@ class LitDFormerV2MidFusion(BaseLitSeg):
     def configure_optimizers(self):
         return torch.optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=0.01)
 
+
+class LitDFormerV2C34ZeroInitModalityAdapter(BaseLitSeg):
+    def __init__(
+        self,
+        num_classes=40,
+        lr=1e-4,
+        dformerv2_pretrained=None,
+        loss_type: str = "ce",
+        dice_weight: float = 0.5,
+    ):
+        super().__init__(
+            num_classes=num_classes,
+            lr=lr,
+            loss_type=loss_type,
+            dice_weight=dice_weight,
+        )
+        self.model = DFormerV2C34ZeroInitModalityAdapterSegmentor(
+            num_classes=num_classes,
+            dformerv2_pretrained=dformerv2_pretrained,
+        )
+
+    def training_step(self, batch, batch_idx):
+        loss = super().training_step(batch, batch_idx)
+        for name, value in self.model.adapter_stats().items():
+            self.log(f"train/{name}", value, prog_bar=False, on_step=False, on_epoch=True)
+        return loss
+
+    def configure_optimizers(self):
+        return torch.optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=0.01)
+
+
 class LitDFormerV2HamDecoder(BaseLitSeg):
     def __init__(
         self,
diff --git a/final daima/train.py b/final daima/train.py
index 8a9df3d..c5f8d17 100644
--- a/final daima/train.py	
+++ b/final daima/train.py	
@@ -33,6 +33,7 @@ from lightning.pytorch.callbacks import Callback, EarlyStopping, TQDMProgressBar
 from src.data_module import NYUDataModule
 from src.models.early_fusion import LitEarlyFusion
 from src.models.mid_fusion import (
+    LitDFormerV2C34ZeroInitModalityAdapter,
     LitDFormerV2HamDecoder,
     LitDFormerV2MidFusion,
     LitMidFusion,
@@ -42,6 +43,7 @@ from src.models.teacher_model import LitDFormerV2GeometryPrimaryHamDecoder
 
 ACTIVE_MODEL_REGISTRY = {
     "dformerv2_mid_fusion": LitDFormerV2MidFusion,
+    "dformerv2_c34_zero_init_modality_adapter": LitDFormerV2C34ZeroInitModalityAdapter,
     "dformerv2_ham_decoder": LitDFormerV2HamDecoder,
     "dformerv2_geometry_primary_ham_decoder": LitDFormerV2GeometryPrimaryHamDecoder,
 }
@@ -139,6 +141,7 @@ def build_model(args):
     model_cls = MODEL_REGISTRY[args.model]
     if args.model in {
         "dformerv2_mid_fusion",
+        "dformerv2_c34_zero_init_modality_adapter",
         "dformerv2_ham_decoder",
         "dformerv2_geometry_primary_ham_decoder",
     }:

```
