# R047 GatedFusion GroupNorm Archived Code

Status: `completed_negative_gn_below_corrected_baseline`

R047 added `GatedFusionGN`, `DFormerV2GatedFusionGNSegmentor`, `LitDFormerV2GatedFusionGN`, and the `dformerv2_gatedfusion_gn` registry entry. The full train reached best val/mIoU `0.528301` and was rejected below R016/R036/R041, so this code is archived here and removed from the active mainline.

## Diff

```diff
diff --git a/final daima/src/models/mid_fusion.py b/final daima/src/models/mid_fusion.py
index 1c1ca54..c84e28d 100644
--- a/final daima/src/models/mid_fusion.py	
+++ b/final daima/src/models/mid_fusion.py	
@@ -31,6 +31,28 @@ class GatedFusion(nn.Module):
         return self.refine(fused)
 
 
+class GatedFusionGN(nn.Module):
+    def __init__(self, rgb_channels, depth_channels, num_groups=32):
+        super().__init__()
+        self.depth_proj = nn.Conv2d(depth_channels, rgb_channels, 1)
+        self.gate = nn.Sequential(
+            nn.Conv2d(rgb_channels * 2, rgb_channels, 1, bias=False),
+            nn.GroupNorm(num_groups, rgb_channels),
+            nn.Sigmoid(),
+        )
+        self.refine = nn.Sequential(
+            nn.Conv2d(rgb_channels, rgb_channels, 3, padding=1, bias=False),
+            nn.GroupNorm(num_groups, rgb_channels),
+            nn.ReLU(inplace=True),
+        )
+
+    def forward(self, rgb_feat, depth_feat):
+        d = self.depth_proj(depth_feat)
+        g = self.gate(torch.cat([rgb_feat, d], dim=1))
+        fused = g * rgb_feat + (1 - g) * d
+        return self.refine(fused)
+
+
 class MidFusionSegmentor(nn.Module):
     def __init__(self, num_classes=40):
         super().__init__()
@@ -95,6 +117,15 @@ class DFormerV2MidFusionSegmentor(nn.Module):
         return self.decoder(fused_feats, input_size=rgb.shape[-2:])
 
 
+class DFormerV2GatedFusionGNSegmentor(DFormerV2MidFusionSegmentor):
+    def __init__(self, num_classes=40, dformerv2_pretrained=None):
+        super().__init__(num_classes=num_classes, dformerv2_pretrained=dformerv2_pretrained)
+        self.fusions = nn.ModuleList([
+            GatedFusionGN(rgb_ch, depth_ch)
+            for rgb_ch, depth_ch in zip(self.rgb_encoder.out_channels, self.depth_encoder.out_channels)
+        ])
+
+
 class DFormerV2HamDecoderSegmentor(DFormerV2MidFusionSegmentor):
     def __init__(self, num_classes=40, dformerv2_pretrained=None):
         super().__init__(num_classes=num_classes, dformerv2_pretrained=dformerv2_pretrained)
@@ -128,6 +159,31 @@ class LitDFormerV2MidFusion(BaseLitSeg):
     def configure_optimizers(self):
         return torch.optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=0.01)
 
+
+class LitDFormerV2GatedFusionGN(BaseLitSeg):
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
+        self.model = DFormerV2GatedFusionGNSegmentor(
+            num_classes=num_classes,
+            dformerv2_pretrained=dformerv2_pretrained,
+        )
+
+    def configure_optimizers(self):
+        return torch.optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=0.01)
+
+
 class LitDFormerV2HamDecoder(BaseLitSeg):
     def __init__(
         self,
diff --git a/final daima/train.py b/final daima/train.py
index 8a9df3d..4be494d 100644
--- a/final daima/train.py	
+++ b/final daima/train.py	
@@ -33,6 +33,7 @@ from lightning.pytorch.callbacks import Callback, EarlyStopping, TQDMProgressBar
 from src.data_module import NYUDataModule
 from src.models.early_fusion import LitEarlyFusion
 from src.models.mid_fusion import (
+    LitDFormerV2GatedFusionGN,
     LitDFormerV2HamDecoder,
     LitDFormerV2MidFusion,
     LitMidFusion,
@@ -42,6 +43,7 @@ from src.models.teacher_model import LitDFormerV2GeometryPrimaryHamDecoder
 
 ACTIVE_MODEL_REGISTRY = {
     "dformerv2_mid_fusion": LitDFormerV2MidFusion,
+    "dformerv2_gatedfusion_gn": LitDFormerV2GatedFusionGN,
     "dformerv2_ham_decoder": LitDFormerV2HamDecoder,
     "dformerv2_geometry_primary_ham_decoder": LitDFormerV2GeometryPrimaryHamDecoder,
 }
@@ -139,6 +141,7 @@ def build_model(args):
     model_cls = MODEL_REGISTRY[args.model]
     if args.model in {
         "dformerv2_mid_fusion",
+        "dformerv2_gatedfusion_gn",
         "dformerv2_ham_decoder",
         "dformerv2_geometry_primary_ham_decoder",
     }:

```
