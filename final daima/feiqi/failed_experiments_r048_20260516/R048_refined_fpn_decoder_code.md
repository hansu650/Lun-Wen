# R048 Refined FPN Decoder Archived Code

Status: `completed_stable_but_below_corrected_baseline`

R048 added `RefinedFPNDecoder`, `DFormerV2RefinedFPNDecoderSegmentor`, `LitDFormerV2RefinedFPNDecoder`, and the `dformerv2_refined_fpn_decoder` registry entry. The full train reached best val/mIoU `0.534154`, with stable final mIoU `0.530318`, but remained below R016/R036/R041, so this code is archived here and removed from the active mainline.

## Diff

```diff
diff --git a/final daima/src/models/decoder.py b/final daima/src/models/decoder.py
index f8b3867..57f2af6 100644
--- a/final daima/src/models/decoder.py	
+++ b/final daima/src/models/decoder.py	
@@ -45,6 +45,55 @@ class ConvBNReLU(nn.Sequential):
         super().__init__(*layers)
 
 
+class RefinedFPNDecoder(nn.Module):
+    """Multi-level FPN refinement decoder for the R048 ablation."""
+
+    def __init__(self, in_channels, out_channels=128, num_classes=40):
+        super().__init__()
+        self.lateral4 = nn.Conv2d(in_channels[3], out_channels, 1)
+        self.lateral3 = nn.Conv2d(in_channels[2], out_channels, 1)
+        self.lateral2 = nn.Conv2d(in_channels[1], out_channels, 1)
+        self.lateral1 = nn.Conv2d(in_channels[0], out_channels, 1)
+
+        self.smooth4 = ConvBNReLU(out_channels, out_channels, 3, padding=1)
+        self.smooth3 = ConvBNReLU(out_channels, out_channels, 3, padding=1)
+        self.smooth2 = ConvBNReLU(out_channels, out_channels, 3, padding=1)
+        self.smooth1 = ConvBNReLU(out_channels, out_channels, 3, padding=1)
+        self.fuse = ConvBNReLU(out_channels * 4, out_channels, 3, padding=1)
+        self.classifier = nn.Conv2d(out_channels, num_classes, 1)
+
+    def forward(self, features, input_size):
+        c1, c2, c3, c4 = features
+
+        p4 = self.smooth4(self.lateral4(c4))
+        p3 = self.smooth3(
+            self.lateral3(c3)
+            + F.interpolate(p4, size=c3.shape[-2:], mode="bilinear", align_corners=False)
+        )
+        p2 = self.smooth2(
+            self.lateral2(c2)
+            + F.interpolate(p3, size=c2.shape[-2:], mode="bilinear", align_corners=False)
+        )
+        p1 = self.smooth1(
+            self.lateral1(c1)
+            + F.interpolate(p2, size=c1.shape[-2:], mode="bilinear", align_corners=False)
+        )
+
+        target_size = p1.shape[-2:]
+        x = torch.cat(
+            [
+                p1,
+                F.interpolate(p2, size=target_size, mode="bilinear", align_corners=False),
+                F.interpolate(p3, size=target_size, mode="bilinear", align_corners=False),
+                F.interpolate(p4, size=target_size, mode="bilinear", align_corners=False),
+            ],
+            dim=1,
+        )
+        x = self.fuse(x)
+        x = F.interpolate(x, size=input_size, mode="bilinear", align_corners=False)
+        return self.classifier(x)
+
+
 class NMF2D(nn.Module):
     """NMF block used by the official DFormer LightHam decoder."""
 
diff --git a/final daima/src/models/mid_fusion.py b/final daima/src/models/mid_fusion.py
index 1c1ca54..40f3c17 100644
--- a/final daima/src/models/mid_fusion.py	
+++ b/final daima/src/models/mid_fusion.py	
@@ -4,7 +4,7 @@ import torch.nn as nn
 import torch.nn.functional as F
 
 from .base_lit import BaseLitSeg
-from .decoder import OfficialHamDecoder, SimpleFPNDecoder
+from .decoder import OfficialHamDecoder, RefinedFPNDecoder, SimpleFPNDecoder
 from .dformerv2_encoder import DFormerv2_S, load_dformerv2_pretrained
 from .encoder import DepthEncoder, RGBEncoder
 
@@ -105,6 +105,15 @@ class DFormerV2HamDecoderSegmentor(DFormerV2MidFusionSegmentor):
         )
 
 
+class DFormerV2RefinedFPNDecoderSegmentor(DFormerV2MidFusionSegmentor):
+    def __init__(self, num_classes=40, dformerv2_pretrained=None):
+        super().__init__(num_classes=num_classes, dformerv2_pretrained=dformerv2_pretrained)
+        self.decoder = RefinedFPNDecoder(
+            self.rgb_encoder.out_channels,
+            num_classes=num_classes,
+        )
+
+
 class LitDFormerV2MidFusion(BaseLitSeg):
     def __init__(
         self,
@@ -150,3 +159,27 @@ class LitDFormerV2HamDecoder(BaseLitSeg):
 
     def configure_optimizers(self):
         return torch.optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=0.01)
+
+
+class LitDFormerV2RefinedFPNDecoder(BaseLitSeg):
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
+        self.model = DFormerV2RefinedFPNDecoderSegmentor(
+            num_classes=num_classes,
+            dformerv2_pretrained=dformerv2_pretrained,
+        )
+
+    def configure_optimizers(self):
+        return torch.optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=0.01)
diff --git a/final daima/train.py b/final daima/train.py
index 8a9df3d..48e1d61 100644
--- a/final daima/train.py	
+++ b/final daima/train.py	
@@ -35,6 +35,7 @@ from src.models.early_fusion import LitEarlyFusion
 from src.models.mid_fusion import (
     LitDFormerV2HamDecoder,
     LitDFormerV2MidFusion,
+    LitDFormerV2RefinedFPNDecoder,
     LitMidFusion,
 )
 from src.models.teacher_model import LitDFormerV2GeometryPrimaryHamDecoder
@@ -43,6 +44,7 @@ from src.models.teacher_model import LitDFormerV2GeometryPrimaryHamDecoder
 ACTIVE_MODEL_REGISTRY = {
     "dformerv2_mid_fusion": LitDFormerV2MidFusion,
     "dformerv2_ham_decoder": LitDFormerV2HamDecoder,
+    "dformerv2_refined_fpn_decoder": LitDFormerV2RefinedFPNDecoder,
     "dformerv2_geometry_primary_ham_decoder": LitDFormerV2GeometryPrimaryHamDecoder,
 }
 
@@ -140,6 +142,7 @@ def build_model(args):
     if args.model in {
         "dformerv2_mid_fusion",
         "dformerv2_ham_decoder",
+        "dformerv2_refined_fpn_decoder",
         "dformerv2_geometry_primary_ham_decoder",
     }:
         return model_cls(

```
