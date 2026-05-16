# R046 DGFusion c4 Depth-Token Archived Code

Status: `completed_negative_depth_token_below_corrected_baseline`

R046 added `DGFusionC4DepthTokenLite`, `DFormerV2DGFusionC4DepthTokenSegmentor`, `LitDFormerV2DGFusionC4DepthToken`, and the `dformerv2_dgfusion_c4_depth_token` registry entry. The full train reached best val/mIoU `0.531838` and was rejected below R016, so this code is archived here and removed from the active mainline.

## Diff

```diff
diff --git a/final daima/src/models/mid_fusion.py b/final daima/src/models/mid_fusion.py
index 1c1ca54..ab5e57a 100644
--- a/final daima/src/models/mid_fusion.py	
+++ b/final daima/src/models/mid_fusion.py	
@@ -95,6 +95,53 @@ class DFormerV2MidFusionSegmentor(nn.Module):
         return self.decoder(fused_feats, input_size=rgb.shape[-2:])
 
 
+class DGFusionC4DepthTokenLite(nn.Module):
+    def __init__(self, channels, token_channels=64, scale=0.1):
+        super().__init__()
+        self.scale = float(scale)
+        self.query_proj = nn.Conv2d(channels, token_channels, 1, bias=False)
+        self.key_proj = nn.Conv2d(channels, token_channels, 1, bias=False)
+        self.value_proj = nn.Conv2d(channels, token_channels, 1, bias=False)
+        self.out_proj = nn.Conv2d(token_channels, channels, 1)
+        nn.init.zeros_(self.out_proj.weight)
+        nn.init.zeros_(self.out_proj.bias)
+        self.last_delta_abs = None
+        self.last_affinity_mean = None
+        self.last_affinity_std = None
+
+    def forward(self, fused_feat, depth_feat):
+        depth_token = F.avg_pool2d(depth_feat, kernel_size=3, stride=1, padding=1)
+        query = F.normalize(self.query_proj(fused_feat), dim=1)
+        key = F.normalize(self.key_proj(depth_token), dim=1)
+        value = self.value_proj(depth_token)
+        affinity = torch.sigmoid((query * key).sum(dim=1, keepdim=True))
+        delta = self.out_proj(affinity * value)
+        self.last_delta_abs = delta.detach().abs().mean()
+        self.last_affinity_mean = affinity.detach().mean()
+        self.last_affinity_std = affinity.detach().std()
+        return fused_feat + self.scale * delta
+
+
+class DFormerV2DGFusionC4DepthTokenSegmentor(DFormerV2MidFusionSegmentor):
+    def __init__(self, num_classes=40, dformerv2_pretrained=None):
+        super().__init__(num_classes=num_classes, dformerv2_pretrained=dformerv2_pretrained)
+        self.c4_depth_token = DGFusionC4DepthTokenLite(self.rgb_encoder.out_channels[3])
+
+    def extract_features(self, rgb, depth):
+        dformer_feats, aligned_depth, fused_feats = super().extract_features(rgb, depth)
+        fused_feats = list(fused_feats)
+        fused_feats[3] = self.c4_depth_token(fused_feats[3], aligned_depth[3])
+        return dformer_feats, aligned_depth, fused_feats
+
+    def token_stats(self):
+        stats = {}
+        if self.c4_depth_token.last_delta_abs is not None:
+            stats["c4_token_delta_abs"] = self.c4_depth_token.last_delta_abs
+            stats["c4_token_affinity_mean"] = self.c4_depth_token.last_affinity_mean
+            stats["c4_token_affinity_std"] = self.c4_depth_token.last_affinity_std
+        return stats
+
+
 class DFormerV2HamDecoderSegmentor(DFormerV2MidFusionSegmentor):
     def __init__(self, num_classes=40, dformerv2_pretrained=None):
         super().__init__(num_classes=num_classes, dformerv2_pretrained=dformerv2_pretrained)
@@ -128,6 +175,37 @@ class LitDFormerV2MidFusion(BaseLitSeg):
     def configure_optimizers(self):
         return torch.optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=0.01)
 
+
+class LitDFormerV2DGFusionC4DepthToken(BaseLitSeg):
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
+        self.model = DFormerV2DGFusionC4DepthTokenSegmentor(
+            num_classes=num_classes,
+            dformerv2_pretrained=dformerv2_pretrained,
+        )
+
+    def training_step(self, batch, batch_idx):
+        loss = super().training_step(batch, batch_idx)
+        for name, value in self.model.token_stats().items():
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
index 8a9df3d..9fb2241 100644
--- a/final daima/train.py	
+++ b/final daima/train.py	
@@ -33,6 +33,7 @@ from lightning.pytorch.callbacks import Callback, EarlyStopping, TQDMProgressBar
 from src.data_module import NYUDataModule
 from src.models.early_fusion import LitEarlyFusion
 from src.models.mid_fusion import (
+    LitDFormerV2DGFusionC4DepthToken,
     LitDFormerV2HamDecoder,
     LitDFormerV2MidFusion,
     LitMidFusion,
@@ -42,6 +43,7 @@ from src.models.teacher_model import LitDFormerV2GeometryPrimaryHamDecoder
 
 ACTIVE_MODEL_REGISTRY = {
     "dformerv2_mid_fusion": LitDFormerV2MidFusion,
+    "dformerv2_dgfusion_c4_depth_token": LitDFormerV2DGFusionC4DepthToken,
     "dformerv2_ham_decoder": LitDFormerV2HamDecoder,
     "dformerv2_geometry_primary_ham_decoder": LitDFormerV2GeometryPrimaryHamDecoder,
 }
@@ -139,6 +141,7 @@ def build_model(args):
     model_cls = MODEL_REGISTRY[args.model]
     if args.model in {
         "dformerv2_mid_fusion",
+        "dformerv2_dgfusion_c4_depth_token",
         "dformerv2_ham_decoder",
         "dformerv2_geometry_primary_ham_decoder",
     }:

```
