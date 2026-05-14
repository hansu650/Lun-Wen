# R018 DropPath 0.25 Contract Gate Archive

R018 temporarily tested official DFormerv2-S NYUDepthv2 `drop_path_rate=0.25` as a separate model entry, `dformerv2_mid_fusion_dpr025`.

Full train evidence:

- Run: `R018_dformerv2_mid_fusion_dpr025_retry1`
- Best val/mIoU: `0.526282` at validation epoch `46`
- Last val/mIoU: `0.522893`
- Evidence: `final daima/miou_list/R018_dformerv2_mid_fusion_dpr025_retry1.md`
- Report: `reports/R018-droppath025-contract-v1.md`

Decision: do not keep this code active. R018 is below the R016 corrected baseline `0.541121`, so the active mainline should stay with `dformerv2_mid_fusion` and default local `drop_path_rate=0.1`.

Failed active-code diff:

```diff
diff --git a/final daima/src/models/mid_fusion.py b/final daima/src/models/mid_fusion.py
@@
-class DFormerV2MidFusionSegmentor(nn.Module):
-    def __init__(self, num_classes=40, dformerv2_pretrained=None):
+class DFormerV2MidFusionSegmentor(nn.Module):
+    def __init__(self, num_classes=40, dformerv2_pretrained=None, drop_path_rate=0.1):
         super().__init__()
-        self.rgb_encoder = DFormerv2_S()
+        self.rgb_encoder = DFormerv2_S(drop_path_rate=drop_path_rate)
@@
 class LitDFormerV2MidFusion(BaseLitSeg):
@@
         loss_type: str = "ce",
         dice_weight: float = 0.5,
+        drop_path_rate: float = 0.1,
@@
         self.model = DFormerV2MidFusionSegmentor(
             num_classes=num_classes,
             dformerv2_pretrained=dformerv2_pretrained,
+            drop_path_rate=drop_path_rate,
         )
+
+class LitDFormerV2MidFusionDropPath025(LitDFormerV2MidFusion):
+    ...
+    drop_path_rate=0.25

diff --git a/final daima/train.py b/final daima/train.py
@@
-from src.models.mid_fusion import LitDFormerV2MidFusion, LitMidFusion
+from src.models.mid_fusion import LitDFormerV2MidFusion, LitDFormerV2MidFusionDropPath025, LitMidFusion
@@
 ACTIVE_MODEL_REGISTRY = {
     "dformerv2_mid_fusion": LitDFormerV2MidFusion,
+    "dformerv2_mid_fusion_dpr025": LitDFormerV2MidFusionDropPath025,
@@
     if args.model in {
         "dformerv2_mid_fusion",
+        "dformerv2_mid_fusion_dpr025",
```
