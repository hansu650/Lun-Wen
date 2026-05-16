# R049 Backbone SyncBN Norm-Eval Code Archive

Archived after full train because `dformerv2_backbone_syncbn_normeval` stayed below the corrected R016 baseline.

Result summary:

- best val/mIoU: `0.537890` at validation epoch `41`
- last val/mIoU: `0.517793`
- best-to-last drop: `0.020097`
- decision: negative contract diagnostic; do not keep this entry in the active registry.

Minimal code that was tested:

```python
class DFormerV2BackboneSyncBNNormEvalSegmentor(DFormerV2MidFusionSegmentor):
    def train(self, mode=True):
        super().train(mode)
        if mode:
            for module in self.rgb_encoder.modules():
                if isinstance(module, (nn.BatchNorm2d, nn.SyncBatchNorm)):
                    module.eval()
        return self
```

```python
class LitDFormerV2BackboneSyncBNNormEval(BaseLitSeg):
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
        self.model = DFormerV2BackboneSyncBNNormEvalSegmentor(
            num_classes=num_classes,
            dformerv2_pretrained=dformerv2_pretrained,
        )

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=0.01)
```

`train.py` temporary registry key:

```python
"dformerv2_backbone_syncbn_normeval": LitDFormerV2BackboneSyncBNNormEval
```

Rationale:

- The official DFormerv2 `train()` method sets `norm_eval=True` but only checks `isinstance(m, nn.BatchNorm2d)`.
- Local smoke verified that PyTorch `nn.SyncBatchNorm` is not an `nn.BatchNorm2d` subclass, so the R016 backbone `SyncBatchNorm` layers stayed in train mode.
- R049 tested whether freezing only DFormerv2 backbone `BatchNorm2d` / `SyncBatchNorm` running stats would reduce fixed-recipe late instability.

Outcome:

- The hypothesis did not hold under the local fixed recipe.
- Peak stayed below R016 `0.541121`, and late drop was worse than R016.
- Do not continue norm-freeze micro-variants unless a future diagnostic isolates a more specific normalization contract issue.
