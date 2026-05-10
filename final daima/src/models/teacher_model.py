"""RGB-only DFormerV2 teacher models."""
import torch
import torch.nn as nn

from .base_lit import BaseLitSeg
from .decoder import SimpleFPNDecoder
from .dformerv2_encoder import DFormerv2_S, load_dformerv2_pretrained


class DFormerV2RGBTeacherSegmentor(nn.Module):
    def __init__(self, num_classes=40, dformerv2_pretrained=None):
        super().__init__()
        self.rgb_encoder = DFormerv2_S()
        if dformerv2_pretrained:
            self.pretrained_load_stats = load_dformerv2_pretrained(self.rgb_encoder, dformerv2_pretrained)
        else:
            self.pretrained_load_stats = None
        self.decoder = SimpleFPNDecoder(self.rgb_encoder.out_channels, num_classes=num_classes)

    def _zero_depth(self, rgb):
        return torch.zeros(
            rgb.shape[0],
            1,
            rgb.shape[2],
            rgb.shape[3],
            device=rgb.device,
            dtype=rgb.dtype,
        )

    def forward(self, rgb, return_features=False):
        depth = self._zero_depth(rgb)
        features = self.rgb_encoder(rgb, depth)
        logits = self.decoder(features, input_size=rgb.shape[-2:])
        if return_features:
            return logits, features
        return logits


class LitDFormerV2RGBTeacher(BaseLitSeg):
    def __init__(
        self,
        num_classes=40,
        lr=1e-4,
        dformerv2_pretrained=None,
        loss_type: str = "ce",
        dice_weight: float = 0.5,
    ):
        super().__init__(num_classes=num_classes, lr=lr, loss_type=loss_type, dice_weight=dice_weight)
        self.model = DFormerV2RGBTeacherSegmentor(
            num_classes=num_classes,
            dformerv2_pretrained=dformerv2_pretrained,
        )

    def forward(self, rgb, depth):
        return self.model(rgb)

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=0.01)
