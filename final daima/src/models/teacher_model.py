"""Geometry-primary DFormerV2 teacher models."""
import torch.nn as nn
import torch

from .base_lit import BaseLitSeg
from .decoder import OfficialHamDecoder, SimpleFPNDecoder
from .dformerv2_encoder import DFormerv2_S, load_dformerv2_pretrained


class DFormerV2GeometryPrimaryTeacherSegmentor(nn.Module):
    def __init__(self, num_classes=40, dformerv2_pretrained=None):
        super().__init__()
        self.rgb_encoder = DFormerv2_S()
        if dformerv2_pretrained:
            self.pretrained_load_stats = load_dformerv2_pretrained(self.rgb_encoder, dformerv2_pretrained)
        else:
            self.pretrained_load_stats = None
        self.decoder = SimpleFPNDecoder(self.rgb_encoder.out_channels, num_classes=num_classes)

    def forward(self, rgb, depth, return_features=False):
        features = self.rgb_encoder(rgb, depth)
        logits = self.decoder(features, input_size=rgb.shape[-2:])
        if return_features:
            return logits, features
        return logits


class DFormerV2GeometryPrimaryHamDecoderSegmentor(nn.Module):
    def __init__(self, num_classes=40, dformerv2_pretrained=None):
        super().__init__()
        self.rgb_encoder = DFormerv2_S()
        if dformerv2_pretrained:
            self.pretrained_load_stats = load_dformerv2_pretrained(self.rgb_encoder, dformerv2_pretrained)
        else:
            self.pretrained_load_stats = None
        self.decoder = OfficialHamDecoder(
            self.rgb_encoder.out_channels,
            channels=512,
            num_classes=num_classes,
        )

    def forward(self, rgb, depth, return_features=False):
        features = self.rgb_encoder(rgb, depth)
        logits = self.decoder(features, input_size=rgb.shape[-2:])
        if return_features:
            return logits, features
        return logits


class LitDFormerV2GeometryPrimaryTeacher(BaseLitSeg):
    def __init__(
        self,
        num_classes=40,
        lr=1e-4,
        dformerv2_pretrained=None,
        loss_type: str = "ce",
        dice_weight: float = 0.5,
    ):
        super().__init__(num_classes=num_classes, lr=lr, loss_type=loss_type, dice_weight=dice_weight)
        self.model = DFormerV2GeometryPrimaryTeacherSegmentor(
            num_classes=num_classes,
            dformerv2_pretrained=dformerv2_pretrained,
        )

    def forward(self, rgb, depth):
        return self.model(rgb, depth)

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=0.01)


class LitDFormerV2GeometryPrimaryHamDecoder(BaseLitSeg):
    def __init__(
        self,
        num_classes=40,
        lr=1e-4,
        dformerv2_pretrained=None,
        loss_type: str = "ce",
        dice_weight: float = 0.5,
    ):
        super().__init__(num_classes=num_classes, lr=lr, loss_type=loss_type, dice_weight=dice_weight)
        self.model = DFormerV2GeometryPrimaryHamDecoderSegmentor(
            num_classes=num_classes,
            dformerv2_pretrained=dformerv2_pretrained,
        )

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=0.01)
