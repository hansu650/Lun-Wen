"""Early Fusion 模型"""
import torch
import torch.nn as nn

from .encoder import EarlyFusionEncoder
from .decoder import SimpleFPNDecoder
from .base_lit import BaseLitSeg


class EarlyFusionSegmentor(nn.Module):
    def __init__(self, num_classes=40):
        super().__init__()
        self.encoder = EarlyFusionEncoder()
        self.decoder = SimpleFPNDecoder(self.encoder.out_channels, num_classes=num_classes)
    
    def forward(self, rgb, depth):
        x = torch.cat([rgb, depth], dim=1)
        feats = self.encoder(x)
        return self.decoder(feats, input_size=rgb.shape[-2:])


class LitEarlyFusion(BaseLitSeg):
    def __init__(self, num_classes=40, lr=1e-4, loss_type: str = "ce", dice_weight: float = 0.5):
        super().__init__(num_classes=num_classes, lr=lr, loss_type=loss_type, dice_weight=dice_weight)
        self.model = EarlyFusionSegmentor(num_classes=num_classes)
