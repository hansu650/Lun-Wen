"""Early Fusion 模型"""
import torch
import torch.nn as nn

from .encoder import EarlyFusionEncoder # 编码器
from .decoder import SimpleFPNDecoder # 解码器
from .base_lit import BaseLitSeg
# 统一训练骨架

class EarlyFusionSegmentor(nn.Module):
    def __init__(self, num_classes=40):
        super().__init__()
        self.encoder = EarlyFusionEncoder()
        self.decoder = SimpleFPNDecoder(self.encoder.out_channels, num_classes=num_classes)
    # 前向传播
    def forward(self, rgb, depth):
        x = torch.cat([rgb, depth], dim=1)# 直接拼接,拼到通道里面所以dim=1,b,c,h,w 取c
        feats = self.encoder(x)# 特征
        return self.decoder(feats, input_size=rgb.shape[-2:])
# 恢复，但是H,W可能不一样，所以需要输入size

class LitEarlyFusion(BaseLitSeg):
    def __init__(self, num_classes=40, lr=1e-4):
        super().__init__(num_classes=num_classes, lr=lr)
        self.model = EarlyFusionSegmentor(num_classes=num_classes)
