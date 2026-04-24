import torch
import torch.nn as nn
import torch.nn.functional as F

from .base_lit import BaseLitSeg
from .decoder import SimpleFPNDecoder
from .encoder import DepthEncoder, RGBEncoder


class GatedFusion(nn.Module):
    def __init__(self, rgb_channels, depth_channels):
        super().__init__()
        self.depth_proj = nn.Conv2d(depth_channels, rgb_channels, 1)
        self.gate = nn.Sequential(
            nn.Conv2d(rgb_channels * 2, rgb_channels, 1, bias=False),
            nn.BatchNorm2d(rgb_channels),
            nn.Sigmoid(),
        )
        self.refine = nn.Sequential(
            nn.Conv2d(rgb_channels, rgb_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(rgb_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, rgb_feat, depth_feat):
        depth_feat = self.depth_proj(depth_feat)
        gate = self.gate(torch.cat([rgb_feat, depth_feat], dim=1))
        fused = gate * rgb_feat + (1.0 - gate) * depth_feat
        return self.refine(fused)


class MidFusionSegmentor(nn.Module):
    def __init__(self, num_classes=40):
        super().__init__()
        self.rgb_encoder = RGBEncoder()
        self.depth_encoder = DepthEncoder()
        # 主模型只保留四层 GatedFusion。
        self.fusions = nn.ModuleList(
            [
                GatedFusion(rgb_ch, depth_ch)
                for rgb_ch, depth_ch in zip(self.rgb_encoder.out_channels, self.depth_encoder.out_channels)
            ]
        )
        self.decoder = SimpleFPNDecoder(self.rgb_encoder.out_channels, num_classes=num_classes)

    def forward(self, rgb, depth):
        rgb_feats = self.rgb_encoder(rgb)
        depth_feats = self.depth_encoder(depth)

        aligned_depth = []
        for rgb_feat, depth_feat in zip(rgb_feats, depth_feats):
            if rgb_feat.shape[-2:] != depth_feat.shape[-2:]:
                depth_feat = F.interpolate(
                    depth_feat,
                    size=rgb_feat.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )
            aligned_depth.append(depth_feat)

        fused_feats = [fusion(rgb_feat, depth_feat) for fusion, rgb_feat, depth_feat in zip(self.fusions, rgb_feats, aligned_depth)]
        return self.decoder(fused_feats, input_size=rgb.shape[-2:])


class LitMidFusion(BaseLitSeg):
    def __init__(self, num_classes=40, lr=1e-4):
        super().__init__(num_classes=num_classes, lr=lr)
        self.model = MidFusionSegmentor(num_classes=num_classes)
