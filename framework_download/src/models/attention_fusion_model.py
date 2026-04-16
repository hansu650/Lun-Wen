"""Attention fusion model adapted to the local training framework."""
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.models.base_lit import BaseLitSeg
from src.models.decoder import SimpleFPNDecoder
from src.models.encoder import DepthEncoder, RGBEncoder


class CrossModalAttentionFusion(nn.Module):
    """Channel attention + spatial attention fusion."""

    def __init__(self, rgb_channels, depth_channels, out_channels=None):
        super().__init__()
        out_channels = out_channels or rgb_channels
        self.out_channels = out_channels  # 两种模态最后都对齐到同一个通道数，后面才好做融合

        # 先把 RGB / Depth 投影到同一个特征维度。
        # 这一步对应课程 05 里的“投影对齐”。
        self.rgb_proj = nn.Sequential(
            nn.Conv2d(rgb_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.depth_proj = nn.Sequential(
            nn.Conv2d(depth_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

        hidden = max(1, out_channels // 4)  # 中间维度做压缩，避免注意力分支太重

        # 通道注意力：
        # 先做全局平均池化，再用 1x1 卷积模拟 MLP，
        # 学到“哪些通道更该信 RGB，哪些通道更该信 Depth”。
        self.channel_attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(out_channels * 2, hidden, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, out_channels * 2, kernel_size=1),
            nn.Sigmoid(),
        )

        # 空间注意力：
        # 对拼接特征做轻量卷积，输出 2 个空间权重图，
        # 分别对应 RGB 和 Depth 在每个位置上的重要性。
        self.spatial_attn = nn.Sequential(
            nn.Conv2d(out_channels * 2, hidden, kernel_size=1),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, 2, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )

        # 注意力加权之后，再用一层交互卷积做进一步融合。
        # 这样不是只靠“乘权重再相加”，而是让两种模态再交流一步。
        self.interaction = nn.Sequential(
            nn.Conv2d(out_channels * 2, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                groups=max(1, out_channels // 4),
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.out_bn = nn.BatchNorm2d(out_channels)

    def forward(self, rgb_feat, depth_feat):
        # 第一步：投影对齐到统一通道数
        rgb = self.rgb_proj(rgb_feat)
        depth = self.depth_proj(depth_feat)
        concat = torch.cat([rgb, depth], dim=1)

        # 第二步：按通道做注意力加权
        channel_weights = self.channel_attn(concat)
        rgb_weighted = rgb * channel_weights[:, : self.out_channels]
        depth_weighted = depth * channel_weights[:, self.out_channels :]

        # 第三步：按空间位置做注意力加权
        spatial_weights = self.spatial_attn(concat)
        fused = (
            spatial_weights[:, 0:1] * rgb_weighted
            + spatial_weights[:, 1:2] * depth_weighted
        )

        # 第四步：交互卷积 + 残差。
        # fused 更像“注意力直接加权结果”，refined 更像“卷积进一步整理后的结果”。
        refined = self.interaction(torch.cat([rgb_weighted, depth_weighted], dim=1))
        return self.out_bn(refined + fused)


class AttentionFusionSegmentor(nn.Module):
    def __init__(self, num_classes=40):
        super().__init__()
        # 保持和 mid_fusion 一样的双编码器主线，
        # 这样你后面对比时更容易看出“只是融合模块变了”。
        self.rgb_encoder = RGBEncoder()
        self.depth_encoder = DepthEncoder()
        self.fusions = nn.ModuleList(
            [
                CrossModalAttentionFusion(rgb_ch, depth_ch)
                for rgb_ch, depth_ch in zip(
                    self.rgb_encoder.out_channels,
                    self.depth_encoder.out_channels,
                )
            ]
        )
        self.decoder = SimpleFPNDecoder(
            self.rgb_encoder.out_channels,
            num_classes=num_classes,
        )

    def forward(self, rgb, depth):
        # 先分别提特征，这里和 mid_fusion 一样
        rgb_feats = self.rgb_encoder(rgb)
        depth_feats = self.depth_encoder(depth)

        aligned_depth = []
        for rgb_feat, depth_feat in zip(rgb_feats, depth_feats):
            # 双编码器有时不同层的空间尺寸不完全一致，
            # 所以先把 depth 特征对齐到 RGB 特征的空间大小。
            if rgb_feat.shape[-2:] != depth_feat.shape[-2:]:
                depth_feat = F.interpolate(
                    depth_feat,
                    size=rgb_feat.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )
            aligned_depth.append(depth_feat)

        # 每个尺度都用一个 attention fusion 模块做融合，
        # 这和课程里“多尺度都可以融”的想法是对齐的。
        fused_feats = [
            fusion(rgb_feat, depth_feat)
            for fusion, rgb_feat, depth_feat in zip(
                self.fusions, rgb_feats, aligned_depth
            )
        ]
        return self.decoder(fused_feats, input_size=rgb.shape[-2:])


class LitAttentionFusion(BaseLitSeg):
    def __init__(self, num_classes=40, lr=1e-4):
        super().__init__(num_classes=num_classes, lr=lr)
        # 训练/验证逻辑直接复用你现在已经跑通的 BaseLitSeg，
        # 这样 attention 模型就自动接进 train/eval/infer 主线了。
        self.model = AttentionFusionSegmentor(num_classes=num_classes)
